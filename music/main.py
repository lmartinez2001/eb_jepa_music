import copy
import importlib
import os
from pathlib import Path
from time import time
from typing import Any

import fire
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_config,
    log_config,
    log_epoch,
    log_model_info,
    setup_device,
    setup_seed,
    setup_wandb,
)
from music.models.encoder import DSTformer
from music.models.predictor import MusicRNNPredictor

logger = get_logger(__name__)


def _import_symbol(path: str):
    module_name, symbol_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), symbol_name)


def _call_builder(builder: Any, cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
    try:
        return builder(**cfg_dict)
    except TypeError:
        return builder(cfg)


def build_dataloaders(cfg):
    """Build train/val loaders from a user-provided callable.

    Expected config:

    data:
      loader: "some.module.build_loaders"

    The callable may return either ``(train_loader, val_loader)`` or a dict with
    ``train`` and optional ``val`` keys.
    """
    if not cfg.data.get("loader"):
        raise ValueError(
            "cfg.data.loader must point to a callable returning train/val loaders"
        )

    result = _call_builder(_import_symbol(cfg.data.loader), cfg.data)
    if isinstance(result, dict):
        return result["train"], result.get("val")
    if isinstance(result, (tuple, list)) and len(result) in (1, 2):
        train_loader = result[0]
        val_loader = result[1] if len(result) == 2 else None
        return train_loader, val_loader
    raise ValueError("data.loader must return (train_loader, val_loader) or a dict")


def build_music_encoder(cfg):
    """Build the music encoder from ``music.video_encoder``.

    By default this expects ``music/video_encoder.py`` to expose
    ``build_music_encoder(cfg)``. Set ``model.music_encoder.target`` to use a
    different callable.
    """
    music_cfg = cfg.model.get("music_encoder", {})
    target = music_cfg.get("target")
    if target:
        return _call_builder(_import_symbol(target), music_cfg)

    module = importlib.import_module("music.video_encoder")
    if not hasattr(module, "build_music_encoder"):
        raise AttributeError(
            "music.video_encoder must define build_music_encoder(cfg), or set "
            "model.music_encoder.target in the config"
        )
    return module.build_music_encoder(music_cfg)


def build_keypoint_encoder(cfg):
    enc_cfg = cfg.model.encoder
    return DSTformer(
        dim_in=enc_cfg.get("dim_in", 3),
        dim_out=enc_cfg.get("dim_out", 3),
        dim_feat=enc_cfg.get("dim_feat", 512),
        dim_rep=enc_cfg.get("dim_rep", 512),
        depth=enc_cfg.get("depth", 5),
        num_heads=enc_cfg.get("num_heads", 8),
        mlp_ratio=enc_cfg.get("mlp_ratio", 2),
        num_joints=enc_cfg.get("num_joints", 17),
        maxlen=enc_cfg.get("maxlen", 243),
        drop_rate=enc_cfg.get("dropout", 0.0),
        attn_drop_rate=enc_cfg.get("attn_dropout", 0.0),
        drop_path_rate=enc_cfg.get("drop_path", 0.0),
        att_fuse=enc_cfg.get("att_fuse", True),
    )


def _get_first(batch, names):
    for name in names:
        if isinstance(batch, dict) and name in batch:
            return batch[name]
    return None


def unpack_batch(batch, cfg):
    """Return ``x_t, x_next, music`` from a dance/music batch.

    Supported batch formats:
    - dict with ``x_t``/``x_next`` and ``music`` or ``music_emb``
    - dict with ``keypoints`` sequence ``[B, T, J, C]`` plus music
    - tuple/list ``(keypoints, music, ...)``

    If only a keypoint sequence is provided, the two MotionBERT windows are:
    ``keypoints[:, :F]`` and ``keypoints[:, horizon:horizon+F]``.
    """
    if isinstance(batch, (tuple, list)):
        keypoints = batch[0]
        music = batch[1]
        batch = {"keypoints": keypoints, "music": music}

    if not isinstance(batch, dict):
        raise TypeError("batch must be a dict or a tuple/list")

    x_t = _get_first(batch, ("x_t", "state", "current", "keypoints_t"))
    x_next = _get_first(batch, ("x_next", "next_state", "target", "keypoints_next"))
    music = _get_first(batch, ("music_emb", "music_embedding", "music", "audio"))

    if x_t is None or x_next is None:
        keypoints = _get_first(batch, ("keypoints", "poses", "dance", "x"))
        if keypoints is None:
            raise KeyError(
                "batch must contain x_t/x_next or a keypoints sequence"
            )
        window = cfg.data.get("window", cfg.model.encoder.get("maxlen", 243))
        horizon = cfg.data.get("horizon", 1)
        if keypoints.shape[1] < window + horizon:
            raise ValueError(
                f"keypoints needs at least window+horizon frames "
                f"({window + horizon}), got {keypoints.shape[1]}"
            )
        x_t = keypoints[:, :window]
        x_next = keypoints[:, horizon : horizon + window]

    if music is None:
        raise KeyError("batch must contain music/audio/music_emb")

    return x_t, x_next, music


def encode_music(music_encoder, music, cfg):
    if cfg.model.get("music_encoder", {}).get("precomputed", False):
        emb = music
    else:
        emb = music_encoder(music)

    if isinstance(emb, (tuple, list)):
        emb = emb[0]
    if isinstance(emb, dict):
        emb = emb.get("embedding", emb.get("embeddings", emb.get("last_hidden_state")))
    if emb is None:
        raise ValueError("music encoder returned no usable embedding")

    if emb.ndim == 3:
        reduce = cfg.model.get("music_encoder", {}).get("pool", "mean")
        if reduce == "last":
            emb = emb[:, -1]
        elif reduce == "mean":
            emb = emb.mean(dim=1)
        else:
            raise ValueError(f"Unknown music_encoder.pool={reduce}")
    return emb


def cls_state(encoder, x):
    z = encoder(x, return_rep=True)
    return z[:, -1, 0]


def variance_covariance_loss(z, std_coeff=0.0, cov_coeff=0.0, eps=1e-4):
    if std_coeff == 0 and cov_coeff == 0:
        return z.new_tensor(0.0), {}

    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1.0 - std))

    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    cov = cov - torch.diag(torch.diag(cov))
    cov_loss = cov.pow(2).sum() / d

    loss = std_coeff * std_loss + cov_coeff * cov_loss
    return loss, {"std_loss": std_loss.detach(), "cov_loss": cov_loss.detach()}


@torch.no_grad()
def update_ema(target, online, momentum):
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.mul_(momentum).add_(online_param.detach(), alpha=1.0 - momentum)


def run(
    fname: str = "music/cfgs/train.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """Train a music-conditioned JEPA on dance keypoint windows.

    The model predicts the target encoder's next-window CLS latent:

    ``DSTformer(x_t)[:, -1, 0] + music_embedding -> DSTformer_target(x_t+1)[:, -1, 0]``.
    """
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    if folder is None:
        if cfg.meta.get("model_folder"):
            folder = Path(cfg.meta.model_folder)
            folder_name = folder.name
            exp_name = folder_name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("music_jepa", cfg)
            folder = get_unified_experiment_dir(
                example_name="music_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        folder = Path(folder)
        folder_name = folder.name
        exp_name = folder_name.rsplit("_seed", 1)[0]
    os.makedirs(folder, exist_ok=True)

    device = setup_device(cfg.meta.get("device", "auto"))
    setup_seed(cfg.meta.seed)

    train_loader, val_loader = build_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    total_steps = cfg.optim.epochs * steps_per_epoch

    wandb_run = setup_wandb(
        project=cfg.logging.get("project", "eb_jepa"),
        config={"example": "music_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=folder,
        run_name=exp_name,
        tags=[f"seed_{cfg.meta.seed}", "music_jepa"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    config_path = folder / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Saved complete config to {config_path}")

    encoder = build_keypoint_encoder(cfg).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    music_encoder = build_music_encoder(cfg).to(device)
    if cfg.model.get("music_encoder", {}).get("freeze", True):
        music_encoder.eval()
        for p in music_encoder.parameters():
            p.requires_grad_(False)

    predictor = MusicRNNPredictor(
        state_dim=cfg.model.encoder.get("dim_rep", 512),
        music_dim=cfg.model.music_encoder.dim,
        num_layers=cfg.model.predictor.get("num_layers", 1),
    ).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    if not cfg.model.get("music_encoder", {}).get("freeze", True):
        params += list(music_encoder.parameters())

    optimizer = AdamW(
        params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-6),
    )
    scheduler = CosineWithWarmup(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=cfg.optim.get("warmup_ratio", 0.1),
    )

    log_model_info(
        predictor,
        {
            "encoder": sum(p.numel() for p in encoder.parameters()),
            "target_encoder": sum(p.numel() for p in target_encoder.parameters()),
            "music_encoder": sum(p.numel() for p in music_encoder.parameters()),
            "predictor": sum(p.numel() for p in predictor.parameters()),
        },
    )
    log_config(cfg)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    use_amp = cfg.training.get("use_amp", True)
    scaler = GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    global_step = 0
    if cfg.meta.get("load_model", False):
        checkpoint = torch.load(
            folder / cfg.meta.get("load_checkpoint", "latest.pth.tar"),
            map_location=device,
            weights_only=False,
        )
        encoder.load_state_dict(checkpoint["encoder"])
        target_encoder.load_state_dict(checkpoint["target_encoder"])
        predictor.load_state_dict(checkpoint["predictor"])
        if "music_encoder" in checkpoint:
            music_encoder.load_state_dict(checkpoint["music_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("step", 0)

    latest_ckpt_path = folder / "latest.pth.tar"
    ema_momentum = cfg.model.get("ema_momentum", 0.99)

    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start = time()
        encoder.train()
        predictor.train()
        if cfg.model.get("music_encoder", {}).get("freeze", True):
            music_encoder.eval()
        else:
            music_encoder.train()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.get("tqdm_silent", False),
        )
        last_logs = {}
        for batch in pbar:
            x_t, x_next, music = unpack_batch(batch, cfg)
            x_t = x_t.to(device, non_blocking=True)
            x_next = x_next.to(device, non_blocking=True)
            music = music.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                z_t = cls_state(encoder, x_t)
                music_emb = encode_music(music_encoder, music, cfg)
                z_pred = predictor(z_t, music_emb)

                with torch.no_grad():
                    z_target = cls_state(target_encoder, x_next)

                pred_loss = F.smooth_l1_loss(z_pred, z_target)
                vc_loss, vc_logs = variance_covariance_loss(
                    torch.cat([z_t, z_target], dim=0).float(),
                    std_coeff=cfg.loss.get("std_coeff", 0.0),
                    cov_coeff=cfg.loss.get("cov_coeff", 0.0),
                )
                loss = pred_loss + vc_loss

            scaler.scale(loss).backward()
            if cfg.optim.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema(target_encoder, encoder, ema_momentum)

            global_step += 1
            last_logs = {
                "loss": loss.detach(),
                "pred_loss": pred_loss.detach(),
                "vc_loss": vc_loss.detach(),
                **vc_logs,
            }
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "pred": f"{pred_loss.item():.4f}",
                    "vc": f"{vc_loss.item():.4f}",
                }
            )

            if wandb_run and global_step % cfg.logging.get("log_every", 100) == 0:
                import wandb

                wandb.log(
                    {f"train/{k}": float(v) for k, v in last_logs.items()}
                    | {"global_step": global_step},
                    step=global_step,
                )

        val_logs = {}
        if val_loader is not None and epoch % cfg.logging.get("val_every", 1) == 0:
            val_logs = validate(val_loader, encoder, target_encoder, music_encoder, predictor, cfg, device)
            if wandb_run:
                import wandb

                wandb.log(val_logs | {"global_step": global_step}, step=global_step)

        log_epoch(
            epoch,
            {
                "loss": float(last_logs.get("loss", torch.tensor(0.0))),
                "pred": float(last_logs.get("pred_loss", torch.tensor(0.0))),
                "val_pred": val_logs.get("val/pred_loss", 0.0),
                "time": time() - epoch_start,
            },
            total_epochs=cfg.optim.epochs,
        )

        torch.save(
            {
                "encoder": encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "music_encoder": music_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
            },
            latest_ckpt_path,
        )
        if epoch % cfg.logging.get("save_every", 10) == 0 and epoch > 0:
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "target_encoder": target_encoder.state_dict(),
                    "music_encoder": music_encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "step": global_step,
                },
                folder / f"epoch_{epoch}.pth.tar",
            )

    return {
        "folder": str(folder),
        "latest_checkpoint": str(latest_ckpt_path),
        "global_step": global_step,
    }


@torch.no_grad()
def validate(loader, encoder, target_encoder, music_encoder, predictor, cfg, device):
    encoder.eval()
    target_encoder.eval()
    music_encoder.eval()
    predictor.eval()
    losses = []
    for batch in loader:
        x_t, x_next, music = unpack_batch(batch, cfg)
        x_t = x_t.to(device, non_blocking=True)
        x_next = x_next.to(device, non_blocking=True)
        music = music.to(device, non_blocking=True)
        z_t = cls_state(encoder, x_t)
        music_emb = encode_music(music_encoder, music, cfg)
        z_pred = predictor(z_t, music_emb)
        z_target = cls_state(target_encoder, x_next)
        losses.append(F.smooth_l1_loss(z_pred, z_target).item())
    mean_loss = sum(losses) / max(len(losses), 1)
    return {"val/pred_loss": mean_loss}


if __name__ == "__main__":
    fire.Fire(run)
