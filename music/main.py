import importlib
import os
from pathlib import Path
from time import time
from typing import Any

import fire
import torch
import torch.nn.functional as F
import wandb
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
from music.models.audio_encoder import AudioEncoder
from music.models.encoder import DSTformer
from music.models.predictor import MusicRNNPredictor

logger = get_logger(__name__)


def _import_symbol(path: str):
    module_name, symbol_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), symbol_name)


def _call_builder(builder: Any, cfg):
    return builder(**OmegaConf.to_container(cfg, resolve=True))


def build_dataloaders(cfg):
    """Build train/val loaders from a user-provided callable.

    Expected config:

    data:
      loader: "some.module.build_loaders"

    The callable may return either ``(train_loader, val_loader)`` or a dict with
    ``train`` and optional ``val`` keys.
    """
    result = _call_builder(_import_symbol(cfg.data.loader), cfg.data)
    return result["train"], result.get("val")


def build_music_encoder(cfg):
    """Build the music/audio encoder.

    By default this uses ``music.models.audio_encoder.AudioEncoder``. Set
    ``model.music_encoder.target`` to point to a custom callable if needed.
    """
    music_cfg = cfg.model.get("music_encoder", {})
    target = music_cfg.get("target")
    if target:
        return _call_builder(_import_symbol(target), music_cfg)

    return AudioEncoder(
        model_name=music_cfg.model_name,
        embed_dim=music_cfg.dim,
        chunk_frames=music_cfg.chunk_frames,
        fps=music_cfg.fps,
        sample_rate=music_cfg.sample_rate,
    )


def build_keypoint_encoder(cfg):
    enc_cfg = cfg.model.encoder
    return DSTformer(
        dim_in=enc_cfg.dim_in,
        dim_out=enc_cfg.dim_out,
        dim_feat=enc_cfg.dim_feat,
        dim_rep=enc_cfg.dim_rep,
        depth=enc_cfg.depth,
        num_heads=enc_cfg.num_heads,
        mlp_ratio=enc_cfg.mlp_ratio,
        num_joints=enc_cfg.num_joints,
        maxlen=enc_cfg.maxlen,
        drop_rate=enc_cfg.dropout,
        attn_drop_rate=enc_cfg.attn_dropout,
        drop_path_rate=enc_cfg.drop_path,
        att_fuse=enc_cfg.att_fuse,
    )


def unpack_batch(batch, cfg):
    """Return ``x_t, x_next, music`` from a dance/music batch.

    Expected batch keys are ``keypoints`` and ``music``. The two MotionBERT
    windows are:
    ``keypoints[:, :F]`` and ``keypoints[:, horizon:horizon+F]``.
    """
    keypoints = batch["keypoints"]
    music = batch["music"]
    window = cfg.data.window
    horizon = cfg.data.horizon
    x_t = keypoints[:, :window]
    x_next = keypoints[:, horizon : horizon + window]

    _validate_keypoint_window("x_t", x_t, cfg)
    _validate_keypoint_window("x_next", x_next, cfg)
    return x_t, x_next, music


def _validate_keypoint_window(name, x, cfg):
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape [B, F, J, C], got {tuple(x.shape)}")
    enc_cfg = cfg.model.encoder
    expected_f = cfg.data.window
    expected_j = enc_cfg.num_joints
    expected_c = enc_cfg.dim_in
    if x.shape[1] != expected_f:
        raise ValueError(f"{name} frame dim must be {expected_f}, got {x.shape[1]}")
    if x.shape[2] != expected_j:
        raise ValueError(f"{name} joint dim must be {expected_j}, got {x.shape[2]}")
    if x.shape[3] != expected_c:
        raise ValueError(f"{name} channel dim must be {expected_c}, got {x.shape[3]}")


def encode_music(music_encoder, music, cfg):
    emb = music_encoder(music)

    if emb.ndim == 3:
        reduce = cfg.model.music_encoder.pool
        if reduce == "last":
            emb = emb[:, -1]
        else:
            emb = emb.mean(dim=1)
    expected_dim = cfg.model.music_encoder.dim
    if emb.ndim != 2:
        raise ValueError(f"music embedding must have shape [B, M], got {tuple(emb.shape)}")
    if emb.shape[-1] != expected_dim:
        raise ValueError(
            f"music embedding dim must match model.music_encoder.dim={expected_dim}, "
            f"got {emb.shape[-1]}"
        )
    return emb


def cls_state(encoder, x):
    z = encoder(x, return_rep=True)
    if z.ndim != 4:
        raise ValueError(f"encoder representation must be [B, F, J+1, D], got {tuple(z.shape)}")
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


def run(
    fname: str = "music/cfgs/train.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """Train a music-conditioned JEPA on dance keypoint windows.

    The model predicts the stop-gradient next-window CLS latent:

    ``DSTformer(x_t)[:, -1, 0] + music_embedding -> stopgrad(DSTformer(x_t+1)[:, -1, 0])``.
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
    music_encoder = build_music_encoder(cfg).to(device)

    predictor = MusicRNNPredictor(
        state_dim=cfg.model.encoder.dim_rep,
        music_dim=cfg.model.music_encoder.dim,
        num_layers=cfg.model.predictor.num_layers,
        final_ln=torch.nn.LayerNorm(cfg.model.encoder.dim_rep)
        if cfg.model.predictor.final_ln
        else None,
    ).to(device)

    params = list(encoder.parameters()) + list(predictor.parameters())
    if not cfg.model.get("music_encoder", {}).get("freeze", True):
        params += [p for p in music_encoder.parameters() if p.requires_grad]

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
        predictor.load_state_dict(checkpoint["predictor"])
        if "music_encoder" in checkpoint:
            music_encoder.load_state_dict(checkpoint["music_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("step", 0)

    latest_ckpt_path = folder / "latest.pth.tar"

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
                    z_target = cls_state(encoder, x_next)

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
                wandb.log(
                    {f"train/{k}": float(v) for k, v in last_logs.items()}
                    | {"global_step": global_step},
                    step=global_step,
                )

        val_logs = {}
        if val_loader is not None and epoch % cfg.logging.get("val_every", 1) == 0:
            val_logs = validate(val_loader, encoder, music_encoder, predictor, cfg, device)
            if wandb_run:
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
def validate(loader, encoder, music_encoder, predictor, cfg, device):
    encoder.eval()
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
        z_target = cls_state(encoder, x_next)
        losses.append(F.smooth_l1_loss(z_pred, z_target).item())
    mean_loss = sum(losses) / max(len(losses), 1)
    return {"val/pred_loss": mean_loss, "val/score": -mean_loss}


if __name__ == "__main__":
    fire.Fire(run)
