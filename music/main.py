# `python music/main.py` puts music/ into sys.path[0], shadowing third-party
# packages (e.g. HuggingFace `datasets` ← music/datasets/).  Strip it upfront.
import sys as _sys, pathlib as _pathlib
_here = _pathlib.Path(__file__).parent.resolve()
_sys.path = [p for p in _sys.path if _pathlib.Path(p).resolve() != _here]
del _sys, _pathlib, _here

import importlib
import os
from pathlib import Path
from time import time
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.losses import BCS
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


@torch.no_grad()
def monitor_collapse(loader, encoder, cfg, device, folder: Path, epoch: int, wandb_run=None):
    """Collect z_t embeddings from the val set and produce collapse diagnostics.

    Saves three figures (per-dim std, PCA explained variance, 2D scatter) to
    folder/collapse/ and logs them + effective_rank to W&B when available.
    """
    encoder.eval()
    zs = []
    for batch in loader:
        x_t, _, _ = unpack_batch(batch, cfg)
        x_t = x_t.to(device, non_blocking=True)
        z = cls_state(encoder, x_t).float().cpu()
        zs.append(z)
        if len(zs) * z.shape[0] >= 2048:  # cap at 2048 samples for speed
            break
    Z = torch.cat(zs, dim=0).numpy()  # [N, D]
    Zc = Z - Z.mean(0)                # centred

    # --- effective rank (scale-invariant) -------------------------
    # Uses entropy of normalised singular values (Roy & Vetterli 2007).
    # Independent of the magnitude of Z — measures dimensionality of the cloud.
    sv = np.linalg.svd(Zc, compute_uv=False)
    sv_norm = sv / (sv.sum() + 1e-9)
    entropy = -(sv_norm * np.log(sv_norm + 1e-9)).sum()
    effective_rank = float(np.exp(entropy))

    # --- raw per-dimension std (scale-sensitive) ------------------
    # Near zero when the cloud is small in magnitude even if well-structured.
    # Useful for checking that VICReg's std hinge (target ≥ 1) is satisfied.
    raw_stds = Zc.std(axis=0)

    # --- normalised per-dimension std (structure, scale-free) -----
    # L2-normalise each sample so ||z_i||=1, then measure per-dim std.
    # This isolates the *shape* of the cloud from its scale.
    # Target: mean_std_norm ≈ 1/sqrt(D) ≈ 0.044 for D=512 when fully spread,
    # and → 0 when collapsed to a single direction.
    norms = np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-9)
    Znorm = Z / norms
    norm_stds = Znorm.std(axis=0)

    idx = np.argsort(raw_stds)[::-1]

    # --- per-dim std bar chart ------------------------------------
    fig_std, axes = plt.subplots(1, 2, figsize=(14, 3))
    axes[0].bar(np.arange(len(raw_stds)), raw_stds[idx], width=1.0)
    axes[0].set_xlabel("dimension (sorted by std)")
    axes[0].set_ylabel("raw std")
    axes[0].set_title(f"Raw per-dim std  (mean={raw_stds.mean():.3f}, min={raw_stds.min():.3f})")
    axes[1].bar(np.arange(len(norm_stds)), norm_stds[np.argsort(norm_stds)[::-1]], width=1.0)
    axes[1].axhline(1 / np.sqrt(Z.shape[1]), color="r", linestyle="--", label="uniform spread")
    axes[1].set_xlabel("dimension (sorted by std)")
    axes[1].set_ylabel("std on unit-sphere")
    axes[1].set_title(f"Normalised per-dim std  (mean={norm_stds.mean():.4f}, min={norm_stds.min():.4f})")
    axes[1].legend()
    fig_std.suptitle(f"Epoch {epoch} — effective rank={effective_rank:.1f} / {Z.shape[1]}")
    fig_std.tight_layout()

    # --- PCA cumulative explained variance ------------------------
    cumvar = np.cumsum(sv**2) / (np.sum(sv**2) + 1e-9)
    fig_pca, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar)
    ax.axhline(0.95, color="r", linestyle="--", label="95%")
    ax.set_xlabel("number of PCA components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"Epoch {epoch} — PCA explained variance")
    ax.legend()
    fig_pca.tight_layout()

    # --- 2D PCA scatter -------------------------------------------
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    Z2 = Zc @ Vt[:2].T  # [N, 2]
    fig_scatter, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Z2[:, 0], Z2[:, 1], s=4, alpha=0.4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Epoch {epoch} — 2D PCA scatter  (N={len(Z)})")
    fig_scatter.tight_layout()

    # --- save / log -----------------------------------------------
    out_dir = folder / "collapse"
    out_dir.mkdir(exist_ok=True)
    fig_std.savefig(out_dir / f"std_epoch{epoch:04d}.png", dpi=100)
    fig_pca.savefig(out_dir / f"pca_epoch{epoch:04d}.png", dpi=100)
    fig_scatter.savefig(out_dir / f"scatter_epoch{epoch:04d}.png", dpi=100)
    plt.close("all")

    logs = {
        "collapse/effective_rank": effective_rank,
        "collapse/mean_std_raw": float(raw_stds.mean()),
        "collapse/min_std_raw": float(raw_stds.min()),
        "collapse/mean_std_norm": float(norm_stds.mean()),
        "collapse/min_std_norm": float(norm_stds.min()),
    }
    if wandb_run:
        wandb.log(logs | {
            "collapse/std_chart": wandb.Image(str(out_dir / f"std_epoch{epoch:04d}.png")),
            "collapse/pca_chart": wandb.Image(str(out_dir / f"pca_epoch{epoch:04d}.png")),
            "collapse/scatter":   wandb.Image(str(out_dir / f"scatter_epoch{epoch:04d}.png")),
        })

    return logs


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
    save_dir = Path(cfg.meta.save_dir) if cfg.meta.get("save_dir") else folder
    save_dir.mkdir(parents=True, exist_ok=True)

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

    bcs = BCS(
        num_slices=cfg.loss.get("bcs_slices", 256),
        lmbd=cfg.loss.get("bcs_coeff", 10.0),
    ).to(device)

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
    scaler = GradScaler(device.type, enabled=use_amp and dtype == torch.float16)

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

    latest_ckpt_path = save_dir / "latest.pth.tar"
    best_ckpt_path = save_dir / "best.pth.tar"
    best_val_loss = float("inf")

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
                loss_type = cfg.loss.get("type", "vicreg")
                if loss_type == "sigreg":
                    bcs_out = bcs(z_t.float(), z_target.float())
                    reg_loss = bcs_out["bcs_loss"]
                    vc_logs = {"bcs_loss": bcs_out["bcs_loss"].detach()}
                else:
                    reg_loss, vc_logs = variance_covariance_loss(
                        torch.cat([z_t, z_target], dim=0).float(),
                        std_coeff=cfg.loss.get("std_coeff", 0.0),
                        cov_coeff=cfg.loss.get("cov_coeff", 0.0),
                    )
                loss = pred_loss + reg_loss

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
                "reg_loss": reg_loss.detach(),
                **vc_logs,
            }
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "pred": f"{pred_loss.item():.4f}",
                    "reg": f"{reg_loss.item():.4f}",
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
            collapse_logs = monitor_collapse(val_loader, encoder, cfg, device, folder, epoch, wandb_run)
            val_logs |= collapse_logs
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

        ckpt_state = {
            "encoder": encoder.state_dict(),
            "music_encoder": music_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
        }
        torch.save(ckpt_state, latest_ckpt_path)

        val_pred = val_logs.get("val/pred_loss")
        if val_pred is not None and val_pred < best_val_loss:
            best_val_loss = val_pred
            torch.save(ckpt_state, best_ckpt_path)
            logger.info(f"New best checkpoint at epoch {epoch} (val/pred_loss={val_pred:.4f})")

        if epoch % cfg.logging.get("save_every", 10) == 0 and epoch > 0:
            torch.save(ckpt_state, folder / f"epoch_{epoch}.pth.tar")

    return {
        "folder": str(folder),
        "save_dir": str(save_dir),
        "latest_checkpoint": str(latest_ckpt_path),
        "best_checkpoint": str(best_ckpt_path),
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
    return {"val/pred_loss": mean_loss}


if __name__ == "__main__":
    fire.Fire(run)
