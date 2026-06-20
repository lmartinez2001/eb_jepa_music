import importlib
import os
from pathlib import Path
from time import time
from typing import Any

import fire
import torch
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
from music.models.motion_decoder import build_motion_decoder

logger = get_logger(__name__)


def _import_symbol(path: str):
    module_name, symbol_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), symbol_name)


def _call_builder(builder: Any, cfg):
    return builder(**OmegaConf.to_container(cfg, resolve=True))


def build_dataloaders(cfg):
    loaders = _call_builder(_import_symbol(cfg.data.loader), cfg.data)
    return loaders["train"], loaders.get("val")


def unpack_batch(batch):
    """Return clean motion clips and conditioning latents.

    The decoder dataset is expected to return:
      poses     : [B, F, J, 3]
      embedding : [B, cond_dim]
    """
    return batch["poses"], batch["embedding"]


@torch.no_grad()
def validate(loader, decoder, device):
    decoder.eval()
    losses = []
    for batch in loader:
        x, z = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)
        loss, _ = decoder.compute_loss(x, z)
        losses.append(loss.item())
    mean_loss = sum(losses) / max(len(losses), 1)
    return {"val/fm_loss": mean_loss}


def run(
    fname: str = "music/cfgs/train_decoder.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """Train the flow-matching motion decoder.

    The dataloader is assumed to provide clean pose clips and their matching
    JEPA/dance latents. It does not build those latents here.
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
            exp_name = get_exp_name("music_decoder", cfg)
            folder = get_unified_experiment_dir(
                example_name="music_decoder",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        folder = Path(folder)
        folder_name = folder.name
        exp_name = folder_name.rsplit("_seed", 1)[0]
    os.makedirs(folder, exist_ok=True)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    train_loader, val_loader = build_dataloaders(cfg)
    total_steps = cfg.optim.epochs * len(train_loader)

    wandb_run = setup_wandb(
        project=cfg.logging.project,
        config={"example": "music_decoder", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=folder,
        run_name=exp_name,
        tags=[f"seed_{cfg.meta.seed}", "music_decoder"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    config_path = folder / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Saved complete config to {config_path}")

    decoder = build_motion_decoder(
        num_frames=cfg.model.decoder.num_frames,
        num_joints=cfg.model.decoder.num_joints,
        cond_dim=cfg.model.decoder.cond_dim,
        dim=cfg.model.decoder.dim,
        depth=cfg.model.decoder.depth,
        num_heads=cfg.model.decoder.num_heads,
        mlp_ratio=cfg.model.decoder.mlp_ratio,
        t_dim=cfg.model.decoder.t_dim,
    ).to(device)

    optimizer = AdamW(
        decoder.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = CosineWithWarmup(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=cfg.optim.warmup_ratio,
    )

    log_model_info(
        decoder,
        {"decoder": sum(p.numel() for p in decoder.parameters())},
    )
    log_config(cfg)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[cfg.training.dtype]
    scaler = GradScaler(device.type, enabled=cfg.training.use_amp)

    start_epoch = 0
    global_step = 0
    if cfg.meta.load_model:
        checkpoint = torch.load(
            folder / cfg.meta.load_checkpoint,
            map_location=device,
            weights_only=False,
        )
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]

    latest_ckpt_path = folder / "latest.pth.tar"

    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start = time()
        decoder.train()
        last_logs = {}

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.tqdm_silent,
        )
        for batch in pbar:
            x, z = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=cfg.training.use_amp, dtype=dtype):
                loss, logs = decoder.compute_loss(x, z)

            scaler.scale(loss).backward()
            if cfg.optim.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            last_logs = {"loss": loss.detach(), **logs}
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if wandb_run and global_step % cfg.logging.log_every == 0:
                wandb.log(
                    {f"train/{k}": float(v) for k, v in last_logs.items()}
                    | {"global_step": global_step},
                    step=global_step,
                )

        val_logs = {}
        if val_loader is not None and epoch % cfg.logging.val_every == 0:
            val_logs = validate(val_loader, decoder, device)
            if wandb_run:
                wandb.log(val_logs | {"global_step": global_step}, step=global_step)

        log_epoch(
            epoch,
            {
                "loss": float(last_logs.get("loss", torch.tensor(0.0))),
                "val_fm": val_logs.get("val/fm_loss", 0.0),
                "time": time() - epoch_start,
            },
            total_epochs=cfg.optim.epochs,
        )

        checkpoint = {
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
        }
        torch.save(checkpoint, latest_ckpt_path)
        if epoch % cfg.logging.save_every == 0 and epoch > 0:
            torch.save(checkpoint, folder / f"epoch_{epoch}.pth.tar")

    return {
        "folder": str(folder),
        "latest_checkpoint": str(latest_ckpt_path),
        "global_step": global_step,
    }


if __name__ == "__main__":
    fire.Fire(run)
