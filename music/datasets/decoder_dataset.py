from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eb_jepa.training_utils import load_config, setup_device, setup_seed
from music.main import (
    build_dataloaders as build_source_dataloaders,
    build_keypoint_encoder,
    build_music_encoder,
    cls_state,
    encode_music,
    unpack_batch,
)
from music.models.predictor import MusicRNNPredictor


class CachedDecoderDataset(Dataset):
    """Shard-backed dataset for diffusion decoder training.

    Expected cache layout:

      cache_dir/
        train/
          manifest.pt
          shard_000000.pt
          ...
        val/
          manifest.pt
          shard_000000.pt
          ...

    Each shard contains:
      poses     : [S, F, J, 3]
      embedding : [S, D]
    """

    def __init__(self, cache_dir: str | Path, split: str = "train"):
        self.split_dir = Path(cache_dir) / split
        manifest = torch.load(self.split_dir / "manifest.pt", map_location="cpu", weights_only=False)
        self.shards = manifest["shards"]

        self.index = []
        for shard_idx, shard in enumerate(self.shards):
            self.index.extend((shard_idx, sample_idx) for sample_idx in range(shard["num_samples"]))

        self._cached_shard_idx = None
        self._cached_shard = None

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, shard_idx: int) -> dict[str, torch.Tensor]:
        if self._cached_shard_idx != shard_idx:
            path = self.split_dir / self.shards[shard_idx]["file"]
            self._cached_shard = torch.load(path, map_location="cpu", weights_only=False)
            self._cached_shard_idx = shard_idx
        return self._cached_shard

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_idx, sample_idx = self.index[idx]
        shard = self._load_shard(shard_idx)
        return {
            "poses": shard["poses"][sample_idx],
            "embedding": shard["embedding"][sample_idx],
        }


def build_loaders(
    cache_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    train_split: str = "train",
    val_split: str = "val",
    **_,
) -> dict[str, DataLoader]:
    train_ds = CachedDecoderDataset(cache_dir, train_split)
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    }

    val_manifest = Path(cache_dir) / val_split / "manifest.pt"
    if val_manifest.exists():
        val_ds = CachedDecoderDataset(cache_dir, val_split)
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return loaders


def build_models(cfg, device: torch.device):
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
    return encoder, music_encoder, predictor


def load_jepa_checkpoint(
    checkpoint_path: str | Path,
    encoder: torch.nn.Module,
    music_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    if "music_encoder" in checkpoint:
        music_encoder.load_state_dict(checkpoint["music_encoder"])
    predictor.load_state_dict(checkpoint["predictor"])


def _save_shard(split_dir: Path, shard_id: int, poses: list[torch.Tensor], embeddings: list[torch.Tensor]):
    data = {
        "poses": torch.cat(poses, dim=0).contiguous(),
        "embedding": torch.cat(embeddings, dim=0).contiguous(),
    }
    file_name = f"shard_{shard_id:06d}.pt"
    torch.save(data, split_dir / file_name)
    return {
        "file": file_name,
        "num_samples": data["poses"].shape[0],
        "poses_shape": list(data["poses"].shape),
        "embedding_shape": list(data["embedding"].shape),
    }


@torch.no_grad()
def generate_split(
    split: str,
    loader: DataLoader,
    out_dir: Path,
    encoder: torch.nn.Module,
    music_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    cfg,
    device: torch.device,
    shard_size: int,
    use_amp: bool,
) -> None:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    encoder.eval()
    music_encoder.eval()
    predictor.eval()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)

    shards = []
    poses_buffer = []
    embedding_buffer = []
    buffered = 0
    shard_id = 0

    for batch in tqdm(loader, desc=f"Generating {split} decoder cache"):
        x_t, x_next, music = unpack_batch(batch, cfg)
        x_t = x_t.to(device, non_blocking=True)
        x_next = x_next.to(device, non_blocking=True)
        music = music.to(device, non_blocking=True)

        with autocast(device.type, enabled=use_amp, dtype=dtype):
            z_t = cls_state(encoder, x_t)
            music_emb = encode_music(music_encoder, music, cfg)
            z_pred = predictor(z_t, music_emb)

        poses_buffer.append(x_next.detach().cpu().float())
        embedding_buffer.append(z_pred.detach().cpu().float())
        buffered += x_next.shape[0]

        if buffered >= shard_size:
            shards.append(_save_shard(split_dir, shard_id, poses_buffer, embedding_buffer))
            poses_buffer = []
            embedding_buffer = []
            buffered = 0
            shard_id += 1

    if buffered:
        shards.append(_save_shard(split_dir, shard_id, poses_buffer, embedding_buffer))

    torch.save({"split": split, "shards": shards}, split_dir / "manifest.pt")


def generate_cache(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg.data.batch_size = args.batch_size or cfg.data.batch_size
    cfg.data.num_workers = args.num_workers if args.num_workers is not None else cfg.data.num_workers

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(args.device or cfg.meta.get("device", "auto"))
    setup_seed(cfg.meta.seed)

    train_loader, val_loader = build_source_dataloaders(cfg)
    encoder, music_encoder, predictor = build_models(cfg, device)
    load_jepa_checkpoint(args.checkpoint, encoder, music_encoder, predictor, device)

    generate_split(
        "train",
        train_loader,
        out_dir,
        encoder,
        music_encoder,
        predictor,
        cfg,
        device,
        args.shard_size,
        args.use_amp,
    )
    if val_loader is not None:
        generate_split(
            "val",
            val_loader,
            out_dir,
            encoder,
            music_encoder,
            predictor,
            cfg,
            device,
            args.shard_size,
            args.use_amp,
        )

    OmegaConf.save(cfg, out_dir / "source_config.yaml")
    metadata = {
        "checkpoint": str(args.checkpoint),
        "window": cfg.data.window,
        "horizon": cfg.data.horizon,
        "num_joints": cfg.model.encoder.num_joints,
        "embedding_dim": cfg.model.encoder.dim_rep,
        "format": {
            "poses": "[B, F, J, 3]",
            "embedding": "[B, embedding_dim]",
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a shard cache for music-conditioned motion decoder training."
    )
    parser.add_argument("--out-dir", required=True, help="Destination directory for the generated cache.")
    parser.add_argument("--checkpoint", required=True, help="Trained music JEPA checkpoint.")
    parser.add_argument("--config", default="music/cfgs/train.yaml", help="Music JEPA config.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, cpu, or auto.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size used while generating embeddings.")
    parser.add_argument("--num-workers", type=int, default=None, help="Source dataloader workers.")
    parser.add_argument("--shard-size", type=int, default=4096, help="Samples per saved shard.")
    parser.add_argument("--use-amp", action="store_true", help="Use autocast while generating embeddings.")
    return parser.parse_args()


if __name__ == "__main__":
    generate_cache(parse_args())
