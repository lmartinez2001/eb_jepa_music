"""End-to-end music → dance inference.

Pipeline
--------
  audio file
      → chunk into pred_horizon segments
      → AudioEncoder (MuQ)      → music_emb  [1, H, T, 1024]
  initial pose NPY (optional)
      → DSTformer encoder       → z_t        [1, 512]   (zeros if omitted)
  predictor.generate(z_t, music_emb)
                                → z_pred     [1, H, 512]
  for each h:
      decoder.predict_motion(z_pred[:, h])
                                → poses_norm [1, F, 25, 3]
      * std + mean              → poses_real [F, 25, 3]

Output: NPZ with key ``poses`` of shape [H*F, 25, 3] in real-world coordinates.

Usage
-----
  python music/infer.py \\
      --audio        /path/to/song.wav \\
      --jepa-ckpt    checkpoints/music_jepa/<run>/best.pth.tar \\
      --decoder-ckpt checkpoints/music_decoder/<run>/best.pth.tar \\
      --cache-dir    datasets/decoder_cache \\
      --out          output/dance.npz

  # Seed the initial pose with a real clip (unnormalised NPY [F, J, 3]):
  python music/infer.py ... --init-pose /path/to/init_pose.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional

from music.datasets.decoder_dataset import build_models as build_jepa_models
from music.datasets.decoder_dataset import load_jepa_checkpoint
from music.main import cls_state, encode_music
from music.models.motion_decoder import build_motion_decoder
from eb_jepa.training_utils import load_config, setup_device


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load any soundfile-readable audio → [1, samples] mono float32."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)        # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)   # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav  # [1, T]


def _chunk_audio(wav: torch.Tensor, chunk_samples: int, n_chunks: int) -> torch.Tensor:
    """Slice wav into n_chunks of exactly chunk_samples, padding the last if needed.

    Returns [n_chunks, chunk_samples].
    """
    chunks = []
    for h in range(n_chunks):
        start = h * chunk_samples
        end   = start + chunk_samples
        chunk = wav[:, start:end]
        if chunk.shape[1] < chunk_samples:
            chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[1]))
        chunks.append(chunk.squeeze(0))  # [chunk_samples]
    return torch.stack(chunks, dim=0)   # [n_chunks, chunk_samples]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_decoder(cfg, decoder_ckpt: str, device: torch.device):
    dec_cfg = cfg.model.decoder
    decoder = build_motion_decoder(
        num_frames=dec_cfg.num_frames,
        num_joints=dec_cfg.num_joints,
        cond_dim=dec_cfg.cond_dim,
        dim=dec_cfg.dim,
        depth=dec_cfg.depth,
        num_heads=dec_cfg.num_heads,
        mlp_ratio=dec_cfg.mlp_ratio,
        t_dim=dec_cfg.get("t_dim", 256),
    ).to(device)
    ckpt = torch.load(decoder_ckpt, map_location=device, weights_only=False)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    return decoder


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(args: argparse.Namespace) -> None:
    jepa_cfg_path = args.jepa_config    or "music/cfgs/train.yaml"
    dec_cfg_path  = args.decoder_config or "music/cfgs/train_decoder.yaml"

    jepa_cfg = load_config(jepa_cfg_path)
    dec_cfg  = load_config(dec_cfg_path)

    device = setup_device(args.device or jepa_cfg.meta.get("device", "auto"))

    # ---- JEPA models -------------------------------------------------------
    encoder, music_encoder, predictor = build_jepa_models(jepa_cfg, device)
    load_jepa_checkpoint(args.jepa_ckpt, encoder, music_encoder, predictor, device)
    encoder.eval()
    music_encoder.eval()
    predictor.eval()

    # ---- Decoder -----------------------------------------------------------
    decoder = _build_decoder(dec_cfg, args.decoder_ckpt, device)

    # ---- Normalizer --------------------------------------------------------
    norm = torch.load(
        Path(args.cache_dir) / "normalizer.pt",
        map_location="cpu",
        weights_only=False,
    )
    mean = torch.from_numpy(norm["mean"]).to(device)  # [J, 3]
    std  = torch.from_numpy(norm["std"]).to(device)   # [J, 3]

    # ---- Audio → music embeddings ------------------------------------------
    sr            = jepa_cfg.data.sample_rate
    chunk_frames  = jepa_cfg.model.music_encoder.chunk_frames
    chunk_samples = chunk_frames * (sr // jepa_cfg.data.fps)
    H             = jepa_cfg.data.pred_horizon

    wav    = _load_audio(args.audio, sr)                          # [1, T]
    chunks = _chunk_audio(wav, chunk_samples, H)                  # [H, chunk_samples]
    music  = chunks.unsqueeze(0).to(device)                       # [1, H, chunk_samples]

    music_emb = encode_music(music_encoder, music, jepa_cfg)      # [1, H, T_muq, 1024]

    # ---- Initial latent z_t ------------------------------------------------
    if args.init_pose is not None:
        poses_np   = np.load(args.init_pose)                      # [F, J, 3] unnormalised
        poses_norm = (poses_np - norm["mean"]) / norm["std"]
        x_t = torch.from_numpy(poses_norm).float().unsqueeze(0).to(device)  # [1, F, J, 3]
        window = jepa_cfg.data.window
        if x_t.shape[1] != window:
            # Linearly interpolate to the expected window length
            x_t = F.interpolate(
                x_t.reshape(1, x_t.shape[1], -1).permute(0, 2, 1),
                size=window,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1).reshape(1, window, x_t.shape[2], 3)
        z_t = cls_state(encoder, x_t)                             # [1, D]
    else:
        z_t = torch.zeros(1, jepa_cfg.model.encoder.dim_rep, device=device)

    # ---- Predict future latents --------------------------------------------
    z_pred = predictor.generate(z_t, music_emb)                   # [1, H, D]

    # ---- Decode each step and unnormalise ----------------------------------
    all_poses = []
    for h in range(H):
        z_h        = z_pred[:, h]                                  # [1, D]
        poses_norm = decoder.predict_motion(z_h, num_steps=args.num_steps)  # [1, F, J, 3]
        poses_real = poses_norm.squeeze(0) * std + mean            # [F, J, 3]
        all_poses.append(poses_real.cpu().numpy())

    output = np.concatenate(all_poses, axis=0)                    # [H*F, J, 3]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, poses=output)
    print(f"Saved {output.shape} poses → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Music-conditioned dance generation.")
    p.add_argument("--audio",          required=True,
                   help="Input music file (any format supported by soundfile).")
    p.add_argument("--jepa-ckpt",      required=True,
                   help="JEPA checkpoint path (e.g. checkpoints/.../best.pth.tar).")
    p.add_argument("--decoder-ckpt",   required=True,
                   help="Decoder checkpoint path (e.g. checkpoints/.../best.pth.tar).")
    p.add_argument("--cache-dir",      required=True,
                   help="Decoder cache directory containing normalizer.pt.")
    p.add_argument("--out",            default="output/dance.npz",
                   help="Output NPZ path (default: output/dance.npz).")
    p.add_argument("--init-pose",      default=None,
                   help="Optional initial pose NPY [F, J, 3] (unnormalised) to seed z_t.")
    p.add_argument("--num-steps",      type=int, default=50,
                   help="Euler steps for the flow-matching ODE (default: 50).")
    p.add_argument("--device",         default=None,
                   help="Device override: cuda / cpu / auto.")
    p.add_argument("--jepa-config",    default=None,
                   help="JEPA config override (default: music/cfgs/train.yaml).")
    p.add_argument("--decoder-config", default=None,
                   help="Decoder config override (default: music/cfgs/train_decoder.yaml).")
    return p.parse_args()


if __name__ == "__main__":
    infer(parse_args())
