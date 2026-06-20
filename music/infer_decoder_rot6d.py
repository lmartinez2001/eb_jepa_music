"""Rot6d decoder inference -> SMPL axis-angle.

Loads N decoder training samples (z_pred latent + GT 6D poses) from the rot6d
cache, generates predictions with the trained x0-prediction flow-matching
decoder, converts BOTH prediction and GT from 6D back to SMPL axis-angle, and
unnormalizes joint-24 translation to raw meters. Saves everything stacked to a
single npz.
"""
import sys, pathlib
_h = pathlib.Path(__file__).parent.resolve()
sys.path = [p for p in sys.path if pathlib.Path(p).resolve() != _h]

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from eb_jepa.training_utils import setup_device, setup_seed, load_config
from music.datasets.decoder_dataset import CachedDecoderDataset
from music.datasets.aistpp_v2 import build_loaders as aistpp_build_loaders
from music.models.motion_decoder import build_motion_decoder
from music.models.rotation_utils import rotation_6d_to_axis_angle

_DEF_CKPT = "/lustre/work/vivatech-4dudes/dpham/checkpoints/music_decoder_rot6d_w75/latest.pth.tar"
_DEF_JEPA_CFG = "/lustre/work/vivatech-4dudes/edugelay/checkpoints/music_jepa/dev_2026-06-20_04-21/exp_seed2025/config.yaml"

def get_normalizer(cache_dir, jepa_cfg_path, device):
    """Return (mean,std) [25,3]. Cache to <cache_dir>/normalizer.npz to avoid
    reloading the full source dataset on every run."""
    npz = Path(cache_dir) / "normalizer.npz"
    if npz.exists():
        d = np.load(npz)
        return (torch.tensor(d["mean"], device=device, dtype=torch.float32),
                torch.tensor(d["std"], device=device, dtype=torch.float32))
    if jepa_cfg_path is None:
        jepa_cfg_path = str(Path(cache_dir) / "source_config.yaml")
    print("computing dataset normalizer from source aistpp (one-time)...")
    cfg = load_config(jepa_cfg_path)
    data_kwargs = OmegaConf.to_container(cfg.data, resolve=True)
    loaders = aistpp_build_loaders(**data_kwargs)
    ds = loaders["train"].dataset
    mean, std = np.asarray(ds.mean), np.asarray(ds.std)
    np.savez(npz, mean=mean, std=std)
    print(f"saved normalizer -> {npz}")
    return (torch.tensor(mean, device=device, dtype=torch.float32),
            torch.tensor(std, device=device, dtype=torch.float32))

def to_smpl(poses6, mean, std):
    """[B,F,25,6] -> (aa[B,F,24,3], trans_raw[B,F,3]).

    Joints 0-23: 6D rotation -> axis-angle. Joint 24: first 3 dims are the
    z-score normalized translation -> unnormalize to raw meters.
    """
    aa = rotation_6d_to_axis_angle(poses6[..., :24, :])     # [B,F,24,3]
    trans_norm = poses6[..., 24, :3]                          # [B,F,3]
    trans_raw = trans_norm * std[24] + mean[24]              # unnormalize
    return aa, trans_raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=_DEF_CKPT)
    ap.add_argument("--config", default=None, help="Decoder config.yaml (default: <ckpt_dir>/config.yaml).")
    ap.add_argument("--jepa-config", default=None, help="JEPA config for source normalizer stats (default: <cache_dir>/source_config.yaml).")
    ap.add_argument("--cache-dir", default=None, help="Override cache dir (default: from decoder config).")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--start", type=int, default=0, help="First sample index.")
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="/lustre/work/vivatech-4dudes/dpham/working/infer_rot6d_10.npz")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    cfg = OmegaConf.load(Path(args.config) if args.config else ckpt_path.parent / "config.yaml")
    dcfg = cfg.model.decoder
    cache_dir = args.cache_dir or cfg.data.cache_dir

    device = setup_device(args.device)
    setup_seed(args.seed)

    decoder = build_motion_decoder(
        # num_frames=dcfg.num_frames, num_joints=dcfg.num_joints, cond_dim=dcfg.cond_dim,
        # dim=dcfg.dim, depth=dcfg.depth, 
        # num_heads=dcfg.num_heads, 
        # mlp_ratio=dcfg.mlp_ratio,
        # t_dim=dcfg.t_dim, 
        # coord_dim=dcfg.get("coord_dim", 6), 
        # sigma_min=dcfg.get("sigma_min", 1e-4),
        num_frames=dcfg.num_frames,
        num_joints=dcfg.num_joints,
        cond_dim=dcfg.cond_dim,
        dim=dcfg.dim,
        depth=dcfg.depth,
        num_heads=dcfg.num_heads,
        mlp_ratio=dcfg.mlp_ratio,
        t_dim=dcfg.t_dim,
        coord_dim=dcfg.get("coord_dim", 3),
        sigma_min=dcfg.get("sigma_min", 1e-4),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    decoder.load_state_dict(state["decoder"])
    decoder.eval()
    ep = state.get("epoch")
    print(f"loaded decoder @ epoch={ep}")

    ds = CachedDecoderDataset(cache_dir, args.split)
    idxs = list(range(args.start, min(args.start + args.num_samples, len(ds))))
    gt6 = torch.stack([ds[i]["poses"] for i in idxs]).to(device).float()       # [N,F,25,6]
    z = torch.stack([ds[i]["embedding"] for i in idxs]).to(device).float()     # [N,512]
    print(f"split={args.split} indices={idxs}  gt6={tuple(gt6.shape)}  z={tuple(z.shape)}")

    with torch.no_grad():
        pred6 = decoder.predict_motion(z, num_frames=dcfg.num_frames, num_steps=args.num_steps)
    mse6 = F.mse_loss(pred6, gt6, reduction="none").mean(dim=(1, 2, 3))        # [N]

    mean, std = get_normalizer(cache_dir, args.jepa_config, device)
    pred_aa, pred_trans = to_smpl(pred6, mean, std)
    gt_aa, gt_trans = to_smpl(gt6, mean, std)

    N, Fr = pred_aa.shape[0], pred_aa.shape[1]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        pred_aa=pred_aa.cpu().numpy(),                 # [N,F,24,3]
        gt_aa=gt_aa.cpu().numpy(),                      # [N,F,24,3]
        pred_trans=pred_trans.cpu().numpy(),           # [N,F,3]
        gt_trans=gt_trans.cpu().numpy(),               # [N,F,3]
        pred_pose72=pred_aa.reshape(N, Fr, 72).cpu().numpy(),   # SMPL body_pose flat
        gt_pose72=gt_aa.reshape(N, Fr, 72).cpu().numpy(),
        pred6=pred6.cpu().numpy(),                      # [N,F,25,6] raw decoder output
        gt6=gt6.cpu().numpy(),
        mse6=mse6.cpu().numpy(),                        # [N] per-sample 6D MSE
        indices=np.asarray(idxs),
        num_steps=np.int64(args.num_steps),
    )
    print(f"per-sample 6D MSE: " + ", ".join(f"{v:.4f}" for v in mse6.tolist()))
    print(f"pred_aa={tuple(pred_aa.shape)} trans={tuple(pred_trans.shape)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()