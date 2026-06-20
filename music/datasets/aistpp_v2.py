"""AIST++ v3 dataset for music-conditioned dance motion learning.

Stored as a HuggingFace Arrow dataset at data_root/train/.
Each row is a 5-second clip (150 frames @ 30 fps) named {base}_w{N}.
Consecutive windows from the same sequence are stitched into 15-second clips
(3 × 150 = 450 frames) at load time.

Returns per sample:
  keypoints : [window + pred_horizon * horizon, 25, 3]  – z-score normalised
  music     : [pred_horizon, chunk_samples]              – one audio chunk per future step

Joint layout (25 joints):
  0–23 : SMPL axis-angle joints (Pelvis … R_Hand)
  24   : root translation (smpl_trans)
"""
import io
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional
from torch.utils.data import DataLoader, Dataset

_DEFAULT_DATA_ROOT = "/lustre/work/vivatech-4dudes/shared/datasets/aistpp_v3"


def _decode_audio(raw_bytes: bytes, target_sr: int, n_frames: int, fps: int) -> torch.Tensor:
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    expected = n_frames * (target_sr // fps)
    if wav.shape[1] < expected:
        wav = F.pad(wav, (0, expected - wav.shape[1]))
    else:
        wav = wav[:, :expected]
    return wav  # [1, expected_samples]


def _load_clips(data_root: str, target_sr: int, fps: int) -> list:
    from datasets import Audio as HFAudio, load_from_disk  # noqa: PLC0415

    ds = load_from_disk(data_root)["train"]
    ds = ds.cast_column("audio", HFAudio(decode=False))

    # Group rows by sequence base name (strip _w{N} suffix) and sort by window index.
    # Each sequence has 3 windows (_w0, _w1, _w2) → 15-second clip after stitching.
    groups: dict[str, list] = defaultdict(list)
    for row in ds:
        base = re.sub(r"_w\d+$", "", row["name"])
        groups[base].append(row)

    clips = []
    for base, rows in groups.items():
        rows.sort(key=lambda r: int(re.search(r"_w(\d+)$", r["name"]).group(1)))
        poses = np.concatenate([np.array(r["smpl_poses"], dtype=np.float32) for r in rows])
        trans = np.concatenate([np.array(r["smpl_trans"], dtype=np.float32) for r in rows])
        n_frames = sum(int(r["n_frames"]) for r in rows)
        audio = torch.cat(
            [_decode_audio(r["audio"]["bytes"], target_sr, int(r["n_frames"]), fps) for r in rows],
            dim=1,
        )
        clips.append({"poses": poses, "trans": trans, "audio": audio, "n_frames": n_frames})
    return clips


def _compute_normalizer(clips: list) -> tuple[np.ndarray, np.ndarray]:
    all_kp = []
    for c in clips:
        kp = np.concatenate([c["poses"].reshape(-1, 24, 3), c["trans"][:, None, :]], axis=1)
        all_kp.append(kp)
    all_kp = np.concatenate(all_kp, axis=0)  # [N_total_frames, 25, 3]
    mean = all_kp.mean(axis=0)
    std  = all_kp.std(axis=0).clip(1e-6)
    return mean, std


class AISTPPV2Dataset(Dataset):
    """Sliding-window dataset over 15-second AIST++ clips.

    For each clip, generates windows of total length kp_len = window + pred_horizon * horizon.
    Returns pred_horizon audio chunks — one per future prediction step.

    Returns per sample:
      keypoints : [kp_len, 25, 3]            normalised
      music     : [pred_horizon, chunk_samples]  one chunk per future step
    """

    def __init__(
        self,
        clips: list,
        mean: np.ndarray,
        std: np.ndarray,
        window: int = 75,
        horizon: int = 75,
        pred_horizon: int = 1,
        stride: int | None = None,
        fps: int = 30,
        target_sr: int = 24_000,
        audio_chunk_frames: int = 150,
    ):
        self.clips = clips
        self.mean = mean
        self.std  = std
        self.window = window
        self.horizon = horizon
        self.pred_horizon = pred_horizon
        self.samples_per_frame = target_sr // fps
        self.audio_chunk_samples = audio_chunk_frames * self.samples_per_frame
        self.kp_len = window + pred_horizon * horizon
        _stride = stride if stride is not None else window

        self._index: list[tuple[int, int]] = []
        for i, c in enumerate(clips):
            for start in range(0, c["n_frames"] - self.kp_len + 1, _stride):
                self._index.append((i, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        clip_idx, start = self._index[idx]
        clip = self.clips[clip_idx]

        poses = clip["poses"][start : start + self.kp_len]
        trans = clip["trans"][start : start + self.kp_len]
        kp = np.concatenate([poses.reshape(-1, 24, 3), trans[:, None, :]], axis=1)
        kp = (kp - self.mean) / self.std

        # One audio chunk per future step.
        # Chunk h ends at the same frame as x_{t+h+1} and extends audio_chunk_frames back.
        full = clip["audio"]  # [1, clip_samples]
        chunks = []
        for h in range(self.pred_horizon):
            step_end = start + self.window + (h + 1) * self.horizon
            s1 = step_end * self.samples_per_frame
            s0 = s1 - self.audio_chunk_samples
            left_pad  = max(0, -s0)
            right_pad = max(0, s1 - full.shape[1])
            chunk = F.pad(full[:, max(0, s0) : min(s1, full.shape[1])], (left_pad, right_pad))
            chunks.append(chunk.squeeze(0))  # [chunk_samples]
        music = torch.stack(chunks, dim=0)  # [pred_horizon, chunk_samples]

        return {
            "keypoints": torch.from_numpy(kp).float(),  # [kp_len, 25, 3]
            "music": music,                              # [pred_horizon, chunk_samples]
        }


def build_loaders(
    batch_size: int = 64,
    num_workers: int = 4,
    window: int = 75,
    horizon: int = 75,
    pred_horizon: int = 1,
    stride: int | None = None,
    data_root: str = _DEFAULT_DATA_ROOT,
    val_fraction: float = 0.1,
    fps: int = 30,
    sample_rate: int = 24_000,
    audio_chunk_frames: int = 150,
    **_,
) -> dict:
    clips = _load_clips(data_root, target_sr=sample_rate, fps=fps)
    n_val = max(1, round(len(clips) * val_fraction))
    train_clips, val_clips = clips[n_val:], clips[:n_val]

    mean, std = _compute_normalizer(train_clips)

    common = dict(
        mean=mean, std=std,
        window=window, horizon=horizon, pred_horizon=pred_horizon, stride=stride,
        fps=fps, target_sr=sample_rate, audio_chunk_frames=audio_chunk_frames,
    )
    train_ds = AISTPPV2Dataset(train_clips, **common)
    val_ds   = AISTPPV2Dataset(val_clips,   **common)

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  **kw),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, **kw),
    }
