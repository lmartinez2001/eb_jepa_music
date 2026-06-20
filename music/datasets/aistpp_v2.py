"""AIST++ v2 dataset for music-conditioned dance motion learning.

Each parquet row is a 5-second clip (150 frames @ 30 fps) with:
  smpl_poses : [150, 72]  axis-angle SMPL pose parameters (24 joints × 3)
  smpl_trans : [150, 3]   root translation
  audio      : WAV bytes @ 48 kHz → resampled to 24 kHz on load

Returns per sample:
  keypoints : [window + horizon, 25, 3]  – 24 SMPL joints + root translation (joint 24)
  music     : [1, chunk_frames * (sample_rate // fps)]  – mono waveform aligned to window

Joint layout (25 joints, 75 total values per frame):
  0–23 : SMPL axis-angle joints (Pelvis … R_Hand), original ordering
  24   : root translation (smpl_trans), appended as a 25th joint
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

_DEFAULT_DATA_ROOT = "/lustre/work/vivatech-4dudes/edugelay/datasets/aistpp_v2/data"


def _decode_audio(raw_bytes: bytes, target_sr: int, n_frames: int, fps: int) -> torch.Tensor:
    wav, sr = torchaudio.load(io.BytesIO(raw_bytes))
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


def _load_clips(data_dir: Path, target_sr: int, fps: int) -> list:
    clips = []
    for pq in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            poses = np.stack(row["smpl_poses"]).astype(np.float32)   # [T, 72]
            trans = np.stack(row["smpl_trans"]).astype(np.float32)   # [T, 3]
            n = len(poses)
            audio = _decode_audio(row["audio"]["bytes"], target_sr, n, fps)
            clips.append({"poses": poses, "trans": trans, "audio": audio, "n_frames": n})
    return clips


class AISTPPV2Dataset(Dataset):
    """Sliding-window dataset over AIST++ v2 clips.

    For each clip of N frames, generates (N - kp_len + 1) windows. Each yields:
      keypoints : [kp_len, 25, 3]   24 SMPL joints + root translation as joint 24
      music     : [1, chunk_samples] mono waveform starting at same frame
    """

    def __init__(
        self,
        clips: list,
        window: int = 120,
        horizon: int = 1,
        stride: int | None = None,
        fps: int = 30,
        target_sr: int = 24_000,
        audio_chunk_frames: int = 150,
    ):
        self.clips = clips
        self.window = window
        self.horizon = horizon
        self.samples_per_frame = target_sr // fps
        self.audio_chunk_samples = audio_chunk_frames * self.samples_per_frame
        self.kp_len = window + horizon
        _stride = stride if stride is not None else window  # non-overlapping by default

        self._index: list[tuple[int, int]] = []
        for i, c in enumerate(clips):
            for start in range(0, c["n_frames"] - self.kp_len + 1, _stride):
                self._index.append((i, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        clip_idx, start = self._index[idx]
        clip = self.clips[clip_idx]

        # [kp_len, 72] → [kp_len, 24, 3]; append trans as joint 24 → [kp_len, 25, 3]
        poses = clip["poses"][start : start + self.kp_len]
        trans = clip["trans"][start : start + self.kp_len]
        kp = np.concatenate([poses.reshape(-1, 24, 3), trans[:, None, :]], axis=1)

        # audio ends at the same frame as x_next, extends audio_chunk_frames backward
        full = clip["audio"]  # [1, clip_samples]
        s1 = (start + self.kp_len) * self.samples_per_frame
        s0 = s1 - self.audio_chunk_samples
        left_pad  = max(0, -s0)
        right_pad = max(0, s1 - full.shape[1])
        music = F.pad(full[:, max(0, s0) : min(s1, full.shape[1])], (left_pad, right_pad))

        return {
            "keypoints": torch.from_numpy(kp),  # [kp_len, 25, 3]
            "music": music,                       # [1, chunk_samples]
        }


def build_loaders(
    batch_size: int = 64,
    num_workers: int = 4,
    window: int = 120,
    horizon: int = 1,
    stride: int | None = None,
    data_root: str = _DEFAULT_DATA_ROOT,
    val_fraction: float = 0.1,
    fps: int = 30,
    sample_rate: int = 24_000,
    audio_chunk_frames: int = 150,
    **_,
) -> dict:
    clips = _load_clips(Path(data_root), target_sr=sample_rate, fps=fps)
    n_val = max(1, round(len(clips) * val_fraction))

    common = dict(window=window, horizon=horizon, stride=stride, fps=fps, target_sr=sample_rate, audio_chunk_frames=audio_chunk_frames)
    train_ds = AISTPPV2Dataset(clips[n_val:], **common)
    val_ds   = AISTPPV2Dataset(clips[:n_val], **common)

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  **kw),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, **kw),
    }
