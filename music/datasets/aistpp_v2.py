"""AIST++ v3 dataset for music-conditioned dance motion learning.

Stored as a HuggingFace Arrow dataset at data_root/train/.
Each row is a 5-second clip (150 frames @ 30 fps) with:
  smpl_poses : Array2D [150, 72]  axis-angle SMPL pose parameters (24 joints × 3)
  smpl_trans : Array2D [150, 3]   root translation
  n_frames   : int                actual frame count
  audio      : Audio @ 48 kHz    decoded via soundfile (torchcodec bypassed)

Returns per sample:
  keypoints : [window + horizon, 25, 3]  – z-score normalised; 24 SMPL joints + root (joint 24)
  music     : [1, chunk_frames * (sample_rate // fps)]  – mono waveform aligned to x_next end

Joint layout (25 joints, 75 total values per frame):
  0–23 : SMPL axis-angle joints (Pelvis … R_Hand), original ordering
  24   : root translation (smpl_trans), appended as a 25th joint
"""
import io
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
    # Lazy import: avoids shadowing by music.datasets (this module's own package)
    # when the editable install resolves top-level 'datasets' ambiguously.
    from datasets import Audio as HFAudio, load_from_disk  # noqa: PLC0415

    ds = load_from_disk(data_root)["train"]
    # Bypass torchcodec — cast Audio to decode=False to get raw bytes
    ds = ds.cast_column("audio", HFAudio(decode=False))

    clips = []
    for row in ds:
        poses = np.array(row["smpl_poses"], dtype=np.float32)  # [T, 72]
        trans = np.array(row["smpl_trans"], dtype=np.float32)  # [T, 3]
        n = int(row["n_frames"])
        audio = _decode_audio(row["audio"]["bytes"], target_sr, n, fps)
        clips.append({"poses": poses, "trans": trans, "audio": audio, "n_frames": n})
    return clips


def _compute_normalizer(clips: list) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std over all frames in the given clips.

    Returns mean, std each of shape [25, 3].
    Std is clamped to 1e-6 to avoid division by zero for constant channels.
    """
    all_kp = []
    for c in clips:
        kp = np.concatenate([c["poses"].reshape(-1, 24, 3), c["trans"][:, None, :]], axis=1)
        all_kp.append(kp)
    all_kp = np.concatenate(all_kp, axis=0)  # [N_total_frames, 25, 3]
    mean = all_kp.mean(axis=0)               # [25, 3]
    std  = all_kp.std(axis=0).clip(1e-6)     # [25, 3]
    return mean, std


class AISTPPV2Dataset(Dataset):
    """Sliding-window dataset over AIST++ clips.

    For each clip of N frames, generates windows of length kp_len = window + horizon.
    Keypoints are z-score normalised using the provided mean/std (computed on training clips).

    Returns per sample:
      keypoints : [kp_len, 25, 3]   normalised
      music     : [1, chunk_samples] mono waveform
    """

    def __init__(
        self,
        clips: list,
        mean: np.ndarray,
        std: np.ndarray,
        window: int = 120,
        horizon: int = 1,
        stride: int | None = None,
        fps: int = 30,
        target_sr: int = 24_000,
        audio_chunk_frames: int = 150,
    ):
        self.clips = clips
        self.mean = mean  # [25, 3]
        self.std  = std   # [25, 3]
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
        kp = (kp - self.mean) / self.std  # z-score normalise

        # audio ends at the same frame as x_next, extends audio_chunk_frames backward
        full = clip["audio"]  # [1, clip_samples]
        s1 = (start + self.kp_len) * self.samples_per_frame
        s0 = s1 - self.audio_chunk_samples
        left_pad  = max(0, -s0)
        right_pad = max(0, s1 - full.shape[1])
        music = F.pad(full[:, max(0, s0) : min(s1, full.shape[1])], (left_pad, right_pad))

        return {
            "keypoints": torch.from_numpy(kp).float(),  # [kp_len, 25, 3]
            "music": music,                              # [1, chunk_samples]
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
    clips = _load_clips(data_root, target_sr=sample_rate, fps=fps)
    n_val = max(1, round(len(clips) * val_fraction))
    train_clips, val_clips = clips[n_val:], clips[:n_val]

    mean, std = _compute_normalizer(train_clips)

    common = dict(
        mean=mean, std=std,
        window=window, horizon=horizon, stride=stride,
        fps=fps, target_sr=sample_rate, audio_chunk_frames=audio_chunk_frames,
    )
    train_ds = AISTPPV2Dataset(train_clips, **common)
    val_ds   = AISTPPV2Dataset(val_clips,   **common)

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  **kw),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, **kw),
    }
