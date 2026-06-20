from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from muq import MuQ


_MUQ_SAMPLE_RATE = 24000
_PRETRAINED_DIR = Path(__file__).parent.parent.parent / "pretrained"


class AttentionPooling(nn.Module):
    """Weighted pooling over a sequence: [B, T, D] → [B, out_dim]."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)  # [B, T, 1]
        pooled = (weights * x).sum(dim=1)              # [B, D]
        return self.proj(pooled)                        # [B, out_dim]


class AudioEncoder(nn.Module):
    """MuQ-based audio encoder for a single fixed-size chunk.

    Returns the full MuQ frame-level sequence — pooling is handled by the
    downstream predictor so it can learn task-specific aggregation.

    Args:
        model_name: HuggingFace model ID for MuQ.
        chunk_frames: Chunk duration in fps-rate frames (e.g. 150 = 5 s at 30 Hz).
        fps: Reference frame rate for chunk_frames.
        sample_rate: Input waveform sample rate (must match MuQ: 24 kHz).
    """

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        chunk_frames: int = 150,
        fps: int = 30,
        sample_rate: int = _MUQ_SAMPLE_RATE,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.chunk_samples = chunk_frames * (sample_rate // fps)

        self.muq: MuQ = MuQ.from_pretrained(model_name, cache_dir=_PRETRAINED_DIR)
        self.muq.eval()
        for p in self.muq.parameters():
            p.requires_grad_(False)

        self._output_dim: int = self.muq.config.encoder_dim  # 1024 for MuQ-large

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.muq.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, chunk_samples] raw audio at self.sample_rate.

        Returns:
            [B, T, output_dim]  — full MuQ frame-level sequence.
        """
        B, C, T = x.shape
        if T != self.chunk_samples:
            raise ValueError(
                f"Expected time dimension {self.chunk_samples} "
                f"({self.chunk_samples // (self.sample_rate // 30)} frames), got {T}"
            )

        x = x.mean(dim=1).to(dtype=torch.float32)  # [B, chunk_samples] mono

        with torch.no_grad():
            out = self.muq(x)

        return out.last_hidden_state.float()  # [B, T, output_dim]


if __name__ == "__main__":
    encoder = AudioEncoder()

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"total parameters   : {total:,}")
    print(f"trainable params   : {trainable:,}")

    B, C, T = 2, 2, encoder.chunk_samples
    x = torch.randn(B, C, T)
    out = encoder(x)
    print(f"\nsmoke test — input : {list(x.shape)}")
    print(f"             output: {list(out.shape)}  (expected [{B}, T, {encoder.output_dim}])")
