from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from muq import MuQ


_MUQ_SAMPLE_RATE = 24000
_PRETRAINED_DIR = Path(__file__).parent.parent.parent / "pretrained"


class AttentionPooling(nn.Module):
    """Weighted pooling over a sequence: learns which frames matter most.

    Produces a single vector from [B, T, D] → [B, out_dim] via a softmax
    over learned per-frame scalar scores, followed by a linear projection.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        weights = torch.softmax(self.score(x), dim=1)  # [B, T, 1]
        pooled = (weights * x).sum(dim=1)              # [B, D]
        return self.proj(pooled)                        # [B, out_dim]


class AudioEncoder(nn.Module):
    """MuQ-based audio encoder for a single fixed-size chunk.

    Expects input of exactly chunk_samples = chunk_frames * (sample_rate // fps)
    samples. Returns one embedding vector per item in the batch.

    Args:
        model_name: HuggingFace model ID for MuQ.
        embed_dim: Output embedding dimension. If None, uses MuQ's hidden size.
        chunk_frames: Chunk duration in fps-rate frames (e.g. 150 = 5 s at 30 Hz).
        fps: Reference frame rate for chunk_frames.
        sample_rate: Input waveform sample rate (must match MuQ: 24 kHz).
    """

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        embed_dim: Optional[int] = 384,
        chunk_frames: int = 150,
        fps: int = 30,
        sample_rate: int = _MUQ_SAMPLE_RATE,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.chunk_samples = chunk_frames * (sample_rate // fps)

        # Load and freeze MuQ, caching weights in pretrained/
        self.muq: MuQ = MuQ.from_pretrained(model_name, cache_dir=_PRETRAINED_DIR)
        self.muq.eval()
        for p in self.muq.parameters():
            p.requires_grad_(False)

        hidden_size = self.muq.config.encoder_dim
        out_dim = embed_dim if embed_dim is not None else hidden_size
        self.attn_pool = AttentionPooling(hidden_size, out_dim)
        self._output_dim = out_dim

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
               Time dimension must equal self.chunk_samples exactly.

        Returns:
            [B, output_dim]
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

        return self.attn_pool(out.last_hidden_state)  # [B, output_dim]


if __name__ == "__main__":
    encoder = AudioEncoder()

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"total parameters   : {total:,}")
    print(f"trainable params   : {trainable:,}")

    # Smoke test — one chunk of stereo audio
    B, C, T = 2, 2, encoder.chunk_samples
    x = torch.randn(B, C, T)
    out = encoder(x)
    print(f"\nsmoke test — input : {list(x.shape)}")
    print(f"             output: {list(out.shape)}  (expected [{B}, {encoder.output_dim}])")
