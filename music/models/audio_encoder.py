from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from muq import MuQ


_MUQ_SAMPLE_RATE = 24000
_PRETRAINED_DIR = Path(__file__).parent.parent / "pretrained"


class AttentionPooling(nn.Module):
    """Weighted pooling over a sequence: learns which frames matter most.

    Produces a single vector from [M, T, D] → [M, out_dim] via a softmax
    over learned per-frame scalar scores, followed by a linear projection.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, T, D]
        weights = torch.softmax(self.score(x), dim=1)  # [M, T, 1]
        pooled = (weights * x).sum(dim=1)              # [M, D]
        return self.proj(pooled)                        # [M, out_dim]


class AudioEncoder(nn.Module):
    """Chunk-based audio encoder backed by a frozen MuQ model.

    Splits the input waveform into overlapping chunks, encodes each with MuQ,
    and mean-pools the per-frame hidden states into a single vector per chunk.

    Args:
        model_name: HuggingFace model ID for MuQ.
        embed_dim: Output embedding dimension D. If None, uses MuQ's hidden size.
        chunk_frames: Chunk duration in fps-rate frames (e.g. 150 = 5 s at 30 Hz).
        stride_frames: Stride between chunk starts in fps-rate frames.
        fps: Reference frame rate that chunk_frames / stride_frames are expressed in.
        sample_rate: Input waveform sample rate (must match MuQ, i.e. 24 kHz).
    """

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        embed_dim: Optional[int] = 384,
        chunk_frames: int = 150,
        stride_frames: int = 15,
        fps: int = 30,
        sample_rate: int = _MUQ_SAMPLE_RATE,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        samples_per_frame = sample_rate // fps
        self.chunk_samples = chunk_frames * samples_per_frame
        self.stride_samples = stride_frames * samples_per_frame

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

    def _encode_chunks(self, chunks: torch.Tensor) -> torch.Tensor:
        """Run MuQ on a batch of audio chunks and return one vector per chunk.

        Args:
            chunks: [M, chunk_samples] float32 mono audio

        Returns:
            [M, hidden_size]
        """
        with torch.no_grad():
            out = self.muq(chunks)

        return self.attn_pool(out.last_hidden_state)  # [M, out_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] raw audio tensor at self.sample_rate.

        Returns:
            [B, N, output_dim]  with N = (T - chunk_samples) // stride_samples + 1
        """
        B, C, T = x.shape

        # Mix to mono
        x = x.mean(dim=1)  # [B, T]

        # Chunk: [B, T] → [B, N, chunk_samples]
        chunks = x.unfold(-1, self.chunk_samples, self.stride_samples)
        B, N, L = chunks.shape

        # Encode all chunks in one batched MuQ forward pass
        chunks_flat = chunks.reshape(B * N, L).to(dtype=torch.float32)
        emb = self._encode_chunks(chunks_flat)  # [B*N, output_dim]

        return emb.reshape(B, N, -1)  # [B, N, output_dim]


if __name__ == "__main__":
    encoder = AudioEncoder()

    # Parameter counts
    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\ntotal parameters   : {total:,}")
    print(f"trainable params   : {trainable:,}")

    # Smoke test — 10 seconds of stereo audio at 24 kHz (must be > chunk_samples = 5 s)
    B, C, T = 2, 2, 10 * _MUQ_SAMPLE_RATE
    x = torch.randn(B, C, T)
    out = encoder(x)
    print(f"\nsmoke test — input : {list(x.shape)}")
    print(f"             output: {list(out.shape)}  (expected [B, N, {encoder.output_dim}])")
    
