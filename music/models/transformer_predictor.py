"""Autoregressive music-conditioned latent predictor (interleaved sequence)."""

import math
from typing import Optional

import torch
import torch.nn as nn

from music.models.audio_encoder import AttentionPooling


def _sinusoidal_pos(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _causal_mask(n: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)


class _Block(nn.Module):
    """Pre-norm: causal self-attention → FFN."""

    def __init__(self, dim: int, num_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm_sa = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        xn = self.norm_sa(x)
        x = x + self.self_attn(xn, xn, xn, attn_mask=mask, need_weights=False)[0]
        x = x + self.ff(self.norm_ff(x))
        return x


class MusicTransformerPredictor(nn.Module):
    """Autoregressive pose-latent predictor via interleaved music/state sequence.

    Architecture
    ------------
    Music and state tokens are interleaved into a single sequence of length 2N:

        [m₀, z_t,   m₁, z_{t+1},   …,   m_{N-1}, z_{t+N-1}]

    Each music token is produced by an attention-pooling head over the full MuQ
    frame-level sequence, so the predictor learns task-specific aggregation of
    the 1024-d MuQ features into the working dimension.

    A single decoder-only transformer (causal self-attention + FFN, no cross-
    attention) processes the full interleaved sequence. Output at every state
    position predicts the *next* latent; music-token outputs are discarded.

    T must equal N (one music chunk per prediction step).

    Usage
    -----
    Single-step / GRU drop-in  — state [B, D],    music [B, T, M]    → [B, D]
    Teacher-forcing training   — state [B, N, D],  music [B, N, T, M] → [B, N, D]
    Autoregressive inference   — ``generate(state, music)``
    """

    is_rnn: bool = False

    def __init__(
        self,
        state_dim: int,
        music_dim: int,
        horizon: int = 3,
        dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        final_ln: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.state_dim = state_dim
        self.music_dim = music_dim
        self.horizon   = horizon
        self.dim       = dim

        self.state_proj  = nn.Linear(state_dim, dim)
        # Attention pooling replaces the linear projection: pools [B, T, music_dim] → [B, dim]
        self.music_pool  = AttentionPooling(music_dim, dim)

        self.blocks   = nn.ModuleList([_Block(dim, num_heads, ff_mult, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, state_dim, bias=False)
        self.final_ln = final_ln if final_ln is not None else nn.Identity()

    def _pool_music(self, music: torch.Tensor) -> torch.Tensor:
        """Pool a batch of music chunks.

        Args:
            music: [B, N, T, music_dim]
        Returns:
            [B, N, dim]
        """
        B, N, T, M = music.shape
        return self.music_pool(music.reshape(B * N, T, M)).reshape(B, N, self.dim)

    def _interleave(self, s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Interleave [B, N, dim] state and music tensors → [B, 2N, dim]."""
        B, N, D = s.shape
        return torch.stack([m, s], dim=2).reshape(B, 2 * N, D)

    def forward(self, state: torch.Tensor, music: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim] or [B, N, state_dim].
            music: [B, T, music_dim] or [B, N, T, music_dim].  N must match state.

        Returns:
            Same leading shape as state. Output at position i predicts z_{t+i+1}.
        """
        squeeze = state.ndim == 2
        if squeeze:
            state = state.unsqueeze(1)   # [B, 1, D]
            music = music.unsqueeze(1)   # [B, 1, T, M]

        B, N, _ = state.shape
        if music.shape[1] != N:
            raise ValueError(f"music sequence length {music.shape[1]} must equal state length {N}")

        s = self.state_proj(state)   # [B, N, dim]
        m = self._pool_music(music)  # [B, N, dim]

        x = self._interleave(s, m)                             # [B, 2N, dim]
        x = x + _sinusoidal_pos(2 * N, self.dim, x.device)
        mask = _causal_mask(2 * N, x.device)

        for block in self.blocks:
            x = block(x, mask)

        # State positions are at odd indices: 1, 3, 5, …
        out = self.out_proj(self.out_norm(x[:, 1::2, :]))      # [B, N, state_dim]
        out = self.final_ln(out)

        return out.squeeze(1) if squeeze else out

    @torch.no_grad()
    def generate(self, state: torch.Tensor, music: torch.Tensor) -> torch.Tensor:
        """Autoregressively predict ``self.horizon`` future latents.

        Args:
            state: [B, state_dim]              starting latent z_t.
            music: [B, horizon, T, music_dim]  one music chunk per future step.

        Returns:
            [B, horizon, state_dim]
        """
        if music.shape[1] != self.horizon:
            raise ValueError(f"music must have {self.horizon} chunks, got {music.shape[1]}")

        m_all = self._pool_music(music)                        # [B, H, dim]
        seq   = torch.cat([                                    # [B, 2, dim]
            m_all[:, :1, :],
            self.state_proj(state).unsqueeze(1),
        ], dim=1)

        preds = []
        for i in range(self.horizon):
            N    = seq.shape[1]
            x    = seq + _sinusoidal_pos(N, self.dim, seq.device)
            mask = _causal_mask(N, seq.device)
            for block in self.blocks:
                x = block(x, mask)

            z_next = self.final_ln(self.out_proj(self.out_norm(x[:, -1:, :])))  # [B, 1, D]
            preds.append(z_next)

            if i < self.horizon - 1:
                seq = torch.cat([seq, m_all[:, i+1:i+2, :], self.state_proj(z_next)], dim=1)

        return torch.cat(preds, dim=1)                         # [B, H, D]
