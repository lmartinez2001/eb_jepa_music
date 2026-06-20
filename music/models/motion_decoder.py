from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding. t in [0, 1] → [B, dim]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        # Scale t to [0, 1000] so low-frequency components vary meaningfully
        args = t[:, None] * freqs[None] * 1000.0
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class AdaLNZero(nn.Module):
    """adaLN-Zero modulation layer.

    Produces (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    from the conditioning vector. Weights are zero-initialized so each DiT block
    is identity at the start of training.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # cond: [B, cond_dim] → 6 × [B, dim]
        return self.linear(cond).chunk(6, dim=-1)


class DiTBlock(nn.Module):
    """Single DiT block: self-attention + MLP, both gated by adaLN-Zero."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cond_dim: int = 512,
    ):
        super().__init__()
        # elementwise_affine=False: scale/shift are provided by adaLN, not learned here
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.adaLN = AdaLNZero(dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, N, dim]   cond: [B, cond_dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond)

        # Unsqueeze → [B, 1, dim] for broadcast over tokens
        def _b(v: torch.Tensor) -> torch.Tensor:
            return v.unsqueeze(1)

        # Self-attention branch
        x_norm = self.norm1(x) * (1 + _b(scale_msa)) + _b(shift_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + _b(gate_msa) * attn_out

        # MLP branch
        x_norm = self.norm2(x) * (1 + _b(scale_mlp)) + _b(shift_mlp)
        x = x + _b(gate_mlp) * self.mlp(x_norm)

        return x


class MotionDiT(nn.Module):
    """Velocity-field network for Flow Matching over 3D keypoint sequences.

    Tokenizes [B, F, J, 3] → [B, F*J, dim], runs DiT blocks conditioned on
    (z, t), then projects back to [B, F, J, 3].

    Note: full self-attention over F*J tokens (PyTorch uses flash-attention
    kernels when available). For very long sequences consider factored ST attention.
    """

    def __init__(
        self,
        num_frames: int = 243,
        num_joints: int = 17,
        coord_dim: int = 3,
        dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        cond_dim: int = 512,
        t_dim: int = 256,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.coord_dim = coord_dim

        # Input projection: 3 coords → dim
        self.input_proj = nn.Linear(coord_dim, dim)

        # Additive 2-D positional embeddings (frame × joint)
        self.frame_embed = nn.Parameter(torch.zeros(1, num_frames, 1, dim))
        self.joint_embed = nn.Parameter(torch.zeros(1, 1, num_joints, dim))
        nn.init.trunc_normal_(self.frame_embed, std=0.02)
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

        # Conditioning: merge encoder latent z and sinusoidal timestep
        self.t_embed = SinusoidalEmbedding(t_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + t_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio, cond_dim)
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(dim)
        # Zero-init output head so velocity starts at zero
        self.head = nn.Linear(dim, coord_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, J, 3]
        B, F, J, _ = x.shape
        x = self.input_proj(x)                               # [B, F, J, dim]
        x = x + self.frame_embed[:, :F] + self.joint_embed   # broadcast PE
        return x.reshape(B, F * J, -1)                       # [B, F*J, dim]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: noisy motion        [B, F, J, 3]
            t: timestep in [0, 1]  [B]
            z: encoder latent      [B, cond_dim]
        Returns:
            velocity field         [B, F, J, 3]
        """
        B, F, J, _ = x.shape

        tokens = self._tokenize(x)                                     # [B, F*J, dim]
        cond = self.cond_proj(torch.cat([z, self.t_embed(t)], dim=-1)) # [B, cond_dim]

        for block in self.blocks:
            tokens = block(tokens, cond)

        v = self.head(self.norm_out(tokens))                           # [B, F*J, 3]
        return v.reshape(B, F, J, self.coord_dim)                      # [B, F, J, 3]


class MotionDecoder(nn.Module):
    """Flow Matching decoder that generates motion clips from a latent z.

    Training objective: x0-prediction CFM with sigma_min noise floor
    (Lipman et al. 2022 / Albergo & Vanden-Eijnden 2023).  The model predicts
    the clean sample x1 directly; inference uses the implied PF-ODE velocity.

    Inference: Euler integration of the PF-ODE from t=0 to t=1.
    """

    def __init__(self, dit: MotionDiT, sigma_min: float = 1e-4):
        super().__init__()
        self.dit = dit
        self.sigma_min = sigma_min

    @torch.no_grad()
    def inference_loss(self, x1, z, num_steps: int = 50):
        """End-to-end generation quality: full ODE sampling from noise, MSE vs GT."""
        gen = self.predict_motion(z, num_frames=x1.shape[1], num_steps=num_steps)
        loss = F.mse_loss(gen, x1)
        return loss, {"inference_mse": loss.detach()}

    def compute_loss(
        self,
        x1: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """CFM loss with x0-prediction and sigma_min noise floor.

        Args:
            x1: clean motion clips  [B, F, J, C]
            z:  encoder latents     [B, cond_dim]
        Returns:
            (loss, log_dict)
        """
        B = x1.shape[0]
        device = x1.device
        sigma = self.sigma_min

        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=device)
        t_b = t.reshape(B, 1, 1, 1)

        # CFM path with sigma_min: x_t = t*x1 + (1-(1-sigma)*t)*noise
        x_t = t_b * x1 + (1.0 - (1.0 - sigma) * t_b) * x0

        # Model predicts the clean sample x1 (x0-prediction)
        x1_pred = self.dit(x_t, t, z)
        loss = F.mse_loss(x1_pred, x1)

        return loss, {"mse": loss.detach()}

    @torch.no_grad()
    def predict_motion(
        self,
        z: torch.Tensor,
        num_frames: Optional[int] = None,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Generate a motion clip from pure noise via PF-ODE Euler integration.

        Args:
            z:          encoder latents  [B, cond_dim]
            num_frames: override F (default: dit.num_frames)
            num_steps:  number of Euler steps (more → smoother, slower)
        Returns:
            generated motion  [B, F, J, C]
        """
        B = z.shape[0]
        F = num_frames or self.dit.num_frames
        J = self.dit.num_joints
        C = self.dit.coord_dim
        device = z.device

        sigma = self.sigma_min
        x = torch.randn(B, F, J, C, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            tval = i * dt
            t = torch.full((B,), tval, device=device)
            x1_hat = self.dit(x, t, z)                    # clean-sample prediction
            denom = max(1.0 - (1.0 - sigma) * tval, 1e-6)
            eps_hat = (x - tval * x1_hat) / denom         # implied noise
            v = x1_hat - (1.0 - sigma) * eps_hat          # PF-ODE velocity
            x = x + v * dt

        return x


def build_motion_decoder(
    num_frames: int = 243,
    num_joints: int = 17,
    cond_dim: int = 512,
    dim: int = 512,
    depth: int = 6,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    t_dim: int = 256,
    coord_dim: int = 6,
    sigma_min: float = 1e-4,
) -> MotionDecoder:
    dit = MotionDiT(
        num_frames=num_frames,
        num_joints=num_joints,
        coord_dim=coord_dim,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        cond_dim=cond_dim,
        t_dim=t_dim,
    )
    return MotionDecoder(dit, sigma_min=sigma_min)


if __name__ == "__main__":
    B, num_frames, num_joints = 2, 243, 17
    cond_dim = 512

    decoder = build_motion_decoder(cond_dim=cond_dim)
    total = sum(p.numel() for p in decoder.parameters())
    print(f"total parameters: {total:,}")

    z = torch.randn(B, cond_dim)
    x1 = torch.randn(B, num_frames, num_joints, 3)

    loss, logs = decoder.compute_loss(x1, z)
    print(f"compute_loss  → loss={loss.item():.4f}  logs={logs}")

    motion = decoder.predict_motion(z, num_steps=10)
    print(f"predict_motion → {list(motion.shape)}  (expected [{B}, {num_frames}, {num_joints}, 3])")
