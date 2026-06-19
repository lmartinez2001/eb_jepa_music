import torch
import torch.nn as nn


class MusicConditionedPredictor(nn.Module):
    """Residual MLP predictor for one-step music-conditioned latent dynamics.

    This is the vector-latent analogue of the action-conditioned JEPA predictor:
    it maps the current dance latent and the aligned music chunk embedding to the
    next dance latent.
    """

    def __init__(
        self,
        state_dim: int = 512,
        music_dim: int = 768,
        hidden_dim: int = 1024,
        depth: int = 3,
        dropout: float = 0.0,
        residual: bool = True,
        final_norm: bool = True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.state_dim = state_dim
        self.music_dim = music_dim
        self.residual = residual
        layers = []
        in_dim = state_dim + music_dim
        for i in range(depth):
            layers.extend(
                [
                    nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, state_dim)
        self.final_norm = nn.LayerNorm(state_dim) if final_norm else nn.Identity()

    def forward(self, state: torch.Tensor, music: torch.Tensor) -> torch.Tensor:
        """Predict the next latent state.

        Args:
            state: Current latent, shape [B, D].
            music: Aligned music chunk embedding, shape [B, M].

        Returns:
            Predicted next latent, shape [B, D].
        """
        if state.ndim != 2:
            raise ValueError(f"state must have shape [B, D], got {tuple(state.shape)}")
        if music.ndim != 2:
            raise ValueError(f"music must have shape [B, M], got {tuple(music.shape)}")
        if state.shape[0] != music.shape[0]:
            raise ValueError("state and music batch sizes must match")
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state last dim must be {self.state_dim}, got {state.shape[-1]}"
            )
        if music.shape[-1] != self.music_dim:
            raise ValueError(
                f"music last dim must be {self.music_dim}, got {music.shape[-1]}"
            )

        x = torch.cat([state, music], dim=-1)
        delta = self.out(self.net(x))
        pred = state + delta if self.residual else delta
        return self.final_norm(pred)


class MusicGRUPredictor(nn.Module):
    """GRU predictor for multi-step music-conditioned latent rollout.

    The initial hidden state is the current dance latent. Each music embedding in
    the sequence drives one latent transition.
    """

    is_rnn = True
    context_length = 0

    def __init__(
        self,
        state_dim: int = 512,
        music_dim: int = 768,
        num_layers: int = 1,
        dropout: float = 0.0,
        final_norm: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.music_dim = music_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size=music_dim,
            hidden_size=state_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(state_dim) if final_norm else nn.Identity()

    def forward(self, state: torch.Tensor, music: torch.Tensor) -> torch.Tensor:
        """Roll latent state forward with one or more music embeddings.

        Args:
            state: Current latent, shape [B, D].
            music: Music embedding sequence, shape [B, T, M]. A one-step input
                with shape [B, M] is also accepted.

        Returns:
            If music is [B, M], returns [B, D].
            If music is [B, T, M], returns [B, T, D].
        """
        if state.ndim != 2:
            raise ValueError(f"state must have shape [B, D], got {tuple(state.shape)}")
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state last dim must be {self.state_dim}, got {state.shape[-1]}"
            )

        single_step = music.ndim == 2
        if single_step:
            music = music.unsqueeze(1)
        elif music.ndim != 3:
            raise ValueError(
                f"music must have shape [B, M] or [B, T, M], got {tuple(music.shape)}"
            )
        if state.shape[0] != music.shape[0]:
            raise ValueError("state and music batch sizes must match")
        if music.shape[-1] != self.music_dim:
            raise ValueError(
                f"music last dim must be {self.music_dim}, got {music.shape[-1]}"
            )

        h0 = state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        pred, _ = self.rnn(music, h0)
        pred = self.final_norm(pred)
        return pred[:, -1] if single_step else pred
