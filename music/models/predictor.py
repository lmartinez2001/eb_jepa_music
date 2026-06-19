from typing import Optional

import torch
import torch.nn as nn


class MusicRNNPredictor(nn.Module):
    """GRU-based one-step predictor conditioned on a music embedding.

    This mirrors ``eb_jepa.architectures.RNNPredictor``: the current latent state
    initializes the GRU hidden state, and the conditioning vector is the GRU
    input for one transition.
    """

    is_rnn = True
    context_length = 0

    def __init__(
        self,
        state_dim: int = 512,
        music_dim: int = 768,
        num_layers: int = 1,
        final_ln: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.music_dim = music_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size=music_dim,
            hidden_size=state_dim,
            num_layers=num_layers,
        )
        self.final_ln = final_ln if final_ln is not None else nn.Identity()

    def forward(self, state: torch.Tensor, music: torch.Tensor) -> torch.Tensor:
        """Predict one next latent state.

        Args:
            state: Current latent, shape ``[B, D]``.
            music: Music condition, shape ``[B, M]`` or ``[B, M, 1]``.

        Returns:
            Next latent prediction, shape ``[B, D]``.
        """
        if state.ndim != 2:
            raise ValueError(f"state must have shape [B, D], got {tuple(state.shape)}")
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state last dim must be {self.state_dim}, got {state.shape[-1]}"
            )

        if music.ndim == 3:
            if music.shape[-1] != 1:
                raise ValueError(
                    f"3D music must have shape [B, M, 1], got {tuple(music.shape)}"
                )
            music = music.squeeze(-1)
        elif music.ndim != 2:
            raise ValueError(
                f"music must have shape [B, M] or [B, M, 1], got {tuple(music.shape)}"
            )

        if state.shape[0] != music.shape[0]:
            raise ValueError("state and music batch sizes must match")
        if music.shape[-1] != self.music_dim:
            raise ValueError(
                f"music last dim must be {self.music_dim}, got {music.shape[-1]}"
            )

        rnn_state = state.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        rnn_input = music.unsqueeze(0).contiguous()
        next_state, _ = self.rnn(rnn_input, rnn_state)
        next_state = self.final_ln(next_state)
        return next_state[-1]


# Backward-compatible names for earlier local imports.
MusicGRUPredictor = MusicRNNPredictor
MusicConditionedPredictor = MusicRNNPredictor
