"""Time embedding for diffusion timestep conditioning."""

import torch
import torch.nn as nn
from torch import Tensor


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding following SDA (Rozet & Louppe, 2023).
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256, n_freqs: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_freqs * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.register_buffer('freqs', torch.pi * torch.arange(1, n_freqs + 1))

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        return self.mlp(t)