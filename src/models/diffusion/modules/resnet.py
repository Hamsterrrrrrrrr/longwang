"""Residual blocks for the 3D UNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .normalize import group_norm


class ResBlock3D(nn.Module):
    """3D residual block with Adaptive Group Normalisation (AdaGN).

    Conditioning (time + optional spatiotemporal) modulates features via learned
    per-channel scale and shift at each norm layer:
        AdaGN(h, cond) = GroupNorm(h) * (1 + gamma(cond)) + beta(cond)
    """

    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = group_norm(channels)
        self.norm2 = group_norm(channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 4),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        gamma1, beta1, gamma2, beta2 = (
            p[..., None, None, None] for p in self.cond_proj(cond).chunk(4, dim=-1)
        )

        h = self.norm1(x) * (1 + gamma1) + beta1
        h = self.conv1(F.silu(h))

        h = self.norm2(h) * (1 + gamma2) + beta2
        h = self.conv2(F.silu(h))

        return x + h