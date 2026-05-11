"""Self-attention layers for the 3D UNet."""

import torch
import torch.nn as nn
from torch import Tensor

from .normalize import group_norm


class SelfAttention3D(nn.Module):
    """Multi-head self-attention over flattened (T, H, W) dimensions.
    
    Args:
        channels: Number of input channels.
        num_heads: Number of attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()

        self.norm = group_norm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:

        B, C, T, H, W = x.shape

        h = self.norm(x)
        h = h.reshape(B, C, T * H * W).permute(0, 2, 1)  # (B, T*H*W, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, T, H, W)

        return x + h