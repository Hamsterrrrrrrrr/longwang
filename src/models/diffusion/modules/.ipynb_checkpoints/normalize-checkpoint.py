"""Normalization layers for the 3D UNet."""

import torch.nn as nn

def group_norm(channels: int, num_groups: int = 32, eps: float = 1e-6) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=eps, affine=True)