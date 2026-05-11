"""Normalisation layers for 3D VAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import CausalConv3d


class GroupNorm3d(nn.GroupNorm):
    """GroupNorm for 5D tensors (B, C, T, H, W); auto-adjusts num_groups to divide num_channels."""

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


class SpatialNorm3D(nn.Module):
    """Spatially-adaptive normalisation: GroupNorm(x) * (1 + scale(z)) + bias(z).

    Re-injects latent information at every norm layer.
    """

    def __init__(
        self,
        f_channels: int,
        z_channels: int,
        num_groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.f_channels = f_channels
        self.z_channels = z_channels
        self.norm = GroupNorm3d(num_channels=f_channels, num_groups=num_groups, eps=eps, affine=False)
        self.conv_scale = CausalConv3d(z_channels, f_channels, kernel_size=1)
        self.conv_bias = CausalConv3d(z_channels, f_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.shape[2:] != x.shape[2:]:
            z = F.interpolate(z, size=x.shape[2:], mode='nearest')
        scale = self.conv_scale(z)
        bias = self.conv_bias(z)
        return self.norm(x) * (1 + scale) + bias