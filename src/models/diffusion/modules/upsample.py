"""Upsampling layers for the 3D UNet."""

from typing import Sequence, Union

import torch.nn as nn
from torch import Tensor

from .normalize import group_norm


class Upsample3D(nn.Module):
    """3D upsample via nearest interpolation + Conv3d, also changing channel count."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        scale_factor: Union[int, Sequence[int]] = 2,
    ):
        super().__init__()
        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor, scale_factor)
        self.scale_factor = scale_factor
        self.norm = group_norm(in_channels)
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return self.conv(x)