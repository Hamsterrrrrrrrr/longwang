"""Downsampling layers for the 3D UNet."""

from typing import Sequence, Union

import torch.nn as nn
from torch import Tensor


class Downsample3D(nn.Module):
    """3D downsample via strided Conv3d, also changing channel count."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, Sequence[int]] = 2,
    ):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)