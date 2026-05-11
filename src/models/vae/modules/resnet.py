"""ResNet blocks for 3D VAE."""

from typing import Optional

import torch
import torch.nn as nn

from .conv import CausalConv3d
from .normalize import GroupNorm3d, SpatialNorm3D
from ..utils import get_activation


class ResnetBlockCausal3D(nn.Module):
    """3D ResNet block with causal convolutions; used in the encoder (no z conditioning)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_groups: int = 32,
        eps: float = 1e-6,
        act_fn: str = "silu",
        output_scale_factor: float = 1.0,
        use_shortcut_conv: Optional[bool] = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor

        self.norm1 = GroupNorm3d(in_channels, num_groups=num_groups, eps=eps)
        self.act1 = get_activation(act_fn)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)

        self.norm2 = GroupNorm3d(out_channels, num_groups=num_groups, eps=eps)
        self.act2 = get_activation(act_fn)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1)

        if use_shortcut_conv is None:
            use_shortcut_conv = in_channels != out_channels
        self.conv_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1)
            if use_shortcut_conv else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.dropout(self.act2(self.norm2(x))))
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(shortcut)
        return (shortcut + x) / self.output_scale_factor


class ResnetBlockSpatialNorm3D(nn.Module):
    """3D ResNet block with SpatialNorm3D conditioning; used in the decoder (conditioned on z)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        z_channels: int = 4,
        dropout: float = 0.0,
        num_groups: int = 32,
        eps: float = 1e-6,
        act_fn: str = "silu",
        output_scale_factor: float = 1.0,
        use_shortcut_conv: Optional[bool] = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor

        self.norm1 = SpatialNorm3D(in_channels, z_channels, num_groups=num_groups, eps=eps)
        self.act1 = get_activation(act_fn)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)

        self.norm2 = SpatialNorm3D(out_channels, z_channels, num_groups=num_groups, eps=eps)
        self.act2 = get_activation(act_fn)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1)

        if use_shortcut_conv is None:
            use_shortcut_conv = in_channels != out_channels
        self.conv_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1)
            if use_shortcut_conv else None
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(self.act1(self.norm1(x, z)))
        x = self.conv2(self.dropout(self.act2(self.norm2(x, z))))
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(shortcut)
        return (shortcut + x) / self.output_scale_factor