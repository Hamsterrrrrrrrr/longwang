"""3D VAE encoder."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .conv import CausalConv3d
from .downsample import Downsample3D
from .normalize import GroupNorm3d
from .resnet import ResnetBlockCausal3D


def nonlinearity(x):
    return x * torch.sigmoid(x)


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        z_channels: int = 4,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        double_z: bool = True,
        compress: int = 4,
    ):
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.compress_level = int(np.log2(compress))

        self.conv_in = CausalConv3d(in_channels, ch, kernel_size=3, stride=1)

        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks):
                block.append(ResnetBlockCausal3D(
                    in_channels=block_in, out_channels=block_out, dropout=dropout,
                ))
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                compress_time = i_level < self.compress_level
                down.downsample = Downsample3D(
                    in_channels=block_in, with_conv=True, compress_time=compress_time,
                )
            else:
                down.downsample = None
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlockCausal3D(
            in_channels=block_in, out_channels=block_in, dropout=dropout,
        )
        self.mid.block_2 = ResnetBlockCausal3D(
            in_channels=block_in, out_channels=block_in, dropout=dropout,
        )

        self.norm_out = GroupNorm3d(block_in, num_groups=32)
        conv_out_channels = 2 * z_channels if double_z else z_channels
        self.conv_out = CausalConv3d(block_in, conv_out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if self.down[i_level].downsample is not None:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h