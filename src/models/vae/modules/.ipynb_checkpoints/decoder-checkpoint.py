"""3D VAE decoder with SpatialNorm3D conditioning."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .conv import CausalConv3d
from .normalize import SpatialNorm3D
from .resnet import ResnetBlockSpatialNorm3D
from .upsample import Upsample3D


def nonlinearity(x):
    return x * torch.sigmoid(x)


class Decoder3D(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        z_channels: int = 4,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        compress: int = 4,
    ):
        super().__init__()

        self.ch = ch
        self.z_channels = z_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.compress_level = int(np.log2(compress))

        block_in = ch * ch_mult[-1]

        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, stride=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlockSpatialNorm3D(
            in_channels=block_in, out_channels=block_in,
            z_channels=z_channels, dropout=dropout,
        )
        self.mid.block_2 = ResnetBlockSpatialNorm3D(
            in_channels=block_in, out_channels=block_in,
            z_channels=z_channels, dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlockSpatialNorm3D(
                    in_channels=block_in, out_channels=block_out,
                    z_channels=z_channels, dropout=dropout,
                ))
                block_in = block_out

            up = nn.Module()
            up.block = block
            if i_level != 0:
                compress_time = i_level >= self.num_resolutions - self.compress_level
                up.upsample = Upsample3D(
                    in_channels=block_in, with_conv=True, compress_time=compress_time,
                )
            else:
                up.upsample = None
            self.up.insert(0, up)

        self.norm_out = SpatialNorm3D(block_in, z_channels, num_groups=32)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_cond = z
        h = self.conv_in(z)

        h = self.mid.block_1(h, z_cond)
        h = self.mid.block_2(h, z_cond)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, z_cond)
            if self.up[i_level].upsample is not None:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h, z_cond)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h