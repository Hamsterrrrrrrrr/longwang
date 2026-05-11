"""Upsampling layers for 3D VAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Upsample3D(nn.Module):
    """3D upsample: always 2x spatial (nearest), optionally 2x temporal (nearest), then optional conv.

    Uses uniform interpolation so encoder/decoder shapes match exactly.
    """

    def __init__(
        self,
        in_channels: int,
        with_conv: bool = True,
        compress_time: bool = False,
    ):
        super().__init__()
        self.with_conv = with_conv
        self.compress_time = compress_time
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            x = F.interpolate(x, scale_factor=(2, 2, 2), mode='nearest')
        else:
            T = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = rearrange(x, '(b t) c h w -> b c t h w', t=T)

        if self.with_conv:
            T = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.conv(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=T)

        return x