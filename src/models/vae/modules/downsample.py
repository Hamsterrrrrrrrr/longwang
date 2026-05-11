"""Downsampling layers for 3D VAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Downsample3D(nn.Module):
    """3D downsample: always 2x spatial (strided conv or avg pool), optionally 2x temporal (avg pool).

    Uses uniform pooling (no first-frame special handling) so encoder/decoder shapes match exactly.
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
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        if self.compress_time:
            x = rearrange(x, 'b c t h w -> (b h w) c t')
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=B, h=H, w=W)

        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.conv(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', b=B)
        else:
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = rearrange(x, '(b t) c h w -> b c t h w', b=B)

        return x