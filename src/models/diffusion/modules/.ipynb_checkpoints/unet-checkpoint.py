"""3D UNet for score-based diffusion in latent space."""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .attention import SelfAttention3D
from .downsample import Downsample3D
from .normalize import group_norm
from .resnet import ResBlock3D
from .time_embedding import TimeEmbedding
from .upsample import Upsample3D


class UNet3D(nn.Module):
    """3D UNet with time conditioning for score-based diffusion.

    Features:
        - Adaptive Group Normalisation (AdaGN) for time/condition injection
        - Concatenative skip connections with 1x1x1 projection
        - Asymmetric stride support for spatiotemporal data
        - Multi-level self-attention (configurable per level)
        - Optional spatiotemporal conditioning (lat, lon, month)
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        hidden_channels: Sequence[int] = (512, 512, 1024),
        hidden_blocks: Sequence[int] = (4, 4),
        strides: Sequence[tuple] = ((1, 2, 2), (2, 2, 2)),
        time_embed_dim: int = 256,
        attention_heads: int = 8,
        attention_levels: Optional[Sequence[bool]] = None,
        kernel_size: int = 3,
        cond_channels: int = 0,
    ):
        super().__init__()

        assert len(hidden_channels) == len(hidden_blocks) + 1
        assert len(strides) == len(hidden_blocks)

        if attention_levels is None:
            attention_levels = [True] * len(hidden_blocks)
        assert len(attention_levels) == len(hidden_blocks)

        self.time_embedding = TimeEmbedding(time_embed_dim)

        if cond_channels > 0:
            self.cond_embedding = nn.Sequential(
                nn.Linear(cond_channels, 128), nn.SiLU(),
                nn.Linear(128, 256), nn.SiLU(),
                nn.Linear(256, time_embed_dim),
            )
        else:
            self.cond_embedding = None

        padding = kernel_size // 2
        self.input_conv = nn.Conv3d(
            in_channels, hidden_channels[0], kernel_size=kernel_size, padding=padding,
        )

        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, n_blocks in enumerate(hidden_blocks):
            self.encoder_blocks.append(nn.ModuleList(
                ResBlock3D(hidden_channels[i], time_embed_dim, kernel_size)
                for _ in range(n_blocks)
            ))
            self.encoder_attns.append(
                SelfAttention3D(hidden_channels[i], attention_heads)
                if attention_levels[i] and attention_heads > 0 else nn.Identity()
            )
            self.downsamples.append(Downsample3D(
                hidden_channels[i], hidden_channels[i + 1], kernel_size, strides[i],
            ))

        bottleneck_ch = hidden_channels[-1]
        self.bottleneck_block1 = ResBlock3D(bottleneck_ch, time_embed_dim, kernel_size)
        self.bottleneck_attn = (
            SelfAttention3D(bottleneck_ch, attention_heads)
            if attention_heads > 0 else nn.Identity()
        )
        self.bottleneck_block2 = ResBlock3D(bottleneck_ch, time_embed_dim, kernel_size)

        self.upsamples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()

        for i in reversed(range(len(hidden_blocks))):
            self.upsamples.append(Upsample3D(
                hidden_channels[i + 1], hidden_channels[i], kernel_size, strides[i],
            ))
            self.skip_projs.append(
                nn.Conv3d(hidden_channels[i] * 2, hidden_channels[i], kernel_size=1)
            )
            self.decoder_blocks.append(nn.ModuleList(
                ResBlock3D(hidden_channels[i], time_embed_dim, kernel_size)
                for _ in range(hidden_blocks[i])
            ))
            self.decoder_attns.append(
                SelfAttention3D(hidden_channels[i], attention_heads)
                if attention_levels[i] and attention_heads > 0 else nn.Identity()
            )

        self.output_norm = group_norm(hidden_channels[0])
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv3d(
            hidden_channels[0], out_channels, kernel_size=kernel_size, padding=padding,
        )

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        t_emb = self.time_embedding(t)
        if c is not None and self.cond_embedding is not None:
            cond = t_emb + self.cond_embedding(c)
        else:
            cond = t_emb

        x = self.input_conv(x)

        skips = []
        for blocks, attn, downsample in zip(
            self.encoder_blocks, self.encoder_attns, self.downsamples
        ):
            for block in blocks:
                x = block(x, cond)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        x = self.bottleneck_block1(x, cond)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_block2(x, cond)

        for upsample, skip_proj, blocks, attn in zip(
            self.upsamples, self.skip_projs, self.decoder_blocks, self.decoder_attns
        ):
            x = torch.cat([upsample(x), skips.pop()], dim=1)
            x = skip_proj(x)
            for block in blocks:
                x = block(x, cond)
            x = attn(x)

        x = self.output_norm(x)
        x = self.output_act(x)
        x = self.output_conv(x)
        return x