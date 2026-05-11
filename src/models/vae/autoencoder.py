"""3D VAE for spatiotemporal compression (CogVideo-style)."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .distributions import DiagonalGaussianDistribution
from .modules.decoder import Decoder3D
from .modules.encoder import Encoder3D
from .utils import crop_to_original_size, pad_to_compatible_size


@dataclass
class VAE3DConfig:
    in_channels: int = 1
    out_channels: Optional[int] = None
    z_channels: int = 4
    ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 4
    compress: int = 4
    dropout: float = 0.0
    sample_posterior: bool = True


class AutoencoderKL3D(nn.Module):
    """3D VAE compressing (B, C, T, H, W) to (B, z_channels, T', H', W'),
    where T' = T / compress and H', W' = H, W / 2^(len(ch_mult) - 1)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        z_channels: int = 4,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 4,
        compress: int = 4,
        dropout: float = 0.0,
        sample_posterior: bool = True,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.compress = compress
        self.spatial_compress = 2 ** (len(ch_mult) - 1)
        self.sample_posterior = sample_posterior

        self.encoder = Encoder3D(
            in_channels=in_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            double_z=True,
            compress=compress,
        )
        self.decoder = Decoder3D(
            out_channels=out_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            compress=compress,
        )

    @classmethod
    def from_config(cls, config: VAE3DConfig) -> "AutoencoderKL3D":
        return cls(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            z_channels=config.z_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            compress=config.compress,
            dropout=config.dropout,
            sample_posterior=config.sample_posterior,
        )

    def encode(
        self,
        x: torch.Tensor,
        sample_posterior: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """Encode (B, C, T, H, W) input. Returns (latent, posterior)."""
        if sample_posterior is None:
            sample_posterior = self.sample_posterior

        x_padded, pad_info = pad_to_compatible_size(
            x,
            temporal_compression=self.compress,
            spatial_compression=self.spatial_compress,
        )
        h = self.encoder(x_padded)
        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample() if sample_posterior else posterior.mode()
        self._last_pad_info = pad_info
        return z, posterior

    def decode(
        self,
        z: torch.Tensor,
        pad_info: Optional[dict] = None,
    ) -> torch.Tensor:
        """Decode latent to (B, C, T, H, W). Uses pad_info from last encode() if not given."""
        x_recon = self.decoder(z)
        if pad_info is None:
            pad_info = getattr(self, "_last_pad_info", None)
        if pad_info is not None:
            x_recon = crop_to_original_size(x_recon, pad_info)
        return x_recon

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        if sample_posterior is None:
            sample_posterior = self.sample_posterior if self.training else False
        z, posterior = self.encode(x, sample_posterior=sample_posterior)
        x_recon = self.decode(z)
        return x_recon, posterior

    def get_latent_size(self, input_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Compute latent (T, H, W) for a given input (T, H, W), accounting for padding."""
        T, H, W = input_size
        sc = self.spatial_compress
        T_pad = ((T + self.compress - 1) // self.compress) * self.compress
        H_pad = ((H + sc - 1) // sc) * sc
        W_pad = ((W + sc - 1) // sc) * sc
        return (T_pad // self.compress, H_pad // sc, W_pad // sc)