"""Observation operators for zero-shot conditional generation."""

import torch
import torch.nn as nn
from torch import Tensor


class SpatioTemporalDownscaleOperator(nn.Module):
    """Conservative operator: HR daily precip → LR monthly total.

    Conservative regridding preserves total precipitation:
        pr_coarse = Σ(pr_fine × area_fine) / Σ(area_fine)
    """

    def __init__(self, lat: Tensor, spatial_scale: int = 4):
        super().__init__()
        self.spatial_scale = spatial_scale

        # Cell area weights: cos(lat) for each HR row.
        area = torch.cos(torch.deg2rad(lat))
        H, s = lat.shape[0], spatial_scale
        area_blocks = area.reshape(H // s, s)
        block_weights = area_blocks / area_blocks.sum(dim=1, keepdim=True)
        self.register_buffer('block_weights', block_weights)

    def spatial_coarsen(self, x: Tensor) -> Tensor:
        """Area-weighted spatial coarsening: (B, T, H, W) → (B, T, H//s, W//s)."""
        B, T, H, W = x.shape
        s = self.spatial_scale
        x = x.reshape(B, T, H // s, s, W // s, s)

        # H-direction: cos(lat) weighted average; broadcast (H//s, s) → (1, 1, H//s, s, 1, 1).
        w = self.block_weights.reshape(1, 1, -1, s, 1, 1)
        x = (x * w).sum(dim=3)

        # W-direction: uniform mean (cells at same lat share the same area).
        return x.mean(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """HR daily (B, T, H, W) → LR monthly total (B, H//s, W//s)."""
        return self.spatial_coarsen(x).sum(dim=1)