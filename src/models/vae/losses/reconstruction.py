"""Reconstruction losses for VAE: L1/L2 + central-difference spatial/temporal gradient losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributions import DiagonalGaussianDistribution


class GradientLoss(nn.Module):
    """Spatial + temporal gradient loss using central differences (boundaries dropped)."""

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type

    def spatial_gradient(self, x: torch.Tensor):
        """Returns (grad_x, grad_y) on the interior, both shaped (B, C, T, H-2, W-2)."""
        grad_x = (x[:, :, :, 1:-1, 2:] - x[:, :, :, 1:-1, :-2]) / 2.0
        grad_y = (x[:, :, :, 2:, 1:-1] - x[:, :, :, :-2, 1:-1]) / 2.0
        return grad_x, grad_y

    def temporal_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Returns grad_t of shape (B, C, T-2, H, W)."""
        return (x[:, :, 2:, :, :] - x[:, :, :-2, :, :]) / 2.0

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return torch.abs(pred - target).mean()
        return ((pred - target) ** 2).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        pred_gx, pred_gy = self.spatial_gradient(pred)
        target_gx, target_gy = self.spatial_gradient(target)
        spatial = (self.compute_loss(pred_gx, target_gx)
                   + self.compute_loss(pred_gy, target_gy)) / 2

        temporal = self.compute_loss(self.temporal_gradient(pred), self.temporal_gradient(target))
        return {'spatial': spatial, 'temporal': temporal}


class VAEReconstructionLoss(nn.Module):
    """L1/L2 reconstruction + spatial/temporal gradient losses + KL."""

    def __init__(
        self,
        rec_weight: float = 1.0,
        spatial_grad_weight: float = 1.0,
        temporal_grad_weight: float = 1.0,
        kl_weight: float = 1e-6,
        rec_loss_type: str = "l1",
        grad_loss_type: str = "l1",
    ):
        super().__init__()
        self.rec_weight = rec_weight
        self.spatial_grad_weight = spatial_grad_weight
        self.temporal_grad_weight = temporal_grad_weight
        self.kl_weight = kl_weight
        self.rec_loss_type = rec_loss_type
        self.gradient_loss = GradientLoss(loss_type=grad_loss_type)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        posterior: DiagonalGaussianDistribution,
    ):
        if self.rec_loss_type == "l1":
            rec_loss = F.l1_loss(pred, target)
        else:
            rec_loss = F.mse_loss(pred, target)

        grad = self.gradient_loss(pred, target)
        kl_loss = posterior.kl().mean()

        total_loss = (
            self.rec_weight * rec_loss
            + self.spatial_grad_weight * grad['spatial']
            + self.temporal_grad_weight * grad['temporal']
            + self.kl_weight * kl_loss
        )

        log = {
            'total_loss': total_loss.detach(),
            'rec_loss': (rec_loss * self.rec_weight).detach(),
            'spatial_grad_loss': (grad['spatial'] * self.spatial_grad_weight).detach(),
            'temporal_grad_loss': (grad['temporal'] * self.temporal_grad_weight).detach(),
            'kl_loss': (kl_loss * self.kl_weight).detach(),
        }
        return total_loss, log