"""Variance Preserving SDE for score-based diffusion."""

import math

import torch
import torch.nn as nn
from torch import Size, Tensor
from tqdm import tqdm


class VPSDE(nn.Module):
    """Variance Preserving SDE noise scheduler.

    Following SDA (Rozet & Louppe, 2023).
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
        noise_d: float = 64,
        image_d: float = 32,
    ):
        super().__init__()
        self.eps = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta
        self.noise_d = noise_d
        self.image_d = image_d
        self._shifted = (alpha == 'shifted_cos')

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'shifted_cos':
            self.alpha = lambda t: torch.sigmoid(
                -2 * torch.log(torch.tan(torch.pi * t.clamp(1e-4, 1 - 1e-4) / 2))
                + 2 * math.log(self.noise_d / self.image_d)
            ).sqrt()
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t ** 2)
        else:
            raise ValueError(f"Unknown alpha schedule: {alpha}")

        self.register_buffer('_device_tracker', torch.empty(()))

    @property
    def device(self):
        return self._device_tracker.device

    def mu(self, t: Tensor) -> Tensor:
        """Signal coefficient μ(t) = α(t)."""
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        """Noise coefficient σ(t)."""
        if self._shifted:
            return (1 - self.alpha(t) ** 2).sqrt()
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        """Sample x(t) from the perturbation kernel; returns (x(t), ε) if train=True."""
        t = t.reshape(t.shape + (1,) * len(self.shape))
        eps = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * eps
        return (x, eps) if train else x

    def loss(self, x: Tensor, c: Tensor = None, w: Tensor = None) -> Tensor:
        """Denoising score matching loss (ε-parameterisation)."""
        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x, eps = self.forward(x, t, train=True)
        err = (self.eps(x, t, c) - eps).square()
        if w is None:
            return err.mean()
        return (err * w).mean() / w.mean()

    @torch.no_grad()
    def sample(
        self,
        shape: Size = (),
        c: Tensor = None,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        """Reverse-SDE sampling: exponential-integrator predictor + optional Langevin corrector."""
        x = torch.randn(shape + self.shape, device=self.device)
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1, device=self.device)
        dt = 1 / steps

        for t in tqdm(time[:-1], ncols=88):
            # Predictor (exponential integrator).
            r = self.mu(t - dt) / self.mu(t)
            x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t, c)

            # Corrector (Langevin Monte Carlo).
            for _ in range(corrections):
                z = torch.randn_like(x)
                eps = self.eps(x, t - dt, c)
                delta = tau / eps.square().mean(dim=self.dims, keepdim=True)
                x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(t - dt)

        return x.reshape(shape + self.shape)