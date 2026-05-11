"""Zero-shot conditioning via likelihood score estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from .vpsde import VPSDE


class GaussianScore(nn.Module):
    """Score module for Gaussian inverse problems.

    Enables zero-shot conditional generation via Bayes' rule.

    Following SDA (Rozet & Louppe, 2023).
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: 'VPSDE',
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
    ):
        super().__init__()
        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))
        self.A = A
        self.sde = sde
        self.detach = detach

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        if self.detach:
            eps = self.sde.eps(x, t, c)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            if not self.detach:
                eps = self.sde.eps(x, t, c)
            # Tweedie's estimate of clean data.
            x_hat = (x - sigma * eps) / mu
            # Stabilised log-likelihood.
            err = self.y - self.A(x_hat)
            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2
            log_p = -(err ** 2 / var).sum() / 2

        s, = torch.autograd.grad(log_p, x)
        return eps - sigma * s