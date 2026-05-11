"""Probability distributions for VAE."""

from typing import Optional, Tuple

import numpy as np
import torch


class DiagonalGaussianDistribution:
    """Diagonal Gaussian over VAE latents, parameterised by concatenated [mean, logvar]."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.deterministic = deterministic

        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reparameterised sample: mean + std * eps."""
        if self.deterministic:
            return self.mean
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        """KL(self || other), or KL(self || N(0, I)) if other is None. Returns shape (B,)."""
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device)

        reduce_dims = list(range(1, self.mean.ndim))

        if other is None:
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=reduce_dims,
            )
        return 0.5 * torch.sum(
            (self.mean - other.mean).pow(2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=reduce_dims,
        )

    def nll(
        self,
        sample: torch.Tensor,
        reduce_dims: Tuple[int, ...] = (1, 2, 3, 4),
    ) -> torch.Tensor:
        """Negative log-likelihood of `sample` under this distribution."""
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device)
        log_two_pi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            log_two_pi + self.logvar + (sample - self.mean).pow(2) / self.var,
            dim=reduce_dims,
        )