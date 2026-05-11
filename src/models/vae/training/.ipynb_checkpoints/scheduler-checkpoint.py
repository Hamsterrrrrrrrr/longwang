"""Learning rate schedulers."""

import numpy as np


class LambdaWarmUpCosineScheduler:
    """Linear warmup from lr_start to lr_max, then cosine decay to lr_min.
    """

    def __init__(
        self,
        warm_up_steps: int,
        lr_min: float,
        lr_max: float,
        lr_start: float,
        max_decay_steps: int,
        verbosity_interval: int = 0,
    ):
        self.warm_up_steps = warm_up_steps
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_start = lr_start
        self.max_decay_steps = max_decay_steps
        self.verbosity_interval = verbosity_interval
        self.last_lr = 0.0

    def schedule(self, n: int) -> float:
        if self.verbosity_interval > 0 and n % self.verbosity_interval == 0:
            print(f"Step {n}, LR multiplier: {self.last_lr:.6f}")

        if n < self.warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.warm_up_steps * n + self.lr_start
        else:
            t = min((n - self.warm_up_steps) / (self.max_decay_steps - self.warm_up_steps), 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))

        self.last_lr = lr
        return lr / self.lr_max

    def __call__(self, n: int) -> float:
        return self.schedule(n)