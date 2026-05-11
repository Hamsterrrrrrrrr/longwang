"""Exponential moving average of model weights (adapted from VQGAN's LitEma)."""

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


class EMA(nn.Module):
    """EMA shadow copy of model weights: shadow = decay * shadow + (1 - decay) * param.

    Use `ema.update()` after each optimiser step. Use `with ema.ema_scope():`
    to temporarily swap the model to EMA weights for inference/validation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, use_num_updates: bool = True):
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")

        self.model = model
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.num_updates = 0

        self.shadow_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup_params = {}

    def get_decay(self) -> float:
        if self.use_num_updates:
            return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        return self.decay

    @torch.no_grad()
    def update(self):
        if not self.training:
            return
        decay = self.get_decay()
        self.num_updates += 1
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def copy_to_model(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def copy_from_model(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].copy_(param.data)

    def store(self):
        self.backup_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params = {}

    @contextmanager
    def ema_scope(self, context: Optional[str] = None):
        """Temporarily swap model to EMA weights; restored on exit."""
        self.store()
        self.copy_to_model()
        if context is not None:
            print(f"{context}: Switched to EMA weights")
        try:
            yield
        finally:
            self.restore()
            if context is not None:
                print(f"{context}: Restored training weights")

    def state_dict(self):
        return {"shadow_params": self.shadow_params, "num_updates": self.num_updates}

    def load_state_dict(self, state_dict):
        self.shadow_params = state_dict["shadow_params"]
        self.num_updates = state_dict["num_updates"]