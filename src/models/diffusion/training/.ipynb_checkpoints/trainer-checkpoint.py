"""Trainer for score-based diffusion model."""

import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..sde.vpsde import VPSDE
from .distributed import cleanup_distributed, is_main_process, setup_distributed
from .ema import EMA
from .scheduler import LambdaWarmUpCosineScheduler


class DiffusionTrainer:
    """Trainer for score-based diffusion model."""

    def __init__(
        self,
        sde: VPSDE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 2e-5,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0,
        max_steps: int = 100000,
        warmup_steps: int = 2000,
        grad_clip: Optional[float] = None,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        log_interval: int = 100,
        val_interval: int = 5000,
        save_interval: int = 5000,
        save_dir: str = "./checkpoints/diffusion",
        distributed: bool = False,
    ):
        if distributed:
            self.rank, self.world_size, self.local_rank = setup_distributed()
            self.device = f"cuda:{self.local_rank}"
        else:
            self.rank, self.world_size, self.local_rank = 0, 1, 0
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.distributed = distributed
        self.is_main = is_main_process(self.rank)

        # VPSDE wraps the score network; DDP wraps only the UNet (sde.eps).
        self.sde = sde.to(self.device)
        if distributed:
            self.sde.eps = DDP(self.sde.eps, device_ids=[self.local_rank], output_device=self.local_rank)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)

        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.train_log_path = self.save_dir / "train_losses.csv"
            self.val_log_path = self.save_dir / "val_losses.csv"
            if not self.train_log_path.exists():
                with open(self.train_log_path, 'w', newline='') as f:
                    csv.writer(f).writerow(['step', 'loss', 'lr'])
            if not self.val_log_path.exists():
                with open(self.val_log_path, 'w', newline='') as f:
                    csv.writer(f).writerow(['step', 'loss'])

        self.optimizer = torch.optim.AdamW(
            self.sde.parameters(), lr=lr, betas=betas, weight_decay=weight_decay,
        )

        self.lr_scheduler = LambdaWarmUpCosineScheduler(
            warm_up_steps=warmup_steps, lr_min=2e-5, lr_max=lr, lr_start=2e-5,
            max_decay_steps=max_steps,
        )
        self.torch_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_scheduler.schedule,
        )

        self.use_ema = use_ema
        if use_ema and self.is_main:
            self.ema = EMA(self.sde, decay=ema_decay)
            self.ema.train()
        else:
            self.ema = None

        self.global_step = 0
        self.epoch = 0

    def _clean_state_dict(self, state_dict: dict) -> dict:
        """Strip DDP 'module.' prefix from eps keys."""
        return {k.replace("eps.module.", "eps."): v for k, v in state_dict.items()}

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            x, c = batch
            return x.to(self.device), c.to(self.device)
        return batch.to(self.device), None

    def train_step(self, batch) -> float:
        x, c = self._unpack_batch(batch)
        self.sde.train()
        self.optimizer.zero_grad()
        loss = self.sde.loss(x, c=c)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.sde.parameters(), self.grad_clip)
        self.optimizer.step()
        self.torch_scheduler.step()
        if self.ema is not None:
            self.ema.update()
        return loss.item()

    @torch.no_grad()
    def val_step(self, batch) -> float:
        x, c = self._unpack_batch(batch)
        self.sde.eval()
        return self.sde.loss(x, c=c).item()

    def train(self):
        if self.is_main:
            print(f"[diffusion] device={self.device} world_size={self.world_size} "
                  f"max_steps={self.max_steps} grad_clip={self.grad_clip} "
                  f"ema={self.use_ema} save_dir={self.save_dir}")

        train_iter = iter(self.train_loader)
        while self.global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(self.epoch)
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            train_loss = self.train_step(batch)

            if self.is_main and self.global_step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"[TRAIN] Step {self.global_step:6d} | loss: {train_loss:.6f} | lr: {lr:.2e}")
                with open(self.train_log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([self.global_step, train_loss, lr])

            # All ranks participate in validation to avoid NCCL desync.
            if (self.val_loader is not None and self.global_step % self.val_interval == 0
                    and self.global_step > 0):
                val_loss = self.validate()
                if self.is_main:
                    print(f"[VAL]   Step {self.global_step:6d} | loss: {val_loss:.6f}")
                    with open(self.val_log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([self.global_step, val_loss])

            if self.is_main and self.global_step % self.save_interval == 0 and self.global_step > 0:
                self.save_checkpoint()

            self.global_step += 1

        if self.is_main:
            self.save_checkpoint(final=True)
            print("[diffusion] training complete")

        if self.distributed:
            cleanup_distributed()

    @torch.no_grad()
    def validate(self) -> float:
        self.sde.eval()
        ctx = self.ema.ema_scope() if self.ema is not None else nullcontext()
        with ctx:
            losses = [self.val_step(batch) for batch in self.val_loader]
        avg_loss = sum(losses) / len(losses)

        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        return avg_loss

    def save_checkpoint(self, final: bool = False):
        suffix = "final" if final else f"step_{self.global_step}"
        ckpt_path = self.save_dir / f"checkpoint_{suffix}.pt"
        ema_path = self.save_dir / f"ema_{suffix}.pt"

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self._clean_state_dict(self.sde.state_dict()),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.torch_scheduler.state_dict(),
        }
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        torch.save(checkpoint, ckpt_path)

        if self.ema is not None:
            with self.ema.ema_scope():
                torch.save(self._clean_state_dict(self.sde.state_dict()), ema_path)

        print(f"[SAVE] {ckpt_path.name}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint.get("epoch", 0)

        model_sd = checkpoint["model_state_dict"]
        if self.distributed:
            # Re-add 'module.' for DDP-wrapped eps.
            model_sd = {k.replace("eps.", "eps.module."): v for k, v in model_sd.items()}
        elif any("eps.module." in k for k in model_sd):
            model_sd = self._clean_state_dict(model_sd)
        self.sde.load_state_dict(model_sd)

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.torch_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.is_main:
            print(f"[LOAD] Resumed from {path} at step {self.global_step}")