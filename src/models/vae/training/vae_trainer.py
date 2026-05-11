"""VAE trainer with L1 + gradient + KL losses, single-GPU and DDP."""

import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..autoencoder import AutoencoderKL3D
from ..losses import VAEReconstructionLoss
from .distributed import cleanup_distributed, is_main_process, setup_distributed
from .ema import EMA
from .scheduler import LambdaWarmUpCosineScheduler


CSV_FIELDS = ['step', 'rec_loss', 'spatial_grad_loss', 'temporal_grad_loss', 'kl_loss', 'total_loss']
LOSS_KEYS = ['rec_loss', 'spatial_grad_loss', 'temporal_grad_loss', 'kl_loss', 'total_loss']


class VAETrainer:
    """Trainer for 3D VAE supporting single-GPU and multi-GPU (DDP) training."""

    def __init__(
        self,
        model: AutoencoderKL3D,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        rec_weight: float = 1.0,
        spatial_grad_weight: float = 1.0,
        temporal_grad_weight: float = 1.0,
        kl_weight: float = 1e-6,
        rec_loss_type: str = "l1",
        grad_loss_type: str = "l1",
        lr: float = 2e-4,
        betas: tuple = (0.9, 0.999),
        max_steps: int = 700000,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        val_interval: int = 10000,
        log_interval: int = 100,
        save_interval: int = 2000,
        save_dir: str = "./checkpoints",
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

        self.model = model.to(self.device)
        if distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_steps = max_steps
        self.val_interval = val_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)

        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.train_log_path = self.save_dir / "train_losses.csv"
            self.val_log_path = self.save_dir / "val_losses.csv"
            for path in (self.train_log_path, self.val_log_path):
                if not path.exists():
                    with open(path, 'w', newline='') as f:
                        csv.writer(f).writerow(CSV_FIELDS)

        self.loss_fn = VAEReconstructionLoss(
            rec_weight=rec_weight,
            spatial_grad_weight=spatial_grad_weight,
            temporal_grad_weight=temporal_grad_weight,
            kl_weight=kl_weight,
            rec_loss_type=rec_loss_type,
            grad_loss_type=grad_loss_type,
        )

        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=betas)
        self.lr_scheduler = LambdaWarmUpCosineScheduler(
            warm_up_steps=60000, lr_min=2e-5, lr_max=lr, lr_start=2e-6,
            max_decay_steps=max_steps, verbosity_interval=0,
        )
        self.torch_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_scheduler.schedule,
        )

        self.use_ema = use_ema
        if use_ema and self.is_main:
            base_model = self.model.module if distributed else self.model
            self.ema = EMA(base_model, decay=ema_decay)
            self.ema.train()
        else:
            self.ema = None

        self.global_step = 0
        self.epoch = 0

    def train_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = batch.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        recon, posterior = self.model(batch)
        loss, log = self.loss_fn(recon, batch, posterior)
        loss.backward()
        self.optimizer.step()
        self.torch_scheduler.step()
        if self.ema is not None:
            self.ema.update()
        return log

    @torch.no_grad()
    def val_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = batch.to(self.device)
        self.model.eval()
        recon, posterior = self.model(batch)
        _, log = self.loss_fn(recon, batch, posterior)
        return log

    def train(self):
        if self.is_main:
            print(f"[VAE] device={self.device} world_size={self.world_size} "
                  f"max_steps={self.max_steps} ema={self.use_ema} save_dir={self.save_dir}")

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

            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            train_log = self.train_step(batch)

            if self.is_main and self.global_step % self.log_interval == 0:
                self._print_log(train_log, mode='train')
                self._save_log(train_log, mode='train')

            # All ranks participate in validation to avoid NCCL timeout.
            if self.val_loader is not None and self.global_step % self.val_interval == 0:
                val_log = self.validate()
                if self.is_main and val_log is not None:
                    self._print_log(val_log, mode='val')
                    self._save_log(val_log, mode='val')

            if self.is_main and self.global_step % self.save_interval == 0 and self.global_step > 0:
                self.save_checkpoint()

            self.global_step += 1

        if self.is_main:
            self.save_checkpoint(final=True)
            print("[VAE] training complete")

        if self.distributed:
            cleanup_distributed()

    def _extract_losses(self, log: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {k: log.get(k, torch.tensor(0.0)).item() for k in LOSS_KEYS}

    def _print_log(self, log: Dict[str, torch.Tensor], mode: str = 'train'):
        v = self._extract_losses(log)
        prefix = "[TRAIN]" if mode == 'train' else "[VAL]  "
        lr = self.optimizer.param_groups[0]['lr']
        print(f"{prefix} Step {self.global_step:6d} | "
              f"total: {v['total_loss']:.4f} | rec: {v['rec_loss']:.4f} | "
              f"grad_s: {v['spatial_grad_loss']:.4f} | grad_t: {v['temporal_grad_loss']:.4f} | "
              f"kl: {v['kl_loss']:.6f} | lr: {lr:.2e}")

    def _save_log(self, log: Dict[str, torch.Tensor], mode: str = 'train'):
        v = self._extract_losses(log)
        path = self.train_log_path if mode == 'train' else self.val_log_path
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow([self.global_step] + [v[k] for k in LOSS_KEYS[:-1]] + [v['total_loss']])

    @torch.no_grad()
    def validate(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.val_loader is None:
            return None

        self.model.eval()
        # Skip EMA in distributed mode to keep all ranks synchronised.
        use_ema = self.ema is not None and not self.distributed
        ctx = self.ema.ema_scope() if use_ema else nullcontext()

        with ctx:
            val_logs = []
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                val_logs.append(self.val_step(batch))

        avg_log = {
            k: torch.stack([l[k] for l in val_logs]).mean()
            for k in ['rec_loss', 'spatial_grad_loss', 'temporal_grad_loss', 'kl_loss']
        }
        avg_log["total_loss"] = sum(avg_log.values())

        self.model.train()
        return avg_log

    def save_checkpoint(self, final: bool = False):
        suffix = "final" if final else f"step_{self.global_step}"
        ckpt_path = self.save_dir / f"checkpoint_{suffix}.pt"
        ema_path = self.save_dir / f"ema_{suffix}.pt"

        base_model = self.model.module if self.distributed else self.model

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.torch_scheduler.state_dict(),
        }
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        torch.save(checkpoint, ckpt_path)

        if self.ema is not None:
            with self.ema.ema_scope():
                torch.save(base_model.state_dict(), ema_path)

        print(f"[SAVE] {ckpt_path.name}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        base_model = self.model.module if self.distributed else self.model

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint.get("epoch", 0)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.torch_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.is_main:
            print(f"[LOAD] Resumed from {path} at step {self.global_step}")