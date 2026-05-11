"""Dataloader for pre-encoded VAE latents with per-channel normalisation."""

import json

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class LatentDataset(Dataset):
    """Pre-encoded VAE latents in Zarr, with per-channel normalisation.

    Optionally loads spatiotemporal metadata (lat, lon, month) and returns
    a conditioning vector with sin/cos encoding alongside the latent.
    """

    def __init__(self, zarr_path: str, stats_path: str,
                 metadata_path: str = None, key: str = 'latents'):
        super().__init__()
        self.data = zarr.open(zarr_path, 'r')[key]

        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.mean = np.array(stats['channel_mean'], dtype=np.float32).reshape(-1, 1, 1, 1)
        self.std = np.array(stats['channel_std'], dtype=np.float32).reshape(-1, 1, 1, 1)

        self.has_metadata = metadata_path is not None
        if self.has_metadata:
            meta = np.load(metadata_path)
            self.center_lat = meta['center_lat']
            self.center_lon = meta['center_lon']
            self.month = meta['month']
            assert len(self.center_lat) == self.data.shape[0], (
                f"Metadata length {len(self.center_lat)} != data length {self.data.shape[0]}"
            )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        """Returns normalised latent, or (latent, cond) if metadata is available."""
        z = (self.data[idx].astype(np.float32) - self.mean) / self.std
        z = torch.from_numpy(z)

        if self.has_metadata:
            cond = torch.tensor([
                self.center_lat[idx] / 90.0,
                np.sin(self.center_lon[idx] * np.pi / 180),
                np.cos(self.center_lon[idx] * np.pi / 180),
                np.sin(2 * np.pi * self.month[idx] / 12),
                np.cos(2 * np.pi * self.month[idx] / 12),
            ], dtype=torch.float32)
            return z, cond

        return z

    def denormalize(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised latent back to original scale."""
        mean = torch.from_numpy(self.mean).to(z_norm.device)
        std = torch.from_numpy(self.std).to(z_norm.device)
        return z_norm * std + mean


def create_latent_dataloaders(
    train_zarr_path: str,
    val_zarr_path: str,
    stats_path: str,
    batch_size: int,
    num_workers: int = 8,
    world_size: int = 1,
    rank: int = 0,
    train_metadata_path: str = None,
    val_metadata_path: str = None,
) -> tuple:
    """Create train + val dataloaders for latent data, optionally with DDP samplers."""
    train_dataset = LatentDataset(train_zarr_path, stats_path, train_metadata_path)
    val_dataset = LatentDataset(val_zarr_path, stats_path, val_metadata_path)

    if rank == 0:
        print(f"[latent] train={len(train_dataset):,} val={len(val_dataset):,} "
              f"shape={train_dataset.data.shape[1:]} cond={train_dataset.has_metadata} "
              f"batch={batch_size} x {world_size} GPUs = {batch_size * world_size}")

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    return train_loader, val_loader