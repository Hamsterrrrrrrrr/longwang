"""ERA5 precipitation dataset and dataloaders for VAE training."""

import json

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset


class ERA5PrecipitationDataset(Dataset):
    """ERA5 precipitation patches in Zarr format.

    Pipeline: meters -> mm -> log(x + eps) - log(eps) -> normalise to [-1, 1].
    """

    def __init__(self, zarr_path, stats_path, transform=None):
        store = zarr.open(zarr_path, 'r')
        self.data = store['precipitation']
        self.metadata = dict(store.attrs)

        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        self.epsilon = self.stats['epsilon']
        self.norm_min = self.stats['norm_min']
        self.norm_max = self.stats['norm_max']
        self.norm_range = self.stats['norm_range']

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """Returns tensor of shape (1, T, H, W) normalised to [-1, 1]."""
        sample = self.data[idx] * 1000.0
        sample = np.log(sample + self.epsilon) - np.log(self.epsilon)
        sample = 2.0 * (sample - self.norm_min) / self.norm_range - 1.0
        sample = torch.from_numpy(sample[np.newaxis, ...]).float()
        if self.transform:
            sample = self.transform(sample)
        return sample

    def denormalize(self, x_norm):
        """Invert normalisation + log transform: [-1, 1] -> millimetres."""
        x_log = (x_norm + 1.0) / 2.0 * self.norm_range + self.norm_min
        return torch.exp(x_log + np.log(self.epsilon)) - self.epsilon


def create_era5_dataloaders(
    train_zarr_path,
    val_zarr_path,
    stats_path,
    batch_size,
    num_workers=8,
    pin_memory=True,
    world_size=1,
    rank=0,
):
    """Create train + val dataloaders, optionally with DistributedSampler for DDP."""
    train_dataset = ERA5PrecipitationDataset(train_zarr_path, stats_path)
    val_dataset = ERA5PrecipitationDataset(val_zarr_path, stats_path)

    if rank == 0:
        print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}  "
              f"Sample: {train_dataset.data.shape[1:]}  "
              f"Batch: {batch_size} x {world_size} GPUs = {batch_size * world_size}")

    train_sampler = val_sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader