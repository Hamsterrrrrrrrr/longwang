"""Distributed training utilities."""

import datetime
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_distributed():
    """Init distributed training. Returns (rank, world_size, local_rank)."""
    if "RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
    world_size: int = 1,
    rank: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
):
    """DataLoader with DistributedSampler when world_size > 1."""
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None

    use_persistent = persistent_workers and num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=use_persistent,
        drop_last=True,
    )