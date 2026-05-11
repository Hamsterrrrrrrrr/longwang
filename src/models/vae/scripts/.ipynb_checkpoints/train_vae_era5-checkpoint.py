#!/usr/bin/env python3
"""Train 3D VAE on ERA5 precipitation patches.

Usage:
    Single GPU:  python train_vae_era5.py
    Multi-GPU:   torchrun --nproc_per_node=8 train_vae_era5.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vae import AutoencoderKL3D, VAE3DConfig
from vae.training import VAETrainer, create_era5_dataloaders


def main():
    DATA_DIR = "/data/era5/patches"
    TRAIN_ZARR = os.path.join(DATA_DIR, "train_patches.zarr")
    VAL_ZARR = os.path.join(DATA_DIR, "val_patches.zarr")
    STATS_JSON = os.path.join(DATA_DIR, "train_stats.json")
    SAVE_DIR = "/data/era5/experiments/vae_zch16_chmul124_res4"
    os.makedirs(SAVE_DIR, exist_ok=True)

    channels = 1
    frames, height, width = 32, 128, 128
    batch_size = 1
    num_workers = 8

    model_config = VAE3DConfig(
        in_channels=channels,
        z_channels=16,
        ch=128,
        ch_mult=(1, 2, 4),
        num_res_blocks=4,
        compress=4,
        dropout=0.0,
        sample_posterior=True,
    )

    max_steps = 700000
    lr = 2e-4
    rec_weight = 1.0
    spatial_grad_weight = 10.0
    temporal_grad_weight = 10.0
    kl_weight = 1e-6

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    train_loader, val_loader = create_era5_dataloaders(
        train_zarr_path=TRAIN_ZARR,
        val_zarr_path=VAL_ZARR,
        stats_path=STATS_JSON,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        world_size=world_size,
        rank=rank,
    )

    model = AutoencoderKL3D.from_config(model_config)

    if rank == 0:
        config_dict = {
            'model': {
                'in_channels': channels,
                'z_channels': model_config.z_channels,
                'ch': model_config.ch,
                'ch_mult': list(model_config.ch_mult),
                'num_res_blocks': model_config.num_res_blocks,
                'compress': model_config.compress,
                'dropout': model_config.dropout,
            },
            'training': {
                'max_steps': max_steps,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'lr': lr,
                'world_size': world_size,
            },
            'loss_weights': {
                'rec_weight': rec_weight,
                'spatial_grad_weight': spatial_grad_weight,
                'temporal_grad_weight': temporal_grad_weight,
                'kl_weight': kl_weight,
            },
            'data': {
                'train_zarr': TRAIN_ZARR,
                'val_zarr': VAL_ZARR,
                'stats_json': STATS_JSON,
                'train_samples': len(train_loader.dataset),
                'val_samples': len(val_loader.dataset),
            },
            'paths': {'save_dir': SAVE_DIR},
        }
        with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[VAE] params={n_params:,} input=(B,{channels},{frames},{height},{width}) "
              f"z_channels={model_config.z_channels} "
              f"compress=t{model_config.compress}x s{model.spatial_compress}x")
        print(f"[VAE] max_steps={max_steps:,} steps_per_epoch~{len(train_loader):,} "
              f"epochs~{max_steps / len(train_loader):.1f}")

    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        rec_weight=rec_weight,
        spatial_grad_weight=spatial_grad_weight,
        temporal_grad_weight=temporal_grad_weight,
        kl_weight=kl_weight,
        lr=lr,
        max_steps=max_steps,
        use_ema=True,
        ema_decay=0.999,
        val_interval=10000,
        log_interval=1000,
        save_interval=10000,
        save_dir=SAVE_DIR,
        distributed=distributed,
    )

    trainer.train()


if __name__ == "__main__":
    main()