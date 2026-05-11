#!/usr/bin/env python3
"""Encode ERA5 patches into VAE latents and compute channel-wise stats.

Input:  (N, 1, 32, 128, 128) precipitation patches.
Output: (N, 16, 8, 32, 32) latents in zarr + latent_stats.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from vae import AutoencoderKL3D, VAE3DConfig
from vae.training.dataloader import ERA5PrecipitationDataset


DATA_DIR = "/data/era5/patches"
OUTPUT_DIR = "/data/era5/latents"
VAE_CHECKPOINT = "/data/era5/experiments/vae_zch16_chmul124_res4/ema_step_140000.pt"

BATCH_SIZE = 16
DEVICE = "cuda"

SPLITS = {
    "train": os.path.join(DATA_DIR, "train_patches.zarr"),
    "val":   os.path.join(DATA_DIR, "val_patches.zarr"),
    "test":  os.path.join(DATA_DIR, "test_patches.zarr"),
}
STATS_JSON = os.path.join(DATA_DIR, "train_stats.json")


def load_vae():
    config = VAE3DConfig(
        in_channels=1, z_channels=16, ch=128, ch_mult=(1, 2, 4),
        num_res_blocks=4, compress=4, dropout=0.0,
    )
    vae = AutoencoderKL3D.from_config(config)

    ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    if isinstance(ckpt, dict):
        state = ckpt.get("vae_state_dict") or ckpt.get("model_state_dict") or ckpt
    else:
        state = ckpt
    vae.load_state_dict(state)

    vae = vae.to(DEVICE).eval()
    print(f"Loaded VAE from {VAE_CHECKPOINT}")
    return vae


def encode_split(vae, split_name, zarr_path):
    dataset = ERA5PrecipitationDataset(zarr_path, STATS_JSON)
    n_samples = len(dataset)

    with torch.no_grad():
        sample = dataset[0].unsqueeze(0).to(DEVICE)
        z, _ = vae.encode(sample, sample_posterior=False)
        latent_shape = z.shape[1:]  # expected (16, 8, 32, 32)

    out_path = os.path.join(OUTPUT_DIR, f"{split_name}_latents.zarr")
    store = zarr.open(out_path, mode='w')
    latents = store.create_dataset(
        'latents',
        shape=(n_samples, *latent_shape),
        chunks=(1, *latent_shape),
        dtype='float32',
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    idx = 0
    for batch in tqdm(loader, desc=f"Encoding {split_name}"):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(DEVICE)
        with torch.no_grad():
            z, _ = vae.encode(batch, sample_posterior=False)
        z_np = z.cpu().numpy()
        latents[idx:idx + z_np.shape[0]] = z_np
        idx += z_np.shape[0]

    print(f"[{split_name}] {n_samples} samples -> {out_path}, latent shape {latent_shape}")
    return out_path


def calculate_stats(train_latent_path):
    """Channel-wise stats from training latents (float64 accumulators, batched)."""
    store = zarr.open(train_latent_path, 'r')
    latents = store['latents']
    n_samples, n_channels = latents.shape[0], latents.shape[1]

    ch_sum    = np.zeros(n_channels, dtype=np.float64)
    ch_sq_sum = np.zeros(n_channels, dtype=np.float64)
    ch_min    = np.full(n_channels,  np.inf, dtype=np.float64)
    ch_max    = np.full(n_channels, -np.inf, dtype=np.float64)
    n_pixels = 0
    pixels_per_sample = None

    batch_size = 500
    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="Computing stats"):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        batch64 = latents[start:end].astype(np.float64, copy=False)

        ch_sum    += batch64.sum(axis=(0, 2, 3, 4))
        ch_sq_sum += (batch64 ** 2).sum(axis=(0, 2, 3, 4))
        ch_min = np.minimum(ch_min, batch64.min(axis=(0, 2, 3, 4)))
        ch_max = np.maximum(ch_max, batch64.max(axis=(0, 2, 3, 4)))

        if pixels_per_sample is None:
            pixels_per_sample = int(np.prod(batch64.shape[2:]))
        n_pixels += (end - start) * pixels_per_sample

    ch_mean = ch_sum / n_pixels
    # var via E[X^2] - E[X]^2; clip tiny negatives from float round-off.
    ch_std = np.sqrt(np.maximum(ch_sq_sum / n_pixels - ch_mean ** 2, 0.0))

    return {
        "vae_checkpoint": VAE_CHECKPOINT,
        "n_samples": int(n_samples),
        "latent_shape": list(latents.shape[1:]),
        "n_channels": int(n_channels),
        "channel_mean": ch_mean.tolist(),
        "channel_std":  ch_std.tolist(),
        "channel_min":  ch_min.tolist(),
        "channel_max":  ch_max.tolist(),
        "global_mean": float(ch_mean.mean()),
        "global_std":  float(ch_std.mean()),
        "global_min":  float(ch_min.min()),
        "global_max":  float(ch_max.max()),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vae = load_vae()

    train_path = encode_split(vae, "train", SPLITS["train"])
    encode_split(vae, "val",  SPLITS["val"])
    encode_split(vae, "test", SPLITS["test"])

    stats = calculate_stats(train_path)
    stats_path = os.path.join(OUTPUT_DIR, "latent_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()