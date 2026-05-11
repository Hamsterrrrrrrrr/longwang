"""Unconditional sampling from trained diffusion model.

Usage:
    python -m src.models.diffusion.scripts.sample \\
        --diffusion_checkpoint checkpoints/diffusion/ema_final.pt \\
        --vae_checkpoint /data/era5/experiments/vae_run2/ema_final.pt \\
        --latent_stats /data/era5/latents/latent_stats.json \\
        --precip_stats /data/era5/patches/train_stats.json \\
        --output_dir ./samples \\
        --n_samples 16
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from diffusion.modules import UNet3D
from diffusion.sde import VPSDE
from vae import AutoencoderKL3D, VAE3DConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional sampling")

    # Checkpoints
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--latent_stats", type=str, required=True,
                        help="Path to latent_stats.json")
    parser.add_argument("--precip_stats", type=str, required=True,
                        help="Path to train_stats.json (precipitation preprocessing)")

    # Sampling
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--corrections", type=int, default=0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # UNet config (must match training)
    parser.add_argument("--in_channels", type=int, default=16)
    parser.add_argument("--hidden_channels", type=int, nargs="+",
                        default=[512, 512, 512, 512])
    parser.add_argument("--hidden_blocks", type=int, nargs="+",
                        default=[4, 4, 4])
    parser.add_argument("--time_embed_dim", type=int, default=256)
    parser.add_argument("--attention_heads", type=int, default=8)

    # Conditioning (must match training)
    parser.add_argument("--cond_channels", type=int, default=0,
                        help="Must match training config (0=unconditional, 5=conditional)")

    # VAE config (must match training)
    parser.add_argument("--z_channels", type=int, default=16)
    parser.add_argument("--vae_ch", type=int, default=128)
    parser.add_argument("--vae_ch_mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--vae_num_res_blocks", type=int, default=4)
    parser.add_argument("--vae_compress", type=int, default=4)

    # Output
    parser.add_argument("--output_dir", type=str, default="./samples")

    return parser.parse_args()


def load_latent_stats(stats_path):
    """Load per-channel mean and std from latent_stats.json."""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    mean = torch.tensor(stats['channel_mean'], dtype=torch.float32).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(stats['channel_std'], dtype=torch.float32).reshape(1, -1, 1, 1, 1)
    return mean, std


def load_precip_stats(stats_path):
    with open(stats_path, 'r') as f:
        return json.load(f)


def latent_to_precipitation(samples, vae, latent_mean, latent_std, precip_stats, device):
    """Decode normalised latents back to precipitation (mm)."""
    z = samples * latent_std.to(device) + latent_mean.to(device)
    with torch.no_grad():
        x = vae.decode(z)

    norm_min = precip_stats['norm_min']
    norm_range = precip_stats['norm_range']
    epsilon = precip_stats['epsilon']

    x_log = (x + 1.0) / 2.0 * norm_range + norm_min
    x_mm = torch.exp(x_log + np.log(epsilon)) - epsilon
    return x_mm.clamp(min=0.0)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    latent_mean, latent_std = load_latent_stats(args.latent_stats)
    precip_stats = load_precip_stats(args.precip_stats)

    strides = [(1, 2, 2), (2, 2, 2), (2, 2, 2)]
    latent_shape = (args.in_channels, 8, 32, 32)

    unet = UNet3D(
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        hidden_channels=tuple(args.hidden_channels),
        hidden_blocks=tuple(args.hidden_blocks),
        strides=strides,
        time_embed_dim=args.time_embed_dim,
        attention_heads=args.attention_heads,
        cond_channels=args.cond_channels,
    )
    sde = VPSDE(eps=unet, shape=latent_shape)
    sde.load_state_dict(torch.load(args.diffusion_checkpoint, map_location=device))
    sde = sde.to(device).eval()

    vae = AutoencoderKL3D.from_config(VAE3DConfig(
        in_channels=1,
        z_channels=args.z_channels,
        ch=args.vae_ch,
        ch_mult=tuple(args.vae_ch_mult),
        num_res_blocks=args.vae_num_res_blocks,
        compress=args.vae_compress,
        sample_posterior=False,
    ))
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
    vae = vae.to(device).eval()

    n_params = sum(p.numel() for p in sde.parameters())
    print(f"[sample] diffusion_params={n_params:,} latent_shape={latent_shape}")
    print(f"[sample] generating {args.n_samples} samples with {args.steps} steps...")

    samples = sde.sample(
        shape=(args.n_samples,),
        steps=args.steps,
        corrections=args.corrections,
        tau=args.tau,
    )
    precip = latent_to_precipitation(samples, vae, latent_mean, latent_std, precip_stats, device)

    output_path = os.path.join(args.output_dir, "unconditional_samples.pt")
    torch.save({'latents': samples.cpu(), 'precipitation': precip.cpu()}, output_path)
    print(f"[sample] saved {precip.shape} samples (range "
          f"[{precip.min():.4f}, {precip.max():.4f}] mm) to {output_path}")


if __name__ == "__main__":
    main()