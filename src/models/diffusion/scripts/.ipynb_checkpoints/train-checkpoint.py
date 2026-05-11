"""Training script for score-based diffusion model.

Usage:
    Single GPU:  python -m src.models.diffusion.scripts.train [args...]
    Multi-GPU:   torchrun --nproc_per_node=8 -m src.models.diffusion.scripts.train [args...]

See README for full conditional / unconditional invocations.
"""

import argparse
import os

import torch

from ..modules import UNet3D
from ..sde import VPSDE
from ..training import DiffusionTrainer, create_latent_dataloaders
from ..training.distributed import setup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion model")

    # Data
    parser.add_argument("--train_zarr", type=str, required=True,
                        help="Path to training latents .zarr")
    parser.add_argument("--val_zarr", type=str, required=True,
                        help="Path to validation latents .zarr")
    parser.add_argument("--stats_path", type=str, required=True,
                        help="Path to latent_stats.json")

    # UNet architecture
    parser.add_argument("--in_channels", type=int, default=16)
    parser.add_argument("--hidden_channels", type=int, nargs="+",
                        default=[512, 512, 512, 512])
    parser.add_argument("--hidden_blocks", type=int, nargs="+",
                        default=[4, 4, 4])
    parser.add_argument("--time_embed_dim", type=int, default=256)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--attention_levels", type=int, nargs="+", default=[1, 1, 1],
                        help="Which levels get attention (0 or 1 per level)")
    parser.add_argument("--strides", type=str, nargs="+",
                        default=["1,2,2", "2,2,2", "2,2,2"],
                        help="Strides per level as T,H,W")

    # SDE
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--alpha", type=str, default="cos",
                        choices=["cos", "lin", "exp", "shifted_cos"])
    parser.add_argument("--noise_d", type=float, default=64)
    parser.add_argument("--image_d", type=float, default=32)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=1)
    parser.add_argument("--num_workers", type=int, default=8)

    # Conditioning
    parser.add_argument("--train_metadata", type=str, default=None,
                        help="Path to training metadata .npz (enables conditioning)")
    parser.add_argument("--val_metadata", type=str, default=None,
                        help="Path to validation metadata .npz")
    parser.add_argument("--cond_channels", type=int, default=0,
                        help="Conditioning vector dimension (0=unconditional, 5=lat/lon/month)")

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--no_ema", action="store_true")

    # Logging / saving
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/diffusion")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def main():
    args = parse_args()

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    strides = [tuple(int(x) for x in s.split(",")) for s in args.strides]
    attention_levels = [bool(x) for x in args.attention_levels]
    latent_shape = (args.in_channels, 8, 32, 32)  # (C, T, H, W)

    unet = UNet3D(
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        hidden_channels=tuple(args.hidden_channels),
        hidden_blocks=tuple(args.hidden_blocks),
        strides=strides,
        time_embed_dim=args.time_embed_dim,
        attention_heads=args.attention_heads,
        attention_levels=attention_levels,
        cond_channels=args.cond_channels,
    )

    sde = VPSDE(
        eps=unet,
        shape=latent_shape,
        alpha=args.alpha,
        eta=args.eta,
        noise_d=args.noise_d,
        image_d=args.image_d,
    )

    if rank == 0:
        n_params = sum(p.numel() for p in sde.parameters() if p.requires_grad)
        print(f"[diffusion] params={n_params:,} latent_shape={latent_shape} "
              f"hidden_channels={args.hidden_channels} hidden_blocks={args.hidden_blocks} "
              f"strides={strides}")

    train_loader, val_loader = create_latent_dataloaders(
        train_zarr_path=args.train_zarr,
        val_zarr_path=args.val_zarr,
        stats_path=args.stats_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
        train_metadata_path=args.train_metadata,
        val_metadata_path=args.val_metadata,
    )

    trainer = DiffusionTrainer(
        sde=sde,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        distributed=distributed,
    )

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()