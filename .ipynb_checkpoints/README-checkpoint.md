# Longwang

Code for *Longwang: Zero-Shot Global Spatiotemporal Precipitation Downscaling with a Latent Generative Prior* (<TODO: authors, year, venue / arXiv link>).

A two-stage latent diffusion pipeline for global spatiotemporal precipitation downscaling. A 3D VAE compresses ERA5 spatiotemporal patches into a latent representation, and a score-based diffusion model is trained on those latents. At inference the prior is combined with a forward operator to perform zero-shot downscaling without retraining.

## Repository layout

```
data/                       Data download, patch extraction, metadata
src/models/vae/             3D VAE: encoder, decoder, training, latent encoding
src/models/diffusion/       3D UNet, VPSDE, training, sampling, evaluation
```

## Installation

Python 3.10+ and a CUDA GPU.

```
pip install -r requirements.txt
```

## Data

ERA5 hourly precipitation is pulled from the public ARCO-ERA5 mirror and aggregated to daily totals. Splits used in the paper: train 1979–2014, val 2015–2019, test 2020–2024.

**Scripts under `data/` contain hardcoded paths under `/data/era5/...`. Edit the constants at the top of each script before running.**

```
python data/download_era5.py            # hourly → daily netCDF
python data/create_patches_zarr.py      # patch extraction to zarr
python data/extract_patch_metadata.py   # per-sample lat/lon/month
```

## Stage 1: VAE

```
torchrun --nproc_per_node=8 -m src.models.vae.scripts.train_vae_era5
python -m src.models.vae.scripts.encode_latents
```

Hyperparameters and paths are set inline. After training, `encode_latents.py` writes `{train,val,test}_latents.zarr` and `latent_stats.json`.

Trained on 8xH100 GPUs for 4 days.

## Stage 2: Diffusion

```
torchrun --nproc_per_node=8 -m src.models.diffusion.scripts.train \
    --train_zarr /data/era5/latents/train_latents.zarr \
    --val_zarr   /data/era5/latents/val_latents.zarr \
    --stats_path /data/era5/latents/latent_stats.json \
    --save_dir   /data/era5/experiments/diffusion_uncond \
    --max_steps 410000 --warmup_steps 10000 --batch_size 4 --lr 2e-4
```

Add `--cond_channels 5 --train_metadata ... --val_metadata ...` for the conditional variant. See `--help` for all options.

Trained on 8xH100 GPUs for 6 days.

## Sampling

```
python -m src.models.diffusion.scripts.uncond_prior_sample \
    --diffusion_checkpoint <path>/ema_final.pt \
    --vae_checkpoint       <path>/ema_final.pt \
    --latent_stats         <path>/latent_stats.json \
    --precip_stats         <path>/train_stats.json \
    --output_dir           ./samples --n_samples 16
```

Zero-shot conditional generation is implemented in `src/models/diffusion/sde/conditioning.py` (`GaussianScore`), following SDA (Rozet & Louppe, 2023).

## Data availability

ERA5 hourly single-level data are available from the Copernicus Climate Data Store, DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47).

## Citation

```
@article{<TODO: bibkey>,
  title   = {Longwang: Zero-Shot Global Spatiotemporal Precipitation Downscaling with a Latent Generative Prior},
  author  = {<TODO>},
  journal = {<TODO>},
  year    = {<TODO>},
  doi     = {<TODO>},
}
```

## License

MIT — see `LICENSE`. © 2026 Yue Wang.