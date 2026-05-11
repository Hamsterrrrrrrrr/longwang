#!/usr/bin/env python3
"""Build per-sample metadata (center lat/lon, month) aligned with patch Zarr files."""

import os

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm

INPUT_DIR = "/data/era5/precipitation_daily"
OUTPUT_DIR = "/data/era5/patches"

TEMPORAL_SIZE = 32
SPATIAL_SIZE = 128
TEMPORAL_STRIDE = 16
SPATIAL_STRIDE = 64

DATA_SPLITS = {
    'train': (1979, 2014),
    'val':   (2015, 2019),
    'test':  (2020, 2024),
}


def get_crop_positions(total_size, crop_size, stride):
    positions = list(range(0, total_size - crop_size + 1, stride))
    edge_pos = total_size - crop_size
    if edge_pos not in positions:
        positions.append(edge_pos)
    return positions


def create_metadata_for_split(split_name, year_range):
    start_year, end_year = year_range
    years = list(range(start_year, end_year + 1))

    lat_starts = get_crop_positions(720, SPATIAL_SIZE, SPATIAL_STRIDE)
    lon_starts = get_crop_positions(1440, SPATIAL_SIZE, SPATIAL_STRIDE)

    sample_file = os.path.join(INPUT_DIR, f"era5_precip_daily_{years[0]}.nc")
    ds = xr.open_dataset(sample_file)
    lat_array = ds.latitude.values[:720]
    lon_array = ds.longitude.values
    ds.close()

    spatial_lats = np.array(
        [lat_array[ls + SPATIAL_SIZE // 2] for ls in lat_starts for _ in lon_starts],
        dtype=np.float32,
    )
    spatial_lons = np.array(
        [lon_array[ls + SPATIAL_SIZE // 2] for _ in lat_starts for ls in lon_starts],
        dtype=np.float32,
    )
    n_spatial = len(spatial_lats)

    all_dates = []
    for year in tqdm(years, desc=f"{split_name}: reading dates"):
        fpath = os.path.join(INPUT_DIR, f"era5_precip_daily_{year}.nc")
        if not os.path.exists(fpath):
            continue
        ds = xr.open_dataset(fpath)
        time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        all_dates.extend(ds[time_dim].values)
        ds.close()

    total_days = len(all_dates)
    temporal_starts = get_crop_positions(total_days, TEMPORAL_SIZE, TEMPORAL_STRIDE)
    total_samples = len(temporal_starts) * n_spatial

    zarr_path = os.path.join(OUTPUT_DIR, f"{split_name}_patches.zarr")
    existing = zarr.open(zarr_path, mode='r')
    assert existing['precipitation'].shape[0] == total_samples, (
        f"Zarr has {existing['precipitation'].shape[0]} samples, expected {total_samples}"
    )

    center_months = np.array([
        int(np.datetime64(all_dates[ts + TEMPORAL_SIZE // 2], 'ns')
            .astype('datetime64[M]').astype(int) % 12 + 1)
        for ts in temporal_starts
    ], dtype=np.int32)

    months = np.repeat(center_months, n_spatial)
    center_lats = np.tile(spatial_lats, len(temporal_starts))
    center_lons = np.tile(spatial_lons, len(temporal_starts))

    out_path = os.path.join(OUTPUT_DIR, f"{split_name}_metadata.npz")
    np.savez_compressed(out_path, center_lat=center_lats, center_lon=center_lons, month=months)
    print(f"{split_name}: {total_samples} samples -> {out_path}")


if __name__ == "__main__":
    for split_name, year_range in DATA_SPLITS.items():
        create_metadata_for_split(split_name, year_range)