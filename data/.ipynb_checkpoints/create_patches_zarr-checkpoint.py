#!/usr/bin/env python3
"""Extract spatiotemporal patches from daily ERA5 precipitation into Zarr."""

import os
import shutil
import time

import numpy as np
import xarray as xr
import zarr
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

DATA_SPLITS = {
    'train': (1979, 2014),
    'val':   (2015, 2019),
    'test':  (2020, 2024),
}

INPUT_DIR = "/data/era5/precipitation_daily_arco"
OUTPUT_DIR = "/data/era5/patches"

TEMPORAL_SIZE = 32
SPATIAL_SIZE = 128
TEMPORAL_STRIDE = 16
SPATIAL_STRIDE = 64


def get_crop_positions(total_size, crop_size, stride):
    positions = list(range(0, total_size - crop_size + 1, stride))
    edge_pos = total_size - crop_size
    if edge_pos not in positions:
        positions.append(edge_pos)
    return positions


def extract_patches_from_year(year, input_dir, lat_starts, lon_starts):
    input_file = os.path.join(input_dir, f"era5_precip_daily_{year}.nc")
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return None

    ds = xr.open_dataset(input_file).load()
    precip = ds.tp.isel(latitude=slice(0, 720)).values.astype(np.float32)
    ds.close()

    # Clip float32 rounding noise 
    precip = np.clip(precip, 0.0, None)

    n_days = precip.shape[0]
    n_lat, n_lon = len(lat_starts), len(lon_starts)

    windows = sliding_window_view(precip, (SPATIAL_SIZE, SPATIAL_SIZE), axis=(1, 2))
    lat_idx = np.asarray(lat_starts)
    lon_idx = np.asarray(lon_starts)
    patches_4d = windows[:, lat_idx[:, None], lon_idx[None, :], :, :]

    # Reshape with lon varying fastest to match (lat outer, lon inner) loop order.
    patches_3d = patches_4d.reshape(n_days, n_lat * n_lon, SPATIAL_SIZE, SPATIAL_SIZE)
    patches_3d = np.ascontiguousarray(patches_3d)

    return {'year': year, 'n_days': n_days, 'patches': patches_3d}


def create_patches_for_split(split_name, year_range):
    start_year, end_year = year_range
    years = list(range(start_year, end_year + 1))

    lat_starts = get_crop_positions(720, SPATIAL_SIZE, SPATIAL_STRIDE)
    lon_starts = get_crop_positions(1440, SPATIAL_SIZE, SPATIAL_STRIDE)
    n_spatial = len(lat_starts) * len(lon_starts)

    print(f"\n[{split_name}] {start_year}-{end_year}, {n_spatial} spatial patches/day")
    start_time = time.time()

    results = []
    for year in tqdm(years, desc=f"{split_name}: extracting"):
        result = extract_patches_from_year(year, INPUT_DIR, lat_starts, lon_starts)
        if result:
            results.append(result)

    if not results:
        raise ValueError(f"No valid data for {split_name}")
    results.sort(key=lambda x: x['year'])

    all_patches = np.concatenate([r['patches'] for r in results], axis=0)
    total_days = all_patches.shape[0]
    assert all_patches.shape[1] == n_spatial

    temporal_starts = get_crop_positions(total_days, TEMPORAL_SIZE, TEMPORAL_STRIDE)
    total_samples = len(temporal_starts) * n_spatial

    output_path = os.path.join(OUTPUT_DIR, f"{split_name}_patches.zarr")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)
    dataset = root.create_dataset(
        'precipitation',
        shape=(total_samples, TEMPORAL_SIZE, SPATIAL_SIZE, SPATIAL_SIZE),
        chunks=(1, TEMPORAL_SIZE, SPATIAL_SIZE, SPATIAL_SIZE),
        dtype=np.float32,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),
    )

    sample_idx = 0
    for temp_start in tqdm(temporal_starts, desc=f"{split_name}: writing"):
        block = all_patches[temp_start:temp_start + TEMPORAL_SIZE]
        block = np.ascontiguousarray(np.transpose(block, (1, 0, 2, 3)))
        dataset[sample_idx:sample_idx + n_spatial] = block
        sample_idx += n_spatial
    assert sample_idx == total_samples

    root.attrs.update({
        'split': split_name,
        'year_start': start_year,
        'year_end': end_year,
        'n_samples': total_samples,
        'temporal_size': TEMPORAL_SIZE,
        'spatial_size': SPATIAL_SIZE,
        'temporal_stride': TEMPORAL_STRIDE,
        'spatial_stride': SPATIAL_STRIDE,
        'lat_crops': len(lat_starts),
        'lon_crops': len(lon_starts),
        'input_source': INPUT_DIR,
        'note_on_years': (
            "Training starts 1979 to match the satellite + in-situ observation era. "
            "ERA5 pre-1979 has far fewer observational constraints for precipitation."
        ),
        'note_on_clipping': "Negative float32 rounding noise (~1e-7 m) clipped to 0.",
    })

    elapsed = time.time() - start_time
    print(f"[{split_name}] {total_samples:,} samples written to {output_path} ({elapsed/60:.1f} min)")
    return output_path


def main():
    for split_name, year_range in DATA_SPLITS.items():
        create_patches_for_split(split_name, year_range)


if __name__ == "__main__":
    main()