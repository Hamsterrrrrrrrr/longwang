#!/usr/bin/env python3
"""Download ARCO-ERA5 hourly total precipitation and aggregate to daily netCDF."""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import logging
for name in ("google", "google.api_core", "urllib3", "gcsfs"):
    logging.getLogger(name).setLevel(logging.ERROR)

import time

import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from tqdm import tqdm


ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
OUTPUT_DIR = "/data/era5/precipitation_daily_arco"
START_YEAR = 1940
END_YEAR = 2024

N_WORKERS = 24
THREADS_PER_WORKER = 4
MEMORY_PER_WORKER = "40GB"


def process_one_year(ds_tp, year, output_dir):
    out_file = os.path.join(output_dir, f"era5_precip_daily_{year}.nc")
    if os.path.exists(out_file):
        return

    tp_year = ds_tp.sel(time=slice(f"{year}-01-01T00:00", f"{year + 1}-01-01T00:00"))
    tp_shifted = tp_year.assign_coords(time=tp_year.time - np.timedelta64(1, "s"))

    daily = tp_shifted.resample(time="1D").sum()
    daily = daily.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    daily = daily.astype("float32").compute()

    daily.attrs["units"] = "m"
    daily.attrs["long_name"] = "Daily total precipitation"
    daily.attrs["note"] = (
        "Sum of hourly tp with -1s time shift for correct UTC day attribution. "
        "Computed from ARCO-ERA5 Zarr."
    )

    ds_out = daily.to_dataset(name="tp")
    ds_out.attrs["source"] = ZARR_URL
    ds_out.attrs["processing"] = f"Year {year}: hourly TP -> daily sum via -1s shift."

    encoding = {"tp": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": np.float32(-9999.0),
    }}

    os.makedirs(output_dir, exist_ok=True)
    ds_out.to_netcdf(out_file, encoding=encoding, engine="netcdf4", format="NETCDF4")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=MEMORY_PER_WORKER,
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    ds = xr.open_zarr(
        ZARR_URL,
        chunks={"time": 48},
        consolidated=True,
        storage_options={"token": "anon"},
    )
    ds_tp = ds["total_precipitation"]

    t0 = time.time()
    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Years"):
        process_one_year(ds_tp, year, OUTPUT_DIR)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()