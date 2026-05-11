"""Compute evaluation metrics for three methods (cond / uncond / trilinear).
"""

import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from diffusion.evaluation.metrics import (
    crps_pointwise,
    domain_mean_lag1,
    dry_day_fraction,
    marginal_wasserstein,
    monthly_consistency,
    monthly_r2,
    r95p_frac,
    wet_day_fraction,
)


SAVE_DIR = "/data/era5/era5_posterior_samples_regional"
COND_PATH = os.path.join(SAVE_DIR, "all_regions_cond.npz")
UNCOND_PATH = os.path.join(SAVE_DIR, "all_regions_uncond.npz")
TRILINEAR_PATH = os.path.join(SAVE_DIR, "all_regions_trilinear.npz")
OUTPUT_DIR = os.path.join(SAVE_DIR, "evaluation")


def load_combined(fpath):
    raw = np.load(fpath)
    results = {}
    for k in raw.files:
        region_month, field = k.split("__")
        results.setdefault(region_month, {})[field] = raw[k]
    return results


def evaluate_model(data, label, compute_crps=True):
    keys = sorted(data.keys())
    N = len(keys)

    results = {
        'region_month': [],
        'r2': np.zeros(N),
        'wasserstein': np.zeros(N),
        'crps': np.full(N, np.nan),
        'monthly_consist': np.zeros(N),
        'wet_frac_pred': np.zeros(N), 'wet_frac_gt': np.zeros(N),
        'dry_frac_pred': np.zeros(N), 'dry_frac_gt': np.zeros(N),
        'r95p_pred': np.zeros(N),     'r95p_gt': np.zeros(N),
        'lag1_pred': np.zeros(N),     'lag1_gt': np.zeros(N),
    }

    def store_pair(i, prefix, pair):
        results[f'{prefix}_pred'][i], results[f'{prefix}_gt'][i] = pair

    for i, key in enumerate(tqdm(keys, desc=label, ncols=88)):
        pred = data[key]['ensemble'].astype(np.float32)  # (E, T, H, W)
        gt = data[key]['gt_mm'].astype(np.float32)       # (T, H, W)

        results['region_month'].append(key)
        results['r2'][i] = monthly_r2(pred, gt)
        results['wasserstein'][i] = marginal_wasserstein(pred, gt)
        if compute_crps:
            results['crps'][i] = crps_pointwise(pred, gt)
        results['monthly_consist'][i] = monthly_consistency(pred, gt)

        store_pair(i, 'wet_frac', wet_day_fraction(pred, gt, threshold=1.0))
        store_pair(i, 'dry_frac', dry_day_fraction(pred, gt, threshold=0.1))
        store_pair(i, 'r95p',     r95p_frac(pred, gt))
        store_pair(i, 'lag1',     domain_mean_lag1(pred, gt))

    return results


def fmt(arr):
    """mean [q05, q95], or '—' if all NaN."""
    if np.all(np.isnan(arr)):
        return "—"
    return f"{np.nanmean(arr):.4f} [{np.nanpercentile(arr, 5):.4f}, {np.nanpercentile(arr, 95):.4f}]"


def fmt_pct(arr):
    """Same as fmt but as percentages."""
    if np.all(np.isnan(arr)):
        return "—"
    return (f"{np.nanmean(arr) * 100:.3f}% "
            f"[{np.nanpercentile(arr, 5) * 100:.3f}%, {np.nanpercentile(arr, 95) * 100:.3f}%]")


def print_table1(tri, cond, uncond, n_entries):
    col = 28
    print(f"\nTable 1 — Performance scores (per ensemble member); "
          f"mean [5th, 95th] over {n_entries} region/month entries")
    print(f"{'Model':<16s}"
          f"{'R² ↑':>{col}s}{'Wasserstein ↓':>{col}s}"
          f"{'CRPS ↓':>{col}s}{'Monthly Consist. ↓':>{col}s}")
    for label, res in [('Trilinear', tri), ('Uncond', uncond), ('Cond', cond)]:
        print(f"{label:<16s}"
              f"{fmt(res['r2']):>{col}s}"
              f"{fmt(res['wasserstein']):>{col}s}"
              f"{fmt(res['crps']):>{col}s}"
              f"{fmt_pct(res['monthly_consist']):>{col}s}")


def print_table2(tri, cond, uncond, n_entries):
    col = 26
    print(f"\nTable 2 — Distributional statistics vs ground truth; "
          f"mean [5th, 95th] over {n_entries} region/month entries")
    print(f"{'Model':<16s}"
          f"{'Wet Day Frac (≥1mm)':>{col}s}{'Dry Day Frac (<0.1mm)':>{col}s}"
          f"{'R95p frac':>{col}s}{'DM Lag-1 Autocorr':>{col}s}")
    print(f"{'GT':<16s}"
          f"{fmt(cond['wet_frac_gt']):>{col}s}"
          f"{fmt(cond['dry_frac_gt']):>{col}s}"
          f"{fmt(cond['r95p_gt']):>{col}s}"
          f"{fmt(cond['lag1_gt']):>{col}s}")
    for label, res in [('Trilinear', tri), ('Uncond', uncond), ('Cond', cond)]:
        print(f"{label:<16s}"
              f"{fmt(res['wet_frac_pred']):>{col}s}"
              f"{fmt(res['dry_frac_pred']):>{col}s}"
              f"{fmt(res['r95p_pred']):>{col}s}"
              f"{fmt(res['lag1_pred']):>{col}s}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cond_data = load_combined(COND_PATH)
    uncond_data = load_combined(UNCOND_PATH)
    # The uncond merge kept "_uncond" in region names; strip for alignment.
    uncond_data = {k.replace("_uncond", ""): v for k, v in uncond_data.items()}
    tri_data = load_combined(TRILINEAR_PATH)

    assert set(cond_data) == set(uncond_data) == set(tri_data), "Key mismatch across methods"
    n_entries = len(cond_data)
    print(f"[eval] {n_entries} region/month entries")

    cond_results = evaluate_model(cond_data, "Conditional", compute_crps=True)
    uncond_results = evaluate_model(uncond_data, "Unconditional", compute_crps=True)
    tri_results = evaluate_model(tri_data, "Trilinear", compute_crps=False)

    out_path = os.path.join(OUTPUT_DIR, "metrics_three_way.npz")
    np.savez_compressed(
        out_path,
        region_month=np.array(cond_results['region_month']),
        **{f"cond_{k}": v for k, v in cond_results.items() if k != 'region_month'},
        **{f"uncond_{k}": v for k, v in uncond_results.items() if k != 'region_month'},
        **{f"tri_{k}": v for k, v in tri_results.items() if k != 'region_month'},
    )
    print(f"[eval] saved to {out_path}")

    print_table1(tri_results, cond_results, uncond_results, n_entries)
    print_table2(tri_results, cond_results, uncond_results, n_entries)


if __name__ == "__main__":
    main()