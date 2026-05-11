"""Evaluation metrics for precipitation downscaling.

Conventions:
- pred: (E, T, H, W) ensemble of E members; (T, H, W) is treated as E=1.
- gt:   (T, H, W).
- All functions return a single scalar (or (pred, gt) pair) per test sample.
- _ensmean variants compute the ensemble mean before applying the metric.
- Default variants compute per member then average over E.
"""

import numpy as np
from scipy.stats import wasserstein_distance


def _ensure_ensemble(pred):
    """Promote (T, H, W) to (1, T, H, W) so all metrics see an ensemble axis."""
    return pred[np.newaxis] if pred.ndim == 3 else pred


def monthly_r2(pred, gt):
    """R² on the monthly total (H, W) field; per member, averaged over E.

    R² = 1 - SS_res / SS_tot, where SS_res = Σ(gt - pred)², SS_tot = Σ(gt - mean(gt))².
    R² = 1 for perfect match, 0 for as-good-as-mean, negative for worse.
    """
    pred = _ensure_ensemble(pred)
    E = pred.shape[0]

    monthly_gt = gt.sum(axis=0).ravel()
    monthly_pred = pred.sum(axis=1).reshape(E, -1)

    ss_tot = ((monthly_gt - monthly_gt.mean()) ** 2).sum()
    ss_res = ((monthly_gt[np.newaxis] - monthly_pred) ** 2).sum(axis=1)

    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return float(r2.mean())


def marginal_wasserstein(pred, gt):
    """Wasserstein-1 between flattened pred and gt distributions; per member, averaged."""
    pred = _ensure_ensemble(pred)
    gt_flat = gt.ravel()
    return np.mean([wasserstein_distance(pred[e].ravel(), gt_flat) for e in range(pred.shape[0])])


def marginal_wasserstein_ensmean(pred, gt):
    return marginal_wasserstein(pred.mean(axis=0, keepdims=True), gt)


def crps_pointwise(pred, gt):
    """Gridpoint CRPS averaged over (T, H, W); fair/unbiased estimator (requires E ≥ 2).

    Uses the closed form
        (1 / (E*(E-1))) Σ_{i≠j} |X_i - X_j|
      = (2 / (E*(E-1))) Σ_i (2i - E + 1) X_(i),
    where X_(i) are the sorted ensemble values (ascending).
    """
    pred = _ensure_ensemble(pred)
    E = pred.shape[0]
    if E < 2:
        raise ValueError("Fair CRPS requires E >= 2 ensemble members.")

    # Skill term: E[|X - y|].
    term1 = np.abs(pred - gt[np.newaxis]).mean(axis=0)

    # Spread term: unbiased E[|X - X'|] / 2 via sorted closed form.
    pred_sorted = np.sort(pred, axis=0)
    weights = (2 * np.arange(E) - E + 1).reshape(E, 1, 1, 1)
    # Factor of 2 absorbed: original spread term is (1/2) E[|X - X'|], net divisor E*(E-1).
    term2 = (weights * pred_sorted).sum(axis=0) / (E * (E - 1))

    return (term1 - term2).mean()


def monthly_consistency(pred, gt):
    """Relative error of total precipitation; per member, averaged over E."""
    pred = _ensure_ensemble(pred)
    gt_total = gt.sum()
    pred_totals = pred.sum(axis=(1, 2, 3))
    return (np.abs(pred_totals - gt_total) / np.maximum(gt_total, 1e-12)).mean()


def monthly_consistency_ensmean(pred, gt):
    return monthly_consistency(pred.mean(axis=0, keepdims=True), gt)


def wet_day_fraction(pred, gt, threshold=1.0):
    """Mean wet-day fraction; returns (pred_frac, gt_frac) — actual values, not errors.

    Threshold defaults to 1.0 mm (standard wet day). Pred fraction is per-member, averaged.
    """
    pred = _ensure_ensemble(pred)
    gt_frac = (gt > threshold).mean()
    pred_frac = np.mean([(pred[e] > threshold).mean() for e in range(pred.shape[0])])
    return pred_frac, gt_frac


def wet_day_fraction_ensmean(pred, gt, threshold=1.0):
    return wet_day_fraction(pred.mean(axis=0, keepdims=True), gt, threshold)


def dry_day_fraction(pred, gt, threshold=0.1):
    """Mean dry-day fraction; returns (pred_frac, gt_frac) — actual values, not errors.

    Threshold defaults to 0.1 mm (trace precipitation). Pred fraction is per-member, averaged.
    """
    pred = _ensure_ensemble(pred)
    gt_frac = (gt < threshold).mean()
    pred_frac = np.mean([(pred[e] < threshold).mean() for e in range(pred.shape[0])])
    return pred_frac, gt_frac


def dry_day_fraction_ensmean(pred, gt, threshold=0.1):
    return dry_day_fraction(pred.mean(axis=0, keepdims=True), gt, threshold)


def extreme_p95(pred, gt):
    """95th percentile of marginal distribution; returns (pred_p95, gt_p95). Per member, averaged."""
    pred = _ensure_ensemble(pred)
    gt_p95 = np.percentile(gt.ravel(), 95)
    pred_p95 = np.mean([np.percentile(pred[e].ravel(), 95) for e in range(pred.shape[0])])
    return pred_p95, gt_p95


def extreme_p95_ensmean(pred, gt):
    return extreme_p95(pred.mean(axis=0, keepdims=True), gt)


def r95p_frac(pred, gt, quantile=0.95, wet_threshold=1.0):
    """Fraction of total precip from cells exceeding the (wet-day) qth percentile of GT.

    Returns (pred_frac, gt_frac), both unitless in [0, 1].
    """
    pred = _ensure_ensemble(pred)
    E = pred.shape[0]

    gt_flat = gt.ravel()
    gt_wet = gt_flat[gt_flat >= wet_threshold]
    if gt_wet.size == 0:
        return 0.0, 0.0

    Q = np.quantile(gt_wet, quantile)

    gt_total = gt_flat.sum()
    gt_frac = gt_flat[gt_flat > Q].sum() / gt_total if gt_total > 0 else 0.0

    pred_fracs = []
    for e in range(E):
        pe = pred[e].ravel()
        pe_total = pe.sum()
        pred_fracs.append(pe[pe > Q].sum() / pe_total if pe_total > 0 else 0.0)

    return float(np.mean(pred_fracs)), float(gt_frac)


def r99p_frac(pred, gt, wet_threshold=1.0):
    """Normalised R99p. See r95p_frac."""
    return r95p_frac(pred, gt, quantile=0.99, wet_threshold=wet_threshold)


def r95p_frac_ensmean(pred, gt, wet_threshold=1.0):
    return r95p_frac(pred.mean(axis=0, keepdims=True), gt, quantile=0.95, wet_threshold=wet_threshold)


def r99p_frac_ensmean(pred, gt, wet_threshold=1.0):
    return r95p_frac(pred.mean(axis=0, keepdims=True), gt, quantile=0.99, wet_threshold=wet_threshold)


def domain_mean_lag1(pred, gt):
    """Lag-1 autocorrelation of domain-mean time series; returns (pred_lag1, gt_lag1)."""
    pred = _ensure_ensemble(pred)

    def _lag1(ts):
        ts = ts - ts.mean()
        return (ts[:-1] * ts[1:]).sum() / max((ts ** 2).sum(), 1e-12)

    gt_ts = gt.mean(axis=(1, 2))
    pred_ts = pred.mean(axis=(2, 3))
    gt_lag1 = _lag1(gt_ts)
    pred_lag1 = np.mean([_lag1(pred_ts[e]) for e in range(pred.shape[0])])
    return pred_lag1, gt_lag1


def domain_mean_lag1_ensmean(pred, gt):
    return domain_mean_lag1(pred.mean(axis=0, keepdims=True), gt)