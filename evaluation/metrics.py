"""
Evaluation Metrics
──────────────────
Standard metrics for weather forecast evaluation.

  RMSE  - Root Mean Squared Error
  MAE   - Mean Absolute Error
  Bias  - Systematic over/under prediction
  CSI   - Critical Success Index (rainfall hit rate)
  ETS   - Equitable Threat Score (skill score)
  ACC   - Anomaly Correlation Coefficient
"""

import numpy as np
import torch
from typing import Dict, List
import logging

log = logging.getLogger(__name__)

CHANNEL_NAMES = ["rainfall", "temp_max", "temp_min"]


def compute_all_metrics(
    predictions: np.ndarray,   # (T, N, 3)  — denormalized
    targets:     np.ndarray,   # (T, N, 3)
    thresholds:  List[float] = [1.0, 5.0, 10.0, 25.0],
) -> Dict:
    """
    Compute the full suite of metrics.

    Args:
        predictions : (T, N, 3) — model output, physical units
        targets     : (T, N, 3) — ground truth, physical units
        thresholds  : rainfall thresholds in mm/day for CSI/ETS

    Returns:
        dict of metric name → value (or per-threshold dict)
    """
    results = {}

    for c, name in enumerate(CHANNEL_NAMES):
        pred = predictions[..., c].flatten()
        true = targets[...,     c].flatten()

        # Basic regression metrics
        results[f"{name}_RMSE"] = rmse(pred, true)
        results[f"{name}_MAE"]  = mae(pred,  true)
        results[f"{name}_Bias"] = bias(pred,  true)
        results[f"{name}_ACC"]  = acc(pred,   true)

        # Rainfall-specific categorical metrics
        if name == "rainfall":
            for thr in thresholds:
                results[f"rainfall_CSI_{thr}mm"]  = csi(pred, true, thr)
                results[f"rainfall_ETS_{thr}mm"]  = ets(pred, true, thr)
                results[f"rainfall_POD_{thr}mm"]  = pod(pred, true, thr)
                results[f"rainfall_FAR_{thr}mm"]  = far(pred, true, thr)

    return results


# ── Regression Metrics ────────────────────────────────────────────────────────

def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - true) ** 2)))

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.nanmean(np.abs(pred - true)))

def bias(pred: np.ndarray, true: np.ndarray) -> float:
    """Positive = model over-predicts, negative = under-predicts."""
    return float(np.nanmean(pred - true))

def acc(pred: np.ndarray, true: np.ndarray) -> float:
    """Anomaly Correlation Coefficient (like Pearson but on anomalies)."""
    clim = np.nanmean(true)
    p_anom = pred - clim
    t_anom = true - clim
    num  = np.nansum(p_anom * t_anom)
    den  = np.sqrt(np.nansum(p_anom**2) * np.nansum(t_anom**2))
    return float(num / (den + 1e-8))


# ── Categorical Rainfall Metrics ──────────────────────────────────────────────

def _contingency(pred, true, threshold):
    """Returns (hits, misses, false_alarms, correct_negatives)"""
    p_yes = pred >= threshold
    t_yes = true >= threshold
    hits   = np.sum( p_yes &  t_yes)
    misses = np.sum(~p_yes &  t_yes)
    fas    = np.sum( p_yes & ~t_yes)
    cns    = np.sum(~p_yes & ~t_yes)
    return hits, misses, fas, cns

def csi(pred, true, threshold) -> float:
    """Critical Success Index = hits / (hits + misses + false_alarms)"""
    h, m, f, _ = _contingency(pred, true, threshold)
    denom = h + m + f
    return float(h / denom) if denom > 0 else 0.0

def ets(pred, true, threshold) -> float:
    """
    Equitable Threat Score — like CSI but corrected for random hits.
    ETS = 0 for no skill, 1 for perfect.
    """
    h, m, f, cn = _contingency(pred, true, threshold)
    n = h + m + f + cn
    h_ref = (h + m) * (h + f) / (n + 1e-8)   # random hits
    denom = h + m + f - h_ref
    return float((h - h_ref) / (denom + 1e-8)) if denom > 0 else 0.0

def pod(pred, true, threshold) -> float:
    """Probability of Detection = hits / (hits + misses)"""
    h, m, _, _ = _contingency(pred, true, threshold)
    return float(h / (h + m + 1e-8))

def far(pred, true, threshold) -> float:
    """False Alarm Rate = false_alarms / (hits + false_alarms)"""
    h, _, f, _ = _contingency(pred, true, threshold)
    return float(f / (h + f + 1e-8))


# ── Evaluation Runner ─────────────────────────────────────────────────────────

def evaluate_model(
    model,
    test_loader,
    graph,
    dataset_stats: dict,
    config:        dict,
    device:        torch.device = None,
) -> Dict:
    """
    Run model on test set and compute all metrics.

    Returns:
        {
          "metrics": {...},
          "predictions": np.array (T, N, 3),
          "targets":     np.array (T, N, 3),
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    edge_index = graph.edge_index.to(device)
    edge_attr  = graph.edge_attr.to(device)

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x, edge_index, edge_attr)  # (B, P, N, 3)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # (T, P, N, 3)
    targets = np.concatenate(all_targets, axis=0)

    # Flatten lead times: (T*P, N, 3)
    T, P, N, C = preds.shape
    preds_flat   = preds.reshape(T * P, N, C)
    targets_flat = targets.reshape(T * P, N, C)

    # Denormalize
    mean = dataset_stats["mean"][0, 0, :3]   # (3,)
    std  = dataset_stats["std"][ 0, 0, :3]   # (3,)
    preds_phys   = preds_flat   * std + mean
    targets_phys = targets_flat * std + mean

    # Clip: rainfall can't be negative
    preds_phys[:, :, 0]   = np.clip(preds_phys[:, :, 0],   0, None)
    targets_phys[:, :, 0] = np.clip(targets_phys[:, :, 0], 0, None)

    thresholds = config["evaluation"]["rainfall_thresholds"]
    metrics    = compute_all_metrics(preds_phys, targets_phys, thresholds)

    _print_metrics(metrics)

    return {
        "metrics":     metrics,
        "predictions": preds_phys,
        "targets":     targets_phys,
    }


def _print_metrics(metrics: dict):
    print("\n" + "="*55)
    print("  EVALUATION RESULTS")
    print("="*55)

    for channel in CHANNEL_NAMES:
        print(f"\n  {channel.upper()}")
        for metric in ["RMSE", "MAE", "Bias", "ACC"]:
            key = f"{channel}_{metric}"
            if key in metrics:
                print(f"    {metric:6s}: {metrics[key]:8.4f}")

    print(f"\n  RAINFALL — Categorical (CSI / ETS / POD / FAR)")
    for thr in [1.0, 5.0, 10.0, 25.0]:
        key = f"rainfall_CSI_{thr}mm"
        if key in metrics:
            print(
                f"    ≥{thr:5.1f}mm | "
                f"CSI={metrics[f'rainfall_CSI_{thr}mm']:.3f} "
                f"ETS={metrics[f'rainfall_ETS_{thr}mm']:.3f} "
                f"POD={metrics[f'rainfall_POD_{thr}mm']:.3f} "
                f"FAR={metrics[f'rainfall_FAR_{thr}mm']:.3f}"
            )
    print("="*55 + "\n")