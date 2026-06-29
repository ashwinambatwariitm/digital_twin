"""
Visualization
─────────────
Forecast maps, loss curves, and metric plots for Mini-GraphCast.

Uses matplotlib + cartopy for geographic maps over India.
Falls back to plain matplotlib if cartopy is unavailable.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# Try cartopy for proper geographic projection
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    log.warning("cartopy not installed — using plain matplotlib maps")
    HAS_CARTOPY = False


# ── India Map ────────────────────────────────────────────────────────────────

def plot_forecast_map(
    prediction:  np.ndarray,   # (N_lat, N_lon)
    target:      np.ndarray,   # (N_lat, N_lon)
    lats:        np.ndarray,
    lons:        np.ndarray,
    variable:    str = "rainfall",
    lead_day:    int = 1,
    date_str:    str = "",
    save_path:   str = None,
):
    """
    Side-by-side map: [Ground Truth] [Prediction] [Error]
    """
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {},
    )
    fig.suptitle(
        f"{variable.title()} Forecast — Day +{lead_day}  {date_str}",
        fontsize=14, fontweight="bold"
    )

    cmap, vmin, vmax, units = _get_colormap(variable)
    error     = prediction - target
    err_max   = np.nanpercentile(np.abs(error), 95)

    panels = [
        (target,     f"Ground Truth ({units})", cmap,       vmin, vmax),
        (prediction, f"Prediction ({units})",   cmap,       vmin, vmax),
        (error,      f"Error ({units})",        "RdBu_r",  -err_max, err_max),
    ]

    for ax, (data, title, cm, vn, vx) in zip(axes, panels):
        im = _plot_panel(ax, data, lats, lons, cm, vn, vx, title)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved forecast map: {save_path}")
    plt.show()
    plt.close()


def plot_multi_day_forecast(
    predictions: np.ndarray,   # (pred_steps, N_lat, N_lon)
    targets:     np.ndarray,   # (pred_steps, N_lat, N_lon)
    lats:        np.ndarray,
    lons:        np.ndarray,
    variable:    str = "rainfall",
    save_path:   str = None,
):
    """
    Grid of maps: rows = lead days, cols = [truth, pred, error]
    """
    P = predictions.shape[0]
    cmap, vmin, vmax, units = _get_colormap(variable)

    fig, axes = plt.subplots(
        P, 3,
        figsize=(18, 5 * P),
        subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {},
    )
    if P == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"{variable.title()} Multi-Day Forecast", fontsize=15, fontweight="bold")

    for d in range(P):
        err     = predictions[d] - targets[d]
        err_max = np.nanpercentile(np.abs(err), 95)

        panels = [
            (targets[d],     f"Day +{d+1} Truth",     cmap,     vmin, vmax),
            (predictions[d], f"Day +{d+1} Prediction", cmap,    vmin, vmax),
            (err,            f"Day +{d+1} Error",      "RdBu_r", -err_max, err_max),
        ]
        for ax, (data, title, cm, vn, vx) in zip(axes[d], panels):
            im = _plot_panel(ax, data, lats, lons, cm, vn, vx, title)
            plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ── Training Curves ──────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, save_path: str = None):
    """Plot train/val loss over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train", color="#2563eb", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val",   color="#dc2626", linewidth=2)
    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # LR
    axes[1].plot(history["lr"], color="#16a34a", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_metrics_bar(metrics: dict, save_path: str = None):
    """Bar chart of RMSE and CSI metrics per variable."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE bars
    vars_   = ["rainfall", "temp_max", "temp_min"]
    rmses   = [metrics.get(f"{v}_RMSE", 0) for v in vars_]
    colors  = ["#3b82f6", "#f97316", "#a855f7"]

    axes[0].bar(vars_, rmses, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("RMSE by Variable", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("RMSE (physical units)")
    for i, v in enumerate(rmses):
        axes[0].text(i, v + 0.01 * max(rmses), f"{v:.3f}",
                     ha="center", fontsize=10, fontweight="bold")

    # CSI bars for rainfall thresholds
    thrs  = [1.0, 5.0, 10.0, 25.0]
    csis  = [metrics.get(f"rainfall_CSI_{t}mm", 0) for t in thrs]
    etss  = [metrics.get(f"rainfall_ETS_{t}mm", 0) for t in thrs]
    x     = np.arange(len(thrs))
    w     = 0.35

    axes[1].bar(x - w/2, csis, w, label="CSI", color="#3b82f6", edgecolor="white")
    axes[1].bar(x + w/2, etss, w, label="ETS", color="#f97316", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"≥{t}mm" for t in thrs])
    axes[1].set_title("Rainfall Skill Scores", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Score (0–1, higher=better)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_time_series(
    predictions: np.ndarray,   # (T, N, 3)
    targets:     np.ndarray,
    node_idx:    int = 0,
    save_path:   str = None,
):
    """Plot prediction vs truth over time at one grid node."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    labels = ["Rainfall (mm/day)", "Max Temp (°C)", "Min Temp (°C)"]
    colors = [("#3b82f6", "#1d4ed8"), ("#f97316", "#c2410c"), ("#a855f7", "#7e22ce")]

    for i, (ax, label, (pc, tc)) in enumerate(zip(axes, labels, colors)):
        ax.plot(targets[:, node_idx, i],     label="Truth",      color=tc, linewidth=1.5)
        ax.plot(predictions[:, node_idx, i], label="Prediction", color=pc,
                linewidth=1.5, linestyle="--", alpha=0.8)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(f"Forecast vs Truth — Grid Node {node_idx}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_colormap(variable: str):
    maps = {
        "rainfall": ("YlGnBu",  0,   50,   "mm/day"),
        "temp_max": ("RdYlBu_r", 15, 45,   "°C"),
        "temp_min": ("RdYlBu_r", 5,  35,   "°C"),
    }
    return maps.get(variable, ("viridis", None, None, ""))


def _plot_panel(ax, data, lats, lons, cmap, vmin, vmax, title):
    if HAS_CARTOPY:
        ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle="--")
        ax.add_feature(cfeature.STATES,    linewidth=0.3, alpha=0.5)
        lon_g, lat_g = np.meshgrid(lons, lats)
        im = ax.pcolormesh(lon_g, lat_g, data, cmap=cmap,
                           vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    else:
        lon_g, lat_g = np.meshgrid(lons, lats)
        im = ax.pcolormesh(lon_g, lat_g, data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    ax.set_title(title, fontsize=10, fontweight="bold")
    return im


def nodes_to_grid_plot(node_data: np.ndarray, graph) -> np.ndarray:
    """Reshape (N_nodes,) → (n_lat, n_lon) for plotting."""
    return node_data.reshape(graph.n_lat, graph.n_lon)