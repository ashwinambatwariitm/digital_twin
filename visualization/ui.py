"""
Gradio UI
─────────
Web interface to run inference and view forecasts.
Runs locally at http://localhost:7860
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)


def launch_ui(model, graph, test_results: dict, config: dict, dataset_stats: dict):
    """
    Launch Gradio interface for exploring model forecasts.

    Args:
        model        : trained MiniGraphCast
        graph        : India graph (torch_geometric Data)
        test_results : output of evaluate_model()
        config       : config dict
        dataset_stats: normalization stats
    """
    predictions = test_results["predictions"]   # (T, N, 3)
    targets     = test_results["targets"]       # (T, N, 3)
    metrics     = test_results["metrics"]

    lats = graph.lats.numpy()
    lons = graph.lons.numpy()
    n_lat = graph.n_lat
    n_lon = graph.n_lon
    T     = predictions.shape[0]

    VARS = ["Rainfall (mm/day)", "Max Temperature (°C)", "Min Temperature (°C)"]
    VAR_KEYS = ["rainfall", "temp_max", "temp_min"]

    # ── Forecast Map Tab ─────────────────────────────────────────
    def make_forecast_map(time_idx: int, variable: str):
        vi   = VARS.index(variable)
        pred = predictions[time_idx, :, vi].reshape(n_lat, n_lon)
        true = targets[time_idx,     :, vi].reshape(n_lat, n_lon)
        err  = pred - true

        cmap_data = {
            0: ("YlGnBu",   0,   50),
            1: ("RdYlBu_r", 15,  45),
            2: ("RdYlBu_r", 5,   35),
        }
        cmap, vmin, vmax = cmap_data[vi]
        err_max = np.nanpercentile(np.abs(err), 95)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        lon_g, lat_g = np.meshgrid(lons, lats)

        for ax, data, title, cm, vn, vx in [
            (axes[0], true, "Ground Truth", cmap, vmin, vmax),
            (axes[1], pred, "Prediction",   cmap, vmin, vmax),
            (axes[2], err,  "Error",        "RdBu_r", -err_max, err_max),
        ]:
            im = ax.pcolormesh(lon_g, lat_g, data, cmap=cm, vmin=vn, vmax=vx)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

        fig.suptitle(f"{variable} — Time Step {time_idx}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    # ── Metrics Tab ──────────────────────────────────────────────
    def show_metrics():
        lines = ["### 📊 Model Evaluation Metrics\n"]
        for ch in ["rainfall", "temp_max", "temp_min"]:
            lines.append(f"**{ch.upper()}**")
            for m in ["RMSE", "MAE", "Bias", "ACC"]:
                key = f"{ch}_{m}"
                if key in metrics:
                    lines.append(f"  - {m}: `{metrics[key]:.4f}`")
            lines.append("")

        lines.append("**RAINFALL — Categorical Skills**")
        for thr in [1.0, 5.0, 10.0, 25.0]:
            csi = metrics.get(f"rainfall_CSI_{thr}mm", 0)
            ets = metrics.get(f"rainfall_ETS_{thr}mm", 0)
            lines.append(f"  - ≥{thr}mm | CSI: `{csi:.3f}` | ETS: `{ets:.3f}`")

        return "\n".join(lines)

    # ── Time Series Tab ──────────────────────────────────────────
    def make_timeseries(lat_in: float, lon_in: float, variable: str):
        # Find nearest node
        vi    = VARS.index(variable)
        lats2 = graph.node_lats.numpy()
        lons2 = graph.node_lons.numpy()
        dist  = (lats2 - lat_in)**2 + (lons2 - lon_in)**2
        node  = int(np.argmin(dist))
        actual_lat = float(lats2[node])
        actual_lon = float(lons2[node])

        pred_ts = predictions[:, node, vi]
        true_ts = targets[:,    node, vi]
        t       = np.arange(T)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, true_ts, label="Truth",      color="#1d4ed8", linewidth=1.5)
        ax.plot(t, pred_ts, label="Prediction", color="#dc2626",
                linewidth=1.5, linestyle="--", alpha=0.85)
        ax.set_xlabel("Time Step (days)"); ax.set_ylabel(variable)
        ax.set_title(
            f"{variable} at ({actual_lat:.2f}°N, {actual_lon:.2f}°E)",
            fontweight="bold"
        )
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # ── Loss Curve Tab ───────────────────────────────────────────
    def show_loss_curves():
        hist_path = Path(config["paths"]["checkpoints"]) / "history.json"
        if not hist_path.exists():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No training history found", ha="center", va="center")
            return fig

        with open(hist_path) as f:
            history = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="Train", color="#2563eb", linewidth=2)
        axes[0].plot(history["val_loss"],   label="Val",   color="#dc2626", linewidth=2)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(history["lr"], color="#16a34a", linewidth=2)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("LR")
        axes[1].set_title("Learning Rate"); axes[1].set_yscale("log"); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    # ── Build UI ─────────────────────────────────────────────────
    with gr.Blocks(title="GraphCast India — PS-5 Hackathon") as app:
        gr.Markdown(
            "# 🌧️ Mini-GraphCast India\n"
            "**PS-5 Bharatiya Antariksh Hackathon 2026** — IMD + INSAT Fusion\n"
        )

        with gr.Tab("🗺️ Forecast Map"):
            gr.Markdown("Compare ground truth vs model prediction spatially.")
            with gr.Row():
                t_slider = gr.Slider(0, T-1, value=0, step=1,
                                     label="Time Step (day)")
                var_dd   = gr.Dropdown(VARS, value=VARS[0], label="Variable")
            map_btn = gr.Button("Generate Map", variant="primary")
            map_out = gr.Plot()
            map_btn.click(make_forecast_map, inputs=[t_slider, var_dd], outputs=map_out)

        with gr.Tab("📈 Time Series"):
            gr.Markdown("Forecast vs truth at a specific grid location.")
            with gr.Row():
                lat_in  = gr.Number(value=19.0,  label="Latitude (°N)")
                lon_in  = gr.Number(value=77.0,  label="Longitude (°E)")
                var_dd2 = gr.Dropdown(VARS, value=VARS[0], label="Variable")
            ts_btn = gr.Button("Plot Time Series", variant="primary")
            ts_out = gr.Plot()
            ts_btn.click(make_timeseries, inputs=[lat_in, lon_in, var_dd2], outputs=ts_out)

        with gr.Tab("📊 Metrics"):
            gr.Markdown("Quantitative evaluation on the held-out test set.")
            metrics_btn = gr.Button("Show Metrics", variant="primary")
            metrics_out = gr.Markdown()
            metrics_btn.click(show_metrics, outputs=metrics_out)

        with gr.Tab("📉 Training Curves"):
            gr.Markdown("Loss and learning rate over training epochs.")
            loss_btn = gr.Button("Show Curves", variant="primary")
            loss_out = gr.Plot()
            loss_btn.click(show_loss_curves, outputs=loss_out)

    log.info("Launching Gradio UI at http://localhost:7860")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)