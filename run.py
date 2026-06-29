"""
run.py — Main Entry Point
─────────────────────────
GraphCast-India: IMD + INSAT Fusion for PS-5 Hackathon

Usage:
    python run.py --mode train       # train with synthetic data
    python run.py --mode train --real_data   # train with real IMD/INSAT files
    python run.py --mode eval        # evaluate best checkpoint
    python run.py --mode ui          # launch Gradio UI
    python run.py --mode all         # train + eval + ui
"""

import argparse
import logging
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("graphcast_india.log"),
    ],
)
log = logging.getLogger("run")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GraphCast India — PS-5 Hackathon")
    parser.add_argument("--mode",      choices=["train", "eval", "ui", "all"],
                        default="all")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--real_data", action="store_true",
                        help="Use real IMD/INSAT files instead of synthetic")
    parser.add_argument("--resume",    default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--year",      type=int, default=2023,
                        help="Year to use for synthetic data")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    console.print(Panel.fit(
        "[bold cyan]🌧️  Mini-GraphCast India[/bold cyan]\n"
        "PS-5 Bharatiya Antariksh Hackathon 2026\n"
        f"Mode: [yellow]{args.mode}[/yellow] | "
        f"Data: [yellow]{'real' if args.real_data else 'synthetic'}[/yellow]",
        border_style="cyan"
    ))

    # ── Device info ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[green]Device:[/green] {device}")
    if device.type == "cuda":
        console.print(f"[green]GPU:[/green] {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]VRAM:[/green] {vram:.1f} GB")

    # ── Step 1: Load / Generate Data ─────────────────────────────
    console.print("\n[bold]Step 1: Loading data...[/bold]")
    imd_data, insat_data = load_data(config, args)

    # ── Step 2: Build Graph ──────────────────────────────────────
    console.print("\n[bold]Step 2: Building graph...[/bold]")
    from graph.builder import load_or_build_graph
    graph = load_or_build_graph(config)
    console.print(f"  Nodes: {graph.num_nodes} | Edges: {graph.edge_index.shape[1]}")

    # ── Step 3: Prepare Datasets ─────────────────────────────────
    console.print("\n[bold]Step 3: Preparing datasets...[/bold]")
    from data.dataset import prepare_datasets, make_dataloaders
    train_ds, val_ds, test_ds, stats = prepare_datasets(
        imd_data, insat_data, graph, config
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds, config
    )
    console.print(
        f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} samples"
    )

    # ── Step 4: Build Model ──────────────────────────────────────
    console.print("\n[bold]Step 4: Building model...[/bold]")
    from model.graphcast import MiniGraphCast
    model = MiniGraphCast(
        node_features    = config["model"]["node_features"],
        hidden_dim       = config["model"]["hidden_dim"],
        n_process_layers = config["model"]["n_process_layers"],
        seq_len          = config["model"]["seq_len"],
        pred_steps       = config["model"]["pred_steps"],
        dropout          = config["model"]["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Parameters: {n_params:,}")

    # ── Step 5: Train ─────────────────────────────────────────────
    if args.mode in ["train", "all"]:
        console.print("\n[bold]Step 5: Training...[/bold]")
        from training.trainer import Trainer
        trainer = Trainer(
            model, train_loader, val_loader, graph, config,
            resume_from=args.resume
        )
        history = trainer.train()
        console.print("[green]✅ Training complete![/green]")

    # ── Step 6: Evaluate ─────────────────────────────────────────
    if args.mode in ["eval", "all"]:
        console.print("\n[bold]Step 6: Evaluating...[/bold]")
        from training.trainer import load_best_model
        from evaluation.metrics import evaluate_model
        from visualization.plotter import (
            plot_loss_curves, plot_metrics_bar,
            plot_multi_day_forecast, nodes_to_grid_plot
        )

        model       = load_best_model(model, config)
        test_results = evaluate_model(
            model, test_loader, graph, stats, config, device
        )

        # Save plots
        fig_dir = Path(config["paths"]["figures"])
        fig_dir.mkdir(parents=True, exist_ok=True)

        hist_path = Path(config["paths"]["checkpoints"]) / "history.json"
        if hist_path.exists():
            import json
            with open(hist_path) as f:
                history = json.load(f)
            plot_loss_curves(history, save_path=str(fig_dir / "loss_curves.png"))

        plot_metrics_bar(
            test_results["metrics"],
            save_path=str(fig_dir / "metrics_bar.png")
        )

        # Sample forecast map (first test sample, all 3 lead days)
        preds   = test_results["predictions"]    # (T, N, 3)
        targets = test_results["targets"]

        for vi, vname in enumerate(["rainfall", "temp_max", "temp_min"]):
            pred_grid = preds[:3,   :, vi].reshape(3, graph.n_lat, graph.n_lon)
            true_grid = targets[:3, :, vi].reshape(3, graph.n_lat, graph.n_lon)
            plot_multi_day_forecast(
                pred_grid, true_grid,
                graph.lats.numpy(), graph.lons.numpy(),
                variable  = vname,
                save_path = str(fig_dir / f"forecast_{vname}.png")
            )

        console.print(f"[green]✅ Figures saved to {fig_dir}[/green]")

    # ── Step 7: Launch UI ─────────────────────────────────────────
    if args.mode in ["ui", "all"]:
        console.print("\n[bold]Step 7: Launching UI...[/bold]")

        if args.mode == "ui":
            # Load model + generate test results if only running UI
            from training.trainer import load_best_model
            from evaluation.metrics import evaluate_model
            model        = load_best_model(model, config)
            test_results = evaluate_model(
                model, test_loader, graph, stats, config, device
            )

        from visualization.ui import launch_ui
        launch_ui(model, graph, test_results, config, stats)


# ── Data Loader Helper ────────────────────────────────────────────────────────

def load_data(config: dict, args) -> tuple:
    """Load real or synthetic IMD + INSAT data."""

    year = args.year

    if args.real_data:
        # Real data — expects files in data/raw/imd/ and data/raw/insat/
        log.info(f"Loading real IMD data for year {year}...")
        from data.imd_reader import load_year
        from data.insat_reader import build_insat_yearly
        from data.insat_reader import generate_synthetic_insat

        imd_data = load_year(
            year    = year,
            imd_dir = config["paths"]["raw_imd"],
        )
        # Check if real INSAT files exist
        insat_dir = config["paths"]["raw_insat"]
        has_insat = all(
            (Path(insat_dir) / product).exists()
            for product in ["LST", "SST", "IMC"]
        )

        if has_insat:
            log.info(f"Loading real INSAT data for year {year}...")
            from data.insat_reader import build_insat_yearly
            insat_data = build_insat_yearly(
                insat_dir = insat_dir,
                year      = year,
            )
        else:
            log.warning("INSAT files not found — using synthetic INSAT")
            from data.insat_reader import generate_synthetic_insat
            insat_data = generate_synthetic_insat(year, seed=config["synthetic"]["seed"])
    else:
        # Synthetic data — no files needed, works immediately
        log.info(f"Generating synthetic data for year {year}...")
        from data.imd_reader import generate_synthetic_imd
        from data.insat_reader import generate_synthetic_insat

        imd_data   = generate_synthetic_imd(year,   seed=config["synthetic"]["seed"])
        insat_data = generate_synthetic_insat(year,  seed=config["synthetic"]["seed"])

    console.print(f"  IMD channels:   {list(imd_data.keys())}")
    console.print(f"  INSAT channels: {list(insat_data.keys())}")
    for name, da in {**imd_data, **insat_data}.items():
        console.print(f"    {name:12s}: {da.shape}")

    return imd_data, insat_data


if __name__ == "__main__":
    main()