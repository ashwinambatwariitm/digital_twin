"""
Climate Dataset
───────────────
Combines IMD (rainfall + temp) and INSAT (LST + SST + IMC) into a
unified PyTorch Dataset for the Mini-GraphCast model.

Each sample:
  x : (seq_len, N_nodes, 6)  — 7 days of history, 6 variables per node
  y : (pred_steps, N_nodes, 3) — 3 days of future rainfall + tmax + tmin
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import pickle
import logging

log = logging.getLogger(__name__)

# Channel ordering — important to keep consistent everywhere
CHANNELS = ["rainfall", "temp_max", "temp_min", "LST", "SST", "IMC"]
TARGET_CHANNELS = ["rainfall", "temp_max", "temp_min"]   # what we predict


class ClimateDataset(Dataset):
    """
    Graph-based climate dataset.

    Each item returns (x, y) where:
      x : (seq_len, N_nodes, n_channels)
      y : (pred_steps, N_nodes, n_targets)
    """

    def __init__(
        self,
        data:       dict,         # {channel_name: np.array (T, N_nodes)}
        seq_len:    int = 7,
        pred_steps: int = 3,
        normalize:  bool = True,
        stats:      dict = None,  # provide for val/test (don't recompute)
    ):
        self.seq_len    = seq_len
        self.pred_steps = pred_steps
        self.channels   = CHANNELS

        # ── Stack channels → (T, N_nodes, C) ─────────────────────
        arrays = []
        for ch in CHANNELS:
            if ch not in data:
                log.warning(f"Channel '{ch}' missing — filling with zeros")
                ref   = next(iter(data.values()))
                arr   = np.zeros_like(ref, dtype=np.float32)
            else:
                arr = data[ch].astype(np.float32)
            arrays.append(arr)

        # shape: (T, N_nodes, C)
        self.raw = np.stack(arrays, axis=-1)
        T, N, C  = self.raw.shape
        log.info(f"Dataset: T={T} days, N={N} nodes, C={C} channels")

        # ── Normalize ────────────────────────────────────────────
        if normalize:
            if stats is None:
                # Compute from this split
                self.mean = self.raw.mean(axis=(0, 1), keepdims=True)  # (1,1,C)
                self.std  = self.raw.std( axis=(0, 1), keepdims=True)  # (1,1,C)
                self.std[self.std < 1e-6] = 1.0   # avoid div-by-zero
            else:
                self.mean = stats["mean"]
                self.std  = stats["std"]

            self.data = (self.raw - self.mean) / self.std
        else:
            self.mean = np.zeros((1, 1, len(CHANNELS)), dtype=np.float32)
            self.std  = np.ones( (1, 1, len(CHANNELS)), dtype=np.float32)
            self.data = self.raw.copy()

        # Target channel indices
        self.target_idx = [CHANNELS.index(c) for c in TARGET_CHANNELS]

        # Valid start indices (need seq_len history + pred_steps future)
        self.valid_starts = T - seq_len - pred_steps + 1
        log.info(f"Valid samples: {self.valid_starts}")

    def __len__(self):
        return self.valid_starts

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]               # (S, N, C)
        y = self.data[idx + self.seq_len :
                      idx + self.seq_len + self.pred_steps]   # (P, N, C)

        y_target = y[:, :, self.target_idx]                   # (P, N, 3)

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y_target),
        )

    def get_stats(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def denormalize(self, y_norm: np.ndarray, target_only: bool = True) -> np.ndarray:
        """Reverse normalization on predictions."""
        if target_only:
            mean = self.mean[:, :, self.target_idx]
            std  = self.std[ :, :, self.target_idx]
        else:
            mean = self.mean
            std  = self.std
        return y_norm * std + mean


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def prepare_datasets(
    imd_data:   dict,
    insat_data: dict,
    graph,
    config:     dict,
) -> tuple:
    """
    Combine IMD + INSAT, flatten to nodes, split into train/val/test.

    Args:
        imd_data   : output of imd_reader.load_year or generate_synthetic_imd
        insat_data : output of insat_reader.generate_synthetic_insat
        graph      : torch_geometric Data object
        config     : full config dict

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    from graph.builder import grid_to_nodes

    n_lat = graph.n_lat
    n_lon = graph.n_lon
    N     = graph.num_nodes

    # ── Regrid all data to graph resolution ──────────────────────
    # IMD might be 0.25° but graph might be 0.5° (laptop mode)
    channel_data = {}

    for name, da in imd_data.items():
        arr = _regrid_to_graph(da.values, da.lat.values, da.lon.values, graph)
        channel_data[name] = grid_to_nodes(arr, graph).numpy()  # (T, N)

    for name, da in insat_data.items():
        arr = _regrid_to_graph(da.values, da.lat.values, da.lon.values, graph)
        channel_data[name] = grid_to_nodes(arr, graph).numpy()

    # ── Fill remaining NaN with channel mean ─────────────────────
    for name, arr in channel_data.items():
        nan_frac = np.isnan(arr).mean()
        if nan_frac > 0:
            log.info(f"Channel '{name}': {nan_frac:.1%} NaN — filling with column mean")
            col_mean = np.nanmean(arr, axis=0, keepdims=True)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            mask = np.isnan(arr)
            arr[mask] = np.broadcast_to(col_mean, arr.shape)[mask]
            channel_data[name] = arr

    # ── Split by time (no shuffling — preserve temporal order) ───
    T          = next(iter(channel_data.values())).shape[0]
    val_frac   = config["training"]["val_split"]
    test_frac  = config["training"]["test_split"]
    train_frac = 1.0 - val_frac - test_frac

    t_train = int(T * train_frac)
    t_val   = int(T * (train_frac + val_frac))

    def slice_data(start, end):
        return {k: v[start:end] for k, v in channel_data.items()}

    seq_len    = config["model"]["seq_len"]
    pred_steps = config["model"]["pred_steps"]

    train_data = slice_data(0,       t_train)
    val_data   = slice_data(t_train, t_val)
    test_data  = slice_data(t_val,   T)

    train_ds = ClimateDataset(train_data, seq_len, pred_steps, normalize=True)
    stats    = train_ds.get_stats()   # use train stats for val + test
    val_ds   = ClimateDataset(val_data,  seq_len, pred_steps, normalize=True, stats=stats)
    test_ds  = ClimateDataset(test_data, seq_len, pred_steps, normalize=True, stats=stats)

    log.info(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} samples")
    return train_ds, val_ds, test_ds, stats


def make_dataloaders(train_ds, val_ds, test_ds, config: dict) -> tuple:
    """Create DataLoaders from datasets."""
    bs  = config["training"]["batch_size"]
    nw  = config["training"]["num_workers"]
    pin = config["training"]["pin_memory"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=1,  shuffle=False,
                              num_workers=nw, pin_memory=pin)
    return train_loader, val_loader, test_loader


# ── Helpers ───────────────────────────────────────────────────────────────────

def _regrid_to_graph(
    data: np.ndarray,      # (T, n_lat_src, n_lon_src)
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    graph,
) -> np.ndarray:
    """
    Bilinear interpolation from source grid to graph node positions.
    Returns (T, n_lat_graph, n_lon_graph).
    """
    from scipy.interpolate import RegularGridInterpolator

    T           = data.shape[0]
    tgt_lats    = graph.lats.numpy()
    tgt_lons    = graph.lons.numpy()
    out         = np.zeros((T, len(tgt_lats), len(tgt_lons)), dtype=np.float32)

    # Need sorted lats for RegularGridInterpolator
    lat_order = np.argsort(src_lats)
    sorted_lats = src_lats[lat_order]

    for t in range(T):
        frame = data[t][lat_order, :]   # sort lats
        # Replace NaN with mean for interpolation
        frame_filled = np.where(np.isnan(frame), np.nanmean(frame), frame)
        interp = RegularGridInterpolator(
            (sorted_lats, src_lons), frame_filled,
            method="linear", bounds_error=False, fill_value=np.nan
        )
        tgt_lat_g, tgt_lon_g = np.meshgrid(tgt_lats, tgt_lons, indexing="ij")
        pts     = np.stack([tgt_lat_g.ravel(), tgt_lon_g.ravel()], axis=1)
        out[t]  = interp(pts).reshape(len(tgt_lats), len(tgt_lons))

    return out