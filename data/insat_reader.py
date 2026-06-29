"""
INSAT-3D / 3DR L2B Product Reader
───────────────────────────────────
Reads HDF5 products from MOSDAC (https://www.mosdac.gov.in)

Products supported:
  3RIMG_L2B_LST  → Land Surface Temperature
  3RIMG_L2B_SST  → Sea Surface Temperature
  3RIMG_L2B_IMC  → INSAT Multi-spectral Rainfall

Each product is a half-hourly HDF5 file.
We aggregate to daily means to match IMD temporal resolution.
"""

import h5py
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, List
import logging
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# Target grid to regrid onto (matches IMD rainfall grid)
TARGET_LATS = np.arange(6.5,  38.75, 0.25)
TARGET_LONS = np.arange(66.5, 100.25, 0.25)


# ── Single File Reader ────────────────────────────────────────────────────────

def read_insat_file(h5_path: str, product: str = "LST") -> Optional[xr.DataArray]:
    """
    Read a single INSAT L2B HDF5 file.

    Args:
        h5_path : path to .h5 file from MOSDAC
        product : "LST", "SST", or "IMC"

    Returns:
        xr.DataArray on native INSAT grid, or None if read fails
    """
    product_keys = {
        "LST": "L2B_LST",
        "SST": "L2B_SST",
        "IMC": "L2B_IMC",
    }
    key = product_keys.get(product)
    if key is None:
        raise ValueError(f"Unknown product: {product}. Choose from LST, SST, IMC")

    path = Path(h5_path)
    if not path.exists():
        log.warning(f"INSAT file not found: {h5_path}")
        return None

    try:
        with h5py.File(str(path), "r") as f:
            # Read the main data variable
            if key not in f:
                # Try alternative key names (MOSDAC format varies slightly)
                available = list(f.keys())
                log.warning(f"Key '{key}' not found. Available: {available}")
                # Try to find a close match
                matches = [k for k in available if product in k.upper()]
                if not matches:
                    return None
                key = matches[0]

            data = f[key][:]

            # Get scale, offset, fill from attributes
            attrs   = f[key].attrs
            scale   = float(attrs.get("scale_factor",  1.0))
            offset  = float(attrs.get("add_offset",    0.0))
            fill    = float(attrs.get("_FillValue",  -999.0))

            # Get geolocation arrays
            if "Latitude" in f and "Longitude" in f:
                lats = f["Latitude"][:]
                lons = f["Longitude"][:]
                # INSAT lat/lon can be 2D (y, x) — take first row/col
                if lats.ndim == 2:
                    lats = lats[:, 0]
                    lons = lons[0, :]
            else:
                # Fallback: INSAT-3DR covers roughly 60°E–105°E, -10°N–45°N
                lats = np.linspace(-10, 45,  data.shape[0])
                lons = np.linspace( 60, 105, data.shape[1])

            # Apply scale + mask fill
            data = data.astype(np.float32)
            data[np.abs(data - fill) < 1e-3] = np.nan
            data = data * scale + offset

            # Extract timestamp from filename: *YYYYMMDD_HHMM*.h5
            timestamp = _extract_timestamp(path.name)

    except Exception as e:
        log.error(f"Failed to read {h5_path}: {e}")
        return None

    da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
        attrs={
            "units":     _units_for(product),
            "long_name": _longname_for(product),
            "source":    "INSAT-3DR MOSDAC",
            "timestamp": str(timestamp),
        },
        name=product,
    )
    return da


# ── Daily Aggregator ──────────────────────────────────────────────────────────

def aggregate_insat_daily(
    h5_files: List[str],
    product:  str,
    target_lats: np.ndarray = TARGET_LATS,
    target_lons: np.ndarray = TARGET_LONS,
) -> xr.DataArray:
    """
    Read multiple INSAT files for one day, regrid, and take daily mean.

    Args:
        h5_files    : list of .h5 files for one day (half-hourly = ~48 files)
        product     : "LST", "SST", or "IMC"
        target_lats : output latitude grid (default: IMD 0.25°)
        target_lons : output longitude grid

    Returns:
        xr.DataArray  shape (n_lat, n_lon)  — daily mean on IMD grid
    """
    slices = []
    for f in h5_files:
        da = read_insat_file(f, product)
        if da is None:
            continue
        # Clip to India domain
        da = da.sel(
            lat=slice(target_lats.min() - 1, target_lats.max() + 1),
            lon=slice(target_lons.min() - 1, target_lons.max() + 1),
        )
        # Regrid to IMD grid
        da_regrid = da.interp(
            lat=target_lats, lon=target_lons, method="linear"
        )
        slices.append(da_regrid.values)

    if not slices:
        log.warning(f"No valid INSAT {product} files for this day — filling NaN")
        return xr.DataArray(
            np.full((len(target_lats), len(target_lons)), np.nan, dtype=np.float32),
            dims=["lat", "lon"],
            coords={"lat": target_lats, "lon": target_lons},
            name=product,
        )

    daily_mean = np.nanmean(np.stack(slices, axis=0), axis=0)
    return xr.DataArray(
        daily_mean.astype(np.float32),
        dims=["lat", "lon"],
        coords={"lat": target_lats, "lon": target_lons},
        attrs={"units": _units_for(product), "source": "INSAT-3DR daily mean"},
        name=product,
    )


# ── Year Builder ──────────────────────────────────────────────────────────────

def build_insat_yearly(
    insat_dir: str,
    year:      int,
    products:  List[str] = ["LST", "SST", "IMC"],
) -> dict:
    """
    Walk an INSAT directory organised as:
        insat_dir/
          LST/YYYYMMDD/*.h5
          SST/YYYYMMDD/*.h5
          IMC/YYYYMMDD/*.h5

    Returns dict of xr.DataArrays, each shape (n_days, n_lat, n_lon).
    """
    import pandas as pd
    n_days = 366 if _is_leap(year) else 365
    times  = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    result = {}

    for product in products:
        prod_dir = Path(insat_dir) / product
        daily_arrays = []

        for day in times:
            date_str  = day.strftime("%Y%m%d")
            day_dir   = prod_dir / date_str
            h5_files  = sorted(day_dir.glob("*.h5")) if day_dir.exists() else []
            daily_da  = aggregate_insat_daily(
                [str(f) for f in h5_files], product
            )
            daily_arrays.append(daily_da.values)

        stack = np.stack(daily_arrays, axis=0)  # (n_days, n_lat, n_lon)
        result[product] = xr.DataArray(
            stack,
            dims=["time", "lat", "lon"],
            coords={
                "time": times,
                "lat":  TARGET_LATS,
                "lon":  TARGET_LONS,
            },
            attrs={"units": _units_for(product), "source": "INSAT-3DR"},
            name=product,
        )
        log.info(f"Built yearly INSAT {product}: {stack.shape}")

    return result


# ── Synthetic INSAT (for testing) ────────────────────────────────────────────

def generate_synthetic_insat(year: int = 2023, seed: int = 42) -> dict:
    """
    Generate synthetic INSAT products for testing.
    Correlates with synthetic IMD data (warmer LST → more rain, etc.)
    """
    from data.imd_reader import generate_synthetic_imd

    rng     = np.random.default_rng(seed + 100)
    imd     = generate_synthetic_imd(year, seed)
    rain_np = imd["rainfall"].values   # (n_days, n_lat, n_lon)
    tmax_np = imd["temp_max"].values

    n_days, n_lat, n_lon = rain_np.shape
    import pandas as pd
    times = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")

    # LST: correlated with surface temp + noise
    lst = tmax_np + rng.normal(0, 3, tmax_np.shape).astype(np.float32)

    # SST: cooler, smoother (mainly coastal/ocean)
    lon_grid = TARGET_LONS[None, None, :] * np.ones((n_days, n_lat, 1))
    coastal  = np.exp(-((lon_grid - 72) ** 2) / 30)   # west coast
    sst_base = 28 + 3 * np.sin(2 * np.pi * np.arange(n_days)[:, None, None] / 365)
    sst      = (sst_base * coastal + rng.normal(0, 1, tmax_np.shape)).astype(np.float32)

    # IMC: satellite rainfall estimate (correlated but noisier than IMD)
    imc = (rain_np * 0.8 + rng.exponential(2, rain_np.shape)).astype(np.float32)

    def make_da(data, name, units):
        return xr.DataArray(
            data, dims=["time", "lat", "lon"],
            coords={"time": times, "lat": TARGET_LATS, "lon": TARGET_LONS},
            attrs={"units": units, "source": "synthetic"},
            name=name,
        )

    log.info(f"Generated synthetic INSAT data for year {year}")
    return {
        "LST": make_da(lst,  "LST", "°C"),
        "SST": make_da(sst,  "SST", "°C"),
        "IMC": make_da(imc,  "IMC", "mm/day"),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_timestamp(filename: str) -> Optional[datetime]:
    """Extract datetime from INSAT filename e.g. 3RIMG_20230615_0000_..."""
    import re
    m = re.search(r"(\d{8})_(\d{4})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
        except ValueError:
            pass
    return None


def _units_for(product: str) -> str:
    return {"LST": "°C", "SST": "°C", "IMC": "mm/day"}.get(product, "unknown")


def _longname_for(product: str) -> str:
    return {
        "LST": "Land Surface Temperature",
        "SST": "Sea Surface Temperature",
        "IMC": "INSAT Multi-spectral Rainfall",
    }.get(product, product)


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)