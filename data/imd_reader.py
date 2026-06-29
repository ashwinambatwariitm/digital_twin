"""
IMD Binary Data Reader
─────────────────────
Reads IMD gridded rainfall (.bin) and temperature (.GRD) binary files.

Rainfall : 0.25° x 0.25° grid, daily, unit = mm/day
Max Temp : 1.0°  x 1.0°  grid, daily, unit = °C
Min Temp : 1.0°  x 1.0°  grid, daily, unit = °C

Download from:
  Rainfall : https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html
  Max Temp : https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html
  Min Temp : https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional
import logging

log = logging.getLogger(__name__)


# ── Grid Definitions ──────────────────────────────────────────────────────────

RAINFALL_GRID = dict(
    lats=np.arange(6.5,  38.75, 0.25),   # 129 latitudes
    lons=np.arange(66.5, 100.25, 0.25),  # 135 longitudes
    missing=99.9,
)

TEMP_GRID = dict(
    lats=np.arange(7.5,  37.5 + 1.0, 1.0),  # 31 latitudes
    lons=np.arange(67.5, 97.5 + 1.0, 1.0),  # 31 longitudes
    missing=99.9,
)


# ── Readers ───────────────────────────────────────────────────────────────────

def read_imd_rainfall(bin_file: str, year: int) -> xr.DataArray:
    """
    Read one year of IMD 0.25° gridded rainfall.

    Args:
        bin_file : path to .bin file  (e.g. "RF25_ind2023_rfp25.bin")
        year     : the year (to determine leap year / n_days)

    Returns:
        xr.DataArray  shape (n_days, n_lat, n_lon)
    """
    g = RAINFALL_GRID
    n_lat, n_lon = len(g["lats"]), len(g["lons"])
    n_days = 366 if _is_leap(year) else 365

    path = Path(bin_file)
    if not path.exists():
        raise FileNotFoundError(f"IMD rainfall file not found: {bin_file}")

    # IMD binary: float32, row-major (lat × lon) × days
    raw = np.fromfile(str(path), dtype=np.float32)
    expected = n_days * n_lat * n_lon

    if raw.size != expected:
        log.warning(
            f"Size mismatch: got {raw.size} floats, expected {expected}. "
            f"Trying to reshape with actual days = {raw.size // (n_lat * n_lon)}"
        )
        n_days = raw.size // (n_lat * n_lon)

    data = raw.reshape(n_days, n_lat, n_lon)
    data[data >= g["missing"]] = np.nan
    data[data < 0] = np.nan

    import pandas as pd
    times = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")

    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": g["lats"], "lon": g["lons"]},
        attrs={"units": "mm/day", "long_name": "Daily Rainfall", "source": "IMD"},
        name="rainfall",
    )
    log.info(f"Loaded IMD rainfall: {da.shape} for year {year}")
    return da


def read_imd_temperature(grd_file: str, year: int, var: str = "max") -> xr.DataArray:
    """
    Read one year of IMD 1.0° gridded temperature.

    Args:
        grd_file : path to .GRD file
        year     : the year
        var      : "max" or "min"

    Returns:
        xr.DataArray  shape (n_days, n_lat, n_lon)
    """
    g = TEMP_GRID
    n_lat, n_lon = len(g["lats"]), len(g["lons"])
    n_days = 366 if _is_leap(year) else 365

    path = Path(grd_file)
    if not path.exists():
        raise FileNotFoundError(f"IMD temperature file not found: {grd_file}")

    raw = np.fromfile(str(path), dtype=np.float32)
    actual_days = raw.size // (n_lat * n_lon)
    if actual_days != n_days:
        log.warning(f"Expected {n_days} days, got {actual_days}")
        n_days = actual_days

    data = raw.reshape(n_days, n_lat, n_lon)
    data[data >= g["missing"]] = np.nan
    data[data <= -g["missing"]] = np.nan

    import pandas as pd
    times = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")

    name = f"temp_{var}"
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": g["lats"], "lon": g["lons"]},
        attrs={
            "units": "°C",
            "long_name": f"Daily {'Maximum' if var == 'max' else 'Minimum'} Temperature",
            "source": "IMD",
        },
        name=name,
    )
    log.info(f"Loaded IMD {var} temp: {da.shape} for year {year}")
    return da


def regrid_temperature_to_rainfall(temp_da: xr.DataArray) -> xr.DataArray:
    """
    Upsample 1.0° temperature grid to 0.25° rainfall grid via bilinear interp.
    Required to stack all variables into one tensor.
    """
    target_lats = RAINFALL_GRID["lats"]
    target_lons = RAINFALL_GRID["lons"]
    return temp_da.interp(lat=target_lats, lon=target_lons, method="linear")


def load_year(
    year: int,
    imd_dir: str,
    rainfall_pattern: str = "rain/{year}.grd",
    tmax_pattern: str    = "tmax/{year}.GRD",
    tmin_pattern: str    = "tmin/{year}.GRD",
) -> dict:
    """
    Load a full year of IMD data (rainfall + tmax + tmin).
    Returns dict with regridded xr.DataArrays all on 0.25° grid.

    Args:
        year        : e.g. 2023
        imd_dir     : folder containing the .bin and .GRD files
        *_pattern   : filename patterns (use {year} placeholder)

    Returns:
        {
          "rainfall": xr.DataArray (365, 129, 135),
          "temp_max": xr.DataArray (365, 129, 135),
          "temp_min": xr.DataArray (365, 129, 135),
        }
    """
    d = Path(imd_dir)

    rain_file = d / rainfall_pattern.format(year=year)
    tmax_file = d / tmax_pattern.format(year=year)
    tmin_file = d / tmin_pattern.format(year=year)

    rain  = read_imd_rainfall(str(rain_file), year)
    tmax  = read_imd_temperature(str(tmax_file), year, var="max")
    tmin  = read_imd_temperature(str(tmin_file), year, var="min")

    # Regrid temperature to 0.25° to match rainfall
    tmax_regrid = regrid_temperature_to_rainfall(tmax)
    tmin_regrid = regrid_temperature_to_rainfall(tmin)

    # Align time dimension (in case of slight mismatch)
    common_times = rain.time.values
    tmax_regrid  = tmax_regrid.sel(time=common_times, method="nearest")
    tmin_regrid  = tmin_regrid.sel(time=common_times, method="nearest")

    return {
        "rainfall": rain,
        "temp_max": tmax_regrid,
        "temp_min": tmin_regrid,
    }


# ── Synthetic Data (for testing without downloading real files) ───────────────

def generate_synthetic_imd(year: int = 2023, seed: int = 42) -> dict:
    """
    Generate realistic-looking synthetic IMD data for testing.
    Uses seasonal patterns so the model has something to learn.
    """
    rng = np.random.default_rng(seed)
    g   = RAINFALL_GRID
    n_lat, n_lon = len(g["lats"]), len(g["lons"])
    n_days = 366 if _is_leap(year) else 365

    import pandas as pd
    times = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    day_of_year = np.arange(n_days)

    # Seasonal envelope: monsoon peaks around day 180 (late June)
    monsoon  = np.exp(-((day_of_year - 180) ** 2) / (2 * 45**2))
    monsoon  = monsoon / monsoon.max()

    # Base rainfall with spatial gradient (wetter in west coast / NE)
    lat_grid = g["lats"][:, None]  * np.ones((1, n_lon))
    lon_grid = np.ones((n_lat, 1)) * g["lons"][None, :]

    # Western Ghats effect
    west_ghats = np.exp(-((lon_grid - 76) ** 2) / (2 * 3**2))
    # NE India effect
    ne_india   = np.exp(-((lat_grid - 26) ** 2 + (lon_grid - 92) ** 2) / 30)

    spatial_mask = 0.6 * west_ghats + 0.4 * ne_india

    # Generate rainfall: (n_days, n_lat, n_lon)
    rain = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    for d in range(n_days):
        base    = monsoon[d] * 20 * spatial_mask
        noise   = rng.exponential(scale=base + 0.5)
        rain[d] = np.clip(noise, 0, 200)
    rain[rain < 1.0] = 0   # dry days

    # Temperature: seasonal + lat gradient
    temp_seasonal = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
    lat_cooling   = (lat_grid - 6.5) / 32 * 5   # cooler in north
    tmax = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    tmin = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    for d in range(n_days):
        tmax[d] = temp_seasonal[d] + 25 - lat_cooling + rng.normal(0, 2, (n_lat, n_lon))
        tmin[d] = temp_seasonal[d] + 15 - lat_cooling + rng.normal(0, 2, (n_lat, n_lon))

    def make_da(data, name, units):
        return xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": g["lats"], "lon": g["lons"]},
            attrs={"units": units, "source": "synthetic"},
            name=name,
        )

    log.info(f"Generated synthetic IMD data for year {year}: {n_days} days")
    return {
        "rainfall": make_da(rain, "rainfall", "mm/day"),
        "temp_max": make_da(tmax, "temp_max", "°C"),
        "temp_min": make_da(tmin, "temp_min", "°C"),
    }


# ── Utils ─────────────────────────────────────────────────────────────────────

def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)