"""
post_process_for_ml.py
─────────────────────
Takes insat_lst_complete.csv and produces a lean, ML-ready training file.

Run:
    python3 post_process_for_ml.py
"""

import pandas as pd
import numpy as np

# ── 1. LOAD ────────────────────────────────────────────────────────────────
df = pd.read_csv('insat_lst_complete.csv')
print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")

# ── 2. DROP CONSTANT / PROVENANCE-ONLY COLUMNS ─────────────────────────────
# These never vary within a single file — useless as ML features
CONSTANT_COLS = [
    'satellite', 'sensor', 'processing_level', 'product_type',
    'unique_id', 'acquisition_date', 'acquisition_start', 'acquisition_end',
    'product_creation_time', 'hdf_source_file', 'output_format',
    'software_version', 'ground_station', 'institute', 'conventions',
    'sat_nominal_lat_deg', 'sat_altitude_km',
    'scene_left_lon', 'scene_right_lon', 'scene_upper_lat', 'scene_lower_lat',  # all 0.0
    'lst_units_raw', 'fill_value_raw', 'scale_factor', 'add_offset',
    'source_file', 'date',  # keeping date_iso instead
]
df.drop(columns=[c for c in CONSTANT_COLS if c in df.columns], inplace=True)
print(f"After dropping constants: {len(df.columns)} columns")

# ── 3. FIX DATETIME ────────────────────────────────────────────────────────
df['date_iso'] = pd.to_datetime(df['date_iso'])

# Temporal features useful for ML
df['year']        = df['date_iso'].dt.year
df['month']       = df['date_iso'].dt.month
df['day']         = df['date_iso'].dt.day
df['day_of_year'] = df['date_iso'].dt.dayofyear

# Cyclical encoding of month and day-of-year (avoids Dec→Jan discontinuity)
df['month_sin']   = np.sin(2 * np.pi * df['month']      / 12)
df['month_cos']   = np.cos(2 * np.pi * df['month']      / 12)
df['doy_sin']     = np.sin(2 * np.pi * df['day_of_year']/ 365)
df['doy_cos']     = np.cos(2 * np.pi * df['day_of_year']/ 365)

# Hour from time_gmt
df['hour'] = pd.to_datetime(df['time_gmt'], format='%H:%M:%S').dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Drop raw datetime strings — keep numeric versions only
df.drop(columns=['date_iso', 'time_gmt', 'time_minutes_epoch'], inplace=True)

# ── 4. FIX sat_nominal_lon (74°E) ──────────────────────────────────────────
# View zenith angle proxy: distance of each pixel from sub-satellite point
# Useful feature — pixels far from 74°E have more atmospheric path distortion
df['delta_lon_from_sat'] = (df['longitude'] - 74.0).abs()
df['delta_lat_from_sat'] = (df['latitude']  - 0.0 ).abs()

# ── 5. HANDLE NaN std (single-pixel cells) ─────────────────────────────────
# Option A: fill with 0 (no spread known → assume uniform)
df['lst_k_std'] = df['lst_k_std'].fillna(0.0)

# Flag single-pixel cells so model can weight them differently
df['is_single_pixel'] = (df['pixel_count'] == 1).astype(int)

# ── 6. QUALITY FLAG ────────────────────────────────────────────────────────
# Low pixel_count cells are less reliable
df['qa_flag'] = pd.cut(
    df['pixel_count'],
    bins=[0, 1, 5, 20, np.inf],
    labels=['single', 'low', 'medium', 'high']
).astype(str)

# ── 7. FINAL COLUMN ORDER ──────────────────────────────────────────────────
feature_cols = [
    # Spatiotemporal identifiers
    'latitude', 'longitude',
    'year', 'month', 'day', 'day_of_year', 'hour',
    'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    'hour_sin', 'hour_cos',
    # Satellite geometry
    'delta_lon_from_sat', 'delta_lat_from_sat',
    # LST features (target + diagnostics)
    'lst_c_mean',     # ← primary TARGET for temperature prediction
    'lst_c_min',
    'lst_c_max',
    'lst_c_median',
    'lst_k_mean',     # keep K versions too for physics-based models
    'lst_k_min',
    'lst_k_max',
    'lst_k_median',
    'lst_k_std',
    # Data quality
    'pixel_count',
    'is_single_pixel',
    'qa_flag',
]

df = df[[c for c in feature_cols if c in df.columns]]

# ── 8. SAVE ────────────────────────────────────────────────────────────────
OUT = 'insat_lst_ml_ready.csv'
df.to_csv(OUT, index=False)

print(f"\nFinal shape : {df.shape}")
print(f"Columns     : {list(df.columns)}")
print(f"\nSample:")
print(df.head(3).to_string(index=False))
print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nSaved → {OUT}")