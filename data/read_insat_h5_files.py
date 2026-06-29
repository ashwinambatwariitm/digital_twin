"""
INSAT-3DR LST → COMPLETE ML-ready CSV
All raw pixels + all metadata fields extracted
"""

import h5py
import numpy as np
import pandas as pd
import datetime
import os

def process_h5_complete(filepath):
    print(f"Processing: {os.path.basename(filepath)}")

    with h5py.File(filepath, 'r') as f:

        # ── DATASETS ──────────────────────────────────────────────
        lst       = f['LST'][0].copy()           # (2816, 2805) float32, Kelvin
        lat       = f['Latitude'][:] * 0.01      # degrees
        lon       = f['Longitude'][:] * 0.01     # degrees
        t_val     = float(f['time'][0])

        fill_lst  = float(f['LST'].attrs['_FillValue'][0])
        lst_units = f['LST'].attrs.get('units', b'K').decode()

        # spatial bounds stored as dataset attributes
        left_lon  = float(f['LST'].attrs.get('left_longitude',  [0])[0])
        right_lon = float(f['LST'].attrs.get('right_longitude', [0])[0])
        upper_lat = float(f['LST'].attrs.get('upper_latitude',  [0])[0])
        lower_lat = float(f['LST'].attrs.get('lower_latitude',  [0])[0])

        # scale/offset if present
        scale  = float(f['LST'].attrs.get('scale_factor', [1.0])[0])
        offset = float(f['LST'].attrs.get('add_offset',   [0.0])[0])

        # ── ROOT ATTRIBUTES ───────────────────────────────────────
        def ga(key, default=''):
            v = f.attrs.get(key, default)
            return v.decode().strip() if isinstance(v, bytes) else str(v)

        satellite      = ga('Satellite_Name',       'INSAT-3DR')
        sensor         = ga('Sensor_Name',           'IMAGER')
        proc_level     = ga('Processing_Level',      'L2B')
        product_type   = ga('Product_Type',          'GEOPHY')
        acq_date       = ga('Acquisition_Date',      '')
        acq_start      = ga('Acquisition_Start_Time','')
        acq_end        = ga('Acquisition_End_Time',  '')
        creation_time  = ga('Product_Creation_Time', '')
        hdf_filename   = ga('HDF_Product_File_Name', '')
        output_format  = ga('Output_Format',         '')
        software_ver   = ga('Software_Version',      '')
        ground_station = ga('Ground_Station',        '')
        institute      = ga('institute',             '')
        conventions    = ga('conventions',           '')
        unique_id      = ga('Unique_Id',             '')

        coords_arr = f.attrs.get(
            'Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude', [0, 74.0])
        sat_lat = float(coords_arr[0])
        sat_lon = float(coords_arr[1])

        altitude_arr = f.attrs.get('Nominal_Altitude(km)', [36000.0])
        sat_altitude_km = float(altitude_arr[0])

    # ── DECODE TIME ───────────────────────────────────────────────
    obs_dt        = datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=t_val)
    date_str      = obs_dt.strftime('%d-%m-%Y')
    date_iso      = obs_dt.strftime('%Y-%m-%d')
    time_gmt      = obs_dt.strftime('%H:%M:%S')
    time_minutes  = t_val  # raw minutes since epoch, useful for ML

    # ── VALID PIXEL MASK (India bbox) ────────────────────────────
    LAT_MIN, LAT_MAX = 6.5,  38.5
    LON_MIN, LON_MAX = 66.5, 100.0

    mask = (
        (lst != fill_lst) &
        (lst > 0) &
        (lat >= LAT_MIN) & (lat <= LAT_MAX) &
        (lon >= LON_MIN) & (lon <= LON_MAX)
    )

    lats_v = lat[mask]
    lons_v = lon[mask]
    lst_v  = lst[mask]

    # Apply scale/offset if not already applied
    # INSAT L2B LST is usually already in K, but check
    if scale != 1.0 or offset != 0.0:
        lst_v = lst_v * scale + offset

    print(f"  Valid pixels: {mask.sum():,} | LST range: {lst_v.min():.1f}–{lst_v.max():.1f} K")

    # ── IMD 0.5° GRID SNAP ───────────────────────────────────────
    IMD_LAT0, IMD_LON0, IMD_STEP = 6.5, 66.5, 0.5

    lat_imd = (np.round((lats_v - IMD_LAT0) / IMD_STEP) * IMD_STEP + IMD_LAT0).round(1)
    lon_imd = (np.round((lons_v - IMD_LON0) / IMD_STEP) * IMD_STEP + IMD_LON0).round(1)

    # ── AGGREGATE TO IMD GRID ─────────────────────────────────────
    df_raw = pd.DataFrame({'lat_imd': lat_imd, 'lon_imd': lon_imd, 'lst_k': lst_v})

    df_grid = (
        df_raw.groupby(['lat_imd', 'lon_imd'])
        .agg(
            lst_k_mean  = ('lst_k', 'mean'),
            lst_k_min   = ('lst_k', 'min'),
            lst_k_max   = ('lst_k', 'max'),
            lst_k_std   = ('lst_k', 'std'),
            lst_k_median= ('lst_k', 'median'),
            pixel_count = ('lst_k', 'count'),
        )
        .reset_index()
        .rename(columns={'lat_imd': 'latitude', 'lon_imd': 'longitude'})
    )

    # ── CELSIUS COLUMNS ───────────────────────────────────────────
    for col in ['mean', 'min', 'max', 'median']:
        df_grid[f'lst_c_{col}'] = (df_grid[f'lst_k_{col}'] - 273.15).round(4)
    for col in ['mean', 'min', 'max', 'std', 'median']:
        df_grid[f'lst_k_{col}'] = df_grid[f'lst_k_{col}'].round(4)
    df_grid['lst_k_std'] = df_grid['lst_k_std'].round(4)

    # ── METADATA COLUMNS (every attribute) ───────────────────────
    n = len(df_grid)
    df_grid.insert(0,  'date',              date_str)
    df_grid.insert(1,  'date_iso',          date_iso)
    df_grid.insert(2,  'time_gmt',          time_gmt)
    df_grid.insert(3,  'time_minutes_epoch',round(time_minutes, 4))

    df_grid['satellite']            = satellite
    df_grid['sensor']               = sensor
    df_grid['processing_level']     = proc_level
    df_grid['product_type']         = product_type
    df_grid['unique_id']            = unique_id
    df_grid['acquisition_date']     = acq_date
    df_grid['acquisition_start']    = acq_start
    df_grid['acquisition_end']      = acq_end
    df_grid['product_creation_time']= creation_time
    df_grid['hdf_source_file']      = hdf_filename
    df_grid['output_format']        = output_format
    df_grid['software_version']     = software_ver
    df_grid['ground_station']       = ground_station
    df_grid['institute']            = institute
    df_grid['conventions']          = conventions
    df_grid['sat_nominal_lat_deg']  = sat_lat
    df_grid['sat_nominal_lon_deg']  = sat_lon
    df_grid['sat_altitude_km']      = sat_altitude_km
    df_grid['scene_left_lon']       = left_lon
    df_grid['scene_right_lon']      = right_lon
    df_grid['scene_upper_lat']      = upper_lat
    df_grid['scene_lower_lat']      = lower_lat
    df_grid['lst_units_raw']        = lst_units
    df_grid['fill_value_raw']       = fill_lst
    df_grid['scale_factor']         = scale
    df_grid['add_offset']           = offset
    df_grid['source_file']          = os.path.basename(filepath)

    df_grid = df_grid.sort_values(['latitude', 'longitude']).reset_index(drop=True)
    return df_grid


# ── RUN ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    SINGLE_FILE = '/home/studinstru/ISRO/data/raw/insat/raw/Order/Jun26_184692/3RIMG_01APR2022_1115_L2B_LST_V01R00.h5'
    OUTPUT_CSV  = 'insat_lst_complete.csv'

    df = process_h5_complete(SINGLE_FILE)

    print(f"\nTotal IMD grid points : {len(df):,}")
    print(f"Total columns          : {len(df.columns)}")
    print(f"\nAll columns:\n{list(df.columns)}")
    print(f"\nSample (5 rows):")
    print(df.head(5).to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")