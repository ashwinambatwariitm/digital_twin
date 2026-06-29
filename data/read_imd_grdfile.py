import numpy as np
import struct
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
FILE_PATH = "/home/studinstru/ISRO/digital_twin/data/raw/imd/rain/2023.grd"   # <-- change to your file path
ROWS      = 129          # latitude  points: 6.5°N to 38.5°N  (0.5° step)
COLS      = 135          # longitude points: 66.5°E to 100°E  (0.5° step)
DAYS      = 365          # daily data for full year
MISSING   = -999.0       # IMD missing/ocean value

LAT_START, LAT_STEP   = 6.5,  0.5
LON_START, LON_STEP   = 66.5, 0.5
# ────────────────────────────────────────────────────────────────────────────

print("=" * 55)
print("   IMD Daily Rainfall Grid Explorer — 2023.grd")
print("=" * 55)

# ── 1. LOAD ──────────────────────────────────────────────────────────────────
size = os.path.getsize(FILE_PATH)
print(f"\n📁 File size   : {size:,} bytes ({size/1e6:.1f} MB)")

with open(FILE_PATH, "rb") as f:
    raw = np.frombuffer(f.read(), dtype=np.float32)

data = raw.reshape(DAYS, ROWS, COLS)   # shape: (365, 129, 135)
print(f"📐 Array shape : {data.shape}  →  (days, lats, lons)")

# ── 2. BASIC STATS ───────────────────────────────────────────────────────────
valid = data[data > MISSING]
print(f"\n── Quick Stats (valid land points only) ──")
print(f"  Min rainfall  : {valid.min():.2f} mm")
print(f"  Max rainfall  : {valid.max():.2f} mm")
print(f"  Mean rainfall : {valid.mean():.2f} mm")
print(f"  Valid pixels  : {valid.size:,} / {data.size:,}")

# ── 3. MONTHLY TOTALS ────────────────────────────────────────────────────────
month_days = [31,28,31,30,31,30,31,31,30,31,30,31]
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

print(f"\n── Monthly Average Rainfall (mm/day, India mean) ──")
start = 0
for m, (name, nd) in enumerate(zip(month_names, month_days)):
    chunk = data[start:start+nd]          # (nd, 129, 135)
    v = chunk[chunk > MISSING]
    avg = v.mean() if v.size else 0
    bar = "█" * int(avg * 2)
    print(f"  {name}  {avg:5.2f} mm/day  {bar}")
    start += nd

# ── 4. PEAK DAY ──────────────────────────────────────────────────────────────
daily_sum = np.where(data > MISSING, data, 0).sum(axis=(1, 2))
peak_day  = int(daily_sum.argmax())
peak_val  = daily_sum[peak_day]

import datetime
peak_date = datetime.date(2023, 1, 1) + datetime.timedelta(days=peak_day)
print(f"\n── Peak Rainfall Day ──")
print(f"  Day {peak_day+1}  →  {peak_date.strftime('%B %d, %Y')}")
print(f"  Total over India grid: {peak_val:,.0f} mm")

# ── 5. SAMPLE: read one lat/lon ───────────────────────────────────────────────
sample_lat, sample_lon = 19.0, 72.5   # Mumbai approx
r = int((sample_lat - LAT_START) / LAT_STEP)
c = int((sample_lon - LON_START) / LON_STEP)
ts = data[:, r, c]
ts_valid = ts.copy(); ts_valid[ts_valid <= MISSING] = 0

print(f"\n── Time-series at ({sample_lat}°N, {sample_lon}°E) ≈ Mumbai ──")
print(f"  Annual total : {ts_valid.sum():.1f} mm")
print(f"  Rainy days   : {(ts_valid > 2.5).sum()} days")
print(f"  First 10 days: {[round(float(v),1) for v in ts_valid[:10]]}")

# ── 6. OPTIONAL PLOT ─────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("IMD Rainfall 2023", fontsize=14, fontweight="bold")

    # Map: Annual total
    annual = np.where(data > MISSING, data, 0).sum(axis=0)
    annual_masked = np.ma.masked_where(data[0] <= MISSING, annual)
    lats = np.arange(LAT_START, LAT_START + ROWS * LAT_STEP, LAT_STEP)
    lons = np.arange(LON_START, LON_START + COLS * LON_STEP, LON_STEP)
    im = axes[0].contourf(lons, lats, annual_masked, levels=20, cmap="YlGnBu")
    plt.colorbar(im, ax=axes[0], label="mm")
    axes[0].set_title("Annual Rainfall Total (mm)")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

    # Time-series: Mumbai
    axes[1].bar(range(DAYS), ts_valid, color="steelblue", width=1, alpha=0.8)
    axes[1].set_title(f"Daily Rainfall at ~Mumbai ({sample_lat}°N, {sample_lon}°E)")
    axes[1].set_xlabel("Day of Year"); axes[1].set_ylabel("Rainfall (mm)")

    plt.tight_layout()
    plt.savefig("imd_rainfall_2023.png", dpi=150)
    print("\n✅  Plot saved → imd_rainfall_2023.png")
except Exception as e:
    print(f"\n(Plot skipped: {e})")

print("\nDone! 🎉")