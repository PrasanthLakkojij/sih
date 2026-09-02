"""
Phase 2 — Visualization
SIH26168 — AI-ML Dead Reckoning

Plots for 2-3 representative trips:
  1. Accelerometer X/Y/Z vs time
  2. Gyroscope X/Y/Z vs time
  3. Magnetometer X/Y/Z vs time
  4. GPS trajectory (lat vs lon)
  5. GPS update pattern — step-function hold (Q5 confirmation)

Representative files chosen:
  - S-Vta1a.csv  (25,676 rows, ~43 min, largest Vta, GPS at 1 Hz — the outlier)
  - S-Vtb5.csv   (64,388 rows, ~107 min, largest Vtb, GPS at 0.11 Hz)
  - S-Vw4.csv    (126,526 rows, ~211 min, largest Vw, GPS at 0.11 Hz)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
from pathlib import Path
import os

DATA_DIR = Path("data") / "S-Dataset"
OUT_DIR = Path("plots") / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENCODING = "latin-1"

TRIPS = ["S-Vta1a.csv", "S-Vtb5.csv", "S-Vw4.csv"]

def get_col(df, keyword):
    """Find column containing keyword (case-insensitive)."""
    for c in df.columns:
        if keyword.upper() in c.upper():
            return c
    return None

def load_trip(fname):
    """Load a trip CSV with standardized time column in seconds."""
    df = pd.read_csv(DATA_DIR / fname, encoding=ENCODING)
    # Strip column name whitespace for cleaner access
    df.columns = [c.strip() for c in df.columns]
    # Compute time in seconds from TIME SINCE START (ms)
    t_col = [c for c in df.columns if 'TIME SINCE START' in c.upper()][0]
    df['t_sec'] = (df[t_col] - df[t_col].iloc[0]) / 1000.0
    return df

# =====================================================================
# Load all trips
# =====================================================================
print("Loading trips...")
trips = {}
for fname in TRIPS:
    df = load_trip(fname)
    trips[fname] = df
    dur = df['t_sec'].iloc[-1]
    print(f"  {fname}: {len(df):,d} rows, {dur:.0f}s ({dur/60:.1f} min)")

# =====================================================================
# PLOT 1: Accelerometer X/Y/Z vs time
# =====================================================================
print("\nPlotting accelerometers...")
fig, axes = plt.subplots(len(TRIPS), 1, figsize=(16, 4*len(TRIPS)), sharex=False)
if len(TRIPS) == 1:
    axes = [axes]

for ax, (fname, df) in zip(axes, trips.items()):
    t = df['t_sec'].values
    ax_col = get_col(df, 'ACCELEROMETER X')
    ay_col = get_col(df, 'ACCELEROMETER Y')
    az_col = get_col(df, 'ACCELEROMETER Z')
    
    ax.plot(t, df[ax_col].values, linewidth=0.3, alpha=0.8, label='Accel X')
    ax.plot(t, df[ay_col].values, linewidth=0.3, alpha=0.8, label='Accel Y')
    ax.plot(t, df[az_col].values, linewidth=0.3, alpha=0.8, label='Accel Z')
    ax.axhline(y=9.81, color='k', linestyle='--', linewidth=0.5, alpha=0.5, label='g=9.81')
    ax.set_ylabel('m/s²')
    ax.set_title(f'Accelerometer — {fname}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'accel_vs_time.png', dpi=150)
plt.close()
print(f"  Saved: {OUT_DIR / 'accel_vs_time.png'}")

# =====================================================================
# PLOT 2: Gyroscope X/Y/Z vs time
# =====================================================================
print("Plotting gyroscopes...")
fig, axes = plt.subplots(len(TRIPS), 1, figsize=(16, 4*len(TRIPS)), sharex=False)
if len(TRIPS) == 1:
    axes = [axes]

for ax, (fname, df) in zip(axes, trips.items()):
    t = df['t_sec'].values
    gx_col = get_col(df, 'GYROSCOPE X')
    gy_col = get_col(df, 'GYROSCOPE Y')
    gz_col = get_col(df, 'GYROSCOPE Z')
    
    ax.plot(t, df[gx_col].values, linewidth=0.3, alpha=0.8, label='Gyro X')
    ax.plot(t, df[gy_col].values, linewidth=0.3, alpha=0.8, label='Gyro Y')
    ax.plot(t, df[gz_col].values, linewidth=0.3, alpha=0.8, label='Gyro Z')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('rad/s')
    ax.set_title(f'Gyroscope — {fname}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'gyro_vs_time.png', dpi=150)
plt.close()
print(f"  Saved: {OUT_DIR / 'gyro_vs_time.png'}")

# =====================================================================
# PLOT 3: Magnetometer X/Y/Z vs time
# =====================================================================
print("Plotting magnetometers...")
fig, axes = plt.subplots(len(TRIPS), 1, figsize=(16, 4*len(TRIPS)), sharex=False)
if len(TRIPS) == 1:
    axes = [axes]

for ax, (fname, df) in zip(axes, trips.items()):
    t = df['t_sec'].values
    mx_col = get_col(df, 'MAGNETIC FIELD X')
    my_col = get_col(df, 'MAGNETIC FIELD Y')
    mz_col = get_col(df, 'MAGNETIC FIELD Z')
    
    ax.plot(t, df[mx_col].values, linewidth=0.3, alpha=0.8, label='Mag X')
    ax.plot(t, df[my_col].values, linewidth=0.3, alpha=0.8, label='Mag Y')
    ax.plot(t, df[mz_col].values, linewidth=0.3, alpha=0.8, label='Mag Z')
    ax.set_ylabel('uT')
    ax.set_title(f'Magnetometer — {fname}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'mag_vs_time.png', dpi=150)
plt.close()
print(f"  Saved: {OUT_DIR / 'mag_vs_time.png'}")

# =====================================================================
# PLOT 4: GPS trajectory (lat vs lon)
# =====================================================================
print("Plotting GPS trajectories...")
fig, axes = plt.subplots(1, len(TRIPS), figsize=(6*len(TRIPS), 6))
if len(TRIPS) == 1:
    axes = [axes]

for ax, (fname, df) in zip(axes, trips.items()):
    lat_col = get_col(df, 'LATITUDE')
    lon_col = get_col(df, 'LONGITUDE')
    lat = df[lat_col].values
    lon = df[lon_col].values
    
    # Color by time for directionality
    colors = np.linspace(0, 1, len(lat))
    sc = ax.scatter(lon, lat, c=colors, cmap='viridis', s=0.5, alpha=0.6)
    ax.plot(lon[0], lat[0], 'go', markersize=8, label='Start')
    ax.plot(lon[-1], lat[-1], 'ro', markersize=8, label='End')
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title(f'GPS Trajectory — {fname}')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Time progression')

plt.tight_layout()
plt.savefig(OUT_DIR / 'gps_trajectory.png', dpi=150)
plt.close()
print(f"  Saved: {OUT_DIR / 'gps_trajectory.png'}")

# =====================================================================
# PLOT 5: GPS update pattern — step-function hold (Q5 confirmation)
# =====================================================================
print("Plotting GPS update pattern (step-function)...")

# Show a 60-second window from each trip to make the staircase visible
fig, axes = plt.subplots(len(TRIPS), 2, figsize=(18, 4*len(TRIPS)))
if len(TRIPS) == 1:
    axes = [axes]

for row_ax, (fname, df) in zip(axes, trips.items()):
    lat_col = get_col(df, 'LATITUDE')
    lon_col = get_col(df, 'LONGITUDE')
    speed_col = get_col(df, 'GPS SPEED')
    t = df['t_sec'].values
    lat = df[lat_col].values
    lon = df[lon_col].values
    speed = df[speed_col].values
    
    # Find a 60-second window in the middle of the trip where there's movement
    mid = len(df) // 2
    # Find window where speed > 5 km/h
    moving = np.where(speed > 5)[0]
    if len(moving) > 600:
        start_idx = moving[len(moving)//2]
    else:
        start_idx = mid
    end_idx = min(start_idx + 600, len(df))  # 600 samples = 60 seconds
    
    sl = slice(start_idx, end_idx)
    t_win = t[sl] - t[start_idx]
    
    # Left: Latitude step-function
    ax_lat = row_ax[0]
    ax_lat.plot(t_win, lat[sl], 'b-', linewidth=1.0, drawstyle='default')
    ax_lat.plot(t_win, lat[sl], 'r.', markersize=2, alpha=0.5)
    
    # Mark where lat actually changes
    lat_diff = np.diff(lat[sl])
    change_idx = np.where(lat_diff != 0)[0]
    if len(change_idx) > 0:
        ax_lat.plot(t_win[change_idx], lat[sl][change_idx], 'gv', markersize=6,
                   label=f'GPS update ({len(change_idx)} in {t_win[-1]:.0f}s)')
    
    ax_lat.set_xlabel('Time (s)')
    ax_lat.set_ylabel('Latitude (deg)')
    ax_lat.set_title(f'GPS Lat step-function — {fname} (60s window)')
    ax_lat.legend(fontsize=8)
    ax_lat.grid(True, alpha=0.3)
    
    # Right: GPS Speed step-function
    ax_spd = row_ax[1]
    ax_spd.plot(t_win, speed[sl], 'b-', linewidth=1.0)
    ax_spd.plot(t_win, speed[sl], 'r.', markersize=2, alpha=0.5)
    
    # Mark where speed changes
    spd_diff = np.diff(speed[sl])
    spd_change = np.where(spd_diff != 0)[0]
    if len(spd_change) > 0:
        ax_spd.plot(t_win[spd_change], speed[sl][spd_change], 'gv', markersize=6,
                   label=f'Speed update ({len(spd_change)} in {t_win[-1]:.0f}s)')
    
    ax_spd.set_xlabel('Time (s)')
    ax_spd.set_ylabel('Speed (km/h)')
    ax_spd.set_title(f'GPS Speed step-function — {fname} (60s window)')
    ax_spd.legend(fontsize=8)
    ax_spd.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'gps_update_pattern.png', dpi=150)
plt.close()
print(f"  Saved: {OUT_DIR / 'gps_update_pattern.png'}")

# =====================================================================
# PLOT 6: Full GPS update interval histogram (all sampled files)
# =====================================================================
print("Plotting GPS update interval histogram...")
fig, axes = plt.subplots(1, len(TRIPS), figsize=(6*len(TRIPS), 4))
if len(TRIPS) == 1:
    axes = [axes]

for ax, (fname, df) in zip(axes, trips.items()):
    lat_col = get_col(df, 'LATITUDE')
    lon_col = get_col(df, 'LONGITUDE')
    lat = df[lat_col].values
    lon = df[lon_col].values
    
    changed = np.where((np.diff(lat) != 0) | (np.diff(lon) != 0))[0]
    if len(changed) < 2:
        continue
    change_indices = np.concatenate([[0], changed + 1])
    run_lengths = np.diff(np.concatenate([change_indices, [len(lat)]]))
    
    # Convert to seconds
    run_sec = run_lengths * 0.1  # 10 Hz -> 0.1s per sample
    
    ax.hist(run_sec, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.median(run_sec), color='r', linestyle='--',
              label=f'Median = {np.median(run_sec):.1f}s')
    ax.set_xlabel('GPS hold duration (seconds)')
    ax.set_ylabel('Count')
    ax.set_title(f'GPS Update Intervals — {fname}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'gps_update_histogram.png', dpi=150)
plt.close()
print(f"  Saved: {OUT_DIR / 'gps_update_histogram.png'}")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 2 COMPLETE")
print("=" * 70)
print(f"\nAll plots saved to: {OUT_DIR.resolve()}")
print(f"\nPlots generated:")
for p in sorted(OUT_DIR.glob("*.png")):
    print(f"  {p.name} ({p.stat().st_size/1024:.0f} KB)")
