"""
Phase 1 — Focused Dataset Inspection
SIH26168 — AI-ML Dead Reckoning

Answers exactly 9 questions across all 72 S-Dataset files.
Does NOT rediscover columns (we already know the 24-column schema from S-M.csv).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data") / "S-Dataset"
EXPECTED_COLS = 24
ENCODING = "latin-1"

csv_files = sorted(DATA_DIR.glob("*.csv"))
print(f"Files found: {len(csv_files)}\n")

# ── Storage for results ─────────────────────────────────────────────────
file_info = []          # (name, rows, cols, readable, schema_match, null_count, dup_count)
ref_columns = None      # reference column list from first file
schema_mismatches = []
unreadable = []

# ── Read reference schema from S-M.csv ──────────────────────────────────
ref_file = DATA_DIR / "S-M.csv"
ref_df = pd.read_csv(ref_file, nrows=0, encoding=ENCODING)
ref_columns = [c.strip() for c in ref_df.columns.tolist()]
print(f"Reference schema (S-M.csv): {len(ref_columns)} columns")

# =====================================================================
# Q1-Q3, Q6-Q7: Readability, schema, rows, nulls, duplicates
# =====================================================================
print("\n" + "=" * 80)
print("Q1-Q3, Q6-Q7: File readability, schema match, row counts, nulls, duplicates")
print("=" * 80)

all_dfs = {}  # keep loaded DataFrames for later questions

for f in csv_files:
    name = f.name
    try:
        df = pd.read_csv(f, encoding=ENCODING)
        all_dfs[name] = df
        cols_stripped = [c.strip() for c in df.columns.tolist()]
        schema_ok = (cols_stripped == ref_columns)
        if not schema_ok:
            # Find what differs
            diff = set(cols_stripped).symmetric_difference(set(ref_columns))
            schema_mismatches.append((name, cols_stripped, diff))
        null_total = int(df.isnull().sum().sum())
        dup_total = int(df.duplicated().sum())
        file_info.append((name, len(df), len(df.columns), True, schema_ok, null_total, dup_total))
    except Exception as e:
        unreadable.append((name, str(e)))
        file_info.append((name, 0, 0, False, False, -1, -1))

# Print summary table
print(f"\n  {'File':<20s} {'Rows':>8s} {'Cols':>5s} {'OK?':>4s} {'Schema':>7s} {'Nulls':>7s} {'Dups':>5s}")
print(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*4} {'-'*7} {'-'*7} {'-'*5}")
total_rows = 0
for name, rows, cols, readable, schema_ok, nulls, dups in file_info:
    total_rows += rows
    print(f"  {name:<20s} {rows:>8,d} {cols:>5d} {'YES' if readable else 'NO':>4s} "
          f"{'MATCH' if schema_ok else 'DIFF':>7s} {nulls:>7d} {dups:>5d}")
print(f"  {'-'*20} {'-'*8}")
print(f"  {'TOTAL':<20s} {total_rows:>8,d}")

# Q1 summary
print(f"\n  Q1 — Readable files: {sum(1 for _,_,_,r,_,_,_ in file_info if r)}/{len(csv_files)}")
if unreadable:
    for name, err in unreadable:
        print(f"       FAILED: {name} — {err}")

# Q2 summary
print(f"\n  Q2 — Schema matches: {sum(1 for _,_,_,_,s,_,_ in file_info if s)}/{len(csv_files)}")
if schema_mismatches:
    for name, cols, diff in schema_mismatches:
        print(f"       MISMATCH: {name}")
        # Show the differing column(s)
        for i, (a, b) in enumerate(zip(cols, ref_columns)):
            if a != b:
                print(f"         col[{i}]: got '{a}' vs expected '{b}'")

# Q6 summary
null_files = [(n, nl) for n, _, _, _, _, nl, _ in file_info if nl > 0]
print(f"\n  Q6 — Files with missing values: {len(null_files)}/{len(csv_files)}")
if null_files:
    for name, nl in null_files:
        df = all_dfs[name]
        null_cols = df.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        print(f"       {name}: {nl} total nulls")
        for col, cnt in null_cols.items():
            print(f"         {col.strip()}: {cnt} ({cnt/len(df)*100:.1f}%)")

# Q7 summary
dup_files = [(n, d) for n, _, _, _, _, _, d in file_info if d > 0]
print(f"\n  Q7 — Files with duplicate rows: {len(dup_files)}/{len(csv_files)}")
if dup_files:
    for name, d in dup_files:
        print(f"       {name}: {d} duplicates")

# =====================================================================
# Q4: Time interval between consecutive rows
# =====================================================================
print("\n" + "=" * 80)
print("Q4: Time intervals (using TIME SINCE START in ms)")
print("=" * 80)

# Find the time column (has leading space in some files)
def get_time_col(df):
    for c in df.columns:
        if 'TIME SINCE START' in c.upper():
            return c
    return None

dt_stats = []
for name, df in all_dfs.items():
    tc = get_time_col(df)
    if tc is None:
        continue
    t = df[tc].values.astype(float)
    if len(t) < 2:
        continue
    dt_ms = np.diff(t)
    dt_stats.append({
        'file': name,
        'n': len(dt_ms),
        'median': np.median(dt_ms),
        'mean': np.mean(dt_ms),
        'min': np.min(dt_ms),
        'max': np.max(dt_ms),
        'std': np.std(dt_ms),
        'pct_near_100': np.mean(np.abs(dt_ms - 100) < 20) * 100,  # within ±20ms of 100ms
    })

print(f"\n  {'File':<20s} {'Median':>8s} {'Mean':>8s} {'Min':>8s} {'Max':>8s} {'Std':>8s} {'%~100ms':>8s}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for s in dt_stats:
    print(f"  {s['file']:<20s} {s['median']:>8.1f} {s['mean']:>8.1f} {s['min']:>8.1f} "
          f"{s['max']:>8.1f} {s['std']:>8.1f} {s['pct_near_100']:>7.1f}%")

medians = [s['median'] for s in dt_stats]
print(f"\n  Overall: median of medians = {np.median(medians):.1f} ms -> {1000/np.median(medians):.1f} Hz")
irregular = [s for s in dt_stats if s['std'] > 20 or s['median'] < 80 or s['median'] > 120]
if irregular:
    print(f"  ⚠ {len(irregular)} file(s) with irregular timing (std>20ms or median outside 80-120ms)")
else:
    print(f"  ✓ All files have consistent ~100ms intervals (10 Hz)")

# =====================================================================
# Q5: GPS update rate — consecutive identical lat/lon
# =====================================================================
print("\n" + "=" * 80)
print("Q5: GPS update rate (consecutive identical lat/lon)")
print("=" * 80)

def get_col(df, keyword):
    for c in df.columns:
        if keyword.upper() in c.upper():
            return c
    return None

gps_update_stats = []
# Check a representative sample: first S-M, a few Vta, Vtb, Vw
sample_files = ['S-M.csv', 'S-Vta1a.csv', 'S-Vtb1.csv', 'S-Vtb5.csv',
                'S-Vw4.csv', 'S-Vw2.csv', 'S-Y1.csv', 'S-Vta29.csv']
sample_files = [f for f in sample_files if f in all_dfs]

for name in sample_files:
    df = all_dfs[name]
    lat_col = get_col(df, 'LATITUDE')
    lon_col = get_col(df, 'LONGITUDE')
    if lat_col is None or lon_col is None:
        continue
    
    lat = df[lat_col].values
    lon = df[lon_col].values
    
    # Find runs of identical lat/lon
    changed = np.where((np.diff(lat) != 0) | (np.diff(lon) != 0))[0]
    if len(changed) == 0:
        gps_update_stats.append((name, len(df), 0, len(df), len(df), len(df)))
        continue
    
    # Run lengths between changes
    change_indices = np.concatenate([[0], changed + 1])
    run_lengths = np.diff(np.concatenate([change_indices, [len(lat)]]))
    
    n_updates = len(run_lengths)
    median_run = np.median(run_lengths)
    mean_run = np.mean(run_lengths)
    max_run = np.max(run_lengths)
    min_run = np.min(run_lengths)
    
    gps_update_stats.append((name, len(df), n_updates, median_run, mean_run, max_run))

print(f"\n  {'File':<20s} {'Rows':>8s} {'GPS upd':>8s} {'Med run':>8s} {'Mean run':>9s} {'Max run':>8s} {'Est GPS Hz':>10s}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*10}")
for name, rows, n_upd, med, mean, mx in gps_update_stats:
    # IMU is 10Hz, so run_length * 0.1s = GPS update interval
    gps_hz = 10.0 / med if med > 0 else 0
    print(f"  {name:<20s} {rows:>8,d} {n_upd:>8,d} {med:>8.1f} {mean:>9.1f} {mx:>8d} {gps_hz:>10.2f}")

print(f"\n  Interpretation: If 'Med run' = 1, GPS updates every IMU sample (10 Hz).")
print(f"  If 'Med run' = 10, GPS updates once per second (1 Hz).")
print(f"  The 'Max run' shows the longest GPS hold — could indicate brief signal loss.")

# =====================================================================
# Q8: Unit sanity check
# =====================================================================
print("\n" + "=" * 80)
print("Q8: Unit sanity check (value ranges)")
print("=" * 80)

# Use S-M.csv for a comprehensive check (largest single file)
df = all_dfs['S-M.csv']

checks = [
    ('Accel X (m/s²)', 'ACCELEROMETER X', -50, 50, 'Should be ±20 m/s² for driving'),
    ('Accel Y (m/s²)', 'ACCELEROMETER Y', -50, 50, 'Should be ±20 m/s² for driving'),
    ('Accel Z (m/s²)', 'ACCELEROMETER Z', -5, 25,  'Should be ~9.8 if phone flat'),
    ('Gravity X',      'GRAVITY X',       -15, 15,  'Magnitude should be ~9.8'),
    ('Gravity Y',      'GRAVITY Y',       -15, 15,  'Magnitude should be ~9.8'),
    ('Gravity Z',      'GRAVITY Z',       -15, 15,  'Magnitude should be ~9.8'),
    ('Gyro X (rad/s)', 'GYROSCOPE X',     -10, 10,  'If >50, units might be deg/s not rad/s'),
    ('Gyro Y (rad/s)', 'GYROSCOPE Y',     -10, 10,  'If >50, units might be deg/s not rad/s'),
    ('Gyro Z (rad/s)', 'GYROSCOPE Z',     -10, 10,  'If >50, units might be deg/s not rad/s'),
    ('Mag X (µT)',     'MAGNETIC FIELD X', -200, 200, 'Earth field is ~25-65 µT'),
    ('Mag Y (µT)',     'MAGNETIC FIELD Y', -200, 200, 'Earth field is ~25-65 µT'),
    ('Mag Z (µT)',     'MAGNETIC FIELD Z', -200, 200, 'Earth field is ~25-65 µT'),
    ('GPS Speed (Kmh)','GPS SPEED',       -1, 200,   'Vehicle speed in km/h'),
    ('GPS Lat (deg)',  'LATITUDE',         -90, 90,   'WGS84 latitude'),
    ('GPS Lon (deg)',  'LONGITUDE',        -180, 180, 'WGS84 longitude'),
    ('Azimuth (°)',    'AZIMUTH',          -1, 361,   'Compass heading 0-360'),
    ('Pitch (°)',      'PITCH',            -181, 181, 'Phone tilt'),
    ('Roll (°)',       'ROLL',             -181, 181, 'Phone tilt'),
]

print(f"\n  {'Check':<22s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'OK?':>5s}  Notes")
print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*5}  {'-'*35}")

for label, keyword, lo, hi, note in checks:
    col = get_col(df, keyword)
    if col is None:
        print(f"  {label:<22s} {'(column not found)':>30s}")
        continue
    vals = df[col].dropna()
    if vals.dtype == 'object':
        continue
    vmin, vmax, vmean = vals.min(), vals.max(), vals.mean()
    ok = lo <= vmin and vmax <= hi
    flag = " ✓" if ok else " ⚠"
    print(f"  {label:<22s} {vmin:>10.3f} {vmax:>10.3f} {vmean:>10.3f} {flag:>5s}  {note}")

# =====================================================================
# Q9: Gravity Z — flat mount or tilted?
# =====================================================================
print("\n" + "=" * 80)
print("Q9: Gravity Z analysis — phone mounting orientation")
print("=" * 80)

# Check S-M.csv and a few others
check_files = ['S-M.csv', 'S-S1.csv', 'S-Vta1a.csv', 'S-Vtb5.csv', 'S-Vw4.csv']
check_files = [f for f in check_files if f in all_dfs]

print(f"\n  {'File':<20s} {'Grav X':>8s} {'Grav Y':>8s} {'Grav Z':>8s} {'|G|':>8s} {'Pitch':>8s} {'Roll':>8s}  Mounting")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")

for name in check_files:
    df = all_dfs[name]
    gx_col = get_col(df, 'GRAVITY X')
    gy_col = get_col(df, 'GRAVITY Y')
    gz_col = get_col(df, 'GRAVITY Z')
    pitch_col = get_col(df, 'PITCH')
    roll_col = get_col(df, 'ROLL')
    
    gx = df[gx_col].mean()
    gy = df[gy_col].mean()
    gz = df[gz_col].mean()
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    pitch = df[pitch_col].mean()
    roll = df[roll_col].mean()
    
    # Determine mounting
    if abs(gz) > 9.0:
        mount = "~FLAT (screen up)" if gz > 0 else "~FLAT (screen down)"
    elif abs(gy) > 9.0:
        mount = "PORTRAIT upright"
    elif abs(gx) > 9.0:
        mount = "LANDSCAPE"
    else:
        mount = "TILTED"
    
    print(f"  {name:<20s} {gx:>8.3f} {gy:>8.3f} {gz:>8.3f} {g_mag:>8.3f} {pitch:>8.1f} {roll:>8.1f}  {mount}")

    # Also check variability of gravity Z
    gz_vals = df[gz_col]
    print(f"  {'':20s} Grav Z: std={gz_vals.std():.4f}, min={gz_vals.min():.3f}, max={gz_vals.max():.3f}")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE — All 9 questions answered")
print("=" * 80)
