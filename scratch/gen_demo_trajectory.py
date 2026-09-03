"""
Generate per-timestep trajectory data for Phase 9 map demo.
Window: S-Vw4, 120s outage, window_idx=0 (s_gnss=28809, e_gnss=30069)

This window shows maximum visual contrast:
  ZUPT v1 error: 2944.1m (large false-positive drift on motorway)
  AI-Hybrid error: 1351.4m (substantial improvement)
  Gyro-only error: 1667.6m

Outputs:
  - plots/phase9/demo_trajectory.json   (per-second lat/lon for all 3 paths)
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

from phase6_outage_sim import get_zupt_v1_mask, get_col, latlon_to_enu
from phase4_orientation import transform_to_vehicle_frame
from phase8_ml_correction import segment_features

PROCESSED_DIR = Path('data/processed_sessions')
CACHE_DIR     = Path('data/phase8_cache')
OUT_DIR       = Path('plots/phase9')
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_EARTH = 6371000.0

# --- Window parameters (from phase9_raw_records.csv) ---
SESSION   = 'S-Vw4.parquet'
S_GNSS    = 28809    # outage start (genuine GNSS fix row index)
E_GNSS    = 30069    # outage end   (genuine GNSS fix row index)
MIN_T     = 2.0
DECIMATE  = 10       # keep every Nth sample (10 Hz -> 1 Hz for browser)

# -------------------------------------------------------
# 1. Load session
# -------------------------------------------------------
print(f"Loading {SESSION}...")
df = pd.read_parquet(PROCESSED_DIR / SESSION)
df, _, _ = transform_to_vehicle_frame(df)
zupt_mask = get_zupt_v1_mask(df)

lat0 = df['gps_lat'].values[S_GNSS]
lon0 = df['gps_lon'].values[S_GNSS]
east_all, north_all = latlon_to_enu(
    df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)

az_col      = get_col(df, 'azimuth')
psi_mag     = np.radians(df[az_col].values)
bearing_col = get_col(df, 'gps_orientation') or get_col(df, 'gps_bearing')
t_arr       = df['time_rel_sec'].values
a_fwd       = df['acc_veh_fwd'].values
dt_arr      = df['dt_sec'].values
gyro_yaw    = df['gyro_veh_yaw_rate'].values
lat_all     = df['gps_lat'].values
lon_all     = df['gps_lon'].values

t0 = t_arr[S_GNSS]
N  = E_GNSS - S_GNSS + 1
print(f"  Outage: [{S_GNSS}, {E_GNSS}], duration={t_arr[E_GNSS]-t0:.1f}s, N={N} samples")

# -------------------------------------------------------
# 2. Train LOTO C1 model (exclude S-Vw4)
# -------------------------------------------------------
print("Training LOTO C1 model (excl. S-Vw4)...")
meta = pd.read_parquet(CACHE_DIR / 'meta_all.parquet')
yC   = np.load(CACHE_DIR / 'yC_all.npy')
X    = np.load(CACHE_DIR / 'X_all.npy')

train_mask = (meta['dur_s'].values >= MIN_T) & (~meta['session'].str.startswith('S-Vw4'))
print(f"  Training on {train_mask.sum():,} segments from {meta[train_mask]['session'].nunique()} other trips")
model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     n_jobs=-1, random_state=42, verbosity=0)
model.fit(X[train_mask], yC[train_mask])

# -------------------------------------------------------
# 3. Init states
# -------------------------------------------------------
b0   = df[bearing_col].values[S_GNSS] if bearing_col else np.nan
psi0 = psi_mag[S_GNSS] if (np.isnan(b0) or b0 == 0) else np.radians(b0)
v0   = df['gps_speed_ms'].values[S_GNSS]

# ZUPT v1 state
vz_cur = v0;  psiz_cur = psi0;  ez = 0.0;  nz = 0.0

# Gyro-only state
vg_cur = v0;  psig_cur = psi0;  eg = 0.0;  ng = 0.0

# AI-Hybrid: uses ZUPT v1 within each interval, then applies ML correction
# We need per-interval correction so we track intervals
INTERVAL_SAMPLES = 90  # ~9s at 10Hz

# Compute interval boundaries with MIN_T guarantee
intervals = []
curr = S_GNSS
while curr < E_GNSS:
    nxt = min(curr + INTERVAL_SAMPLES, E_GNSS)
    if nxt < E_GNSS and (t_arr[E_GNSS] - t_arr[nxt]) < MIN_T:
        nxt = E_GNSS
    intervals.append((curr, nxt))
    curr = nxt

print(f"  AI-Hybrid intervals ({len(intervals)} total):")
for ia, ib in intervals:
    print(f"    [{ia}, {ib}] dur={t_arr[ib]-t_arr[ia]:.2f}s")

# For AI-Hybrid, compute cumulative ENU correction applied after each interval
# Between-interval positions are ZUPT v1 positions (we track ZUPT v1 step-by-step anyway)
# At each interval end, we apply ML correction to the ZUPT displacement for that interval.

# Strategy:
# - Run ZUPT v1 sample-by-sample for the full outage (gives per-sample positions)
# - After each interval, compute ds_imu and ds_corr_pred, build corrected displacement
# - The AI-Hybrid position after interval k = start_of_interval + corrected_vector

# -------------------------------------------------------
# 4. Per-sample integration (1 Hz decimated output)
# -------------------------------------------------------

def enu_to_latlon(e, n, lat0, lon0):
    lat = lat0 + np.degrees(n / R_EARTH)
    lon = lon0 + np.degrees(e / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon

frames = []

# AI-Hybrid tracks cumulatively; we compute per-interval then track
# Build per-interval: AI-hybrid cumulative offset per interval end
interval_ends  = {ib: None for _, ib in intervals}  # row_idx -> (e_cumulative, n_cumulative)

# Precompute per-interval cumulative ZUPT v1 vector within outage
# After applying ML correction at interval end, AI-Hybrid has a fixed offset vs pure ZUPT v1
ai_cumulative_e = 0.0
ai_cumulative_n = 0.0
v_iv = v0; psi_iv = psi0  # state for AI-Hybrid interval-level integration

for ia, ib in intervals:
    de_sub = 0.0; dn_sub = 0.0
    v_step = v_iv; psi_step = psi_iv
    for t in range(ia + 1, ib + 1):
        dt = dt_arr[t]
        if zupt_mask[t]: v_step = 0.0
        else: v_step = max(0.0, v_step + a_fwd[t] * dt)
        psi_step += gyro_yaw[t] * dt
        de_sub += v_step * np.sin(psi_step) * dt
        dn_sub += v_step * np.cos(psi_step) * dt
    ds_imu = np.sqrt(de_sub**2 + dn_sub**2)
    h_dir  = np.arctan2(de_sub, dn_sub) if ds_imu > 1e-3 else psi_step
    feat   = segment_features(df, ia, ib).reshape(1, -1)
    ds_corr_pred  = float(model.predict(feat)[0])
    ds_final      = max(0.0, ds_imu + ds_corr_pred)
    ai_cumulative_e += ds_final * np.sin(h_dir)
    ai_cumulative_n += ds_final * np.cos(h_dir)
    interval_ends[ib] = (ai_cumulative_e, ai_cumulative_n)
    v_iv = v_step; psi_iv = psi_step

# Build a lookup: for each row index in [S_GNSS, E_GNSS], what is the AI-Hybrid cumulative ENU?
# Between interval ends, AI-Hybrid position = linear interpolation along ZUPT v1 direction
# (simpler: at end of each interval, we know exact AI ENU; in between, use ZUPT v1 as approximate)
# For animation purposes: track ZUPT v1 sample-by-sample for all 3, apply AI correction at interval ends.

# We'll now do the actual per-sample pass:
vz2 = v0; psiz2 = psi0; ez2 = 0.0; nz2 = 0.0  # ZUPT v1 running state
vg2 = v0; psig2 = psi0; eg2 = 0.0; ng2 = 0.0  # Gyro-only running state

# AI-Hybrid: within each interval it follows ZUPT v1 from interval start,
# but with cumulative AI correction applied at each interval start as a base offset.
# For smooth per-frame interpolation:
# At each sample s_i within interval k: AI_pos = ai_base[k] + (ZUPT progress within interval k)
# Where ai_base[k] = AI cumulative ENU at end of interval k-1

# Build ai_base_e/n per interval
ai_base_e, ai_base_n = 0.0, 0.0
interval_base = {}  # ia -> (base_e, base_n)
interval_base[intervals[0][0]] = (0.0, 0.0)
for idx, (ia, ib) in enumerate(intervals):
    interval_base[ia] = (ai_base_e, ai_base_n)
    if ib in interval_ends and interval_ends[ib] is not None:
        ai_base_e, ai_base_n = interval_ends[ib]

# Now: within interval ia..ib, AI position at sample t =
#   base_e + (ZUPT v1 progress from ia to t)
#   base_n + (ZUPT v1 progress from ia to t)

# Reset ZUPT v1 states per interval
interval_state = {}  # ia -> (v, psi) at interval start
v_tmp = v0; psi_tmp = psi0
for ia, ib in intervals:
    interval_state[ia] = (v_tmp, psi_tmp)
    for t in range(ia + 1, ib + 1):
        dt = dt_arr[t]
        if zupt_mask[t]: v_tmp = 0.0
        else: v_tmp = max(0.0, v_tmp + a_fwd[t] * dt)
        psi_tmp += gyro_yaw[t] * dt

# Main per-sample pass (decimated)
vz3 = v0; psiz3 = psi0; ez3 = 0.0; nz3 = 0.0  # running ZUPT v1 (for AI intra-interval)
vz_run = v0; psiz_run = psi0; ez_run = 0.0; nz_run = 0.0  # running ZUPT v1 (for ZUPT path)
vg_run = v0; psig_run = psi0; eg_run = 0.0; ng_run = 0.0  # running Gyro-only

# Current interval tracking for AI
cur_interval_idx = 0
cur_ia, cur_ib = intervals[0]
intra_e, intra_n = 0.0, 0.0  # ZUPT v1 progress within current interval from ia
v_intra = v0; psi_intra = psi0

print("\nBuilding per-sample trajectory (decimated to 1 Hz)...")
for idx, t_abs in enumerate(range(S_GNSS, E_GNSS + 1)):
    elapsed = t_arr[t_abs] - t0

    # GPS ground truth: use actual lat/lon at this sample
    gt_lat = lat_all[t_abs]
    gt_lon = lon_all[t_abs]

    if t_abs > S_GNSS:
        dt = dt_arr[t_abs]

        # ZUPT v1 step
        if zupt_mask[t_abs]: vz_run = 0.0
        else: vz_run = max(0.0, vz_run + a_fwd[t_abs] * dt)
        psiz_run += gyro_yaw[t_abs] * dt
        ez_run   += vz_run * np.sin(psiz_run) * dt
        nz_run   += vz_run * np.cos(psiz_run) * dt

        # Gyro-only step
        vg_run = max(0.0, vg_run + a_fwd[t_abs] * dt)
        psig_run += gyro_yaw[t_abs] * dt
        eg_run   += vg_run * np.sin(psig_run) * dt
        ng_run   += vg_run * np.cos(psig_run) * dt

        # AI-Hybrid: intra-interval ZUPT progress
        if zupt_mask[t_abs]: v_intra = 0.0
        else: v_intra = max(0.0, v_intra + a_fwd[t_abs] * dt)
        psi_intra += gyro_yaw[t_abs] * dt
        intra_e   += v_intra * np.sin(psi_intra) * dt
        intra_n   += v_intra * np.cos(psi_intra) * dt

        # Advance interval when we hit boundary
        if t_abs == cur_ib and cur_interval_idx < len(intervals) - 1:
            cur_interval_idx += 1
            cur_ia, cur_ib = intervals[cur_interval_idx]
            intra_e, intra_n = 0.0, 0.0
            v_intra, psi_intra = vz_run, psiz_run  # carry ZUPT v1 state

    # AI-Hybrid ENU = base of current interval + intra progress
    base_e, base_n = interval_base.get(cur_ia, (0.0, 0.0))
    ai_e = base_e + intra_e
    ai_n = base_n + intra_n

    # Convert ENU offsets to lat/lon
    physics_lat, physics_lon = enu_to_latlon(ez_run, nz_run, lat0, lon0)
    gyro_lat, gyro_lon       = enu_to_latlon(eg_run, ng_run, lat0, lon0)
    ai_lat, ai_lon           = enu_to_latlon(ai_e, ai_n, lat0, lon0)

    if idx % DECIMATE == 0:
        frames.append({
            'elapsed_s':    round(float(elapsed), 1),
            'gnss_lat':     round(float(gt_lat),      7),
            'gnss_lon':     round(float(gt_lon),      7),
            'physics_lat':  round(float(physics_lat), 7),
            'physics_lon':  round(float(physics_lon), 7),
            'gyro_lat':     round(float(gyro_lat),    7),
            'gyro_lon':     round(float(gyro_lon),    7),
            'ai_lat':       round(float(ai_lat),      7),
            'ai_lon':       round(float(ai_lon),      7),
        })

print(f"  Generated {len(frames)} frames (1 Hz) from {N} samples")

# Save
out_path = OUT_DIR / 'demo_trajectory.json'
with open(out_path, 'w') as f:
    json.dump(frames, f)
print(f"  Saved: {out_path}")

# Quick sanity
last = frames[-1]
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

zupt_final_err = haversine_m(last['gnss_lat'], last['gnss_lon'], last['physics_lat'], last['physics_lon'])
ai_final_err   = haversine_m(last['gnss_lat'], last['gnss_lon'], last['ai_lat'], last['ai_lon'])
print(f"\nFinal position errors (cross-check vs phase9_raw_records.csv):")
print(f"  ZUPT v1:   {zupt_final_err:.1f}m  (CSV says 2944.1m)")
print(f"  AI-Hybrid: {ai_final_err:.1f}m  (CSV says 1351.4m)")
print(f"\nFirst frame: {frames[0]}")
print(f"Last  frame: {frames[-1]}")
