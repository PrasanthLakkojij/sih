"""
Phase 5B — ZUPT Threshold Calibration & Benchmark
SIH26168 — AI-ML Dead Reckoning System

Step 1: Calibrate a_th and omega_th strictly on static/stationary sessions (S-S1, S-S2, S-S3a, S-S3b, S-S3c, S-S4).
Step 2: Lock thresholds and implement ZUPT detector (window duration = 0.8s / 8 consecutive samples).
Step 3: Evaluate Open-loop (Phase 5A) vs ZUPT-enhanced (Phase 5B) on S-Vw4, S-Vta10, S-Vtb2, S-Vta1a, S-Vtb5.
Step 4: Generate comparison table and plots.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data") / "processed_sessions"
OUT_DIR = Path("plots") / "phase5b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# STEP 1: CALIBRATE THRESHOLDS FROM STATIC SESSIONS ONLY
# -------------------------------------------------------------------
def calibrate_zupt_thresholds():
    static_files = [
        "S-S1.parquet",
        "S-S2_sub01.parquet",
        "S-S2_sub02.parquet",
        "S-S3a.parquet",
        "S-S3b_sub01.parquet",
        "S-S3b_sub02.parquet",
        "S-S3c.parquet",
        "S-S4_sub01.parquet",
        "S-S4_sub02.parquet",
        "S-S4_sub03.parquet",
    ]
    
    a_lin_mags = []
    omega_mags = []
    
    print("=" * 80)
    print("CALIBRATING ZUPT THRESHOLDS ON GROUND-TRUTH STATIC SESSIONS")
    print("=" * 80)
    
    total_static_samples = 0
    
    for sf in static_files:
        p = PROCESSED_DIR / sf
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        
        # Linear acceleration magnitude
        ax = df['acc_lin_x'].values
        ay = df['acc_lin_y'].values
        az = df['acc_lin_z'].values
        a_mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        # Gyroscope magnitude
        gx = df['gyro_x'].values
        gy = df['gyro_y'].values
        gz = df['gyro_z'].values
        w_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        
        a_lin_mags.append(a_mag)
        omega_mags.append(w_mag)
        total_static_samples += len(df)
        print(f"  Loaded {sf:<20s}: {len(df):>7,d} samples | a_mag p95={np.percentile(a_mag, 95):.4f} m/s^2, w_mag p95={np.percentile(w_mag, 95):.4f} rad/s")
        
    all_a_mag = np.concatenate(a_lin_mags)
    all_w_mag = np.concatenate(omega_mags)
    
    a_p95 = np.percentile(all_a_mag, 95)
    a_p99 = np.percentile(all_a_mag, 99)
    a_max = np.max(all_a_mag)
    
    w_p95 = np.percentile(all_w_mag, 95)
    w_p99 = np.percentile(all_w_mag, 99)
    w_max = np.max(all_w_mag)
    
    print("\n--- STATIC NOISE FLOOR DISTRIBUTION ({:,d} samples) ---".format(total_static_samples))
    print(f"  |a_lin|: Mean={np.mean(all_a_mag):.4f}, Std={np.std(all_a_mag):.4f}, 95th%={a_p95:.4f}, 99th%={a_p99:.4f}, Max={a_max:.4f} m/s^2")
    print(f"  |omega|: Mean={np.mean(all_w_mag):.4f}, Std={np.std(all_w_mag):.4f}, 95th%={w_p95:.4f}, 99th%={w_p99:.4f}, Max={w_max:.4f} rad/s")
    
    # Selection rule:
    # Set threshold at 99th percentile + small safety margin to tolerate slight vehicle engine idle vibrations.
    # In stationary vehicle with engine running, noise is slightly higher than desk static, so 1.25 * p99 is scientifically justified.
    a_th = float(np.round(a_p99 * 1.25, 3))
    w_th = float(np.round(w_p99 * 1.25, 3))
    
    # Enforce standard practical minimums:
    a_th = max(0.25, a_th)   # min 0.25 m/s^2
    w_th = max(0.05, w_th)   # min 0.05 rad/s (~2.8 deg/s)
    
    print("\n--- LOCKED CALIBRATED THRESHOLDS ---")
    print(f"  a_th (Linear Accel Threshold): {a_th:.3f} m/s^2")
    print(f"  w_th (Gyroscope Threshold):    {w_th:.3f} rad/s (~{np.degrees(w_th):.2f} deg/s)")
    print(f"  Window Duration:               0.80 seconds (8 consecutive samples at 10 Hz)")
    print("=" * 80 + "\n")
    
    return a_th, w_th

# -------------------------------------------------------------------
# STEP 2: ZUPT-ENHANCED DEAD RECKONING ENGINE
# -------------------------------------------------------------------
R_EARTH = 6371000.0

def latlon_to_enu(lat, lon, lat0, lon0):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    east = R_EARTH * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    north = R_EARTH * (lat_rad - lat0_rad)
    return east, north

def get_col_name(df, keyword):
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None

def run_dead_reckoning_comparison(df, a_th, w_th, window_len=8):
    """
    Runs both Open-loop (Phase 5A) and ZUPT-enhanced (Phase 5B) dead reckoning.
    """
    N = len(df)
    dt = df['dt_sec'].values
    
    # Coordinate transformation
    if 'acc_veh_fwd' not in df.columns:
        from phase4_orientation import transform_to_vehicle_frame
        df, _, _ = transform_to_vehicle_frame(df)
        
    a_fwd = df['acc_veh_fwd'].values
    
    # Raw 3D linear acceleration & gyro magnitude for ZUPT detection
    ax_lin = df['acc_lin_x'].values
    ay_lin = df['acc_lin_y'].values
    az_lin = df['acc_lin_z'].values
    a_mag_3d = np.sqrt(ax_lin**2 + ay_lin**2 + az_lin**2)
    
    gx = df['gyro_x'].values
    gy = df['gyro_y'].values
    gz = df['gyro_z'].values
    w_mag_3d = np.sqrt(gx**2 + gy**2 + gz**2)
    
    # Detect Instantaneous Stationary Condition
    raw_stationary = (a_mag_3d < a_th) & (w_mag_3d < w_th)
    
    # Enforce Temporal Window (rolling window of window_len consecutive samples)
    # Series of 1s if all samples in the past window_len are stationary
    is_stationary = np.zeros(N, dtype=bool)
    consec_count = 0
    for i in range(N):
        if raw_stationary[i]:
            consec_count += 1
            if consec_count >= window_len:
                is_stationary[i] = True
        else:
            consec_count = 0
            
    # Initial conditions
    lat0 = df['gps_lat'].iloc[0]
    lon0 = df['gps_lon'].iloc[0]
    v0 = df['gps_speed_ms'].iloc[0]
    
    az_col = get_col_name(df, 'azimuth')
    mag_azimuth = np.radians(df[az_col].values)
    
    bearing_col = get_col_name(df, 'gps_orientation') or get_col_name(df, 'gps_bearing')
    init_bearing = df[bearing_col].iloc[0] if bearing_col else np.nan
    if np.isnan(init_bearing) or init_bearing == 0:
        psi0 = mag_azimuth[0]
    else:
        psi0 = np.radians(init_bearing)
        
    gyro_yaw_rate = df['gyro_veh_yaw_rate'].values
    
    # -------------------------------------------------------------
    # 1. RUN OPEN-LOOP (Phase 5A)
    # -------------------------------------------------------------
    v_open = np.zeros(N)
    v_open[0] = v0
    e_open = np.zeros(N)
    n_open = np.zeros(N)
    psi_gyro_open = np.zeros(N)
    psi_gyro_open[0] = psi0
    
    for t in range(1, N):
        v_open[t] = max(0.0, v_open[t-1] + a_fwd[t] * dt[t])
        psi_gyro_open[t] = psi_gyro_open[t-1] + gyro_yaw_rate[t] * dt[t]
        e_open[t] = e_open[t-1] + v_open[t] * np.sin(psi_gyro_open[t]) * dt[t]
        n_open[t] = n_open[t-1] + v_open[t] * np.cos(psi_gyro_open[t]) * dt[t]
        
    # -------------------------------------------------------------
    # 2. RUN ZUPT-ENHANCED (Phase 5B)
    # -------------------------------------------------------------
    v_zupt = np.zeros(N)
    v_zupt[0] = v0
    e_zupt = np.zeros(N)
    n_zupt = np.zeros(N)
    psi_gyro_zupt = np.zeros(N)
    psi_gyro_zupt[0] = psi0
    
    zupt_events_count = 0
    in_zupt_block = False
    
    for t in range(1, N):
        if is_stationary[t]:
            # ZUPT Triggered: Force velocity to 0 (reset integration state)
            v_zupt[t] = 0.0
            if not in_zupt_block:
                zupt_events_count += 1
                in_zupt_block = True
        else:
            in_zupt_block = False
            v_cand = v_zupt[t-1] + a_fwd[t] * dt[t]
            v_zupt[t] = max(0.0, v_cand)
            
        psi_gyro_zupt[t] = psi_gyro_zupt[t-1] + gyro_yaw_rate[t] * dt[t]
        e_zupt[t] = e_zupt[t-1] + v_zupt[t] * np.sin(psi_gyro_zupt[t]) * dt[t]
        n_zupt[t] = n_zupt[t-1] + v_zupt[t] * np.cos(psi_gyro_zupt[t]) * dt[t]
        
    # -------------------------------------------------------------
    # 3. EVALUATION AGAINST GNSS GROUND TRUTH
    # -------------------------------------------------------------
    gps_east, gps_north = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)
    gnss_mask = df['is_gnss_update'].values
    gnss_idx = np.where(gnss_mask)[0]
    
    g_e_fixes = gps_east[gnss_idx]
    g_n_fixes = gps_north[gnss_idx]
    
    if len(gnss_idx) > 1:
        step_distances = np.sqrt(np.diff(g_e_fixes)**2 + np.diff(g_n_fixes)**2)
        total_dist_traveled = float(np.sum(step_distances))
    else:
        total_dist_traveled = 1.0
        
    err_open = np.sqrt((e_open[gnss_idx] - g_e_fixes)**2 + (n_open[gnss_idx] - g_n_fixes)**2)
    err_zupt = np.sqrt((e_zupt[gnss_idx] - g_e_fixes)**2 + (n_zupt[gnss_idx] - g_n_fixes)**2)
    
    dur_min = df['time_rel_sec'].iloc[-1] / 60.0
    total_stat_time_s = np.sum(is_stationary * dt)
    
    final_err_open = err_open[-1]
    final_err_zupt = err_zupt[-1]
    improvement_pct = ((final_err_open - final_err_zupt) / max(1.0, final_err_open)) * 100.0
    
    metrics = {
        'duration_min': dur_min,
        'distance_m': total_dist_traveled,
        'open_term_v_ms': v_open[-1],
        'zupt_term_v_ms': v_zupt[-1],
        'open_max_v_ms': np.max(v_open),
        'zupt_max_v_ms': np.max(v_zupt),
        'open_final_err_m': final_err_open,
        'zupt_final_err_m': final_err_zupt,
        'open_drift_pct': (final_err_open / max(1.0, total_dist_traveled)) * 100.0,
        'zupt_drift_pct': (final_err_zupt / max(1.0, total_dist_traveled)) * 100.0,
        'zupt_events': zupt_events_count,
        'stat_time_min': total_stat_time_s / 60.0,
        'stat_pct': (total_stat_time_s / df['time_rel_sec'].iloc[-1]) * 100.0,
        'improvement_pct': improvement_pct
    }
    
    return {
        'df': df,
        'gnss_idx': gnss_idx,
        'gps_east': gps_east,
        'gps_north': gps_north,
        'e_open': e_open,
        'n_open': n_open,
        'e_zupt': e_zupt,
        'n_zupt': n_zupt,
        'v_open': v_open,
        'v_zupt': v_zupt,
        'err_open': err_open,
        'err_zupt': err_zupt,
        'is_stationary': is_stationary,
        'metrics': metrics
    }

# -------------------------------------------------------------------
# STEP 3: PLOTTING & BENCHMARK SUITE
# -------------------------------------------------------------------
def plot_zupt_comparison(res, s_name):
    m = res['metrics']
    t_all = res['df']['time_rel_sec'].values
    v_gps = res['df']['gps_speed_ms'].values
    gnss_idx = res['gnss_idx']
    t_gnss = t_all[gnss_idx]
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=False)
    
    # 1. Velocity Comparison (Open-loop vs ZUPT vs GPS)
    axes[0].plot(t_all, v_gps, 'k-', linewidth=1.2, label='GPS Speed Ground Truth')
    axes[0].plot(t_all, res['v_open'], 'r--', linewidth=0.8, alpha=0.7, label=f'Open-Loop Velocity (Max: {m["open_max_v_ms"]:.1f} m/s)')
    axes[0].plot(t_all, res['v_zupt'], 'b-', linewidth=1.0, label=f'ZUPT Velocity (Max: {m["zupt_max_v_ms"]:.1f} m/s)')
    # Highlight stationary intervals
    axes[0].fill_between(t_all, 0, max(10, m['zupt_max_v_ms']), where=res['is_stationary'], color='green', alpha=0.15, label=f'ZUPT Active ({m["zupt_events"]} events, {m["stat_time_min"]:.1f} min)')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].set_title(f'Velocity Evolution & ZUPT Resets — {s_name} (Duration: {m["duration_min"]:.1f} min)')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Position Error over Time
    axes[1].plot(t_gnss, res['err_open'], 'r--', label=f'Open-Loop Error (Final: {m["open_final_err_m"]:.0f}m, Drift: {m["open_drift_pct"]:.1f}%)')
    axes[1].plot(t_gnss, res['err_zupt'], 'b-', label=f'ZUPT Error (Final: {m["zupt_final_err_m"]:.0f}m, Drift: {m["zupt_drift_pct"]:.1f}%) — Improvement: {m["improvement_pct"]:.1f}%')
    axes[1].set_ylabel('Position Error (m)')
    axes[1].set_title('Position Error (Bias-driven drift evaluated at GNSS fixes)')
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # 3. 2D Trajectory Map
    ax_map = axes[2]
    ax_map.plot(res['gps_east'][gnss_idx], res['gps_north'][gnss_idx], 'k.-', linewidth=1.5, label='GNSS Fixes Reference')
    ax_map.plot(res['e_open'], res['n_open'], 'r--', linewidth=0.8, alpha=0.7, label='Open-Loop Trajectory')
    ax_map.plot(res['e_zupt'], res['n_zupt'], 'b-', linewidth=1.2, label='ZUPT-Enhanced Trajectory')
    ax_map.plot(0, 0, 'go', markersize=8, label='Start')
    ax_map.set_xlabel('East (meters)')
    ax_map.set_ylabel('North (meters)')
    ax_map.set_title('2D Trajectory Comparison (Local ENU)')
    ax_map.legend(loc='upper right', fontsize=8)
    ax_map.grid(True, alpha=0.3)
    ax_map.axis('equal')
    
    plt.tight_layout()
    out_file = OUT_DIR / f"zupt_vs_open_{Path(s_name).stem}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Saved plot: {out_file}")

def main():
    # 1. Calibrate
    a_th, w_th = calibrate_zupt_thresholds()
    
    # 2. Test Suite
    test_files = [
        "S-Vw4.parquet",    # 210.9 min long trip
        "S-Vta1a.parquet",  # 42.8 min trip
        "S-Vtb5.parquet",  # 107.3 min trip
        "S-Vta10.parquet",  # 2.5 min short trip
        "S-Vtb2.parquet"    # 9.5 min medium trip
    ]
    
    results = []
    
    for s_name in test_files:
        p = PROCESSED_DIR / s_name
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        res = run_dead_reckoning_comparison(df, a_th, w_th)
        m = res['metrics']
        
        results.append({
            'Session': s_name,
            'Duration (min)': round(m['duration_min'], 1),
            'Distance (km)': round(m['distance_m'] / 1000.0, 2),
            'Open Term V (m/s)': round(m['open_term_v_ms'], 1),
            'ZUPT Term V (m/s)': round(m['zupt_term_v_ms'], 1),
            'Open Max V (m/s)': round(m['open_max_v_ms'], 1),
            'ZUPT Max V (m/s)': round(m['zupt_max_v_ms'], 1),
            'Open Final Err (m)': round(m['open_final_err_m'], 1),
            'ZUPT Final Err (m)': round(m['zupt_final_err_m'], 1),
            'ZUPT Events': m['zupt_events'],
            'Stat Time (min)': round(m['stat_time_min'], 1),
            'Improvement (%)': round(m['improvement_pct'], 1)
        })
        
        plot_zupt_comparison(res, s_name)
        
    res_df = pd.DataFrame(results)
    
    print("\n" + "=" * 100)
    print("PHASE 5B: OPEN-LOOP (5A) VS ZUPT-ENHANCED (5B) BENCHMARK RESULTS")
    print("=" * 100)
    print(res_df.to_string(index=False))
    
    res_df.to_csv(OUT_DIR / "zupt_benchmark_results.csv", index=False)
    print(f"\nSaved benchmark table to: {OUT_DIR / 'zupt_benchmark_results.csv'}")

if __name__ == "__main__":
    main()
