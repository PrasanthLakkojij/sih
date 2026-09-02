"""
Phase 5 — Physics Dead Reckoning Baseline
SIH26168 — AI-ML Dead Reckoning System

Goal:
Measure the uncorrected, physics-only inertial dead reckoning drift over entire continuous sessions.
Compare two distinct heading integration baselines:
  - Baseline A: Gyroscope-only heading integration: psi_t = psi_{t-1} + omega_z * dt
  - Baseline B: Magnetometer / Orientation azimuth: psi_t = ori_azimuth_deg

Initial Conditions (from first valid GNSS fix):
  - Position: p_0 = [lat_0, lon_0] -> local ENU (0, 0)
  - Velocity: v_0 = gps_speed_ms[0]
  - Heading:  psi_0 = gps_bearing_deg[0] (or ori_azimuth[0] if gps bearing unavailable)

Integration Pipeline:
  1. For each time step t:
     a_fwd(t) = acc_veh_fwd(t)
     v(t) = max(0, v(t-1) + a_fwd(t) * dt)  (Non-holonomic constraint: no backward driving in normal trip)
  2. Baseline A Heading:
     psi_gyro(t) = psi_gyro(t-1) + gyro_veh_yaw_rate(t) * dt
  3. Baseline B Heading:
     psi_mag(t) = ori_azimuth_deg(t)
  4. Local Coordinates (East, North in meters):
     dx_gyro = v(t) * sin(psi_gyro(t)) * dt;  dy_gyro = v(t) * cos(psi_gyro(t)) * dt
     dx_mag  = v(t) * sin(psi_mag(t)) * dt;   dy_mag  = v(t) * cos(psi_mag(t)) * dt
  5. Error Evaluation:
     Evaluated ONLY at discrete GNSS update points (is_gnss_update == True).
     GNSS reference converted to local ENU (East, North) using equirectangular projection centered at (lat_0, lon_0).

Metrics Recorded:
  - Final Position Error (m)
  - Mean Position Error (m)
  - Max Position Error (m)
  - Total Traveled Distance (m)
  - Drift % = (Final Error / Traveled Distance) * 100
  - Velocity RMSE (m/s)
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
OUT_DIR = Path("plots") / "phase5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Geodetic to Local ENU (Meters)
# -------------------------------------------------------------------
R_EARTH = 6371000.0  # meters

def latlon_to_enu(lat, lon, lat0, lon0):
    """
    Convert lat/lon array to local flat East/North coordinates (meters)
    relative to reference (lat0, lon0).
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    east = R_EARTH * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    north = R_EARTH * (lat_rad - lat0_rad)
    return east, north

# -------------------------------------------------------------------
# Physics Dead Reckoning Engine
# -------------------------------------------------------------------
def get_col_name(df, keyword):
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None

def run_physics_dead_reckoning(df):
    """
    Executes dead reckoning on a single preprocessed & coordinate-transformed session.
    """
    N = len(df)
    dt = df['dt_sec'].values
    
    # 1. Transform coordinate frame if not already present
    if 'acc_veh_fwd' not in df.columns:
        from phase4_orientation import transform_to_vehicle_frame
        df, _, _ = transform_to_vehicle_frame(df)
        
    a_fwd = df['acc_veh_fwd'].values
    gyro_yaw_rate = df['gyro_veh_yaw_rate'].values  # rad/s
    
    az_col = get_col_name(df, 'azimuth')
    bearing_col = get_col_name(df, 'gps_orientation') or get_col_name(df, 'gps_bearing')
    
    mag_azimuth = np.radians(df[az_col].values) # rad (0=North, pi/2=East)
    
    # 2. Initial Conditions
    # First valid GNSS fix
    lat0 = df['gps_lat'].iloc[0]
    lon0 = df['gps_lon'].iloc[0]
    v0 = df['gps_speed_ms'].iloc[0]
    
    # Initial heading (from GPS bearing or azimuth)
    init_bearing = df[bearing_col].iloc[0] if bearing_col else np.nan
    if np.isnan(init_bearing) or init_bearing == 0:
        psi0 = mag_azimuth[0]
    else:
        psi0 = np.radians(init_bearing)
        
    # 3. State Arrays
    v_dr = np.zeros(N)
    v_dr[0] = v0
    
    # Baseline A: Gyroscope integration
    psi_gyro = np.zeros(N)
    psi_gyro[0] = psi0
    e_gyro = np.zeros(N)
    n_gyro = np.zeros(N)
    
    # Baseline B: Magnetometer / Orientation azimuth
    psi_mag = mag_azimuth.copy()
    e_mag = np.zeros(N)
    n_mag = np.zeros(N)
    
    # 4. Integrate Step-by-Step
    for t in range(1, N):
        # Velocity integration with non-holonomic zero-floor
        # (Vehicles don't drive backwards on road routes)
        v_cand = v_dr[t-1] + a_fwd[t] * dt[t]
        v_dr[t] = max(0.0, v_cand)
        
        # Baseline A: Integrate Gyro yaw rate
        psi_gyro[t] = psi_gyro[t-1] + gyro_yaw_rate[t] * dt[t]
        
        # Baseline A Position propagation
        # Standard navigation frame: 0 = North (+Y), 90 deg = East (+X)
        de_g = v_dr[t] * np.sin(psi_gyro[t]) * dt[t]
        dn_g = v_dr[t] * np.cos(psi_gyro[t]) * dt[t]
        e_gyro[t] = e_gyro[t-1] + de_g
        n_gyro[t] = n_gyro[t-1] + dn_g
        
        # Baseline B Position propagation
        de_m = v_dr[t] * np.sin(psi_mag[t]) * dt[t]
        dn_m = v_dr[t] * np.cos(psi_mag[t]) * dt[t]
        e_mag[t] = e_mag[t-1] + de_m
        n_mag[t] = n_mag[t-1] + dn_m
        
    # 5. GNSS Ground Truth Trajectory in Local ENU (meters)
    gps_east, gps_north = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)
    
    # Calculate GNSS traveled distance (cumulative sum of discrete GNSS fix updates)
    gnss_mask = df['is_gnss_update'].values
    gnss_idx = np.where(gnss_mask)[0]
    
    # GNSS discrete positions
    g_e_fixes = gps_east[gnss_idx]
    g_n_fixes = gps_north[gnss_idx]
    
    if len(gnss_idx) > 1:
        step_distances = np.sqrt(np.diff(g_e_fixes)**2 + np.diff(g_n_fixes)**2)
        total_dist_traveled = float(np.sum(step_distances))
    else:
        total_dist_traveled = 1.0
        
    # 6. Errors evaluated strictly at GNSS Fixes
    pos_err_gyro = np.sqrt((e_gyro[gnss_idx] - g_e_fixes)**2 + (n_gyro[gnss_idx] - g_n_fixes)**2)
    pos_err_mag  = np.sqrt((e_mag[gnss_idx] - g_e_fixes)**2  + (n_mag[gnss_idx] - g_n_fixes)**2)
    
    # Velocity error against GPS speed
    v_gps = df['gps_speed_ms'].values
    v_rmse = float(np.sqrt(np.mean((v_dr[gnss_idx] - v_gps[gnss_idx])**2)))
    
    metrics = {
        'total_dist_m': total_dist_traveled,
        'duration_s': df['time_rel_sec'].iloc[-1],
        'v_rmse_ms': v_rmse,
        # Baseline A (Gyro)
        'gyro_final_err_m': float(pos_err_gyro[-1]),
        'gyro_mean_err_m': float(np.mean(pos_err_gyro)),
        'gyro_max_err_m': float(np.max(pos_err_gyro)),
        'gyro_drift_pct': float((pos_err_gyro[-1] / max(1.0, total_dist_traveled)) * 100.0),
        # Baseline B (Mag)
        'mag_final_err_m': float(pos_err_mag[-1]),
        'mag_mean_err_m': float(np.mean(pos_err_mag)),
        'mag_max_err_m': float(np.max(pos_err_mag)),
        'mag_drift_pct': float((pos_err_mag[-1] / max(1.0, total_dist_traveled)) * 100.0),
    }
    
    dr_results = {
        'df': df,
        'gnss_idx': gnss_idx,
        'gps_east': gps_east,
        'gps_north': gps_north,
        'e_gyro': e_gyro,
        'n_gyro': n_gyro,
        'e_mag': e_mag,
        'n_mag': n_mag,
        'v_dr': v_dr,
        'pos_err_gyro': pos_err_gyro,
        'pos_err_mag': pos_err_mag,
        'metrics': metrics
    }
    return dr_results

# -------------------------------------------------------------------
# Plotting & Multi-Session Comparison
# -------------------------------------------------------------------
def plot_session_dr(res, s_name):
    gnss_idx = res['gnss_idx']
    t_gnss = res['df']['time_rel_sec'].values[gnss_idx]
    m = res['metrics']
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    
    # 1. 2D Trajectory Overlay (Local ENU Meters)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_traj.plot(res['gps_east'][gnss_idx], res['gps_north'][gnss_idx], 'k.-', label='GNSS Reference Fixes', alpha=0.8, linewidth=1.5)
    ax_traj.plot(res['e_gyro'], res['n_gyro'], 'r-', label=f'Baseline A (Gyro DR, Final Err: {m["gyro_final_err_m"]:.0f}m)', linewidth=1.0)
    ax_traj.plot(res['e_mag'], res['n_mag'], 'b--', label=f'Baseline B (Mag/Azimuth DR, Final Err: {m["mag_final_err_m"]:.0f}m)', linewidth=1.0)
    ax_traj.plot(0, 0, 'go', markersize=8, label='Start (0, 0)')
    ax_traj.set_xlabel('East (meters)')
    ax_traj.set_ylabel('North (meters)')
    ax_traj.set_title(f'Physics Dead Reckoning Trajectory — {s_name}\nDistance: {m["total_dist_m"]:.0f}m, Duration: {m["duration_s"]/60:.1f}min')
    ax_traj.legend(fontsize=8)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')
    
    # 2. Position Error vs Time
    ax_err = fig.add_subplot(gs[0, 1])
    ax_err.plot(t_gnss, res['pos_err_gyro'], 'r-', label=f'Baseline A (Gyro) — Drift: {m["gyro_drift_pct"]:.1f}%')
    ax_err.plot(t_gnss, res['pos_err_mag'], 'b--', label=f'Baseline B (Mag) — Drift: {m["mag_drift_pct"]:.1f}%')
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Position Error (meters)')
    ax_err.set_title('Inertial Position Drift Over Time (Evaluated at GNSS fixes)')
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.3)
    
    # 3. Velocity Comparison
    ax_v = fig.add_subplot(gs[1, 0])
    t_all = res['df']['time_rel_sec'].values
    v_gps = res['df']['gps_speed_ms'].values
    ax_v.plot(t_all, v_gps, 'k-', alpha=0.6, label='GPS Speed (m/s)')
    ax_v.plot(t_all, res['v_dr'], 'r-', alpha=0.7, linewidth=0.8, label=f'Inertial Integrated Speed (RMSE: {m["v_rmse_ms"]:.2f} m/s)')
    ax_v.set_xlabel('Time (s)')
    ax_v.set_ylabel('Speed (m/s)')
    ax_v.set_title('Velocity: Physics Integration vs GNSS Reference')
    ax_v.legend(fontsize=8)
    ax_v.grid(True, alpha=0.3)
    
    # 4. Heading Comparison
    ax_h = fig.add_subplot(gs[1, 1])
    az_col = get_col_name(res['df'], 'azimuth')
    bearing_col = get_col_name(res['df'], 'gps_orientation') or get_col_name(res['df'], 'gps_bearing')
    
    psi0_val = np.radians(res['df'][az_col].values[0])
    psi_gyro_deg = np.degrees(np.unwrap(psi0_val + np.cumsum(res['df']['gyro_veh_yaw_rate'].values * res['df']['dt_sec'].values)))
    psi_mag_deg = res['df'][az_col].values
    gps_heading = res['df'][bearing_col].values if bearing_col else np.zeros(len(res['df']))
    
    ax_h.plot(t_all, psi_mag_deg, 'b--', alpha=0.7, label='Mag / Orientation Azimuth (°)')
    ax_h.plot(t_all, psi_gyro_deg % 360, 'r-', alpha=0.6, label='Gyro Integrated Heading (°)')
    ax_h.plot(t_gnss, gps_heading[gnss_idx], 'k.', markersize=4, label='GPS Bearing Fixes')
    ax_h.set_xlabel('Time (s)')
    ax_h.set_ylabel('Heading (degrees)')
    ax_h.set_title('Heading Evolution')
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = OUT_DIR / f"dr_baseline_{Path(s_name).stem}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Plot saved: {out_file}")

def main():
    print("=" * 80)
    print("PHASE 5: PHYSICS-ONLY DEAD RECKONING BASELINE")
    print("=" * 80)
    
    test_sessions = [
        "S-Vta1a.parquet",
        "S-Vtb5.parquet",
        "S-Vw4.parquet",
        "S-Vta10.parquet",
        "S-Vtb2.parquet",
        "S-Vw1.parquet"
    ]
    
    summary_records = []
    
    for s_name in test_sessions:
        s_path = PROCESSED_DIR / s_name
        if not s_path.exists():
            continue
        df = pd.read_parquet(s_path)
        
        res = run_physics_dead_reckoning(df)
        m = res['metrics']
        
        summary_records.append({
            'session': s_name,
            'duration_min': round(m['duration_s'] / 60.0, 1),
            'distance_m': round(m['total_dist_m'], 1),
            'v_rmse_ms': round(m['v_rmse_ms'], 2),
            'gyro_final_err_m': round(m['gyro_final_err_m'], 1),
            'gyro_drift_pct': round(m['gyro_drift_pct'], 1),
            'mag_final_err_m': round(m['mag_final_err_m'], 1),
            'mag_drift_pct': round(m['mag_drift_pct'], 1),
        })
        
        plot_session_dr(res, s_name)
        
    summary_df = pd.DataFrame(summary_records)
    print("\n" + "=" * 80)
    print("PHYSICS DEAD RECKONING BENCHMARK RESULTS")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(OUT_DIR / "physics_dr_benchmark.csv", index=False)
    print(f"\nResults table saved to: {OUT_DIR / 'physics_dr_benchmark.csv'}")

if __name__ == "__main__":
    main()
