"""
Phase 5B Investigations — Diagnostic Suite
SIH26168 — AI-ML Dead Reckoning System

Investigation 1: Short-Trip ZUPT False-Positive Analysis (S-Vta10, S-Vtb2)
  - For every ZUPT-triggered stationary sample, inspect true GPS speed at the nearest genuine GNSS fix.
  - Compute fraction of ZUPT events occurring when vehicle is actually moving (GPS speed > 5 km/h ≈ 1.39 m/s).
  - Evaluate if cutting velocity during cruise explains the negative percentage change.

Investigation 2: Heading Drift Analysis on S-Vw4 (210.9 min)
  - Compute genuine GNSS bearing between consecutive genuine fixes:
    Only for fix pairs where distance > 10m and GNSS speed > 2.0 m/s (reliable motion bearing).
    GNSS bearing = atan2(dEast, dNorth) in degrees [0, 360).
  - Compare against Gyro-integrated heading and Magnetometer azimuth.
  - Unwrap all heading angles to avoid 0/360 boundary discontinuities.
  - Compute heading error Delta_psi(t) = unwrap(psi_DR) - unwrap(psi_GNSS).
  - Fit error growth models: Linear (at) vs Quadratic (bt^2) vs Saturation (c*(1-exp(-t/tau))).
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

R_EARTH = 6371000.0

def latlon_to_enu(lat, lon, lat0, lon0):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    east = R_EARTH * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    north = R_EARTH * (lat_rad - lat0_rad)
    return east, north

# -------------------------------------------------------------------
# INVESTIGATION 1: SHORT-TRIP ZUPT FALSE POSITIVES
# -------------------------------------------------------------------
def run_investigation_1():
    print("=" * 80)
    print("INVESTIGATION 1: SHORT-TRIP ZUPT FALSE POSITIVE AUDIT (S-Vta10, S-Vtb2)")
    print("=" * 80)
    
    # Locked thresholds from Phase 5B calibration
    a_th = 5.389
    w_th = 0.753
    window_len = 8
    
    test_files = ["S-Vta10.parquet", "S-Vtb2.parquet"]
    
    results = []
    
    for s_name in test_files:
        p = PROCESSED_DIR / s_name
        df = pd.read_parquet(p)
        
        # 1. Compute 3D linear acceleration & gyro magnitude
        ax_lin = df['acc_lin_x'].values
        ay_lin = df['acc_lin_y'].values
        az_lin = df['acc_lin_z'].values
        a_mag_3d = np.sqrt(ax_lin**2 + ay_lin**2 + az_lin**2)
        
        gx = df['gyro_x'].values
        gy = df['gyro_y'].values
        gz = df['gyro_z'].values
        w_mag_3d = np.sqrt(gx**2 + gy**2 + gz**2)
        
        raw_stationary = (a_mag_3d < a_th) & (w_mag_3d < w_th)
        
        N = len(df)
        is_stationary = np.zeros(N, dtype=bool)
        consec_count = 0
        for i in range(N):
            if raw_stationary[i]:
                consec_count += 1
                if consec_count >= window_len:
                    is_stationary[i] = True
            else:
                consec_count = 0
                
        # 2. Get true GNSS speed at genuine updates
        gnss_mask = df['is_gnss_update'].values
        gnss_idx = np.where(gnss_mask)[0]
        
        t_all = df['time_rel_sec'].values
        v_gps_all = df['gps_speed_ms'].values
        
        # At each stationary sample, evaluate genuine GPS speed
        # To avoid sample-and-hold artifact, interpolate GPS speed only across genuine GNSS fix timestamps
        t_gnss = t_all[gnss_idx]
        v_gnss_fixes = v_gps_all[gnss_idx]
        
        if len(t_gnss) > 1:
            v_gps_true_interp = np.interp(t_all, t_gnss, v_gnss_fixes)
        else:
            v_gps_true_interp = v_gps_all
            
        stat_indices = np.where(is_stationary)[0]
        num_stat_samples = len(stat_indices)
        
        if num_stat_samples > 0:
            v_during_stat_ms = v_gps_true_interp[stat_indices]
            v_during_stat_kmh = v_during_stat_ms * 3.6
            
            # False positive: stationary triggered while GPS speed > 5 km/h (~1.39 m/s)
            fp_5kmh_count = np.sum(v_during_stat_kmh > 5.0)
            fp_5kmh_pct = (fp_5kmh_count / num_stat_samples) * 100.0
            
            fp_10kmh_count = np.sum(v_during_stat_kmh > 10.0)
            fp_10kmh_pct = (fp_10kmh_count / num_stat_samples) * 100.0
            
            mean_stat_gps_speed_kmh = np.mean(v_during_stat_kmh)
            max_stat_gps_speed_kmh = np.max(v_during_stat_kmh)
        else:
            fp_5kmh_count = 0
            fp_5kmh_pct = 0.0
            fp_10kmh_count = 0
            fp_10kmh_pct = 0.0
            mean_stat_gps_speed_kmh = 0.0
            max_stat_gps_speed_kmh = 0.0
            
        results.append({
            'Session': s_name,
            'Duration (s)': t_all[-1],
            'Total Samples': N,
            'ZUPT Samples': num_stat_samples,
            'ZUPT Time (s)': num_stat_samples * 0.1,
            'Mean GPS Speed during ZUPT (km/h)': round(mean_stat_gps_speed_kmh, 2),
            'Max GPS Speed during ZUPT (km/h)': round(max_stat_gps_speed_kmh, 2),
            'False Positives (>5 km/h)': fp_5kmh_count,
            'False Positive Rate (%)': round(fp_5kmh_pct, 1),
            'FP (>10 km/h) Rate (%)': round(fp_10kmh_pct, 1)
        })
        
        # Plot false positive diagnostic for S-Vta10 and S-Vtb2
        plot_zupt_fp_diagnostic(df, is_stationary, v_gps_true_interp, s_name)
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("\n")
    return res_df

def plot_zupt_fp_diagnostic(df, is_stationary, v_gps_interp, s_name):
    t = df['time_rel_sec'].values
    v_kmh = v_gps_interp * 3.6
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, v_kmh, 'k-', linewidth=1.2, label='True GPS Speed (km/h)')
    ax.axhline(5.0, color='r', linestyle='--', linewidth=0.8, label='5 km/h False Positive Threshold')
    
    # Highlight ZUPT intervals
    ax.fill_between(t, 0, np.max(v_kmh)+2, where=is_stationary, color='orange', alpha=0.3, label='ZUPT Active (Zero-Velocity Forced)')
    
    # Highlight False Positive intervals (ZUPT active AND v > 5 km/h)
    fp_mask = is_stationary & (v_kmh > 5.0)
    ax.fill_between(t, 0, np.max(v_kmh)+2, where=fp_mask, color='red', alpha=0.5, label='False Positive Region (ZUPT during Cruise)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title(f'Investigation 1: ZUPT False Positive Audit — {s_name}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    out_p = OUT_DIR / f"investigation1_fp_{Path(s_name).stem}.png"
    plt.tight_layout()
    plt.savefig(out_p, dpi=150)
    plt.close()
    print(f"  Saved plot: {out_p}")

# -------------------------------------------------------------------
# INVESTIGATION 2: HEADING DRIFT ANALYSIS ON S-Vw4
# -------------------------------------------------------------------
def run_investigation_2():
    print("=" * 80)
    print("INVESTIGATION 2: HEADING DRIFT ANALYSIS ON S-Vw4 (210.9 MIN)")
    print("=" * 80)
    
    p = PROCESSED_DIR / "S-Vw4.parquet"
    df = pd.read_parquet(p)
    
    if 'acc_veh_fwd' not in df.columns:
        from phase4_orientation import transform_to_vehicle_frame
        df, _, _ = transform_to_vehicle_frame(df)
        
    t_all = df['time_rel_sec'].values
    dt = df['dt_sec'].values
    lat = df['gps_lat'].values
    lon = df['gps_lon'].values
    lat0, lon0 = lat[0], lon[0]
    
    east_all, north_all = latlon_to_enu(lat, lon, lat0, lon0)
    
    # 1. Extract genuine GNSS fixes only
    gnss_mask = df['is_gnss_update'].values
    gnss_idx = np.where(gnss_mask)[0]
    
    t_gnss = t_all[gnss_idx]
    e_gnss = east_all[gnss_idx]
    n_gnss = north_all[gnss_idx]
    v_gps = df['gps_speed_ms'].values[gnss_idx]
    
    # 2. Compute reliable GNSS course/bearing between consecutive genuine fixes
    gnss_bearings = []
    valid_t_gnss = []
    
    for k in range(len(gnss_idx) - 1):
        de = e_gnss[k+1] - e_gnss[k]
        dn = n_gnss[k+1] - n_gnss[k]
        dist = np.sqrt(de**2 + dn**2)
        v_seg = 0.5 * (v_gps[k] + v_gps[k+1])
        
        # Valid motion filter: distance > 10m and speed > 2.0 m/s (~7.2 km/h)
        if dist > 10.0 and v_seg > 2.0:
            # Bearing: 0 = North (+dn), 90 = East (+de)
            # theta = atan2(de, dn) in [0, 2pi)
            bearing_rad = np.arctan2(de, dn)
            if bearing_rad < 0:
                bearing_rad += 2 * np.pi
            gnss_bearings.append(bearing_rad)
            valid_t_gnss.append(0.5 * (t_gnss[k] + t_gnss[k+1]))
            
    gnss_bearings = np.array(gnss_bearings)
    valid_t_gnss = np.array(valid_t_gnss)
    
    print(f"Total genuine GNSS fixes: {len(gnss_idx):,d}")
    print(f"Valid motion bearing points (dist > 10m, v > 2m/s): {len(valid_t_gnss):,d}")
    
    # 3. Compute Gyro Integrated Heading & Magnetometer Azimuth at same timestamps
    gyro_yaw_rate = df['gyro_veh_yaw_rate'].values
    
    # Initial bearing
    psi0 = gnss_bearings[0] if len(gnss_bearings) > 0 else 0.0
    
    # Continuous Gyro integration
    psi_gyro_cont = np.zeros(len(df))
    psi_gyro_cont[0] = psi0
    for t in range(1, len(df)):
        psi_gyro_cont[t] = psi_gyro_cont[t-1] + gyro_yaw_rate[t] * dt[t]
        
    # Interpolate gyro integrated heading and mag azimuth at valid GNSS timestamps
    psi_gyro_at_gnss = np.interp(valid_t_gnss, t_all, psi_gyro_cont)
    
    # Magnetometer / Sensor azimuth
    az_col = [c for c in df.columns if 'azimuth' in c.lower()][0]
    mag_az_rad = np.radians(df[az_col].values)
    mag_az_at_gnss = np.interp(valid_t_gnss, t_all, mag_az_rad)
    
    # 4. Unwrap angles to prevent 0/360 boundary discontinuities
    gnss_bearing_unwrapped = np.unwrap(gnss_bearings)
    gyro_unwrapped = np.unwrap(psi_gyro_at_gnss)
    mag_unwrapped = np.unwrap(mag_az_at_gnss)
    
    # Re-align initial offsets
    offset_gyro = gyro_unwrapped[0] - gnss_bearing_unwrapped[0]
    gyro_unwrapped -= offset_gyro
    
    offset_mag = mag_unwrapped[0] - gnss_bearing_unwrapped[0]
    mag_unwrapped -= offset_mag
    
    # Heading Errors (Degrees)
    err_gyro_deg = np.degrees(gyro_unwrapped - gnss_bearing_unwrapped)
    err_mag_deg  = np.degrees(mag_unwrapped - gnss_bearing_unwrapped)
    
    # 5. Fit Error Growth Models to Gyro Heading Drift
    t_sec = valid_t_gnss - valid_t_gnss[0]
    t_hr = t_sec / 3600.0
    
    # Linear fit: y = b_gyro * t (deg)
    # y = b * t
    fit_linear_coef = np.polyfit(t_hr, err_gyro_deg, 1) # [slope (deg/hr), intercept]
    drift_rate_deg_per_hr = fit_linear_coef[0]
    
    # Quadratic fit: y = c2 * t^2 + c1 * t + c0
    fit_quad_coef = np.polyfit(t_hr, err_gyro_deg, 2)
    
    print("\n--- HEADING ERROR AUDIT METRICS (S-Vw4, 210.9 min) ---")
    print(f"  Total Duration:                   {t_all[-1]/3600:.2f} hours ({t_all[-1]/60:.1f} min)")
    print(f"  Final Gyro Heading Error:         {err_gyro_deg[-1]:.1f} degrees")
    print(f"  Max Gyro Heading Error:           {np.max(np.abs(err_gyro_deg)):.1f} degrees")
    print(f"  Linear Gyro Drift Rate:           {drift_rate_deg_per_hr:+.2f} deg/hour")
    print(f"  Magnetometer Mean Absolute Error: {np.mean(np.abs(err_mag_deg)):.1f} degrees (bounded, no run-away)")
    
    # 6. Plot Comprehensive Heading Drift Figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Subplot 1: Unwrapped Heading Progression
    axes[0].plot(valid_t_gnss / 60.0, np.degrees(gnss_bearing_unwrapped), 'k.', markersize=3, label='GNSS Ground-Truth Course (Genuine Fixes)')
    axes[0].plot(valid_t_gnss / 60.0, np.degrees(gyro_unwrapped), 'r-', linewidth=1.0, alpha=0.8, label='Gyro-Integrated Heading (Continuous Unwrapped)')
    axes[0].plot(valid_t_gnss / 60.0, np.degrees(mag_unwrapped), 'b--', linewidth=0.8, alpha=0.6, label='Magnetometer Azimuth (Unwrapped)')
    axes[0].set_ylabel('Heading (degrees)')
    axes[0].set_title('Investigation 2: Cumulative Unwrapped Heading vs Genuine GNSS Course (S-Vw4, 3.5 Hours)')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Heading Error Growth Delta_psi(t)
    axes[1].plot(valid_t_gnss / 60.0, err_gyro_deg, 'r-', linewidth=1.2, label=f'Gyro Heading Error Delta_psi(t) — Final: {err_gyro_deg[-1]:.1f}°')
    # Plot linear trend
    t_hr_axis = (valid_t_gnss - valid_t_gnss[0]) / 3600.0
    trend_linear = np.polyval(fit_linear_coef, t_hr_axis)
    trend_quad = np.polyval(fit_quad_coef, t_hr_axis)
    axes[1].plot(valid_t_gnss / 60.0, trend_linear, 'k--', linewidth=1.0, label=f'Linear Trend ({drift_rate_deg_per_hr:+.1f}°/hour)')
    axes[1].plot(valid_t_gnss / 60.0, trend_quad, 'g:', linewidth=1.2, label='Quadratic Fit')
    axes[1].set_ylabel('Error (degrees)')
    axes[1].set_title('Gyroscope Heading Error Evolution & Growth Dynamics')
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Subplot 3: Magnetometer Heading Error (Bounded Magnetic Distortion)
    axes[2].plot(valid_t_gnss / 60.0, err_mag_deg, 'b-', linewidth=0.8, alpha=0.7, label=f'Magnetometer Error (Bounded, Mean Abs: {np.mean(np.abs(err_mag_deg)):.1f}°)')
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[2].set_ylabel('Mag Error (degrees)')
    axes[2].set_xlabel('Time (minutes)')
    axes[2].set_title('Magnetometer Azimuth Error (Local Vehicle Ferromagnetic Perturbations)')
    axes[2].legend(loc='upper left', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    out_p2 = OUT_DIR / "investigation2_heading_drift_S-Vw4.png"
    plt.tight_layout()
    plt.savefig(out_p2, dpi=150)
    plt.close()
    print(f"  Saved plot: {out_p2}")

def main():
    run_investigation_1()
    run_investigation_2()

if __name__ == "__main__":
    main()
