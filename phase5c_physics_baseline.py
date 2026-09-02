"""
Phase 5C — ZUPT v2 & Circular Complementary Heading Filter
SIH26168 — AI-ML Dead Reckoning System

Step 1: Calibrate 4 thresholds strictly on static sessions (S-S1 through S-S4):
  1. |a_lin| magnitude threshold
  2. |omega| magnitude threshold
  3. rolling variance of |a_lin| (window = 8 samples = 0.8s)
  4. rolling variance of |omega| (window = 8 samples = 0.8s)

Step 2: Re-run False-Positive Audit on S-Vta10 and S-Vtb2 with ZUPT v2.

Step 3: Circular Complementary Heading Filter:
  - Unit vector representation:
      v_gyro = [cos(psi_gyro), sin(psi_gyro)]
      v_mag  = [cos(psi_mag),  sin(psi_mag)]
      v_fused = alpha * v_gyro + (1 - alpha) * v_mag
      psi_fused = atan2(v_fused_y, v_fused_x)
  - Sweep alpha in {0.95, 0.97, 0.98} + Gyro-only baseline on S-Vw4.
  - Evaluate Heading MAE, final heading error, drift rate using genuine GNSS fixes (dist > 10m, v > 2m/s).

Step 4: Full Pipeline Comparison on S-Vw4 (210.9 min):
  - Phase 5A: Open-Loop
  - Phase 5B: ZUPT v1 (magnitude-only)
  - Phase 5C: ZUPT v2 + Circular Complementary Heading Filter
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
OUT_DIR = Path("plots") / "phase5c"
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

def rolling_var_1d(arr, window=8):
    """Compute rolling variance with minimum window size 1."""
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=1).var().fillna(0.0).values

# -------------------------------------------------------------------
# STEP 1: CALIBRATE ALL 4 THRESHOLDS FROM STATIC SESSIONS ONLY
# -------------------------------------------------------------------
def calibrate_zupt_v2_thresholds(window_len=8):
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
    
    a_mags = []
    w_mags = []
    a_vars = []
    w_vars = []
    
    print("=" * 80)
    print("PHASE 5C - STEP 1: CALIBRATING 4 THRESHOLDS ON STATIC SESSIONS")
    print("=" * 80)
    
    total_samples = 0
    
    for sf in static_files:
        p = PROCESSED_DIR / sf
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        
        ax = df['acc_lin_x'].values
        ay = df['acc_lin_y'].values
        az = df['acc_lin_z'].values
        a_mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        gx = df['gyro_x'].values
        gy = df['gyro_y'].values
        gz = df['gyro_z'].values
        w_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        
        a_var = rolling_var_1d(a_mag, window=window_len)
        w_var = rolling_var_1d(w_mag, window=window_len)
        
        a_mags.append(a_mag)
        w_mags.append(w_mag)
        a_vars.append(a_var)
        w_vars.append(w_var)
        total_samples += len(df)
        
    all_a_mag = np.concatenate(a_mags)
    all_w_mag = np.concatenate(w_mags)
    all_a_var = np.concatenate(a_vars)
    all_w_var = np.concatenate(w_vars)
    
    # 99th percentiles
    a_mag_p99 = np.percentile(all_a_mag, 99)
    w_mag_p99 = np.percentile(all_w_mag, 99)
    a_var_p99 = np.percentile(all_a_var, 99)
    w_var_p99 = np.percentile(all_w_var, 99)
    
    # Locked thresholds: 1.25 * p99
    a_th = float(np.round(a_mag_p99 * 1.25, 3))
    w_th = float(np.round(w_mag_p99 * 1.25, 3))
    a_var_th = float(np.round(a_var_p99 * 1.25, 4))
    w_var_th = float(np.round(w_var_p99 * 1.25, 4))
    
    print(f"Static samples analyzed: {total_samples:,d}")
    print(f"1. a_mag (m/s²):      p95={np.percentile(all_a_mag,95):.4f}, p99={a_mag_p99:.4f}  --> Locked a_th:     {a_th:.3f} m/s²")
    print(f"2. w_mag (rad/s):     p95={np.percentile(all_w_mag,95):.4f}, p99={w_mag_p99:.4f}  --> Locked w_th:     {w_th:.3f} rad/s")
    print(f"3. a_var (m²/s⁴):     p95={np.percentile(all_a_var,95):.4f}, p99={a_var_p99:.4f}  --> Locked a_var_th: {a_var_th:.4f} m²/s⁴")
    print(f"4. w_var (rad²/s²):   p95={np.percentile(all_w_var,95):.4f}, p99={w_var_p99:.4f}  --> Locked w_var_th: {w_var_th:.4f} rad²/s²")
    print("=" * 80 + "\n")
    
    return a_th, w_th, a_var_th, w_var_th

# -------------------------------------------------------------------
# STEP 2: ZUPT v2 DETECTOR & FALSE-POSITIVE AUDIT
# -------------------------------------------------------------------
def get_zupt_v2_mask(df, a_th, w_th, a_var_th, w_var_th, window_len=8):
    ax = df['acc_lin_x'].values
    ay = df['acc_lin_y'].values
    az = df['acc_lin_z'].values
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    
    gx = df['gyro_x'].values
    gy = df['gyro_y'].values
    gz = df['gyro_z'].values
    w_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    a_var = rolling_var_1d(a_mag, window=window_len)
    w_var = rolling_var_1d(w_mag, window=window_len)
    
    # Two-condition detector: magnitude AND variance below calibrated thresholds
    cond_mag = (a_mag < a_th) & (w_mag < w_th)
    cond_var = (a_var < a_var_th) & (w_var < w_var_th)
    raw_stationary = cond_mag & cond_var
    
    N = len(df)
    is_stationary = np.zeros(N, dtype=bool)
    consec = 0
    for i in range(N):
        if raw_stationary[i]:
            consec += 1
            if consec >= window_len:
                is_stationary[i] = True
        else:
            consec = 0
    return is_stationary

def run_false_positive_audit_v2(a_th, w_th, a_var_th, w_var_th):
    print("=" * 80)
    print("PHASE 5C - STEP 2: ZUPT v2 FALSE POSITIVE AUDIT (S-Vta10, S-Vtb2)")
    print("=" * 80)
    
    test_files = ["S-Vta10.parquet", "S-Vtb2.parquet"]
    results = []
    
    for s_name in test_files:
        df = pd.read_parquet(PROCESSED_DIR / s_name)
        is_stat_v2 = get_zupt_v2_mask(df, a_th, w_th, a_var_th, w_var_th)
        
        # Genuine GNSS speed interpolation
        gnss_idx = np.where(df['is_gnss_update'].values)[0]
        t_all = df['time_rel_sec'].values
        v_gps_all = df['gps_speed_ms'].values
        t_gnss = t_all[gnss_idx]
        v_gnss_fixes = v_gps_all[gnss_idx]
        
        v_true = np.interp(t_all, t_gnss, v_gnss_fixes) if len(t_gnss) > 1 else v_gps_all
        v_kmh = v_true * 3.6
        
        num_stat = np.sum(is_stat_v2)
        if num_stat > 0:
            v_stat = v_kmh[is_stat_v2]
            fp_5 = np.sum(v_stat > 5.0)
            fp_5_pct = (fp_5 / num_stat) * 100.0
            mean_v_stat = np.mean(v_stat)
            max_v_stat = np.max(v_stat)
        else:
            fp_5 = 0
            fp_5_pct = 0.0
            mean_v_stat = 0.0
            max_v_stat = 0.0
            
        results.append({
            'Session': s_name,
            'Duration (s)': t_all[-1],
            'ZUPT v2 Samples': int(num_stat),
            'ZUPT v2 Time (s)': round(num_stat * 0.1, 1),
            'Mean GPS Speed during ZUPT (km/h)': round(mean_v_stat, 2),
            'Max GPS Speed during ZUPT (km/h)': round(max_v_stat, 2),
            'False Positives (>5 km/h)': int(fp_5),
            'False Positive Rate (%)': round(fp_5_pct, 1)
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("=" * 80 + "\n")
    return res_df

# -------------------------------------------------------------------
# STEP 3: CIRCULAR COMPLEMENTARY HEADING FILTER
# -------------------------------------------------------------------
def run_circular_complementary_filter(df, alpha):
    """
    Circular Complementary Heading Filter:
      v_gyro = [cos(psi_gyro_t), sin(psi_gyro_t)]
      v_mag  = [cos(psi_mag_t),  sin(psi_mag_t)]
      v_fused = alpha * v_gyro + (1 - alpha) * v_mag
      psi_fused = atan2(v_fused_y, v_fused_x)
    """
    N = len(df)
    dt = df['dt_sec'].values
    gyro_yaw_rate = df['gyro_veh_yaw_rate'].values
    
    az_col = [c for c in df.columns if 'azimuth' in c.lower()][0]
    psi_mag = np.radians(df[az_col].values)
    
    # Initial heading
    bearing_col = [c for c in df.columns if 'gps_orientation' in c.lower() or 'gps_bearing' in c.lower()]
    init_bearing = df[bearing_col[0]].iloc[0] if bearing_col else np.nan
    psi0 = psi_mag[0] if (np.isnan(init_bearing) or init_bearing == 0) else np.radians(init_bearing)
    
    psi_fused = np.zeros(N)
    psi_fused[0] = psi0
    
    for t in range(1, N):
        # 1. Propagate previous fused heading using gyroscope
        psi_gyro_step = psi_fused[t-1] + gyro_yaw_rate[t] * dt[t]
        
        # 2. Convert to unit vectors (handles 0/360 boundary safely)
        gx_unit = np.cos(psi_gyro_step)
        gy_unit = np.sin(psi_gyro_step)
        
        mx_unit = np.cos(psi_mag[t])
        my_unit = np.sin(psi_mag[t])
        
        # 3. Weighted vector fusion
        fx = alpha * gx_unit + (1.0 - alpha) * mx_unit
        fy = alpha * gy_unit + (1.0 - alpha) * my_unit
        
        # 4. Extract continuous angle
        psi_fused[t] = np.arctan2(fy, fx)
        if psi_fused[t] < 0:
            psi_fused[t] += 2 * np.pi
            
    return psi_fused

def evaluate_heading_sweep_on_svw4():
    print("=" * 80)
    print("PHASE 5C - STEP 3: CIRCULAR COMPLEMENTARY HEADING SWEEP ON S-Vw4 (210.9 min)")
    print("=" * 80)
    
    df = pd.read_parquet(PROCESSED_DIR / "S-Vw4.parquet")
    if 'acc_veh_fwd' not in df.columns:
        from phase4_orientation import transform_to_vehicle_frame
        df, _, _ = transform_to_vehicle_frame(df)
        
    t_all = df['time_rel_sec'].values
    dt = df['dt_sec'].values
    lat = df['gps_lat'].values
    lon = df['gps_lon'].values
    lat0, lon0 = lat[0], lon[0]
    east_all, north_all = latlon_to_enu(lat, lon, lat0, lon0)
    
    # Genuine GNSS motion bearings
    gnss_mask = df['is_gnss_update'].values
    gnss_idx = np.where(gnss_mask)[0]
    t_gnss = t_all[gnss_idx]
    e_gnss = east_all[gnss_idx]
    n_gnss = north_all[gnss_idx]
    v_gps = df['gps_speed_ms'].values[gnss_idx]
    
    gnss_bearings = []
    valid_t_gnss = []
    for k in range(len(gnss_idx) - 1):
        de = e_gnss[k+1] - e_gnss[k]
        dn = n_gnss[k+1] - n_gnss[k]
        dist = np.sqrt(de**2 + dn**2)
        v_seg = 0.5 * (v_gps[k] + v_gps[k+1])
        if dist > 10.0 and v_seg > 2.0:
            b_rad = np.arctan2(de, dn)
            if b_rad < 0:
                b_rad += 2 * np.pi
            gnss_bearings.append(b_rad)
            valid_t_gnss.append(0.5 * (t_gnss[k] + t_gnss[k+1]))
            
    gnss_bearings = np.array(gnss_bearings)
    valid_t_gnss = np.array(valid_t_gnss)
    gnss_unwrapped = np.unwrap(gnss_bearings)
    
    t_hr = (valid_t_gnss - valid_t_gnss[0]) / 3600.0
    
    alphas = [1.0, 0.95, 0.97, 0.98]  # 1.0 = Gyro-only
    sweep_results = []
    fused_headings = {}
    
    for a in alphas:
        label = "Gyro-only" if a == 1.0 else f"alpha={a:.2f}"
        if a == 1.0:
            # Gyro only continuous integration
            psi_cont = np.zeros(len(df))
            psi_cont[0] = gnss_bearings[0]
            gyro_yaw = df['gyro_veh_yaw_rate'].values
            for t in range(1, len(df)):
                psi_cont[t] = psi_cont[t-1] + gyro_yaw[t] * dt[t]
            psi_fused = psi_cont
        else:
            psi_fused = run_circular_complementary_filter(df, alpha=a)
            
        fused_headings[label] = psi_fused
        
        # Sample at valid GNSS timestamps
        psi_at_gnss = np.interp(valid_t_gnss, t_all, psi_fused)
        psi_unwrapped = np.unwrap(psi_at_gnss)
        psi_unwrapped -= (psi_unwrapped[0] - gnss_unwrapped[0])
        
        err_deg = np.degrees(psi_unwrapped - gnss_unwrapped)
        mae = float(np.mean(np.abs(err_deg)))
        final_err = float(err_deg[-1])
        
        # Linear drift rate (deg/hour)
        fit_coef = np.polyfit(t_hr, err_deg, 1)
        drift_rate = float(fit_coef[0])
        
        sweep_results.append({
            'alpha': label,
            'Heading MAE (deg)': round(mae, 1),
            'Final Heading Error (deg)': round(final_err, 1),
            'Drift Rate (deg/hour)': round(drift_rate, 2)
        })
        
    sweep_df = pd.DataFrame(sweep_results)
    print(sweep_df.to_string(index=False))
    print("=" * 80 + "\n")
    
    # Save heading comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    for a in alphas:
        label = "Gyro-only" if a == 1.0 else f"alpha={a:.2f}"
        psi_at_gnss = np.interp(valid_t_gnss, t_all, fused_headings[label])
        psi_unw = np.unwrap(psi_at_gnss)
        psi_unw -= (psi_unw[0] - gnss_unwrapped[0])
        err_deg = np.degrees(psi_unw - gnss_unwrapped)
        ax.plot(valid_t_gnss / 60.0, err_deg, label=f"{label} (Final Err: {err_deg[-1]:.1f}°)")
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Heading Error (degrees)')
    ax.set_title('Circular Complementary Heading Filter: Alpha Parameter Sweep (S-Vw4, 3.5 Hours)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    out_p = OUT_DIR / "heading_alpha_sweep_S-Vw4.png"
    plt.tight_layout()
    plt.savefig(out_p, dpi=150)
    plt.close()
    print(f"Saved heading sweep plot: {out_p}")
    
    return sweep_df, fused_headings

# -------------------------------------------------------------------
# STEP 4: FULL PIPELINE BENCHMARK (5A vs 5B vs 5C) ON S-Vw4
# -------------------------------------------------------------------
def run_full_pipeline_comparison(a_th, w_th, a_var_th, w_var_th, best_alpha=0.98):
    print("=" * 80)
    print(f"PHASE 5C - STEP 4: FULL DEAD RECKONING BENCHMARK (5A vs 5B vs 5C) ON S-Vw4")
    print("=" * 80)
    
    df = pd.read_parquet(PROCESSED_DIR / "S-Vw4.parquet")
    if 'acc_veh_fwd' not in df.columns:
        from phase4_orientation import transform_to_vehicle_frame
        df, _, _ = transform_to_vehicle_frame(df)
        
    N = len(df)
    dt = df['dt_sec'].values
    a_fwd = df['acc_veh_fwd'].values
    gyro_yaw = df['gyro_veh_yaw_rate'].values
    
    lat0, lon0 = df['gps_lat'].iloc[0], df['gps_lon'].iloc[0]
    v0 = df['gps_speed_ms'].iloc[0]
    
    # 1. ZUPT v1 mask (magnitude only)
    ax_l, ay_l, az_l = df['acc_lin_x'].values, df['acc_lin_y'].values, df['acc_lin_z'].values
    a_mag_3d = np.sqrt(ax_l**2 + ay_l**2 + az_l**2)
    gx, gy, gz = df['gyro_x'].values, df['gyro_y'].values, df['gyro_z'].values
    w_mag_3d = np.sqrt(gx**2 + gy**2 + gz**2)
    raw_stat_v1 = (a_mag_3d < a_th) & (w_mag_3d < w_th)
    is_stat_v1 = np.zeros(N, dtype=bool)
    consec = 0
    for i in range(N):
        if raw_stat_v1[i]:
            consec += 1
            if consec >= 8:
                is_stat_v1[i] = True
        else:
            consec = 0
            
    # 2. ZUPT v2 mask (magnitude AND variance)
    is_stat_v2 = get_zupt_v2_mask(df, a_th, w_th, a_var_th, w_var_th, window_len=8)
    
    # 3. Headings:
    # Heading A: Gyro-only
    bearing_col = [c for c in df.columns if 'gps_orientation' in c.lower() or 'gps_bearing' in c.lower()]
    az_col = [c for c in df.columns if 'azimuth' in c.lower()][0]
    init_bearing = df[bearing_col[0]].iloc[0] if bearing_col else np.nan
    psi0 = np.radians(df[az_col].iloc[0]) if (np.isnan(init_bearing) or init_bearing == 0) else np.radians(init_bearing)
    
    psi_gyro_only = np.zeros(N)
    psi_gyro_only[0] = psi0
    for t in range(1, N):
        psi_gyro_only[t] = psi_gyro_only[t-1] + gyro_yaw[t] * dt[t]
        
    # Heading C: Circular Complementary Fused Heading
    psi_fused_c = run_circular_complementary_filter(df, alpha=best_alpha)
    
    # -------------------------------------------------------------
    # RUN PHASE 5A: Open-Loop
    # -------------------------------------------------------------
    v_5a = np.zeros(N); v_5a[0] = v0
    e_5a = np.zeros(N); n_5a = np.zeros(N)
    for t in range(1, N):
        v_5a[t] = max(0.0, v_5a[t-1] + a_fwd[t] * dt[t])
        e_5a[t] = e_5a[t-1] + v_5a[t] * np.sin(psi_gyro_only[t]) * dt[t]
        n_5a[t] = n_5a[t-1] + v_5a[t] * np.cos(psi_gyro_only[t]) * dt[t]
        
    # -------------------------------------------------------------
    # RUN PHASE 5B: ZUPT v1 only (with gyro-only heading)
    # -------------------------------------------------------------
    v_5b = np.zeros(N); v_5b[0] = v0
    e_5b = np.zeros(N); n_5b = np.zeros(N)
    for t in range(1, N):
        if is_stat_v1[t]:
            v_5b[t] = 0.0
        else:
            v_5b[t] = max(0.0, v_5b[t-1] + a_fwd[t] * dt[t])
        e_5b[t] = e_5b[t-1] + v_5b[t] * np.sin(psi_gyro_only[t]) * dt[t]
        n_5b[t] = n_5b[t-1] + v_5b[t] * np.cos(psi_gyro_only[t]) * dt[t]
        
    # -------------------------------------------------------------
    # RUN PHASE 5C: ZUPT v2 + Circular Complementary Heading
    # -------------------------------------------------------------
    v_5c = np.zeros(N); v_5c[0] = v0
    e_5c = np.zeros(N); n_5c = np.zeros(N)
    for t in range(1, N):
        if is_stat_v2[t]:
            v_5c[t] = 0.0
        else:
            v_5c[t] = max(0.0, v_5c[t-1] + a_fwd[t] * dt[t])
        e_5c[t] = e_5c[t-1] + v_5c[t] * np.sin(psi_fused_c[t]) * dt[t]
        n_5c[t] = n_5c[t-1] + v_5c[t] * np.cos(psi_fused_c[t]) * dt[t]
        
    # -------------------------------------------------------------
    # EVALUATION AGAINST GENUINE GNSS FIXES
    # -------------------------------------------------------------
    east_all, north_all = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)
    gnss_mask = df['is_gnss_update'].values
    gnss_idx = np.where(gnss_mask)[0]
    
    g_e_fixes = east_all[gnss_idx]
    g_n_fixes = north_all[gnss_idx]
    
    total_dist = float(np.sum(np.sqrt(np.diff(g_e_fixes)**2 + np.diff(g_n_fixes)**2)))
    
    err_5a = np.sqrt((e_5a[gnss_idx] - g_e_fixes)**2 + (n_5a[gnss_idx] - g_n_fixes)**2)
    err_5b = np.sqrt((e_5b[gnss_idx] - g_e_fixes)**2 + (n_5b[gnss_idx] - g_n_fixes)**2)
    err_5c = np.sqrt((e_5c[gnss_idx] - g_e_fixes)**2 + (n_5c[gnss_idx] - g_n_fixes)**2)
    
    comp_results = [
        {
            'Baseline': 'Phase 5A (Open-Loop)',
            'Terminal Velocity (m/s)': round(v_5a[-1], 1),
            'Max Velocity (m/s)': round(np.max(v_5a), 1),
            'Final Position Error (m)': round(err_5a[-1], 1),
            'Final Drift %': round((err_5a[-1] / total_dist) * 100.0, 1),
            'Improvement vs 5A (%)': '0.0%'
        },
        {
            'Baseline': 'Phase 5B (ZUPT v1 only)',
            'Terminal Velocity (m/s)': round(v_5b[-1], 1),
            'Max Velocity (m/s)': round(np.max(v_5b), 1),
            'Final Position Error (m)': round(err_5b[-1], 1),
            'Final Drift %': round((err_5b[-1] / total_dist) * 100.0, 1),
            'Improvement vs 5A (%)': f"{((err_5a[-1] - err_5b[-1]) / err_5a[-1])*100.0:.1f}%"
        },
        {
            'Baseline': f'Phase 5C (ZUPT v2 + Comp Heading a={best_alpha:.2f})',
            'Terminal Velocity (m/s)': round(v_5c[-1], 1),
            'Max Velocity (m/s)': round(np.max(v_5c), 1),
            'Final Position Error (m)': round(err_5c[-1], 1),
            'Final Drift %': round((err_5c[-1] / total_dist) * 100.0, 1),
            'Improvement vs 5A (%)': f"{((err_5a[-1] - err_5c[-1]) / err_5a[-1])*100.0:.1f}%"
        }
    ]
    
    comp_df = pd.DataFrame(comp_results)
    print(comp_df.to_string(index=False))
    print("=" * 80 + "\n")
    
    # Plot Trajectory & Error Evolution
    t_gnss = df['time_rel_sec'].values[gnss_idx] / 60.0
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 1. 2D Trajectory Map
    ax_map = axes[0]
    ax_map.plot(g_e_fixes, g_n_fixes, 'k.-', linewidth=1.5, label='GNSS Fixes Ground Truth')
    ax_map.plot(e_5a, n_5a, 'r:', linewidth=0.8, alpha=0.6, label=f'5A Open-Loop (Err: {err_5a[-1]:.0f}m)')
    ax_map.plot(e_5b, n_5b, 'm--', linewidth=0.9, alpha=0.7, label=f'5B ZUPT v1 (Err: {err_5b[-1]:.0f}m)')
    ax_map.plot(e_5c, n_5c, 'b-', linewidth=1.3, label=f'5C ZUPT v2 + Comp Heading (Err: {err_5c[-1]:.0f}m)')
    ax_map.plot(0, 0, 'go', markersize=8, label='Start')
    ax_map.set_xlabel('East (meters)')
    ax_map.set_ylabel('North (meters)')
    ax_map.set_title(f'Dead Reckoning Trajectory Progression — S-Vw4 (210.9 min, {total_dist/1000:.1f} km)')
    ax_map.legend(loc='upper right', fontsize=8)
    ax_map.grid(True, alpha=0.3)
    ax_map.axis('equal')
    
    # 2. Position Error over Time
    ax_err = axes[1]
    ax_err.plot(t_gnss, err_5a, 'r:', linewidth=1.0, label='5A Open-Loop')
    ax_err.plot(t_gnss, err_5b, 'm--', linewidth=1.2, label='5B ZUPT v1')
    ax_err.plot(t_gnss, err_5c, 'b-', linewidth=1.5, label='5C ZUPT v2 + Complementary Heading')
    ax_err.set_xlabel('Time (minutes)')
    ax_err.set_ylabel('Position Error (meters)')
    ax_err.set_title('Cumulative Position Drift Comparison (Evaluated at GNSS Fixes)')
    ax_err.legend(loc='upper left', fontsize=9)
    ax_err.grid(True, alpha=0.3)
    
    out_comp = OUT_DIR / "phase5c_dr_progression_S-Vw4.png"
    plt.tight_layout()
    plt.savefig(out_comp, dpi=150)
    plt.close()
    print(f"Saved Phase 5C comparison plot: {out_comp}")

def main():
    a_th, w_th, a_var_th, w_var_th = calibrate_zupt_v2_thresholds()
    run_false_positive_audit_v2(a_th, w_th, a_var_th, w_var_th)
    sweep_df, _ = evaluate_heading_sweep_on_svw4()
    run_full_pipeline_comparison(a_th, w_th, a_var_th, w_var_th, best_alpha=0.98)

if __name__ == "__main__":
    main()
