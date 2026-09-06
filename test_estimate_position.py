"""
Test and Verification Script for estimate_position()
SIH26168 — AI-ML Dead Reckoning System

Requirements:
  1. Windowing and segment boundaries must EXACTLY match phase6_outage_sim.py / phase9_ai_hybrid.py
     (aligned to genuine GNSS fixes and identical interval boundaries).
  2. Pick a 300-second outage stretch from S-Vw4 (Window 0: s_gnss=88439, e_gnss=91490).
  3. Call estimate_position() sequentially across the stretch.
  4. Compare the final position error against the Phase 9 benchmark (expected ~1,854.15 m).
  5. Plot estimated path against real GNSS path.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from phase6_outage_sim import (
    get_zupt_v1_mask, latlon_to_enu, get_col,
    sample_outage_windows, find_genuine_gnss_updates,
    OUTAGE_DURATIONS_S, OUTAGES_PER_DURATION
)
from phase4_orientation import transform_to_vehicle_frame
from phase9_ai_hybrid import slice_outage_intervals, MIN_T
from estimate_position import estimate_position

PROCESSED_DIR = Path("data") / "processed_sessions"
OUT_DIR = Path("plots") / "step1_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("data") / "phase8_cache"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def main():
    print("=" * 80)
    print("STEP 1 VERIFICATION: Testing estimate_position() Pipeline")
    print("=" * 80)

    # 1. Load S-Vw4 session
    session_file = "S-Vw4.parquet"
    print(f"[1] Loading session {session_file}...")
    df = pd.read_parquet(PROCESSED_DIR / session_file)
    df, _, _ = transform_to_vehicle_frame(df)
    zupt_mask = get_zupt_v1_mask(df)
    gnss_idx = find_genuine_gnss_updates(df)

    lat0, lon0 = df['gps_lat'].iloc[0], df['gps_lon'].iloc[0]
    east_all, north_all = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)
    t_arr = df['time_rel_sec'].values

    # 2. Train LOTO Phase 9 model (Target C1) excluding S-Vw4
    print("[2] Loading / Training LOTO C1 XGBoost model...")
    meta_all = pd.read_parquet(CACHE_DIR / "meta_all.parquet")
    yC_all = np.load(CACHE_DIR / "yC_all.npy")
    X_all = np.load(CACHE_DIR / "X_all.npy")

    train_mask = (meta_all['dur_s'].values >= MIN_T) & (~meta_all['session'].str.startswith('S-Vw4'))
    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
    model.fit(X_all[train_mask], yC_all[train_mask])
    print(f"    Trained on {train_mask.sum():,} segments from other trips.")

    # 3. Exact Outage Window from Phase 6/9:
    # Phase 6 & Phase 9 sample durations in order [30, 60, 120, 300] using random seed 42.
    # Replicating the exact sampling loop guarantees identical window selection:
    np.random.seed(RANDOM_SEED)
    windows_dict = {}
    for dur in OUTAGE_DURATIONS_S:
        windows_dict[dur] = sample_outage_windows(df, gnss_idx, dur, OUTAGES_PER_DURATION)

    s_gnss, e_gnss = windows_dict[300][0]  # window_idx=0 -> (88439, 91490)
    actual_dur = t_arr[e_gnss] - t_arr[s_gnss]
    print(f"[3] Selected 300s Outage Window 0:")
    print(f"    s_gnss: {s_gnss}, e_gnss: {e_gnss}, duration: {actual_dur:.1f} s")

    # Initial GNSS-known state at start of stretch
    bearing_col = get_col(df, 'gps_orientation') or get_col(df, 'gps_bearing')
    az_col = get_col(df, 'azimuth')
    psi_mag = np.radians(df[az_col].values)
    b0 = df[bearing_col].values[s_gnss] if bearing_col else np.nan
    psi0 = psi_mag[s_gnss] if (np.isnan(b0) or b0 == 0) else np.radians(b0)

    initial_state = {
        'x': float(east_all[s_gnss]),
        'y': float(north_all[s_gnss]),
        'heading': float(psi0),
        'velocity': float(df['gps_speed_ms'].values[s_gnss])
    }
    print(f"    Initial State: x={initial_state['x']:.1f}m, y={initial_state['y']:.1f}m, "
          f"v={initial_state['velocity']:.2f}m/s, heading={np.degrees(initial_state['heading']):.1f} deg")

    # 4. Partition into intervals using EXACT Phase 9 logic
    intervals = slice_outage_intervals(s_gnss, e_gnss, t_arr, nominal_step_samples=90, min_t_s=MIN_T)
    print(f"[4] Sliced into {len(intervals)} consecutive ~9s intervals (matching Phase 9 exactly).")

    # 5. Sequentially call estimate_position() across intervals
    current_state = dict(initial_state)
    est_path_x = [current_state['x']]
    est_path_y = [current_state['y']]

    columns_needed = [
        'acc_x', 'acc_y', 'acc_z',
        'acc_lin_x', 'acc_lin_y', 'acc_lin_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'acc_veh_fwd', 'gyro_veh_yaw_rate',
        'dt_sec'
    ]

    for ia, ib in intervals:
        # Window slice spanning ia to ib inclusive
        window_df = df.iloc[ia : ib + 1][columns_needed]
        window_zupt = zupt_mask[ia : ib + 1]
        seg_dur = float(t_arr[ib] - t_arr[ia])

        # Call unified pipeline function
        current_state = estimate_position(
            imu_window=window_df,
            last_known_state=current_state,
            model=model,
            zupt_mask=window_zupt,
            seg_dur_s=seg_dur
        )

        est_path_x.append(current_state['x'])
        est_path_y.append(current_state['y'])

    # 6. Evaluation against ground-truth GNSS
    e_true_final = east_all[e_gnss]
    n_true_final = north_all[e_gnss]
    final_pos_err = np.sqrt((current_state['x'] - e_true_final)**2 + (current_state['y'] - n_true_final)**2)

    # Reference value from phase9_raw_records.csv
    benchmark_ref_err = 1854.1543654183824

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS:")
    print("=" * 80)
    print(f"Refactored estimate_position() Final Error : {final_pos_err:.4f} meters")
    print(f"Phase 9 Benchmark Reference Error          : {benchmark_ref_err:.4f} meters")
    discrepancy = abs(final_pos_err - benchmark_ref_err)
    print(f"Numerical Discrepancy                      : {discrepancy:.6e} meters")

    if discrepancy < 1e-4:
        print(">>> SUCCESS: Refactored function produces BIT-EXACT identical output to Phase 9 benchmark!")
    else:
        print(">>> WARNING: Discrepancy detected! Check interval bounds or feature order.")

    # 7. Plot Estimated Path vs Ground Truth GNSS
    gt_x = east_all[s_gnss : e_gnss + 1]
    gt_y = north_all[s_gnss : e_gnss + 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt_x - gt_x[0], gt_y - gt_y[0], 'g-', linewidth=2.5, label='GNSS Ground Truth Path')
    ax.plot(np.array(est_path_x) - gt_x[0], np.array(est_path_y) - gt_y[0],
            'b--D', markersize=5, linewidth=2.0, label=f'estimate_position() Path (Final Err: {final_pos_err:.1f}m)')

    ax.plot(0, 0, 'ko', markersize=10, label='Outage Start')
    ax.plot(gt_x[-1] - gt_x[0], gt_y[-1] - gt_y[0], 'go', markersize=9, label='GNSS Fix End')
    ax.plot(est_path_x[-1] - gt_x[0], est_path_y[-1] - gt_y[0], 'bo', markersize=9, label='Estimated End')

    ax.set_xlabel('East Displacement (m)', fontweight='bold')
    ax.set_ylabel('North Displacement (m)', fontweight='bold')
    ax.set_title(f'Verification of estimate_position(): 300s Outage on S-Vw4\nFinal Position Error = {final_pos_err:.2f} m (Phase 9 Benchmark = {benchmark_ref_err:.2f} m)',
                 fontweight='bold', pad=12)
    ax.legend(fontsize=9.5, loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')

    plt.tight_layout()
    plot_file = OUT_DIR / "verification_path_comparison.png"
    plt.savefig(plot_file, dpi=180)
    plt.close()
    print(f"\nSaved verification plot: {plot_file.resolve()}")

if __name__ == "__main__":
    main()
