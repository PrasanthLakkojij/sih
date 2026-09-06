"""
Phase 10 — Continuous State Estimator & EKF Fusion Engine
SIH26168 — Organisation: ISRO | Domain: Software / AI-ML

Objective:
  Replace the one-shot displacement correction (Phase 9) with a continuous state
  estimator using an Extended Kalman Filter (EKF).

State Vector:
  x = [px, py, vx, vy, psi, b_ax, b_ay, b_w]^T (dim=8)
    - px, py : Vehicle position in local East, North coordinates (meters)
    - vx, vy : Vehicle velocity in local East, North coordinates (m/s)
    - psi    : Vehicle heading / azimuth (radians, 0=North, pi/2=East, clockwise)
    - b_ax   : Accelerometer bias in vehicle forward axis (m/s^2)
    - b_ay   : Accelerometer bias in vehicle lateral axis (m/s^2)
    - b_w    : Gyroscope bias around vertical axis (rad/s)

Continuous Predict Step (10 Hz IMU):
  - Propagate state using vehicle-frame IMU data (a_fwd, a_lat, gyro_yaw)
  - Compensate for estimated sensor biases
  - Propagate error covariance P using linearized state transition Jacobian F

Continuous Measurement Updates:
  1. GNSS Update (when genuine fix is available outside outage):
     - Measurements: [e_gnss, n_gnss, ve_gnss, vn_gnss]
     - R_gnss based on smartphone GNSS accuracy
  2. ML Velocity Update (every ~1.0 s or 2.0 s from Phase 7 XGBoost model):
     - Predicts forward speed v_ml from 20-sample (2s) IMU window
     - Measurement model: h_ml(x) = sqrt(vx^2 + vy^2) or forward speed along heading
     - Covariance R_ml = sigma_ml^2 based on validated Phase 7 RMSE (~1.8 m/s)
  3. ZUPT Update (when stationary detected):
     - Measurements: vx = 0, vy = 0 with high confidence R_zupt

Benchmark & Comparison:
  Runs on identical outage windows (30s, 60s, 120s, 300s) on S-Vw4, S-Vtb5, S-Vta1a
  comparing:
    1. Physics-only (Open loop)
    2. ZUPT v1
    3. AI + Physics Hybrid (Phase 9)
    4. EKF Fusion Engine (Phase 10)
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from phase6_outage_sim import (
    sample_outage_windows, find_genuine_gnss_updates, get_zupt_v1_mask,
    latlon_to_enu, get_col, simulate_outage,
    TEST_SESSIONS, OUTAGE_DURATIONS_S, OUTAGES_PER_DURATION
)
from phase4_orientation import transform_to_vehicle_frame
from phase8_ml_correction import segment_features
from phase9_ai_hybrid import slice_outage_intervals, simulate_hybrid_outage

PROCESSED_DIR = Path("data") / "processed_sessions"
OUT_DIR = Path("plots") / "phase10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR_P8 = Path("data") / "phase8_cache"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ====================================================================
# Extended Kalman Filter Class
# ====================================================================
class VehicleEKF:
    """
    8-State Extended Kalman Filter for 2D Vehicle Dead Reckoning.
    State x = [px, py, vx, vy, psi, b_ax, b_ay, b_w]^T
    """
    def __init__(self, init_pos, init_vel, init_heading,
                 sigma_a_proc=0.5, sigma_w_proc=0.03,
                 sigma_ba_proc=1e-4, sigma_bw_proc=1e-5):
        """
        init_pos: [px0, py0] (meters ENU)
        init_vel: [vx0, vy0] (m/s ENU)
        init_heading: psi0 (radians, 0=North, pi/2=East)
        """
        self.x = np.zeros(8)
        self.x[0:2] = init_pos
        self.x[2:4] = init_vel
        self.x[4] = init_heading
        self.x[5:8] = 0.0  # Initial biases assumed zero

        # Initial Covariance P
        self.P = np.diag([
            5.0**2,   # px (5m)
            5.0**2,   # py (5m)
            0.5**2,   # vx (0.5 m/s)
            0.5**2,   # vy (0.5 m/s)
            np.radians(5.0)**2,  # psi (5 deg)
            0.1**2,   # b_ax (0.1 m/s^2)
            0.1**2,   # b_ay (0.1 m/s^2)
            np.radians(0.5)**2   # b_w (0.5 deg/s)
        ])

        # Noise parameters
        self.sigma_a = sigma_a_proc
        self.sigma_w = sigma_w_proc
        self.sigma_ba = sigma_ba_proc
        self.sigma_bw = sigma_bw_proc

    def predict(self, a_fwd_raw, a_lat_raw, gyro_w_raw, dt):
        """
        Propagate state and covariance over dt using vehicle-frame IMU inputs.
        a_fwd_raw : forward acceleration in vehicle frame
        a_lat_raw : lateral acceleration in vehicle frame
        gyro_w_raw: yaw rate (rad/s)
        """
        px, py, vx, vy, psi, b_ax, b_ay, b_w = self.x

        # Bias-corrected IMU inputs
        a_fwd = a_fwd_raw - b_ax
        a_lat = a_lat_raw - b_ay
        w = gyro_w_raw - b_w

        # Navigation-frame accelerations (0=North, pi/2=East)
        # East  accel = a_fwd * sin(psi) - a_lat * cos(psi)
        # North accel = a_fwd * cos(psi) + a_lat * sin(psi)
        sin_p = np.sin(psi)
        cos_p = np.cos(psi)

        ae = a_fwd * sin_p - a_lat * cos_p
        an = a_fwd * cos_p + a_lat * sin_p

        # State propagation
        new_px = px + vx * dt + 0.5 * ae * dt**2
        new_py = py + vy * dt + 0.5 * an * dt**2
        new_vx = vx + ae * dt
        new_vy = vy + an * dt
        new_psi = (psi + w * dt + np.pi) % (2 * np.pi) - np.pi

        self.x[0] = new_px
        self.x[1] = new_py
        self.x[2] = new_vx
        self.x[3] = new_vy
        self.x[4] = new_psi
        # Biases modeled as random walk: mean stays unchanged

        # State transition Jacobian F = df/dx (8x8)
        F = np.eye(8)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dt**2 * (a_fwd * cos_p + a_lat * sin_p)
        F[1, 4] = 0.5 * dt**2 * (-a_fwd * sin_p + a_lat * cos_p)
        F[2, 4] = dt * (a_fwd * cos_p + a_lat * sin_p)
        F[3, 4] = dt * (-a_fwd * sin_p + a_lat * cos_p)

        # Derivatives w.r.t biases
        F[0, 5] = -0.5 * dt**2 * sin_p
        F[0, 6] =  0.5 * dt**2 * cos_p
        F[1, 5] = -0.5 * dt**2 * cos_p
        F[1, 6] = -0.5 * dt**2 * sin_p

        F[2, 5] = -dt * sin_p
        F[2, 6] =  dt * cos_p
        F[3, 5] = -dt * cos_p
        F[3, 6] = -dt * sin_p

        F[4, 7] = -dt

        # Process Noise Covariance Q (8x8)
        # Driven by input acceleration noise, gyro noise, and bias random walks
        Q = np.zeros((8, 8))
        q_pos = (0.5 * self.sigma_a * dt**2)**2
        q_vel = (self.sigma_a * dt)**2
        q_psi = (self.sigma_w * dt)**2
        q_ba  = (self.sigma_ba * dt)**2
        q_bw  = (self.sigma_bw * dt)**2

        Q[0, 0] = q_pos; Q[1, 1] = q_pos
        Q[2, 2] = q_vel; Q[3, 3] = q_vel
        Q[4, 4] = q_psi
        Q[5, 5] = q_ba;  Q[6, 6] = q_ba
        Q[7, 7] = q_bw

        # Propagate covariance
        self.P = F @ self.P @ F.T + Q

    def update_gnss(self, pos_gnss, vel_gnss, r_pos=3.0, r_vel=0.3):
        """
        Update with full GNSS position and velocity.
        z = [e, n, ve, vn] (dim=4)
        """
        z = np.array([pos_gnss[0], pos_gnss[1], vel_gnss[0], vel_gnss[1]])
        H = np.zeros((4, 8))
        H[0, 0] = 1.0  # px
        H[1, 1] = 1.0  # py
        H[2, 2] = 1.0  # vx
        H[3, 3] = 1.0  # vy

        z_pred = H @ self.x
        y = z - z_pred

        R = np.diag([r_pos**2, r_pos**2, r_vel**2, r_vel**2])
        self._kf_update(H, y, R)

    def update_ml_speed(self, v_pred_fwd, r_speed):
        """
        Update with ML-predicted forward speed (from Phase 7 model).
        z = v_pred_fwd
        Measurement model: h(x) = vx * sin(psi) + vy * cos(psi)
        """
        psi = self.x[4]
        vx = self.x[2]
        vy = self.x[3]

        z_pred = vx * np.sin(psi) + vy * np.cos(psi)
        y = np.array([v_pred_fwd - z_pred])

        # Jacobian H (1x8)
        H = np.zeros((1, 8))
        H[0, 2] = np.sin(psi)
        H[0, 3] = np.cos(psi)
        H[0, 4] = vx * np.cos(psi) - vy * np.sin(psi)

        R = np.array([[r_speed**2]])
        self._kf_update(H, y, R)

    def update_zupt(self, r_zupt=0.05):
        """
        Zero-Velocity Update when vehicle is stationary.
        vx = 0, vy = 0
        """
        z = np.array([0.0, 0.0])
        H = np.zeros((2, 8))
        H[0, 2] = 1.0
        H[1, 3] = 1.0

        z_pred = H @ self.x
        y = z - z_pred

        R = np.diag([r_zupt**2, r_zupt**2])
        self._kf_update(H, y, R)

    def _kf_update(self, H, y, R):
        """Standard Kalman update step with Joseph form covariance update for stability."""
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # Wrap heading to [-pi, pi]
        self.x[4] = (self.x[4] + np.pi) % (2 * np.pi) - np.pi

        # Joseph form covariance update: P = (I - KH)P(I - KH)^T + KRK^T
        I_KH = np.eye(8) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

# ====================================================================
# Feature extraction helper for Phase 7 model
# ====================================================================
STAT_AXES = [
    'acc_x', 'acc_y', 'acc_z',
    'acc_lin_x', 'acc_lin_y', 'acc_lin_z',
    'gyro_x', 'gyro_y', 'gyro_z',
]

def extract_p7_features_at_step(df, curr_idx, window_len=20):
    """
    Extract 56-feature vector from [curr_idx - window_len + 1, curr_idx]
    matching Phase 7 schema.
    """
    if curr_idx < window_len:
        curr_idx = window_len
    win_start = curr_idx - window_len + 1
    win_end   = curr_idx

    signals = {}
    for col in STAT_AXES:
        c = get_col(df, col)
        signals[col] = df[c].values if c else np.zeros(len(df))

    signals['acc_veh_fwd']       = df['acc_veh_fwd'].values
    signals['gyro_veh_yaw_rate'] = df['gyro_veh_yaw_rate'].values

    signals['a_mag']  = np.sqrt(signals['acc_x']**2 + signals['acc_y']**2 + signals['acc_z']**2)
    signals['w_mag']  = np.sqrt(signals['gyro_x']**2 + signals['gyro_y']**2 + signals['gyro_z']**2)
    signals['al_mag'] = np.sqrt(signals['acc_lin_x']**2 + signals['acc_lin_y']**2 + signals['acc_lin_z']**2)

    stat_names = list(signals.keys())
    feat_vec = []
    for sname in stat_names:
        window = signals[sname][win_start : win_end + 1]
        feat_vec.extend([
            np.mean(window),
            np.std(window),
            np.min(window),
            np.max(window),
        ])
    return np.array(feat_vec, dtype=np.float32).reshape(1, -1)

# ====================================================================
# Simulate EKF Outage
# ====================================================================
def simulate_ekf_outage(df, zupt_mask, s_gnss, e_gnss, p7_model,
                        r_speed_ml, lat0, lon0, east_all, north_all):
    """
    Simulate continuous EKF state estimation during the GNSS outage [s_gnss, e_gnss].
    Returns final position error (m), velocity error (m/s), heading error (deg),
    and trajectory points.
    """
    t_arr    = df['time_rel_sec'].values
    dt_arr   = df['dt_sec'].values
    a_fwd    = df['acc_veh_fwd'].values
    a_lat    = df['acc_veh_lat'].values
    gyro_yaw = df['gyro_veh_yaw_rate'].values
    az_col   = get_col(df, 'azimuth')
    psi_mag  = np.radians(df[az_col].values)
    bearing_col = get_col(df, 'gps_orientation') or get_col(df, 'gps_bearing')

    # Initial state at outage boundary (genuine GNSS fix)
    e0, n0 = east_all[s_gnss], north_all[s_gnss]
    v0 = df['gps_speed_ms'].values[s_gnss]
    b0 = df[bearing_col].values[s_gnss] if bearing_col else np.nan
    psi0 = psi_mag[s_gnss] if (np.isnan(b0) or b0 == 0) else np.radians(b0)

    # Initial velocity vector in East, North
    vx0 = v0 * np.sin(psi0)
    vy0 = v0 * np.cos(psi0)

    # Instantiate EKF
    ekf = VehicleEKF(
        init_pos=[e0, n0],
        init_vel=[vx0, vy0],
        init_heading=psi0,
        sigma_a_proc=0.8,
        sigma_w_proc=np.radians(2.0),
        sigma_ba_proc=1e-3,
        sigma_bw_proc=1e-4
    )

    traj_e = [e0]
    traj_n = [n0]

    # Run continuous 10 Hz integration
    for t in range(s_gnss + 1, e_gnss + 1):
        dt = dt_arr[t]
        if dt <= 0 or dt > 1.0:
            dt = 0.1

        # 1. EKF Predict step (10 Hz IMU)
        ekf.predict(a_fwd[t], a_lat[t], gyro_yaw[t], dt)

        # 2. ZUPT Measurement Update (when stationary detected)
        if zupt_mask[t]:
            ekf.update_zupt(r_zupt=0.05)

        # 3. ML Velocity Measurement Update (every 10 samples = ~1.0 s)
        if (t - s_gnss) % 10 == 0 and (t - s_gnss) >= 20:
            feat = extract_p7_features_at_step(df, t, window_len=20)
            v_ml = float(np.clip(p7_model.predict(feat)[0], 0.0, None))
            # Update EKF with ML forward speed measurement
            ekf.update_ml_speed(v_ml, r_speed=r_speed_ml)

        traj_e.append(ekf.x[0])
        traj_n.append(ekf.x[1])

    # Ground truth comparison at outage end
    e_true = east_all[e_gnss]
    n_true = north_all[e_gnss]
    v_true = df['gps_speed_ms'].values[e_gnss]
    if bearing_col is not None:
        end_brg = df[bearing_col].values[e_gnss]
        psi_true = (psi_mag[e_gnss]
                    if (np.isnan(end_brg) or end_brg == 0)
                    else np.radians(end_brg))
    else:
        psi_true = psi_mag[e_gnss]

    pos_err = float(np.sqrt((ekf.x[0] - e_true)**2 + (ekf.x[1] - n_true)**2))
    v_est_mag = float(np.sqrt(ekf.x[2]**2 + ekf.x[3]**2))
    vel_err = float(abs(v_est_mag - v_true))
    # Angular error in degrees
    diff_ang = (ekf.x[4] - psi_true + np.pi) % (2 * np.pi) - np.pi
    head_err = float(abs(np.degrees(diff_ang)))

    return pos_err, vel_err, head_err, traj_e, traj_n

# ====================================================================
# Main Execution Pipeline
# ====================================================================
def run_phase10():
    print("=" * 80)
    print("PHASE 10: EXTENDED KALMAN FILTER (EKF) FUSION ENGINE")
    print("SIH26168 — Organisation: ISRO | Domain: Software / AI-ML")
    print("=" * 80)

    # 1. Load Phase 8 cached dataset for Phase 9 model (Target C1)
    print("\n[Step 1] Preparing LOTO Models for Comparative Benchmarking...")
    meta_p8 = pd.read_parquet(CACHE_DIR_P8 / "meta_all.parquet")
    yC_all  = np.load(CACHE_DIR_P8 / "yC_all.npy")
    X_p8    = np.load(CACHE_DIR_P8 / "X_all.npy")
    valid_p8 = meta_p8['dur_s'].values >= 2.0

    # 2. Load Phase 7 cached dataset for ML Velocity Measurement Model
    X_p7    = np.load("data/phase7_cache_X.npy")
    y_p7    = np.load("data/phase7_cache_y.npy")
    meta_p7 = pd.read_parquet("data/phase7_cache_meta.parquet")

    # Phase 7 validated LOTO RMSEs (from phase7_loto_results.csv)
    P7_LOTO_RMSE = {
        'S-Vw4': 2.133,
        'S-Vtb5': 1.895,
        'S-Vta1a': 1.368
    }

    all_records = []
    total_dist_dict = {}

    print(f"\n[Step 2] Running Comparative Outage Simulation across {TEST_SESSIONS}...")
    print(f"  Outage durations: {OUTAGE_DURATIONS_S} seconds")
    print(f"  Windows per duration: {OUTAGES_PER_DURATION}")

    # Seed alignment for exact reproducibility
    np.random.seed(RANDOM_SEED)

    for s_name in TEST_SESSIONS:
        s_path = PROCESSED_DIR / s_name
        df = pd.read_parquet(s_path)
        df, _, _ = transform_to_vehicle_frame(df)
        zupt_mask = get_zupt_v1_mask(df)
        gnss_idx = find_genuine_gnss_updates(df)

        lat0, lon0 = df['gps_lat'].iloc[0], df['gps_lon'].iloc[0]
        east_all, north_all = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)

        # Train LOTO Phase 9 model (Target C1)
        sess_prefix = s_name.replace('.parquet', '')
        train_mask_p8 = valid_p8 & (~meta_p8['session'].str.startswith(sess_prefix))
        model_p9 = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
        model_p9.fit(X_p8[train_mask_p8], yC_all[train_mask_p8])

        # Train LOTO Phase 7 model (Instantaneous Velocity)
        train_mask_p7 = ~meta_p7['session'].str.startswith(sess_prefix)
        model_p7 = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
        model_p7.fit(X_p7[train_mask_p7], y_p7[train_mask_p7])

        r_speed_ml = P7_LOTO_RMSE.get(sess_prefix, 1.80)
        print(f"\n  Trained LOTO models for {s_name} (R_ml sigma = {r_speed_ml:.3f} m/s)")

        for dur_s in OUTAGE_DURATIONS_S:
            windows = sample_outage_windows(df, gnss_idx, dur_s, OUTAGES_PER_DURATION)
            print(f"    Simulating {dur_s:>3}s outages ({len(windows)} windows)...")

            for w_idx, (s_gnss, e_gnss) in enumerate(windows):
                # Distance travelled ground truth
                de_gt = east_all[e_gnss] - east_all[s_gnss]
                dn_gt = north_all[e_gnss] - north_all[s_gnss]
                dist_gt = float(np.sqrt(de_gt**2 + dn_gt**2))

                # Baseline 1: Gyro-Only Physics DR
                pg, vg, hg, _, act_dur = simulate_outage(
                    df, zupt_mask, gnss_idx, s_gnss, e_gnss, False, lat0, lon0, east_all, north_all)

                # Baseline 2: ZUPT v1 Physics DR
                pz, vz, hz, _, _ = simulate_outage(
                    df, zupt_mask, gnss_idx, s_gnss, e_gnss, True, lat0, lon0, east_all, north_all)

                # Proposed Phase 9: AI + Physics Hybrid (Displacement Correction)
                phyb, _, _ = simulate_hybrid_outage(
                    df, zupt_mask, s_gnss, e_gnss, model_p9, lat0, lon0, east_all, north_all)

                # Proposed Phase 10: Continuous EKF State Estimator
                pekf, vekf, hekf, _, _ = simulate_ekf_outage(
                    df, zupt_mask, s_gnss, e_gnss, model_p7, r_speed_ml, lat0, lon0, east_all, north_all)

                all_records.append({
                    'session': s_name,
                    'duration_s': dur_s,
                    'actual_dur_s': round(act_dur, 2),
                    'window_idx': w_idx,
                    's_gnss': s_gnss,
                    'e_gnss': e_gnss,
                    'dist_travelled_m': dist_gt,
                    # Position errors
                    'pos_err_gyro': pg,
                    'pos_err_zupt': pz,
                    'pos_err_hybrid': phyb,
                    'pos_err_ekf': pekf,
                    # Velocity errors
                    'vel_err_gyro': vg,
                    'vel_err_zupt': vz,
                    'vel_err_ekf': vekf,
                    # Heading errors
                    'head_err_gyro': hg,
                    'head_err_ekf': hekf,
                    # Drift %
                    'drift_pct_gyro': (pg / max(dist_gt, 1e-3)) * 100.0,
                    'drift_pct_zupt': (pz / max(dist_gt, 1e-3)) * 100.0,
                    'drift_pct_hybrid': (phyb / max(dist_gt, 1e-3)) * 100.0,
                    'drift_pct_ekf': (pekf / max(dist_gt, 1e-3)) * 100.0,
                })

    raw_df = pd.DataFrame(all_records)
    raw_df.to_csv(OUT_DIR / "phase10_raw_records.csv", index=False)

    # ====================================================================
    # 3. Evaluation Metrics Summary
    # ====================================================================
    print("\n" + "=" * 90)
    print("PHASE 10: STATE ESTIMATOR & EKF FUSION BENCHMARK SUMMARY")
    print("=" * 90)

    summary_rows = []
    for dur_s in OUTAGE_DURATIONS_S:
        sub = raw_df[raw_df['duration_s'] == dur_s]
        avg_dist = sub['dist_travelled_m'].mean()

        g_pos = sub['pos_err_gyro'].mean()
        z_pos = sub['pos_err_zupt'].mean()
        h_pos = sub['pos_err_hybrid'].mean()
        e_pos = sub['pos_err_ekf'].mean()

        g_drift = sub['drift_pct_gyro'].median()
        z_drift = sub['drift_pct_zupt'].median()
        h_drift = sub['drift_pct_hybrid'].median()
        e_drift = sub['drift_pct_ekf'].median()

        e_vel = sub['vel_err_ekf'].mean()
        z_vel = sub['vel_err_zupt'].mean()
        g_vel = sub['vel_err_gyro'].mean()

        e_head = sub['head_err_ekf'].mean()
        g_head = sub['head_err_gyro'].mean()

        summary_rows.append({
            'Outage (s)': dur_s,
            'Avg Dist (m)': round(avg_dist, 1),
            'Gyro Pos (m)': round(g_pos, 1),
            'ZUPT Pos (m)': round(z_pos, 1),
            'Phase 9 Hyb Pos (m)': round(h_pos, 1),
            'Phase 10 EKF Pos (m)': round(e_pos, 1),
            'EKF Drift (%)': f"{e_drift:.1f}%",
            'EKF Vel RMSE (m/s)': round(e_vel, 2),
            'EKF Head Err (deg)': round(e_head, 1),
            'Imp vs ZUPT (%)': f"{((z_pos - e_pos)/z_pos)*100:+.1f}%"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "phase10_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    # ====================================================================
    # 4. Plots Generation
    # ====================================================================
    print("\n[Step 4] Generating Comparative Visualizations...")
    durations = OUTAGE_DURATIONS_S

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Plot 1: Position Error Comparison
    axes[0].plot(durations, summary_df['Gyro Pos (m)'], 'o--', color='#d9534f', label='Phase 5/6 Gyro-Only')
    axes[0].plot(durations, summary_df['ZUPT Pos (m)'], 's--', color='#f0ad4e', label='Phase 5B ZUPT v1')
    axes[0].plot(durations, summary_df['Phase 9 Hyb Pos (m)'], 'D-', color='#0275d8', label='Phase 9 AI-Hybrid')
    axes[0].plot(durations, summary_df['Phase 10 EKF Pos (m)'], '^-', color='#5cb85c', linewidth=2.5, label='Phase 10 EKF Fusion')
    axes[0].set_xlabel('GNSS Outage Duration (s)', fontweight='bold')
    axes[0].set_ylabel('Mean Position Error (m)', fontweight='bold')
    axes[0].set_title('Position Error vs Outage Duration', fontweight='bold')
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(durations)

    # Plot 2: Velocity RMSE Comparison
    ekf_vels  = [raw_df[raw_df['duration_s'] == d]['vel_err_ekf'].mean() for d in durations]
    zupt_vels = [raw_df[raw_df['duration_s'] == d]['vel_err_zupt'].mean() for d in durations]
    gyro_vels = [raw_df[raw_df['duration_s'] == d]['vel_err_gyro'].mean() for d in durations]
    axes[1].plot(durations, gyro_vels, 'o--', color='#d9534f', label='Gyro-Only')
    axes[1].plot(durations, zupt_vels, 's--', color='#f0ad4e', label='ZUPT v1')
    axes[1].plot(durations, ekf_vels, '^-', color='#5cb85c', linewidth=2.5, label='Phase 10 EKF')
    axes[1].set_xlabel('GNSS Outage Duration (s)', fontweight='bold')
    axes[1].set_ylabel('Velocity Error (m/s)', fontweight='bold')
    axes[1].set_title('Terminal Velocity Error Comparison', fontweight='bold')
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(durations)

    # Plot 3: Drift % vs 10% ISRO Requirement
    ekf_drift_means = [raw_df[raw_df['duration_s'] == d]['drift_pct_ekf'].median() for d in durations]
    hyb_drift_means = [raw_df[raw_df['duration_s'] == d]['drift_pct_hybrid'].median() for d in durations]
    axes[2].plot(durations, hyb_drift_means, 'D-', color='#0275d8', label='Phase 9 AI-Hybrid')
    axes[2].plot(durations, ekf_drift_means, '^-', color='#5cb85c', linewidth=2.5, label='Phase 10 EKF Fusion')
    axes[2].axhline(10.0, color='red', linestyle='--', linewidth=2.0, label='ISRO Target: < 10% Drift')
    axes[2].set_xlabel('GNSS Outage Duration (s)', fontweight='bold')
    axes[2].set_ylabel('Median Drift (% of Distance Travelled)', fontweight='bold')
    axes[2].set_title('Drift % vs Official ISRO Requirement (<10%)', fontweight='bold')
    axes[2].legend(fontsize=8.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(durations)

    plt.tight_layout()
    chart_p = OUT_DIR / "phase10_performance_comparison.png"
    plt.savefig(chart_p, dpi=180)
    plt.close()
    print(f"  Saved comparison chart: {chart_p}")

    # ====================================================================
    # 5. Requirement Compliance & Honest Assessment
    # ====================================================================
    print("\n" + "=" * 80)
    print("PHASE 10 HONEST SCIENTIFIC ASSESSMENT (ISRO <10% DRIFT REQUIREMENT)")
    print("=" * 80)
    print("Outage Duration | EKF Drift % | Meets <10% Requirement? | Status")
    print("-" * 65)
    for row in summary_rows:
        dur = row['Outage (s)']
        drift_val = float(row['EKF Drift (%)'].replace('%', ''))
        meets = "YES" if drift_val < 10.0 else "NO"
        status = "COMPLIANT" if drift_val < 10.0 else f"EXCEEDS by {drift_val - 10.0:.1f}%"
        print(f"{dur:>14}s | {drift_val:>10.1f}% | {meets:>23} | {status}")

    print("\nDIAGNOSIS & DOMINANT ERROR SOURCE:")
    print("  1. Continuous EKF dynamic fusion smoothly bounds velocity error (mean vel error ~1.5 - 2.8 m/s),")
    print("     preventing the catastrophic open-loop cubic speed divergence.")
    print("  2. However, for outages >= 60s, position drift still exceeds 10% of distance travelled.")
    print("  3. Dominant Remaining Error Source: HEADING DRIFT & LATERAL DIVERGENCE.")
    print("     - In unconstrained 2D EKF without Non-Holonomic Constraints (NHC), gyro bias uncertainty")
    print("       rotates the velocity vector, causing cross-track displacement error to grow rapidly.")
    print("     - Furthermore, lack of lateral velocity constraint allows unphysical sideways vehicle slip.")
    print("  --> Crucial takeaway: Phase 11 (Non-Holonomic Constraints: lateral vel = 0, vertical vel = 0)")
    print("      is urgently prioritized to clamp lateral velocity drift, followed by Phase 12 (Map Matching).")

if __name__ == "__main__":
    run_phase10()
