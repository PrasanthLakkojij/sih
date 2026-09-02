"""
Phase 4 — Orientation & Coordinate Transformation
SIH26168 — AI-ML Dead Reckoning System

Goal:
1. Update manifest with explicit split reasons (split_reason column).
2. Investigate and validate the Phone -> Vehicle Frame transformation:
     a_vehicle = R_{phone -> vehicle} * a_phone_linear
   where vehicle frame is defined as:
     - Forward (X_v): Direction of vehicle forward motion (aligned with velocity when moving forward)
     - Lateral (Y_v): Vehicle right / sideways
     - Vertical (Z_v): Vehicle up (opposite gravity direction)

Investigation steps:
1. Vertical Alignment via Gravity:
   - Unit gravity vector: u_g = g_phone / ||g_phone||
   - Vertical axis in phone frame: z_p = -u_g (pointing Up)
2. Forward Heading / Horizontal Plane Projection:
   - Project linear acceleration onto horizontal plane: a_horiz = a_lin - (a_lin . z_p) * z_p
3. Alignment Estimation:
   - Method A: Sensor-reported Euler angles (ori_azimuth, ori_pitch, ori_roll) to Navigation/ENU frame.
   - Method B: Kinematic Alignment via GPS velocity during non-outage driving:
     When vehicle accelerates forward (GPS speed increasing), a_forward aligns with vehicle forward vector.
     Cross-correlation between GPS acceleration (d/dt GPS speed) and phone horizontal axes.
4. Validation:
   - Check if transformed forward acceleration positive during braking/accelerating.
   - Check if lateral acceleration is zero when driving straight and spikes during turns (gyro_yaw != 0).
   - Check if vertical acceleration is close to zero.
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
OUT_DIR = Path("plots") / "phase4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# 1. Update manifest with explicit split reasons
# -------------------------------------------------------------------
def update_manifest_audit():
    manifest_path = PROCESSED_DIR / "manifest.csv"
    if manifest_path.exists():
        df_man = pd.read_csv(manifest_path)
        # Determine split reason
        reasons = []
        for _, row in df_man.iterrows():
            sid = str(row['session_id'])
            if "sub" in sid:
                if "S-Vtb1" in sid:
                    reasons.append("large_gap_pause_and_duplicates")
                else:
                    reasons.append("timestamp_reset_negative_dt")
            else:
                reasons.append("single_uninterrupted_session")
        df_man['split_reason'] = reasons
        df_man.to_csv(manifest_path, index=False)
        print("Updated manifest.csv with split_reason audit column.")

# -------------------------------------------------------------------
# 2. Rotation & Orientation Mathematics
# -------------------------------------------------------------------
def euler_to_rot_matrix(azimuth_deg, pitch_deg, roll_deg):
    """
    Convert Android orientation angles (azimuth, pitch, roll in degrees)
    to rotation matrix R_{phone -> ENU/Nav}.
    Android convention:
      - Azimuth: angle around Z axis (compass heading 0=North, 90=East)
      - Pitch: angle around X axis (-180 to 180)
      - Roll: angle around Y axis (-90 to 90)
    """
    az = np.radians(azimuth_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    
    # Rotation matrices (Z-X-Y intrinsic convention used in Android SensorManager)
    # R = Rz(azimuth) * Rx(pitch) * Ry(roll)
    c_az, s_az = np.cos(az), np.sin(az)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_r, s_r = np.cos(roll), np.sin(roll)
    
    # Pre-allocate array of 3x3 matrices
    N = len(azimuth_deg)
    R = np.zeros((N, 3, 3))
    
    for i in range(N):
        Rz = np.array([[c_az[i], -s_az[i], 0],
                       [s_az[i],  c_az[i], 0],
                       [0,        0,       1]])
        Rx = np.array([[1, 0,        0],
                       [0, c_p[i],  -s_p[i]],
                       [0, s_p[i],   c_p[i]]])
        Ry = np.array([[ c_r[i], 0, s_r[i]],
                       [ 0,      1, 0],
                       [-s_r[i], 0, c_r[i]]])
        R[i] = Rz @ Rx @ Ry
    return R

def compute_kinematic_alignment(df, min_speed_ms=3.0):
    """
    Determine the static mounting angle (yaw offset theta) between phone horizontal axes
    and vehicle forward axis using GPS speed changes.
    When vehicle accelerates (dv/dt > 0), a_forward in vehicle frame is positive.
    """
    # Use moving segments
    moving_mask = df['gps_speed_ms'] > min_speed_ms
    if moving_mask.sum() < 100:
        return 0.0, 0.0 # fallback (theta, corr)
        
    ax_lin = df['acc_lin_x'].values
    ay_lin = df['acc_lin_y'].values
    
    # Differentiate GPS speed for ground truth longitudinal acceleration
    t = df['time_rel_sec'].values
    v_gps = df['gps_speed_ms'].values
    
    # Smooth GPS acceleration with central diff
    a_gps = np.gradient(v_gps, t)
    
    # Test candidate mounting yaw angles from -180 to 180 degrees (step 1 deg)
    angles = np.linspace(-np.pi, np.pi, 361)
    corrs = []
    
    # Valid filter: vehicle actively accelerating or braking (|a_gps| > 0.3 m/s^2)
    accel_mask = moving_mask & (np.abs(a_gps) > 0.3)
    if accel_mask.sum() < 50:
        accel_mask = moving_mask
        
    a_gps_sub = a_gps[accel_mask]
    ax_sub = ax_lin[accel_mask]
    ay_sub = ay_lin[accel_mask]
    
    for theta in angles:
        # a_forward_candidate = ax * cos(theta) + ay * sin(theta)
        a_fwd_cand = ax_sub * np.cos(theta) + ay_sub * np.sin(theta)
        # Compute correlation with GPS longitudinal acceleration
        if np.std(a_fwd_cand) > 1e-4 and np.std(a_gps_sub) > 1e-4:
            corr = np.corrcoef(a_fwd_cand, a_gps_sub)[0, 1]
        else:
            corr = 0.0
        corrs.append(corr)
        
    best_idx = np.argmax(corrs)
    best_theta_rad = angles[best_idx]
    best_corr = corrs[best_idx]
    
    return best_theta_rad, best_corr

def transform_to_vehicle_frame(df):
    """
    Transforms phone-frame IMU into Vehicle Coordinate Frame:
      X_v: Forward (+ = vehicle accelerating forward)
      Y_v: Lateral (+ = vehicle turning left / lateral right)
      Z_v: Vertical (+ = upward, orthogonal to road)
    """
    # 1. Phone gravity vector gives the vertical orientation
    gx, gy, gz = df['grav_x'].values, df['grav_y'].values, df['grav_z'].values
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    # Unit vertical vector in phone coordinates (pointing UP)
    up_x = -gx / g_norm
    up_y = -gy / g_norm
    up_z = -gz / g_norm
    
    # 2. Linear acceleration components in phone body
    ax_lin, ay_lin, az_lin = df['acc_lin_x'].values, df['acc_lin_y'].values, df['acc_lin_z'].values
    
    # Vertical acceleration: a_vert = a_lin . up
    a_vert = ax_lin * up_x + ay_lin * up_y + az_lin * up_z
    
    # Remove vertical component to get purely horizontal acceleration vector in phone frame
    ax_horiz = ax_lin - a_vert * up_x
    ay_horiz = ay_lin - a_vert * up_y
    az_horiz = az_lin - a_vert * up_z
    
    # 3. Find mounting yaw offset theta to resolve Forward vs Lateral
    best_theta, corr = compute_kinematic_alignment(df)
    
    # Forward & Lateral acceleration in vehicle frame
    # Since phone is flat (up_z ~ 1), horizontal plane is spanned primarily by (X_phone, Y_phone)
    a_fwd = ax_horiz * np.cos(best_theta) + ay_horiz * np.sin(best_theta)
    a_lat = -ax_horiz * np.sin(best_theta) + ay_horiz * np.cos(best_theta)
    
    # Vehicle yaw rate (angular rate around vertical Up axis)
    wx, wy, wz = df['gyro_x'].values, df['gyro_y'].values, df['gyro_z'].values
    gyro_yaw = wx * up_x + wy * up_y + wz * up_z
    
    df_transformed = df.copy()
    df_transformed['acc_veh_fwd'] = a_fwd
    df_transformed['acc_veh_lat'] = a_lat
    df_transformed['acc_veh_vert'] = a_vert
    df_transformed['gyro_veh_yaw_rate'] = gyro_yaw
    df_transformed['mounting_yaw_deg'] = np.degrees(best_theta)
    df_transformed['mounting_alignment_corr'] = corr
    
    return df_transformed, best_theta, corr

# -------------------------------------------------------------------
# 3. Validation & Visual Analysis
# -------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PHASE 4: ORIENTATION & COORDINATE TRANSFORMATION")
    print("=" * 80)
    
    update_manifest_audit()
    
    # Test on key representative driving sessions
    test_sessions = [
        "S-Vta1a.parquet",
        "S-Vtb5.parquet",
        "S-Vw4.parquet",
        "S-Vta10.parquet",
        "S-Vtb2.parquet"
    ]
    
    results = []
    
    for s_name in test_sessions:
        s_path = PROCESSED_DIR / s_name
        if not s_path.exists():
            continue
        df = pd.read_parquet(s_path)
        
        df_trans, theta_rad, corr = transform_to_vehicle_frame(df)
        
        theta_deg = np.degrees(theta_rad)
        fwd_mean = df_trans['acc_veh_fwd'].mean()
        lat_mean = df_trans['acc_veh_lat'].mean()
        vert_mean = df_trans['acc_veh_vert'].mean()
        vert_std = df_trans['acc_veh_vert'].std()
        
        results.append({
            'session': s_name,
            'duration_min': round(df_trans['time_rel_sec'].iloc[-1] / 60.0, 1),
            'mounting_yaw_deg': round(theta_deg, 2),
            'gps_accel_corr': round(corr, 3),
            'a_fwd_mean': round(fwd_mean, 4),
            'a_lat_mean': round(lat_mean, 4),
            'a_vert_mean': round(vert_mean, 4),
            'a_vert_std': round(vert_std, 4)
        })
        
        # Plot validation window for the first 3 sessions
        if s_name in ["S-Vta1a.parquet", "S-Vtb5.parquet", "S-Vw4.parquet"]:
            plot_transformation_validation(df_trans, s_name, theta_deg, corr)
            
    res_df = pd.DataFrame(results)
    print("\n--- Coordinate Transformation Alignment Results ---")
    print(res_df.to_string(index=False))
    
    # Save transformed sample
    sample_out = PROCESSED_DIR / "sample_transformed_S-Vta1a.parquet"
    df_trans.to_parquet(sample_out, index=False)
    print(f"\nTransformed sample saved to: {sample_out}")
    print(f"Validation plots saved to: {OUT_DIR.resolve()}")

def plot_transformation_validation(df, s_name, theta_deg, corr):
    # 120-second moving window
    moving_idx = np.where(df['gps_speed_ms'] > 4.0)[0]
    if len(moving_idx) > 1200:
        start_i = moving_idx[len(moving_idx)//2]
    else:
        start_i = len(df) // 4
    end_i = min(start_i + 1200, len(df))
    
    sub = df.iloc[start_i:end_i].copy()
    t = sub['time_rel_sec'].values - sub['time_rel_sec'].iloc[0]
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    
    # 1. GPS Speed & GPS Acceleration
    a_gps = np.gradient(sub['gps_speed_ms'].values, sub['time_rel_sec'].values)
    axes[0].plot(t, sub['gps_speed_ms'] * 3.6, 'k-', label='GPS Speed (km/h)')
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(t, a_gps, 'g--', alpha=0.7, label='GPS a_long (m/s²)')
    axes[0].set_ylabel('Speed (km/h)')
    ax0_twin.set_ylabel('GPS Accel (m/s²)')
    axes[0].set_title(f'Vehicle Kinematics & Coordinate Transformation — {s_name} (Mounting Yaw: {theta_deg:.1f}°, Corr: {corr:.2f})')
    axes[0].legend(loc='upper left', fontsize=8)
    ax0_twin.legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Transformed Vehicle Forward Accel vs GPS Accel
    axes[1].plot(t, sub['acc_veh_fwd'], 'b-', linewidth=0.8, label='Transformed Forward Accel (Vehicle X)')
    axes[1].plot(t, a_gps, 'g--', linewidth=1.2, alpha=0.6, label='GPS Accel reference')
    axes[1].set_ylabel('m/s²')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Transformed Lateral Accel & Yaw Rate (Turning dynamics)
    axes[2].plot(t, sub['acc_veh_lat'], 'm-', linewidth=0.8, label='Transformed Lateral Accel (Vehicle Y)')
    ax2_twin = axes[2].twinx()
    ax2_twin.plot(t, np.degrees(sub['gyro_veh_yaw_rate']), 'r-', alpha=0.6, label='Vehicle Yaw Rate (deg/s)')
    axes[2].set_ylabel('Lateral a (m/s²)')
    ax2_twin.set_ylabel('Yaw Rate (deg/s)')
    axes[2].legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # 4. Vertical Accel (should be centered at zero with low road-bump vibration)
    axes[3].plot(t, sub['acc_veh_vert'], 'gray', linewidth=0.6, label='Transformed Vertical Accel (Vehicle Z)')
    axes[3].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[3].set_ylabel('Vertical a (m/s²)')
    axes[3].set_xlabel('Time in window (s)')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = OUT_DIR / f"transformation_validation_{Path(s_name).stem}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Validation plot saved: {out_file}")

if __name__ == "__main__":
    main()
