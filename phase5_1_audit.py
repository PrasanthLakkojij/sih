"""
Phase 5.1 — Numerical Sanity Audit & Precision Diagnosis
SIH26168 — AI-ML Dead Reckoning System

Investigates the 5 key factors behind velocity / position calculation:
1. dt values: min, max, median, mean, std, unit check (seconds vs ms).
2. a_fwd values: min, max, mean, bias offset in m/s^2.
   Compute: If a_fwd has mean bias b_a, v(T) = b_a * T.
   For S-Vw4 (T = 12,652 s), if b_a = +0.09 m/s^2 -> v(T) = 0.09 * 12652 = 1,138 m/s!
3. GPS speed units: verify km/h vs m/s in dataframe.
4. Heading angles: verify radians in sin/cos and verify gyro in rad/s.
5. Zero-Velocity Detection (ZUPT / Stationary bias correction):
   When vehicle is stationary (GPS speed == 0), measure residual a_fwd bias.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data") / "processed_sessions"

def audit_session(s_name):
    s_path = PROCESSED_DIR / s_name
    if not s_path.exists():
        print(f"File not found: {s_path}")
        return
        
    df = pd.read_parquet(s_path)
    from phase4_orientation import transform_to_vehicle_frame
    df_trans, theta_rad, corr = transform_to_vehicle_frame(df)
    
    print("=" * 80)
    print(f"AUDIT REPORT FOR: {s_name}")
    print("=" * 80)
    
    # 1. dt audit
    dt = df_trans['dt_sec'].values
    t_rel = df_trans['time_rel_sec'].values
    T_total = t_rel[-1]
    print(f"1. TIME & DT AUDIT:")
    print(f"   Total Duration:    {T_total:.1f} s ({T_total/60:.1f} min)")
    print(f"   Number of Samples: {len(dt):,d}")
    print(f"   dt min:            {np.min(dt):.6f} s")
    print(f"   dt max:            {np.max(dt):.6f} s")
    print(f"   dt median:         {np.median(dt):.6f} s")
    print(f"   dt mean:           {np.mean(dt):.6f} s")
    print(f"   dt std:            {np.std(dt):.6f} s")
    print(f"   Sum(dt):           {np.sum(dt):.1f} s (matches T_total? {abs(np.sum(dt) - T_total) < 1.0})")
    
    # 2. Forward Acceleration Audit
    a_fwd = df_trans['acc_veh_fwd'].values
    a_mean = np.mean(a_fwd)
    a_std = np.std(a_fwd)
    a_min = np.min(a_fwd)
    a_max = np.max(a_fwd)
    print(f"\n2. ACCELERATION AUDIT:")
    print(f"   a_fwd min:         {a_min:.4f} m/s^2")
    print(f"   a_fwd max:         {a_max:.4f} m/s^2")
    print(f"   a_fwd std:         {a_std:.4f} m/s^2")
    print(f"   a_fwd mean (bias): {a_mean:.6f} m/s^2")
    
    # Calculate pure integration of mean bias
    v_bias_predicted = a_mean * T_total
    print(f"   --> Predicted terminal velocity purely from mean bias (a_mean * T): {v_bias_predicted:.1f} m/s")
    
    # 3. GPS Speed Unit Audit
    v_gps_ms = df_trans['gps_speed_ms'].values
    v_gps_kmh = df_trans['gps_speed_kmh'].values
    print(f"\n3. GPS SPEED AUDIT:")
    print(f"   gps_speed_kmh max: {np.max(v_gps_kmh):.2f} km/h")
    print(f"   gps_speed_ms max:  {np.max(v_gps_ms):.2f} m/s")
    print(f"   Ratio (kmh / ms):  {np.max(v_gps_kmh) / max(0.001, np.max(v_gps_ms)):.4f} (expected 3.6000)")
    
    # 4. Stationary Period / ZUPT Analysis
    # Periods where GPS speed == 0
    zero_speed_mask = v_gps_ms < 0.2
    if zero_speed_mask.sum() > 50:
        a_fwd_stationary = a_fwd[zero_speed_mask]
        bias_stationary = np.mean(a_fwd_stationary)
        std_stationary = np.std(a_fwd_stationary)
        print(f"\n4. STATIONARY (ZERO-VELOCITY) BIAS AUDIT:")
        print(f"   Stationary Samples: {zero_speed_mask.sum():,d} ({zero_speed_mask.mean()*100:.1f}% of trip)")
        print(f"   Stationary a_fwd Bias: {bias_stationary:.6f} m/s^2 (std: {std_stationary:.4f})")
    else:
        print(f"\n4. STATIONARY AUDIT: No prolonged zero-speed periods in this file.")
        
    # 5. Raw Integration vs Bias-Subtracted Integration
    v_raw = np.zeros(len(df_trans))
    v_raw[0] = v_gps_ms[0]
    for i in range(1, len(df_trans)):
        v_raw[i] = max(0.0, v_raw[i-1] + a_fwd[i] * dt[i])
        
    print(f"\n5. VELOCITY INTEGRATION COMPARISON:")
    print(f"   v_raw[0]:          {v_raw[0]:.2f} m/s")
    print(f"   v_raw[-1]:         {v_raw[-1]:.2f} m/s")
    print(f"   v_raw max:         {np.max(v_raw):.2f} m/s")
    print(f"   v_gps mean:        {np.mean(v_gps_ms):.2f} m/s")
    print(f"   v_raw RMSE vs GPS: {np.sqrt(np.mean((v_raw - v_gps_ms)**2)):.2f} m/s")
    
    # If stationary bias is removed:
    if zero_speed_mask.sum() > 50:
        a_fwd_corrected = a_fwd - bias_stationary
        v_corr = np.zeros(len(df_trans))
        v_corr[0] = v_gps_ms[0]
        for i in range(1, len(df_trans)):
            v_corr[i] = max(0.0, v_corr[i-1] + a_fwd_corrected[i] * dt[i])
        print(f"\n   [Hypothetical] With Stationary Bias ({bias_stationary:+.4f} m/s^2) Removed:")
        print(f"   v_corr[-1]:        {v_corr[-1]:.2f} m/s")
        print(f"   v_corr max:        {np.max(v_corr):.2f} m/s")
        print(f"   v_corr RMSE:       {np.sqrt(np.mean((v_corr - v_gps_ms)**2)):.2f} m/s")

def main():
    test_files = ["S-Vta10.parquet", "S-Vtb2.parquet", "S-Vta1a.parquet", "S-Vtb5.parquet", "S-Vw4.parquet"]
    for tf in test_files:
        audit_session(tf)

if __name__ == "__main__":
    main()
