import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from phase6_outage_sim import (
    sample_outage_windows, find_genuine_gnss_updates, get_zupt_v1_mask,
    latlon_to_enu, get_col, simulate_outage,
    TEST_SESSIONS, OUTAGE_DURATIONS_S, OUTAGES_PER_DURATION
)
from phase4_orientation import transform_to_vehicle_frame
from phase8_ml_correction import segment_features

# Load cached data
meta_all = pd.read_parquet('data/phase8_cache/meta_all.parquet')
yC_all = np.load('data/phase8_cache/yC_all.npy')
X_all = np.load('data/phase8_cache/X_all.npy')

# Safety fix: tighten MIN_T filter to >= 2.0s
valid_mask = meta_all['dur_s'].values >= 2.0
print(f"Total cached segments: {len(meta_all)}, after MIN_T >= 2.0s filter: {valid_mask.sum()}")

MIN_T_INFERENCE = 2.0

np.random.seed(42)

all_results = []

for s_name in TEST_SESSIONS:
    df = pd.read_parquet(f"data/processed_sessions/{s_name}")
    df, _, _ = transform_to_vehicle_frame(df)
    zupt_mask = get_zupt_v1_mask(df)
    gnss_idx = find_genuine_gnss_updates(df)
    lat0, lon0 = df['gps_lat'].iloc[0], df['gps_lon'].iloc[0]
    east_all, north_all = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)
    
    t_arr = df['time_rel_sec'].values
    a_fwd = df['acc_veh_fwd'].values
    dt_arr = df['dt_sec'].values
    gyro_yaw = df['gyro_veh_yaw_rate'].values
    az_col = get_col(df, 'azimuth')
    psi_mag = np.radians(df[az_col].values)
    bearing_col = get_col(df, 'gps_orientation') or get_col(df, 'gps_bearing')
    
    # Train LOTO model for this test session (train on all other sessions with dur >= 2.0s)
    session_prefix = s_name.replace('.parquet', '')
    train_mask = valid_mask & (~meta_all['session'].str.startswith(session_prefix))
    print(f"\nTraining C1 model for {s_name} (train on {train_mask.sum()} segments from other trips)...")
    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         n_jobs=-1, random_state=42, verbosity=0)
    model.fit(X_all[train_mask], yC_all[train_mask])
    
    for dur_s in OUTAGE_DURATIONS_S:
        windows = sample_outage_windows(df, gnss_idx, dur_s, OUTAGES_PER_DURATION)
        for w_idx, (s_gnss, e_gnss) in enumerate(windows):
            # Baselines from Phase 6
            pg, vg, hg, _, act_dur = simulate_outage(df, zupt_mask, gnss_idx, s_gnss, e_gnss, False, lat0, lon0, east_all, north_all)
            pz, vz, hz, _, _ = simulate_outage(df, zupt_mask, gnss_idx, s_gnss, e_gnss, True, lat0, lon0, east_all, north_all)
            
            # AI + Physics Hybrid
            e0, n0 = east_all[s_gnss], north_all[s_gnss]
            e_true, n_true = east_all[e_gnss], north_all[e_gnss]
            v0 = df['gps_speed_ms'].values[s_gnss]
            b0 = df[bearing_col].values[s_gnss] if bearing_col else np.nan
            psi0 = psi_mag[s_gnss] if (np.isnan(b0) or b0 == 0) else np.radians(b0)
            
            # Divide into ~9s intervals (90 samples at 10 Hz)
            intervals = []
            curr = s_gnss
            while curr < e_gnss:
                nxt = min(curr + 90, e_gnss)
                if nxt < e_gnss and (t_arr[e_gnss] - t_arr[nxt]) < MIN_T_INFERENCE:
                    nxt = e_gnss
                intervals.append((curr, nxt))
                curr = nxt
                
            e_hyb, n_hyb = e0, n0
            v_curr, psi_curr = v0, psi0
            
            for ia, ib in intervals:
                de_sub, dn_sub = 0.0, 0.0
                v_step = v_curr
                psi_step = psi_curr
                for t in range(ia + 1, ib + 1):
                    dt = dt_arr[t]
                    if zupt_mask[t]: v_step = 0.0
                    else: v_step = max(0.0, v_step + a_fwd[t] * dt)
                    psi_step += gyro_yaw[t] * dt
                    de_sub += v_step * np.sin(psi_step) * dt
                    dn_sub += v_step * np.cos(psi_step) * dt
                
                ds_imu = np.sqrt(de_sub**2 + dn_sub**2)
                h_dir = np.arctan2(de_sub, dn_sub) if ds_imu > 1e-3 else psi_step
                
                feat = segment_features(df, ia, ib).reshape(1, -1)
                ds_corr = model.predict(feat)[0]
                ds_final = max(0.0, ds_imu + ds_corr)
                
                e_hyb += ds_final * np.sin(h_dir)
                n_hyb += ds_final * np.cos(h_dir)
                
                v_curr = v_step
                psi_curr = psi_step
                
            phyb = float(np.sqrt((e_hyb - e_true)**2 + (n_hyb - n_true)**2))
            
            all_results.append({
                'session': s_name,
                'duration_s': dur_s,
                'window_idx': w_idx,
                'pos_err_gyro': pg,
                'pos_err_zupt': pz,
                'pos_err_hybrid': phyb,
            })

df_res = pd.DataFrame(all_results)
print("\n" + "="*75)
print("OVERALL SUMMARY BY DURATION:")
print("="*75)
summary = df_res.groupby('duration_s')[['pos_err_gyro', 'pos_err_zupt', 'pos_err_hybrid']].mean()
summary['imp_vs_gyro_%'] = (summary['pos_err_gyro'] - summary['pos_err_hybrid']) / summary['pos_err_gyro'] * 100
summary['imp_vs_zupt_%'] = (summary['pos_err_zupt'] - summary['pos_err_hybrid']) / summary['pos_err_zupt'] * 100
print(summary.round(1).to_string())

print("\n" + "="*75)
print("PER-TRIP BREAKDOWN BY DURATION:")
print("="*75)
trip_summary = df_res.groupby(['session', 'duration_s'])[['pos_err_gyro', 'pos_err_zupt', 'pos_err_hybrid']].mean()
print(trip_summary.round(1).to_string())
