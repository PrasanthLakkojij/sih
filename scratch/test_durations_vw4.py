import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from phase6_outage_sim import sample_outage_windows, find_genuine_gnss_updates, get_zupt_v1_mask, latlon_to_enu, get_col
from phase4_orientation import transform_to_vehicle_frame
from phase8_ml_correction import segment_features

# Let's test on S-Vw4 for 30s and 60s
df = pd.read_parquet('data/processed_sessions/S-Vw4.parquet')
df, _, _ = transform_to_vehicle_frame(df)
gnss_idx = find_genuine_gnss_updates(df)
zupt_mask = get_zupt_v1_mask(df)
lat0, lon0 = df['gps_lat'].iloc[0], df['gps_lon'].iloc[0]
east_all, north_all = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)
t_arr = df['time_rel_sec'].values
a_fwd = df['acc_veh_fwd'].values
dt_arr = df['dt_sec'].values
gyro_yaw = df['gyro_veh_yaw_rate'].values
az_col = get_col(df, 'azimuth')
psi_mag = np.radians(df[az_col].values)
bearing_col = get_col(df, 'gps_orientation') or get_col(df, 'gps_bearing')

# Load cached data
meta_all = pd.read_parquet('data/phase8_cache/meta_all.parquet')
yC_all = np.load('data/phase8_cache/yC_all.npy')
X_all = np.load('data/phase8_cache/X_all.npy')

# Filter MIN_T >= 2.0s
valid = meta_all['dur_s'].values >= 2.0
# For clean LOTO-style test, train on all sessions except S-Vw4
train_mask = valid & (~meta_all['session'].str.startswith('S-Vw4'))

model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     n_jobs=-1, random_state=42, verbosity=0)
model.fit(X_all[train_mask], yC_all[train_mask])

np.random.seed(42)
for dur in [30, 60, 120, 300]:
    windows = sample_outage_windows(df, gnss_idx, dur, 8)
    errs_gyro, errs_zupt, errs_hyb = [], [], []
    for s, e in windows:
        e_true, n_true = east_all[e], north_all[e]
        e0, n0 = east_all[s], north_all[s]
        v0 = df['gps_speed_ms'].values[s]
        b0 = df[bearing_col].values[s] if bearing_col else np.nan
        psi0 = psi_mag[s] if (np.isnan(b0) or b0 == 0) else np.radians(b0)
        
        # Gyro-only
        eg, ng, vg, psig = e0, n0, v0, psi0
        for t in range(s+1, e+1):
            dt = dt_arr[t]
            vg = max(0.0, vg + a_fwd[t]*dt)
            psig += gyro_yaw[t]*dt
            eg += vg*np.sin(psig)*dt
            ng += vg*np.cos(psig)*dt
        errs_gyro.append(np.sqrt((eg - e_true)**2 + (ng - n_true)**2))
        
        # ZUPT v1
        ez, nz, vz, psiz = e0, n0, v0, psi0
        for t in range(s+1, e+1):
            dt = dt_arr[t]
            if zupt_mask[t]: vz = 0.0
            else: vz = max(0.0, vz + a_fwd[t]*dt)
            psiz += gyro_yaw[t]*dt
            ez += vz*np.sin(psiz)*dt
            nz += vz*np.cos(psiz)*dt
        errs_zupt.append(np.sqrt((ez - e_true)**2 + (nz - n_true)**2))
        
        # AI-Hybrid with ~9s intervals
        curr = s
        MIN_T = 2.0
        intervals = []
        while curr < e:
            nxt = min(curr + 90, e)
            if nxt < e and (t_arr[e] - t_arr[nxt]) < MIN_T:
                nxt = e
            intervals.append((curr, nxt))
            curr = nxt
            
        e_hyb, n_hyb = e0, n0
        v_curr, psi_curr = v0, psi0
        for ia, ib in intervals:
            de_sub, dn_sub = 0.0, 0.0
            v_step = v_curr
            psi_step = psi_curr
            for t in range(ia+1, ib+1):
                dt = dt_arr[t]
                if zupt_mask[t]: v_step = 0.0
                else: v_step = max(0.0, v_step + a_fwd[t]*dt)
                psi_step += gyro_yaw[t]*dt
                de_sub += v_step*np.sin(psi_step)*dt
                dn_sub += v_step*np.cos(psi_step)*dt
            ds_imu = np.sqrt(de_sub**2 + dn_sub**2)
            h_dir = np.arctan2(de_sub, dn_sub) if ds_imu > 1e-3 else psi_step
            feat = segment_features(df, ia, ib).reshape(1, -1)
            ds_corr = model.predict(feat)[0]
            ds_final = max(0.0, ds_imu + ds_corr)
            e_hyb += ds_final * np.sin(h_dir)
            n_hyb += ds_final * np.cos(h_dir)
            v_curr = v_step
            psi_curr = psi_step
        errs_hyb.append(np.sqrt((e_hyb - e_true)**2 + (n_hyb - n_true)**2))
        
    print(f"Dur {dur:>3}s: Gyro={np.mean(errs_gyro):.1f}m, ZUPT={np.mean(errs_zupt):.1f}m, Hybrid={np.mean(errs_hyb):.1f}m")
