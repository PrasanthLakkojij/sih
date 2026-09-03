import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from phase6_outage_sim import sample_outage_windows, find_genuine_gnss_updates, get_zupt_v1_mask, latlon_to_enu, get_col
from phase4_orientation import transform_to_vehicle_frame
from phase8_ml_correction import segment_features

# Load cached data
meta_all = pd.read_parquet('data/phase8_cache/meta_all.parquet')
yC_all = np.load('data/phase8_cache/yC_all.npy')
X_all = np.load('data/phase8_cache/X_all.npy')

# Filter MIN_T >= 2.0s
valid = meta_all['dur_s'].values >= 2.0
X_tr = X_all[valid]
y_tr = yC_all[valid]

print(f"Training XGBoost on {len(y_tr)} segments (t >= 2.0s)...")
model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     n_jobs=-1, random_state=42, verbosity=0)
model.fit(X_tr, y_tr)
print("Trained.")

# Now test on Window 0 of S-Vw4 (30s)
df = pd.read_parquet('data/processed_sessions/S-Vw4.parquet')
df, _, _ = transform_to_vehicle_frame(df)
gnss_idx = find_genuine_gnss_updates(df)
zupt_mask = get_zupt_v1_mask(df)

lat0, lon0 = df['gps_lat'].iloc[0], df['gps_lon'].iloc[0]
east_all, north_all = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)

np.random.seed(42)
windows = sample_outage_windows(df, gnss_idx, 30, 1)
s, e = windows[0]
t_arr = df['time_rel_sec'].values
a_fwd = df['acc_veh_fwd'].values
dt_arr = df['dt_sec'].values
gyro_yaw = df['gyro_veh_yaw_rate'].values
az_col = get_col(df, 'azimuth')
psi_mag = np.radians(df[az_col].values)
bearing_col = get_col(df, 'gps_orientation') or get_col(df, 'gps_bearing')

b0 = df[bearing_col].values[s] if bearing_col else np.nan
psi0 = psi_mag[s] if (np.isnan(b0) or b0 == 0) else np.radians(b0)
v0 = df['gps_speed_ms'].values[s]

e_true = east_all[e]
n_true = north_all[e]
e0 = east_all[s]
n0 = north_all[s]
gt_dist = np.sqrt((e_true - e0)**2 + (n_true - n0)**2)
print(f"Ground truth displacement: {gt_dist:.2f}m")

# Physics DR (ZUPT v1) full
e_z, n_z, v_z, psi_z = e0, n0, v0, psi0
for t in range(s+1, e+1):
    dt = dt_arr[t]
    if zupt_mask[t]: v_z = 0.0
    else: v_z = max(0.0, v_z + a_fwd[t]*dt)
    psi_z += gyro_yaw[t]*dt
    e_z += v_z * np.sin(psi_z) * dt
    n_z += v_z * np.cos(psi_z) * dt
zupt_err = np.sqrt((e_z - e_true)**2 + (n_z - n_true)**2)
print(f"ZUPT v1 position error: {zupt_err:.2f}m")

# Gyro-only DR full
e_g, n_g, v_g, psi_g = e0, n0, v0, psi0
for t in range(s+1, e+1):
    dt = dt_arr[t]
    v_g = max(0.0, v_g + a_fwd[t]*dt)
    psi_g += gyro_yaw[t]*dt
    e_g += v_g * np.sin(psi_g) * dt
    n_g += v_g * np.cos(psi_g) * dt
gyro_err = np.sqrt((e_g - e_true)**2 + (n_g - n_true)**2)
print(f"Gyro-only position error: {gyro_err:.2f}m")

# Now AI-Hybrid over ~9s intervals
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
    # Run ZUPT v1 for this interval
    de_sub, dn_sub = 0.0, 0.0
    v_step = v_curr
    psi_step = psi_curr
    for t in range(ia+1, ib+1):
        dt = dt_arr[t]
        if zupt_mask[t]: v_step = 0.0
        else: v_step = max(0.0, v_step + a_fwd[t]*dt)
        psi_step += gyro_yaw[t]*dt
        de_sub += v_step * np.sin(psi_step) * dt
        dn_sub += v_step * np.cos(psi_step) * dt
    
    ds_imu = np.sqrt(de_sub**2 + dn_sub**2)
    heading_dir = np.arctan2(de_sub, dn_sub) if ds_imu > 1e-3 else psi_step
    
    feat = segment_features(df, ia, ib).reshape(1, -1)
    ds_corr_pred = model.predict(feat)[0]
    ds_corr_final = max(0.0, ds_imu + ds_corr_pred)
    
    e_hyb += ds_corr_final * np.sin(heading_dir)
    n_hyb += ds_corr_final * np.cos(heading_dir)
    
    v_curr = v_step
    psi_curr = psi_step

hyb_err = np.sqrt((e_hyb - e_true)**2 + (n_hyb - n_true)**2)
print(f"AI-Hybrid position error: {hyb_err:.2f}m")
