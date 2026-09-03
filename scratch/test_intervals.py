import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from phase6_outage_sim import sample_outage_windows, find_genuine_gnss_updates, get_zupt_v1_mask, latlon_to_enu, get_col
from phase4_orientation import transform_to_vehicle_frame
from phase8_ml_correction import segment_features

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

# Let's slice into intervals of 90 samples (~9s)
intervals = []
curr = s
MIN_T = 2.0
while curr < e:
    nxt = min(curr + 90, e)
    if nxt < e and (t_arr[e] - t_arr[nxt]) < MIN_T:
        nxt = e
    intervals.append((curr, nxt))
    curr = nxt

print(f"Outage [{s}, {e}], total dur={t_arr[e]-t_arr[s]:.1f}s, num intervals={len(intervals)}")
for idx, (ia, ib) in enumerate(intervals):
    print(f"  Interval {idx}: [{ia}, {ib}], dur={t_arr[ib]-t_arr[ia]:.2f}s")
