import numpy as np
import pandas as pd
from phase6_outage_sim import sample_outage_windows, find_genuine_gnss_updates, get_zupt_v1_mask, latlon_to_enu, get_col
from phase4_orientation import transform_to_vehicle_frame

df = pd.read_parquet('data/processed_sessions/S-Vw4.parquet')
df, _, _ = transform_to_vehicle_frame(df)
gnss_idx = find_genuine_gnss_updates(df)

np.random.seed(42)
windows = sample_outage_windows(df, gnss_idx, 30, 4)
t_arr = df['time_rel_sec'].values
v_gps = df['gps_speed_ms'].values

for idx, (s, e) in enumerate(windows):
    dur = t_arr[e] - t_arr[s]
    print(f"Window {idx}: [{s}, {e}], dur={dur:.2f}s, start_speed={v_gps[s]:.2f}m/s")
