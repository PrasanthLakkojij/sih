import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from phase6_outage_sim import (
    sample_outage_windows, find_genuine_gnss_updates, get_zupt_v1_mask,
    latlon_to_enu, get_col, simulate_outage,
    TEST_SESSIONS, OUTAGE_DURATIONS_S, OUTAGES_PER_DURATION
)
from phase4_orientation import transform_to_vehicle_frame

PROCESSED_DIR = pd.Series([None]) # placeholder
# Let's verify that running the exact loop reproduces phase6 results
np.random.seed(42)
for s_name in TEST_SESSIONS:
    df = pd.read_parquet(f"data/processed_sessions/{s_name}")
    df, _, _ = transform_to_vehicle_frame(df)
    zupt_mask = get_zupt_v1_mask(df)
    gnss_idx = find_genuine_gnss_updates(df)
    lat0, lon0 = df['gps_lat'].iloc[0], df['gps_lon'].iloc[0]
    east_all, north_all = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)
    
    for dur_s in OUTAGE_DURATIONS_S:
        windows = sample_outage_windows(df, gnss_idx, dur_s, OUTAGES_PER_DURATION)
        g_errs = []
        z_errs = []
        for s_gnss, e_gnss in windows:
            pg, _, _, _, _ = simulate_outage(df, zupt_mask, gnss_idx, s_gnss, e_gnss, False, lat0, lon0, east_all, north_all)
            pz, _, _, _, _ = simulate_outage(df, zupt_mask, gnss_idx, s_gnss, e_gnss, True, lat0, lon0, east_all, north_all)
            g_errs.append(pg)
            z_errs.append(pz)
        print(f"{s_name} {dur_s:>3}s: Gyro={np.mean(g_errs):.1f}m, ZUPT={np.mean(z_errs):.1f}m")
