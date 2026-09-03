import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from phase6_outage_sim import get_zupt_v1_mask, get_col, latlon_to_enu, find_genuine_gnss_updates
from phase4_orientation import transform_to_vehicle_frame

# Let's test on 3 sessions (S-Vw4, S-Vtb5, S-Vta1a)
# Compute ds_zupt vs ds_open for genuine segments
for s_name in ["S-Vw4.parquet", "S-Vtb5.parquet"]:
    df = pd.read_parquet(f"data/processed_sessions/{s_name}")
    df, _, _ = transform_to_vehicle_frame(df)
    zupt_mask = get_zupt_v1_mask(df)
    idx = find_genuine_gnss_updates(df)
    lat0, lon0 = df['gps_lat'].iloc[0], df['gps_lon'].iloc[0]
    east, north = latlon_to_enu(df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)
    
    a_fwd = df['acc_veh_fwd'].values
    dt_arr = df['dt_sec'].values
    gyro_yaw = df['gyro_veh_yaw_rate'].values
    az_col = get_col(df, 'azimuth')
    psi_mag = np.radians(df[az_col].values)
    bearing_col = get_col(df, 'gps_orientation') or get_col(df, 'gps_bearing')
    
    zupt_diffs = []
    open_diffs = []
    for k in range(min(100, len(idx)-1)):
        i0, i1 = idx[k], idx[k+1]
        dur = df['time_rel_sec'].iloc[i1] - df['time_rel_sec'].iloc[i0]
        if dur < 2.0 or dur > 30: continue
        
        de, dn = east[i1]-east[i0], north[i1]-north[i0]
        ds_gps = np.sqrt(de**2 + dn**2)
        
        v0 = df['gps_speed_ms'].values[i0]
        b0 = df[bearing_col].values[i0] if bearing_col else np.nan
        psi0 = psi_mag[i0] if (np.isnan(b0) or b0 == 0) else np.radians(b0)
        
        # open
        v_o, psi_o, eo, no = v0, psi0, 0.0, 0.0
        # zupt
        v_z, psi_z, ez, nz = v0, psi0, 0.0, 0.0
        
        for t in range(i0+1, i1+1):
            dt = dt_arr[t]
            v_o = max(0.0, v_o + a_fwd[t]*dt)
            psi_o += gyro_yaw[t]*dt
            eo += v_o*np.sin(psi_o)*dt
            no += v_o*np.cos(psi_o)*dt
            
            if zupt_mask[t]: v_z = 0.0
            else: v_z = max(0.0, v_z + a_fwd[t]*dt)
            psi_z += gyro_yaw[t]*dt
            ez += v_z*np.sin(psi_z)*dt
            nz += v_z*np.cos(psi_z)*dt
            
        ds_o = np.sqrt(eo**2 + no**2)
        ds_z = np.sqrt(ez**2 + nz**2)
        
        open_diffs.append(ds_gps - ds_o)
        zupt_diffs.append(ds_gps - ds_z)
        
    print(f"{s_name}: open diff mean={np.mean(open_diffs):.2f}m, zupt diff mean={np.mean(zupt_diffs):.2f}m")
