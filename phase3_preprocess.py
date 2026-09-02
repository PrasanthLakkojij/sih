"""
Phase 3 — Data Preprocessing
SIH26168 — AI-ML Dead Reckoning System

Transforms the raw 72 CSV files into clean, verified, session-level structured datasets.

Pipeline per file:
1. Fix column schema typos (e.g. malformed DATE in S-Vfa01/02).
2. Standardize column names (strip whitespace).
3. Drop exact duplicate rows (e.g. S-Vtb1.csv ~16.5% duplicates).
4. Detect timestamp resets (negative dt) and long interruptions (> 5s) to split into valid continuous sub-sessions.
5. Handle explicit missing values (e.g. S-Y1.csv).
6. Compute precise sample-to-sample dt.
7. Compute body-frame linear acceleration: a_lin = a_raw - g_recorded (preserving both raw and gravity vectors).
8. Identify actual GNSS fix transitions (is_gnss_update boolean flag + gnss_fix_id), without interpolating GPS.
9. Save clean session files to data/processed_sessions/ with metadata manifest.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

RAW_DATA_DIR = Path("data") / "S-Dataset"
PROCESSED_DIR = Path("data") / "processed_sessions"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ENCODING = "latin-1"
TIME_RESET_THRESHOLD_MS = 0     # any negative dt is a session reset
GAP_SPLIT_THRESHOLD_MS = 5000   # gaps > 5.0 seconds indicate a break/pause

# Mapping for column standardization
COLUMN_MAP = {
    'GPS LATITUDE (degrees)': 'gps_lat',
    'GPS LONGITUDE (degrees)': 'gps_lon',
    'GPS ALTITUDE (m)': 'gps_alt',
    'GPS SPEED (Kmh)': 'gps_speed_kmh',
    'GPS ACCURACY (m)': 'gps_accuracy_m',
    'GPS ORIENTATION (°)': 'gps_bearing_deg',
    'GPS ORIENTATION (A)': 'gps_bearing_deg',
    'GPS SATELLITES IN RANGE': 'gps_satellites',
    'TIME SINCE START (ms)': 'time_ms',
    'DATE (YYYY-MO-DD HH-MI-SS_SSS)': 'date_str',
    'DATE (YYYY-MO-DD HH-MI-SS_SSS': 'date_str',  # Fix typo in S-Vfa01/02
    'ACCELEROMETER X (m/s²)': 'acc_x',
    'ACCELEROMETER Y (m/s²)': 'acc_y',
    'ACCELEROMETER Z (m/s²)': 'acc_z',
    'ACCELEROMETER X (m/s)': 'acc_x',
    'ACCELEROMETER Y (m/s)': 'acc_y',
    'ACCELEROMETER Z (m/s)': 'acc_z',
    'ACCELEROMETER X (m/s) ': 'acc_x',
    'GRAVITY X (m/s²)': 'grav_x',
    'GRAVITY Y (m/s²)': 'grav_y',
    'GRAVITY Z (m/s²)': 'grav_z',
    'GRAVITY X (m/s)': 'grav_x',
    'GRAVITY Y (m/s)': 'grav_y',
    'GRAVITY Z (m/s)': 'grav_z',
    'GYROSCOPE X (rad/s)': 'gyro_x',
    'GYROSCOPE Y (rad/s)': 'gyro_y',
    'GYROSCOPE Z (rad/s)': 'gyro_z',
    'MAGNETIC FIELD X (μT)': 'mag_x',
    'MAGNETIC FIELD Y (μT)': 'mag_y',
    'MAGNETIC FIELD Z (μT)': 'mag_z',
    'MAGNETIC FIELD X (IT)': 'mag_x',
    'MAGNETIC FIELD Y (IT)': 'mag_y',
    'MAGNETIC FIELD Z (IT)': 'mag_z',
    'ORIENTATION (Azimuth) (°)': 'ori_azimuth_deg',
    'ORIENTATION (Pitch) (°)': 'ori_pitch_deg',
    'ORIENTATION (Roll ) (°)': 'ori_roll_deg',
    'ORIENTATION (Azimuth) (A)': 'ori_azimuth_deg',
    'ORIENTATION (Pitch) (A)': 'ori_pitch_deg',
    'ORIENTATION (Roll ) (A)': 'ori_roll_deg',
}

def clean_column_names(df):
    """Normalize messy headers, remove weird encoding artifacts."""
    new_cols = []
    for c in df.columns:
        c_clean = c.strip()
        matched = None
        for k, v in COLUMN_MAP.items():
            if k.strip().upper() == c_clean.upper():
                matched = v
                break
        if matched:
            new_cols.append(matched)
        else:
            # Fallback normalization
            clean_name = c_clean.lower().replace(' ', '_').replace('(', '').replace(')', '')
            new_cols.append(clean_name)
    df.columns = new_cols
    return df

def process_file(csv_path):
    """Process a single raw CSV file into one or more valid sub-sessions."""
    raw_filename = csv_path.name
    base_name = csv_path.stem
    
    # 1. Load CSV with latin-1
    df = pd.read_csv(csv_path, encoding=ENCODING)
    initial_rows = len(df)
    
    # 2. Rename columns
    df = clean_column_names(df)
    
    # 3. Deduplicate
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_rows = len(df)
    duplicates_removed = initial_rows - dedup_rows
    
    # 4. Handle Missing values (e.g. S-Y1)
    # Forward-fill and backward-fill small gaps in GPS or numeric columns
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    # 5. Detect session breaks
    # Time diff in ms
    t_vals = df['time_ms'].values.astype(float)
    dt_vals = np.diff(t_vals)
    
    # Break occurs if dt < 0 (timestamp reset) or dt > 5000 ms (recording pause)
    break_indices = np.where((dt_vals < TIME_RESET_THRESHOLD_MS) | (dt_vals > GAP_SPLIT_THRESHOLD_MS))[0]
    
    # Segment boundaries: [0, break_1+1, break_2+1, ..., len(df)]
    split_points = [0] + list(break_indices + 1) + [len(df)]
    
    sub_sessions = []
    
    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i+1]
        
        session_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        
        # Discard tiny segments (< 5 seconds = 50 rows)
        if len(session_df) < 50:
            continue
            
        # Compute dt in seconds
        t_arr = session_df['time_ms'].values.astype(float)
        # Re-zero time for this sub-session
        session_df['time_rel_sec'] = (t_arr - t_arr[0]) / 1000.0
        
        # Sample dt (first dt is approximated as median dt)
        dt_arr = np.zeros(len(session_df))
        if len(session_df) > 1:
            diff_sec = np.diff(session_df['time_rel_sec'].values)
            median_dt = np.median(diff_sec)
            dt_arr[0] = median_dt
            dt_arr[1:] = diff_sec
        else:
            dt_arr[0] = 0.1
        session_df['dt_sec'] = dt_arr
        
        # 6. Compute Linear Acceleration: a_lin = a_raw - g
        session_df['acc_lin_x'] = session_df['acc_x'] - session_df['grav_x']
        session_df['acc_lin_y'] = session_df['acc_y'] - session_df['grav_y']
        session_df['acc_lin_z'] = session_df['acc_z'] - session_df['grav_z']
        
        # Convert GPS speed from km/h to m/s
        session_df['gps_speed_ms'] = session_df['gps_speed_kmh'] / 3.6
        
        # 7. Identify Discrete GNSS Fix Changes (no interpolation!)
        lat = session_df['gps_lat'].values
        lon = session_df['gps_lon'].values
        
        # Boolean flag where lat or lon changed
        is_gnss_update = np.zeros(len(session_df), dtype=bool)
        is_gnss_update[0] = True # First sample is an initial fix
        
        if len(session_df) > 1:
            changed = (np.diff(lat) != 0) | (np.diff(lon) != 0)
            is_gnss_update[1:] = changed
            
        session_df['is_gnss_update'] = is_gnss_update
        # Assign GNSS fix index (cumulative sum of updates)
        session_df['gnss_fix_id'] = np.cumsum(is_gnss_update) - 1
        
        sub_sessions.append(session_df)
        
    return base_name, initial_rows, duplicates_removed, sub_sessions

def main():
    print("=" * 80)
    print("PHASE 3: DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} raw CSV files to process.\n")
    
    manifest = []
    total_saved_sessions = 0
    total_clean_rows = 0
    
    for f in csv_files:
        base_name, init_rows, dups_removed, sessions = process_file(f)
        
        is_split = len(sessions) > 1
        split_note = f" (SPLIT into {len(sessions)} sub-sessions)" if is_split else ""
        dup_note = f" | {dups_removed} dups removed" if dups_removed > 0 else ""
        
        print(f"Processing {f.name:<20s} [{init_rows:>7,d} rows]{dup_note}{split_note}")
        
        for s_idx, s_df in enumerate(sessions):
            if is_split:
                session_name = f"{base_name}_sub{s_idx+1:02d}"
            else:
                session_name = base_name
                
            out_file = PROCESSED_DIR / f"{session_name}.parquet"
            s_df.to_parquet(out_file, index=False)
            
            # Record manifest info
            dur_sec = s_df['time_rel_sec'].iloc[-1]
            n_gnss_fixes = int(s_df['is_gnss_update'].sum())
            mean_dt = float(s_df['dt_sec'].mean())
            
            manifest.append({
                'session_id': session_name,
                'source_file': f.name,
                'rows': len(s_df),
                'duration_sec': round(dur_sec, 2),
                'duration_min': round(dur_sec / 60.0, 2),
                'gnss_fixes': n_gnss_fixes,
                'gnss_update_rate_hz': round(n_gnss_fixes / dur_sec, 3) if dur_sec > 0 else 0,
                'mean_imu_dt_ms': round(mean_dt * 1000, 2),
                'file_path': str(out_file)
            })
            
            total_saved_sessions += 1
            total_clean_rows += len(s_df)
            
    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(PROCESSED_DIR / "manifest.csv", index=False)
    with open(PROCESSED_DIR / "manifest.json", "w") as jf:
        json.dump(manifest, jf, indent=2)
        
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total raw CSVs:            {len(csv_files)}")
    print(f"Total processed sessions:  {total_saved_sessions}")
    print(f"Total clean IMU samples:   {total_clean_rows:,d}")
    print(f"Output directory:          {PROCESSED_DIR.resolve()}")
    print(f"Manifest created at:       {PROCESSED_DIR / 'manifest.csv'}")
    
    print("\nSample of generated sessions from manifest:")
    print(manifest_df[['session_id', 'rows', 'duration_min', 'gnss_fixes', 'gnss_update_rate_hz']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
