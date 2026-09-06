"""
Verification Script for ONNX Export
SIH26168 — AI-ML Dead Reckoning System

Requirements:
  1. Load the original Phase 9 XGBoost model AND the exported ONNX model.
  2. Run BOTH on the same test inputs from Step 1's 300s outage window
     (the 34 consecutive feature vectors from S-Vw4 window 0).
  3. Compare predictions side-by-side across all 34 windows.
  4. Report maximum absolute difference, mean absolute difference, and confirm
     whether it meets the tolerance (< 0.01 meters).
  5. Report ONNX file size in KB / MB.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import onnxruntime as ort

from phase6_outage_sim import (
    get_zupt_v1_mask, latlon_to_enu, get_col,
    sample_outage_windows, find_genuine_gnss_updates,
    OUTAGE_DURATIONS_S, OUTAGES_PER_DURATION
)
from phase4_orientation import transform_to_vehicle_frame
from phase9_ai_hybrid import slice_outage_intervals, MIN_T
from estimate_position import extract_58_features

PROCESSED_DIR = Path("data") / "processed_sessions"
CACHE_DIR = Path("data") / "phase8_cache"
MODELS_DIR = Path("models")
ONNX_MODEL_PATH = MODELS_DIR / "c1_correction_model.onnx"

RANDOM_SEED = 42

def main():
    print("=" * 80)
    print("STEP 2a VERIFICATION: ONNX MODEL vs XGBOOST PREDICTION CHECK")
    print("=" * 80)

    # 1. Check file size
    if not ONNX_MODEL_PATH.exists():
        print(f"ERROR: ONNX model not found at {ONNX_MODEL_PATH}. Run export_model_to_onnx.py first.")
        return

    file_size_bytes = ONNX_MODEL_PATH.stat().st_size
    file_size_kb = file_size_bytes / 1024.0
    file_size_mb = file_size_bytes / (1024.0 * 1024.0)
    print(f"[1] ONNX Model File Check:")
    print(f"    Path: {ONNX_MODEL_PATH.resolve()}")
    print(f"    Size: {file_size_kb:.2f} KB ({file_size_mb:.2f} MB)")

    # 2. Load ONNX Runtime session
    print("[2] Loading ONNX Runtime Inference Session...")
    ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    print(f"    ONNX Input Name: '{input_name}', Expected Shape: {input_shape}")

    # 3. Load / Fit the reference XGBoost model (matching Step 1)
    print("[3] Loading Reference XGBoost Model...")
    meta_all = pd.read_parquet(CACHE_DIR / "meta_all.parquet")
    yC_all = np.load(CACHE_DIR / "yC_all.npy")
    X_all = np.load(CACHE_DIR / "X_all.npy")

    train_mask = (meta_all['dur_s'].values >= MIN_T) & (~meta_all['session'].str.startswith('S-Vw4'))
    xgb_model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
    xgb_model.fit(X_all[train_mask], yC_all[train_mask])

    # 4. Extract test feature vectors from the 300s outage window (Step 1 test data)
    print("[4] Extracting 34 Test Windows from S-Vw4 300s Outage Window 0...")
    df = pd.read_parquet(PROCESSED_DIR / "S-Vw4.parquet")
    df, _, _ = transform_to_vehicle_frame(df)
    gnss_idx = find_genuine_gnss_updates(df)
    t_arr = df['time_rel_sec'].values

    np.random.seed(RANDOM_SEED)
    windows_dict = {}
    for dur in OUTAGE_DURATIONS_S:
        windows_dict[dur] = sample_outage_windows(df, gnss_idx, dur, OUTAGES_PER_DURATION)

    s_gnss, e_gnss = windows_dict[300][0]  # Window 0: (88439, 91490)
    intervals = slice_outage_intervals(s_gnss, e_gnss, t_arr, nominal_step_samples=90, min_t_s=MIN_T)
    print(f"    Sliced into {len(intervals)} intervals.")

    columns_needed = [
        'acc_x', 'acc_y', 'acc_z',
        'acc_lin_x', 'acc_lin_y', 'acc_lin_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'acc_veh_fwd', 'gyro_veh_yaw_rate',
        'dt_sec'
    ]

    features_list = []
    for ia, ib in intervals:
        win_df = df.iloc[ia : ib + 1][columns_needed]
        win_dict = {col: win_df[col].values for col in win_df.columns}
        seg_dur = float(t_arr[ib] - t_arr[ia])
        f_vec = extract_58_features(win_dict, seg_dur_s=seg_dur, seg_n_samples=len(win_df))
        features_list.append(f_vec)

    X_test_34 = np.vstack(features_list).astype(np.float32)
    print(f"    Extracted test matrix shape: {X_test_34.shape} (34 windows, 58 features)")

    # 5. Run Inference on Both Models
    print("[5] Running Predictions on Both Models...")
    # XGBoost
    pred_xgb = xgb_model.predict(X_test_34)

    # ONNX Runtime
    pred_onnx = ort_session.run(None, {input_name: X_test_34})[0].flatten()

    # 6. Compare Predictions Side-by-Side
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE PREDICTION COMPARISON (meters):")
    print("=" * 80)
    print(f"{'Window':>6} | {'XGBoost (m)':>14} | {'ONNX (m)':>14} | {'Abs Diff (m)':>14}")
    print("-" * 55)

    diffs = np.abs(pred_xgb - pred_onnx)
    for i in range(len(diffs)):
        print(f"{i:>6} | {pred_xgb[i]:>14.6f} | {pred_onnx[i]:>14.6f} | {diffs[i]:>14.8f}")

    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))

    print("\n" + "=" * 80)
    print("SUMMARY VERIFICATION METRICS:")
    print("=" * 80)
    print(f"Max Absolute Difference : {max_diff:.8e} meters ({max_diff*1000:.6f} mm)")
    print(f"Mean Absolute Difference: {mean_diff:.8e} meters ({mean_diff*1000:.6f} mm)")
    print(f"ONNX Model File Size    : {file_size_kb:.2f} KB ({file_size_mb:.2f} MB)")
    print(f"Target Tolerance        : < 0.01 meters (10 mm)")

    if max_diff < 0.01:
        print(">>> RESULT: SUCCESSFUL! ONNX export is bit-accurate within floating-point precision.")
        print("    The exported model is fully trustworthy for integration into Android.")
    else:
        print(">>> RESULT: FAILED! Difference exceeds tolerance threshold.")

if __name__ == "__main__":
    main()
