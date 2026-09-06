"""
Export Trained Phase 9 C1 XGBoost Model to ONNX Format
SIH26168 — AI-ML Dead Reckoning System

This script:
1. Loads the Phase 8 cached dataset (with safety filter MIN_T >= 2.0s).
2. Fits the production Phase 9 C1 XGBoost model (400 trees, max_depth 6, lr 0.05).
   (Optionally trained with LOTO or on full verified dataset for deployment).
3. Converts the trained model to ONNX format via onnxmltools, specifying the
   correct input shape: FloatTensorType([None, 58]).
4. Saves the ONNX model to `models/c1_correction_model.onnx`.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

CACHE_DIR = Path("data") / "phase8_cache"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MIN_T = 2.0
OUTPUT_ONNX_PATH = MODELS_DIR / "c1_correction_model.onnx"

def export_model():
    print("=" * 80)
    print("STEP 2a: EXPORTING PHASE 9 C1 XGBOOST MODEL TO ONNX")
    print("=" * 80)

    # 1. Load cached dataset
    print("[1] Loading Phase 8 segment dataset...")
    meta_all = pd.read_parquet(CACHE_DIR / "meta_all.parquet")
    yC_all = np.load(CACHE_DIR / "yC_all.npy")
    X_all = np.load(CACHE_DIR / "X_all.npy")

    valid_mask = meta_all['dur_s'].values >= MIN_T
    print(f"    Total segments: {len(meta_all):,}")
    print(f"    Valid segments (dur >= {MIN_T:.1f}s): {valid_mask.sum():,}")

    # For deployment model on S-Vw4 benchmark (or full production training):
    # To test exactly against test_estimate_position.py, we train excluding S-Vw4
    # and we also provide the option to train on all data.
    # Here we train the exact LOTO model for S-Vw4 to verify against Step 1,
    # or export the model trained for deployment.
    # Let's train the model matching Step 1's benchmark (excl S-Vw4):
    train_mask = valid_mask & (~meta_all['session'].str.startswith('S-Vw4'))
    print(f"[2] Training Phase 9 C1 XGBoost Regressor...")
    print(f"    Training samples: {train_mask.sum():,} segments from {meta_all[train_mask]['session'].nunique()} trips")

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbosity=0
    )
    model.fit(X_all[train_mask], yC_all[train_mask])
    print("    Model training complete.")

    # 2. Convert to ONNX
    print("[3] Converting model to ONNX with onnxmltools...")
    initial_types = [('float_input', FloatTensorType([None, 58]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types, target_opset=15)

    # 3. Save ONNX model
    onnxmltools.utils.save_model(onnx_model, str(OUTPUT_ONNX_PATH))
    file_size_bytes = OUTPUT_ONNX_PATH.stat().st_size
    file_size_kb = file_size_bytes / 1024.0
    file_size_mb = file_size_bytes / (1024.0 * 1024.0)

    print(f"[4] Model successfully exported to: {OUTPUT_ONNX_PATH.resolve()}")
    print(f"    File Size: {file_size_kb:.2f} KB ({file_size_mb:.2f} MB)")

    # Also save native booster JSON for reference
    json_path = MODELS_DIR / "c1_correction_model.json"
    model.save_model(str(json_path))
    print(f"    Native JSON saved to: {json_path.resolve()} ({json_path.stat().st_size / 1024:.2f} KB)")

    return OUTPUT_ONNX_PATH, file_size_kb

if __name__ == "__main__":
    export_model()
