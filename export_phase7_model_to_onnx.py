"""
Export Trained Phase 7 XGBoost Velocity Model to ONNX Format
SIH26168 — AI-ML Dead Reckoning System

This is the instantaneous forward-speed model that phase10_fusion_engine.py's
EKF fusion validated as its continuous measurement source (56 features, ~2s
window). It was never exported before this session's Android EKF port work --
only the Phase 8/9 C1 *displacement* model was.

Trained on ALL sessions (not LOTO-excluded like the research/validation runs):
a deployed model doesn't know in advance which "session" live phone data
belongs to, so it should see every available session during training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
OUTPUT_ONNX_PATH = MODELS_DIR / "c1_velocity_model.onnx"


def export_model():
    print("=" * 80)
    print("EXPORTING PHASE 7 XGBOOST VELOCITY MODEL TO ONNX")
    print("=" * 80)

    print("[1] Loading Phase 7 cached dataset...")
    X_all = np.load("data/phase7_cache_X.npy")
    y_all = np.load("data/phase7_cache_y.npy")
    meta_all = pd.read_parquet("data/phase7_cache_meta.parquet")
    print(f"    Total windows: {len(meta_all):,} from {meta_all['session'].nunique()} sessions")
    print(f"    Feature dim: {X_all.shape[1]} (expected 56)")
    assert X_all.shape[1] == 56, f"Unexpected feature dimension: {X_all.shape[1]}"

    print("[2] Training production XGBoost Regressor on ALL sessions...")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    model.fit(X_all, y_all)
    print("    Model training complete.")

    print("[3] Converting model to ONNX with onnxmltools...")
    initial_types = [("float_input", FloatTensorType([None, 56]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types, target_opset=15)

    onnxmltools.utils.save_model(onnx_model, str(OUTPUT_ONNX_PATH))
    file_size_kb = OUTPUT_ONNX_PATH.stat().st_size / 1024.0
    print(f"[4] Model successfully exported to: {OUTPUT_ONNX_PATH.resolve()}")
    print(f"    File Size: {file_size_kb:.2f} KB")

    json_path = MODELS_DIR / "c1_velocity_model.json"
    model.save_model(str(json_path))
    print(f"    Native JSON saved to: {json_path.resolve()}")

    return OUTPUT_ONNX_PATH


if __name__ == "__main__":
    export_model()
