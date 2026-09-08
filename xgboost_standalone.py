"""
Standalone XGBoost Dead Reckoning Correction Model
SIH26168 — AI-ML Dead Reckoning System

Use this standalone script to:
1. Train and save the XGBoost model to disk (`xgboost_model.json`).
2. Test predictions with dummy or custom 58-feature inputs.
3. Integrate into your own pipeline or Google Colab.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# ==============================================================================
# 1. LOAD TRAINING DATA (58 FEATURES & CORRECTION TARGET)
# ==============================================================================
CACHE_DIR = Path("data/phase8_cache")

print("Loading cached 58-feature dataset...")
X = np.load(CACHE_DIR / "X_all.npy")           # Shape: (8082, 58)
y = np.load(CACHE_DIR / "yC_all.npy")          # Shape: (8082,) -> Δs_corr in meters
meta = pd.read_parquet(CACHE_DIR / "meta_all.parquet")

# Apply safety filter: exclude sub-2-second noise fragments
valid_idx = meta['dur_s'].values >= 2.0
X_clean = X[valid_idx]
y_clean = y[valid_idx]

print(f"Dataset ready: {X_clean.shape[0]:,} segments, {X_clean.shape[1]} features each.")

# ==============================================================================
# 2. TRAIN / TEST SPLIT
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.20, random_state=42
)

# ==============================================================================
# 3. CONFIGURE & TRAIN THE ONLY XGBOOST MODEL
# ==============================================================================
print("\nTraining XGBoost Regressor...")
model = XGBRegressor(
    n_estimators=400,        # 400 boosted trees
    max_depth=6,             # Tree depth
    learning_rate=0.05,      # Step size (shrinkage)
    subsample=0.8,           # Row subsampling (prevents overfit)
    colsample_bytree=0.8,    # Feature subsampling
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

model.fit(X_train, y_train)
print("Training Complete!")

# ==============================================================================
# 4. EVALUATION METRICS
# ==============================================================================
predictions = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
mae = float(mean_absolute_error(y_test, predictions))
naive_rmse = float(np.sqrt(mean_squared_error(y_test, np.zeros_like(y_test))))

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS:")
print("="*60)
print(f"Naive Zero Baseline RMSE : {naive_rmse:.2f} meters")
print(f"XGBoost Test RMSE        : {rmse:.2f} meters")
print(f"Mean Absolute Error (MAE): {mae:.2f} meters")
print(f"Improvement Over Naive   : {((naive_rmse - rmse) / naive_rmse) * 100:.1f}%")

# Save model to disk so you can load it anywhere without retraining!
model_path = "xgboost_model.json"
model.save_model(model_path)
print(f"\nModel saved to disk as: '{model_path}'")

# ==============================================================================
# 5. HOW TO TEST WITH YOUR OWN DATA (INFERENCE EXAMPLE)
# ==============================================================================
print("\n" + "="*60)
print("TESTING WITH SAMPLE INPUT DATA:")
print("="*60)

# Example: Take a single 58-feature vector from your own data
# (Here we take the first test sample as an example)
my_custom_input = X_test[0:1]   # Shape must be (1, 58) or (N, 58)

# Load the saved model (shows how to use it in another script/app)
loaded_model = XGBRegressor()
loaded_model.load_model(model_path)

# Predict the displacement correction!
predicted_correction = loaded_model.predict(my_custom_input)[0]
actual_correction = y_test[0]

print(f"Input Shape                 : {my_custom_input.shape} (1 window, 58 features)")
print(f"Predicted Correction Output : {predicted_correction:+.2f} meters")
print(f"Actual Ground Truth Needed  : {actual_correction:+.2f} meters")
print(f"Absolute Prediction Error   : {abs(actual_correction - predicted_correction):.2f} meters")
