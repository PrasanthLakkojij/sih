"""
Phase 8 — Physics-Error-Correction ML Formulation Experiment
SIH26168 — AI-ML Dead Reckoning System

FORMULATION SHIFT (from Phase 7):
  Phase 7 target = instantaneous velocity at GPS fix → trivially beaten by naive.
  Phase 8 targets = per-segment residuals/corrections:
    A. v_GNSS   (Phase 7 reference — kept only for direct comparison)
    B. Δv       = v_(k+1) - v_k  (velocity change over the segment)
    C. Δs_corr  = Δs_GNSS - Δs_IMU  (displacement correction the physics model needs)

  Naive baseline for B and C = predict zero (assume physics is perfect).
  This is the correct naive for a correction/residual target.

FEATURES: Same 56 windowed IMU stats as Phase 7, computed over the full
          inter-GNSS segment (variable length, stats capture the distribution).

LOTO rounds (same 4 as Phase 7):
  R1 test=S-Vw4     R2 test=S-Vtb5
  R3 test=S-Vta1a   R4 test=S-Vtb2

HEADING NOTE (corrected wording):
  The Phase 7 diagnostic showed average heading MAE = 76.45° (vs ~90° random).
  Conclusion: The current short-window ML formulation does not provide sufficiently
  generalizable heading estimates; therefore Phase 9 will retain physics-based gyro
  heading as the primary heading estimator.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

PROCESSED_DIR = Path("data") / "processed_sessions"
OUT_DIR       = Path("plots") / "phase8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
LOTO_ROUNDS = [
    ("S-Vw4",   "driver W, 210 min, motorway"),
    ("S-Vtb5",  "driver B, 107 min, suburban"),
    ("S-Vta1a", "driver A,  43 min, 1 Hz GPS"),
    ("S-Vtb2",  "driver B,  10 min, short urban"),
]

# ====================================================================
# Helpers
# ====================================================================
def get_col(df, keyword):
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None

def find_genuine_gnss_idx(df):
    if 'is_gnss_update' in df.columns:
        idx = np.where(df['is_gnss_update'].values.astype(bool))[0]
        if len(idx) > 10:
            return idx
    lat = df['gps_lat'].values
    lon = df['gps_lon'].values
    changed = (np.abs(np.diff(lat)) > 1e-7) | (np.abs(np.diff(lon)) > 1e-7)
    return np.where(np.concatenate([[True], changed]))[0]

def latlon_to_enu(lat, lon, lat0, lon0):
    R = 6371000.0
    east  = R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    north = R * np.radians(lat - lat0)
    return east, north

# ====================================================================
# IMU Physics Integration over a segment (Phase 5A / no-ZUPT method)
# Returns scalar: total displacement (m) estimated by pure IMU.
# ====================================================================
def imu_displacement_segment(df, seg_start, seg_end):
    """
    Open-loop IMU displacement from seg_start to seg_end (row indices).
    Initialises velocity from GPS speed at seg_start (genuine fix).
    Uses acc_veh_fwd + gyro heading integration -> dx,dy.
    Returns (delta_s_imu, v_end_imu).
    """
    a_fwd    = df['acc_veh_fwd'].values
    dt_arr   = df['dt_sec'].values
    gyro_yaw = df['gyro_veh_yaw_rate'].values
    az_col   = get_col(df, 'azimuth')
    psi_mag  = np.radians(df[az_col].values)
    bearing_col = get_col(df, 'gps_orientation') or get_col(df, 'gps_bearing')

    v0 = df['gps_speed_ms'].values[seg_start]
    if bearing_col:
        b0 = df[bearing_col].values[seg_start]
        psi0 = psi_mag[seg_start] if (np.isnan(b0) or b0 == 0) else np.radians(b0)
    else:
        psi0 = psi_mag[seg_start]

    v_dr  = v0
    psi_dr = psi0
    e_dr  = 0.0
    n_dr  = 0.0

    for t in range(seg_start + 1, seg_end + 1):
        if t >= len(dt_arr):
            break
        dt = dt_arr[t]
        v_dr   = max(0.0, v_dr + a_fwd[t] * dt)
        psi_dr += gyro_yaw[t] * dt
        e_dr   += v_dr * np.sin(psi_dr) * dt
        n_dr   += v_dr * np.cos(psi_dr) * dt

    delta_s_imu = float(np.sqrt(e_dr**2 + n_dr**2))
    return delta_s_imu, float(v_dr)

# ====================================================================
# Feature extraction — per segment (between two genuine GNSS fixes)
# ====================================================================
SIGNAL_KEYS = [
    'acc_x', 'acc_y', 'acc_z',
    'acc_lin_x', 'acc_lin_y', 'acc_lin_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'acc_veh_fwd', 'gyro_veh_yaw_rate',
]
STATS = ['mean', 'std', 'min', 'max']

def _safe(df, key):
    c = get_col(df, key)
    return df[c].values if c else np.zeros(len(df))

def segment_features(df, seg_start, seg_end):
    """
    Compute windowed statistics over the IMU samples in [seg_start, seg_end].
    Also include magnitude signals and segment duration as features.
    Returns 1D feature vector.
    """
    sl = slice(seg_start, seg_end + 1)
    feats = []

    # Per-axis stats
    for key in SIGNAL_KEYS:
        w = _safe(df, key)[sl]
        if len(w) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([np.mean(w), np.std(w), np.min(w), np.max(w)])

    # Magnitude signals
    ax = _safe(df, 'acc_x')[sl]; ay = _safe(df, 'acc_y')[sl]; az = _safe(df, 'acc_z')[sl]
    lx = _safe(df, 'acc_lin_x')[sl]; ly = _safe(df, 'acc_lin_y')[sl]; lz = _safe(df, 'acc_lin_z')[sl]
    gx = _safe(df, 'gyro_x')[sl]; gy = _safe(df, 'gyro_y')[sl]; gz = _safe(df, 'gyro_z')[sl]

    for mag in [np.sqrt(ax**2+ay**2+az**2),
                np.sqrt(lx**2+ly**2+lz**2),
                np.sqrt(gx**2+gy**2+gz**2)]:
        if len(mag) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([np.mean(mag), np.std(mag), np.min(mag), np.max(mag)])

    # Segment duration and sample count (physics-informed context)
    t_arr = df['time_rel_sec'].values
    dur_s = t_arr[seg_end] - t_arr[seg_start]
    n_samp = seg_end - seg_start + 1
    feats.extend([dur_s, float(n_samp)])

    return np.array(feats, dtype=np.float32)

# Build feature names list
_feat_names = []
for k in SIGNAL_KEYS:
    for s in STATS:
        _feat_names.append(f"{k}_{s}")
for mag_name in ['a_mag', 'al_mag', 'w_mag']:
    for s in STATS:
        _feat_names.append(f"{mag_name}_{s}")
_feat_names += ['seg_dur_s', 'seg_n_samples']
FEAT_NAMES = _feat_names

# ====================================================================
# Build dataset — one row per inter-GNSS segment
# ====================================================================
def build_segment_dataset():
    print("\n" + "="*72)
    print("BUILDING PER-SEGMENT DATASET")
    print("="*72)

    all_X, all_yA, all_yB, all_yC, all_meta = [], [], [], [], []
    SKIP = {'S-Vfa01.parquet', 'S-Vfa02.parquet'}

    for sf in sorted(PROCESSED_DIR.glob("S-V*.parquet")):
        if sf.name in SKIP:
            continue
        try:
            df = pd.read_parquet(sf)
        except Exception as ex:
            print(f"  SKIP {sf.stem}: {ex}")
            continue

        if 'acc_veh_fwd' not in df.columns:
            try:
                from phase4_orientation import transform_to_vehicle_frame
                df, _, _ = transform_to_vehicle_frame(df)
            except Exception as ex:
                print(f"  SKIP {sf.stem} (Phase 4 transform failed): {ex}")
                continue

        gnss_idx = find_genuine_gnss_idx(df)
        v_gps    = df['gps_speed_ms'].values
        t_arr    = df['time_rel_sec'].values
        lat0, lon0 = df['gps_lat'].values[0], df['gps_lon'].values[0]
        east_all, north_all = latlon_to_enu(
            df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)

        n_segs = 0
        for k in range(len(gnss_idx) - 1):
            i0, i1 = int(gnss_idx[k]), int(gnss_idx[k+1])
            if i1 - i0 < 2:
                continue
            dur = t_arr[i1] - t_arr[i0]
            if dur <= 0 or dur > 30:   # skip very long/invalid gaps
                continue

            v_k   = float(v_gps[i0])
            v_k1  = float(v_gps[i1])
            if v_k < 0.5 and v_k1 < 0.5:
                continue   # skip purely stationary segments

            # Target A: instantaneous GPS speed at end of segment
            yA = v_k1

            # Target B: velocity change
            yB = v_k1 - v_k

            # Target C: displacement correction
            de = east_all[i1]  - east_all[i0]
            dn = north_all[i1] - north_all[i0]
            ds_gnss = float(np.sqrt(de**2 + dn**2))
            ds_imu, v_end_imu = imu_displacement_segment(df, i0, i1)
            yC = ds_gnss - ds_imu

            feat = segment_features(df, i0, i1)
            all_X.append(feat)
            all_yA.append(yA)
            all_yB.append(yB)
            all_yC.append(yC)
            all_meta.append({
                'session':    sf.stem,
                'seg_start':  i0,
                'seg_end':    i1,
                't_start_s':  t_arr[i0],
                't_end_s':    t_arr[i1],
                'dur_s':      dur,
                'v_k':        v_k,
                'v_k1':       v_k1,
                'ds_gnss':    ds_gnss,
                'ds_imu':     ds_imu,
            })
            n_segs += 1

        if n_segs > 0:
            print(f"  {sf.stem:<22}  {n_segs:>4} segments  "
                  f"v=[{v_gps.min():.1f},{v_gps.max():.1f}] m/s")

    X_all    = np.vstack(all_X)
    yA_all   = np.array(all_yA, dtype=np.float32)
    yB_all   = np.array(all_yB, dtype=np.float32)
    yC_all   = np.array(all_yC, dtype=np.float32)
    meta_all = pd.DataFrame(all_meta)

    print(f"\n  Total segments: {len(yA_all):,}  |  Features: {X_all.shape[1]}")
    print(f"  yA (v_GNSS):     mean={yA_all.mean():.2f}  std={yA_all.std():.2f} m/s")
    print(f"  yB (Δv):         mean={yB_all.mean():.3f}  std={yB_all.std():.3f} m/s")
    print(f"  yC (Δs_corr):    mean={yC_all.mean():.2f}  std={yC_all.std():.2f} m")

    return X_all, yA_all, yB_all, yC_all, meta_all

# ====================================================================
# Model factory
# ====================================================================
def build_models():
    return {
        'Ridge':        Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_SEED),
        'XGBoost':      XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=RANDOM_SEED, verbosity=0),
        'MLP':          MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            max_iter=300, early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_SEED, learning_rate_init=1e-3),
    }

# ====================================================================
# LOTO CV for one target
# ====================================================================
def loto_cv_target(X_all, y_all, meta_all, target_name,
                   naive_strategy='zero'):
    """
    naive_strategy:
      'zero'     → predict 0 for every sample (correct for correction targets B, C)
      'last'     → predict last known value per session (for reference target A)
    """
    print(f"\n  --- Target {target_name} (naive={naive_strategy}) ---")
    round_results = []

    for test_trip, trip_desc in LOTO_ROUNDS:
        is_test  = meta_all['session'].str.startswith(test_trip)
        is_train = ~is_test
        n_test, n_train = is_test.sum(), is_train.sum()

        if n_test == 0:
            continue

        X_tr, y_tr = X_all[is_train], y_all[is_train]
        X_te, y_te = X_all[is_test],  y_all[is_test]

        # Naive baseline
        if naive_strategy == 'zero':
            naive_pred = np.zeros_like(y_te)
        else:  # 'last' — persistence within session
            naive_pred = np.zeros_like(y_te)
            meta_te = meta_all[is_test].reset_index(drop=True)
            for sess in meta_te['session'].unique():
                m = meta_te['session'] == sess
                vals = y_te[m.values]
                naive_pred[m.values] = np.concatenate([[vals[0]], vals[:-1]])

        naive_rmse = float(np.sqrt(mean_squared_error(y_te, naive_pred)))
        round_results.append({
            'round': test_trip, 'model': f'Naive ({naive_strategy})',
            'RMSE': naive_rmse, 'pct_imp': 0.0, 'n_test': int(n_test)
        })

        # Scale for Ridge / MLP
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_te_sc  = scaler.transform(X_te)

        models = build_models()
        for mname, model in models.items():
            X_tr_in = X_tr_sc if mname in ('Ridge', 'MLP') else X_tr
            X_te_in = X_te_sc if mname in ('Ridge', 'MLP') else X_te
            model.fit(X_tr_in, y_tr)
            pred    = model.predict(X_te_in)
            rmse    = float(np.sqrt(mean_squared_error(y_te, pred)))
            pct_imp = (naive_rmse - rmse) / max(abs(naive_rmse), 1e-9) * 100
            round_results.append({
                'round': test_trip, 'model': mname,
                'RMSE': rmse, 'pct_imp': pct_imp, 'n_test': int(n_test)
            })

    df_res = pd.DataFrame(round_results)
    return df_res

# ====================================================================
# Feature importance + generalization risk check
# ====================================================================
def feature_importance_report(X_all, y_all, meta_all, target_name, naive_rmse_avg):
    """Train XGBoost on full dataset, report top features, flag generalization risk."""
    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
    model.fit(X_all, y_all)
    imp = model.feature_importances_

    top5_idx = np.argsort(imp)[::-1][:10]
    print(f"\n  Top-10 Features for target {target_name} (XGBoost, full dataset):")
    vibration_features = {'acc_lin_z_std', 'acc_z_std', 'acc_lin_z_max',
                          'acc_lin_z_mean', 'a_mag_std', 'al_mag_std'}
    fwd_features       = {'acc_veh_fwd_mean', 'acc_veh_fwd_std',
                          'acc_lin_x_mean', 'acc_lin_x_std', 'al_mag_mean'}
    generalization_risk = False
    top5_names = []
    for rank, i in enumerate(top5_idx, 1):
        fname = FEAT_NAMES[i] if i < len(FEAT_NAMES) else f"feat_{i}"
        flag  = ''
        if fname in vibration_features and rank <= 5:
            flag = '  ⚠ VIBRATION FEATURE'
            generalization_risk = True
        top5_names.append(fname)
        print(f"    {rank:>2}. {fname:<35}  importance={imp[i]:.4f}{flag}")

    # Generalization risk assessment
    vib_in_top5 = any(FEAT_NAMES[i] in vibration_features
                      for i in top5_idx[:5] if i < len(FEAT_NAMES))
    fwd_in_top5 = any(FEAT_NAMES[i] in fwd_features
                      for i in top5_idx[:5] if i < len(FEAT_NAMES))

    print(f"\n  GENERALIZATION RISK ASSESSMENT for target {target_name}:")
    if vib_in_top5:
        print("  *** RISK FLAGGED: Vertical vibration features dominate top-5. ***")
        print("     This suggests the model may be learning road-surface signatures")
        print("     specific to training trips' road types (e.g. UK motorway vs city).")
        print("     Performance on held-out trips with different road surfaces may")
        print("     degrade beyond what LOTO-CV alone can reveal (all trips are UK).")
    else:
        print("  OK: Forward-acceleration features dominate — physics-driven signal.")
    if fwd_in_top5:
        print("  POSITIVE: Forward-acceleration features present in top-5 —")
        print("     model is partly learning true vehicle kinematic signal.")

    # Importance bar chart
    fig, ax = plt.subplots(figsize=(9, 6))
    top_vals  = imp[top5_idx]
    top_names = [FEAT_NAMES[i] if i < len(FEAT_NAMES) else f"feat_{i}"
                 for i in top5_idx]
    colors    = ['tomato' if n in vibration_features else
                 'steelblue' if n in fwd_features else 'lightgray'
                 for n in top_names]
    ax.barh(range(10), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(10))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (gain)', fontsize=10)
    ax.set_title(f'Top-10 Features — Target {target_name}\n'
                 f'(red=vibration risk, blue=fwd-accel physics)', fontsize=10)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"phase8_feature_imp_{target_name}.png", dpi=150)
    plt.close()
    return vib_in_top5

# ====================================================================
# Summary pivot table
# ====================================================================
def print_pivot(df_res, target_name):
    models = df_res['model'].unique()
    rounds = [r[0] for r in LOTO_ROUNDS]
    rows   = []
    for m in models:
        row = {'Model': m}
        rmses = []
        for r in rounds:
            sub = df_res[(df_res['round'] == r) & (df_res['model'] == m)]
            if not sub.empty:
                v = sub['RMSE'].values[0]
                row[r] = round(float(v), 4)
                rmses.append(v)
            else:
                row[r] = '—'
        row['Avg'] = round(float(np.mean(rmses)), 4) if rmses else '—'
        rows.append(row)
    tbl = pd.DataFrame(rows)
    print(f"\n  Target {target_name} — RMSE per round:")
    print(tbl.to_string(index=False))
    return tbl

# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("PHASE 8: PHYSICS-ERROR-CORRECTION ML FORMULATION EXPERIMENT")
    print("=" * 72)
    print(f"  Features: {len(FEAT_NAMES)} per segment")
    print(f"  Targets: A=v_GNSS (ref)  B=Δv  C=Δs_correction")
    print(f"  Naive: A='last-known', B='zero', C='zero'")
    print(f"\n  HEADING NOTE:")
    print(f"  The current short-window ML formulation does not provide sufficiently")
    print(f"  generalizable heading estimates; therefore Phase 9 will retain")
    print(f"  physics-based gyro heading as the primary heading estimator.")

    # Build dataset
    X_all, yA_all, yB_all, yC_all, meta_all = build_segment_dataset()

    # Persist cache for downstream phases (Phase 9 reads this)
    CACHE_DIR = Path("data") / "phase8_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_all.to_parquet(CACHE_DIR / "meta_all.parquet")
    np.save(CACHE_DIR / "X_all.npy", X_all)
    np.save(CACHE_DIR / "yA_all.npy", yA_all)
    np.save(CACHE_DIR / "yB_all.npy", yB_all)
    np.save(CACHE_DIR / "yC_all.npy", yC_all)
    print(f"\n  Cache written to {CACHE_DIR} for Phase 9.")

    # ----------------------------------------------------------------
    # Run LOTO CV for all three targets
    # ----------------------------------------------------------------
    print("\n" + "="*72)
    print("LOTO CROSS-VALIDATION RESULTS")
    print("="*72)

    res_A = loto_cv_target(X_all, yA_all, meta_all, "A: v_GNSS",   naive_strategy='last')
    res_B = loto_cv_target(X_all, yB_all, meta_all, "B: Δv",        naive_strategy='zero')
    res_C = loto_cv_target(X_all, yC_all, meta_all, "C: Δs_corr",   naive_strategy='zero')

    tbl_A = print_pivot(res_A, "A: v_GNSS (ref)")
    tbl_B = print_pivot(res_B, "B: Δv")
    tbl_C = print_pivot(res_C, "C: Δs_corr")

    # Save CSVs
    res_A.to_csv(OUT_DIR / "phase8_results_A.csv", index=False)
    res_B.to_csv(OUT_DIR / "phase8_results_B.csv", index=False)
    res_C.to_csv(OUT_DIR / "phase8_results_C.csv", index=False)

    # ----------------------------------------------------------------
    # % improvement summary
    # ----------------------------------------------------------------
    print("\n" + "="*72)
    print("IMPROVEMENT OVER NAIVE BASELINE (avg across LOTO rounds)")
    print("="*72)
    for res, tname in [(res_A, "A: v_GNSS"), (res_B, "B: Δv"), (res_C, "C: Δs_corr")]:
        print(f"\n  {tname}:")
        for mname in ['Ridge', 'RandomForest', 'XGBoost', 'MLP']:
            sub = res[res['model'] == mname]
            if sub.empty:
                continue
            avg_imp = sub['pct_imp'].mean()
            min_imp = sub['pct_imp'].min()
            max_imp = sub['pct_imp'].max()
            print(f"    {mname:<18}  avg={avg_imp:+.1f}%  "
                  f"min={min_imp:+.1f}%  max={max_imp:+.1f}%")

    # ----------------------------------------------------------------
    # Feature importance for B and C (the new targets)
    # ----------------------------------------------------------------
    print("\n" + "="*72)
    print("FEATURE IMPORTANCE + GENERALIZATION RISK")
    print("="*72)
    vib_B = feature_importance_report(X_all, yB_all, meta_all, "B_delta_v",   0)
    vib_C = feature_importance_report(X_all, yC_all, meta_all, "C_delta_s",   0)

    # ----------------------------------------------------------------
    # Combined RMSE comparison plot (all targets, all models, avg across rounds)
    # ----------------------------------------------------------------
    print("\nGenerating comparison plots ...")

    targets_data = [
        ("A: v_GNSS", res_A),
        ("B: Δv",     res_B),
        ("C: Δs_corr",res_C),
    ]
    model_order  = ['Naive (zero)', 'Naive (last)', 'Ridge', 'RandomForest', 'XGBoost', 'MLP']
    colours_map  = {
        'Naive (zero)': 'gray', 'Naive (last)': 'dimgray',
        'Ridge': 'lightblue', 'RandomForest': 'green',
        'XGBoost': 'steelblue', 'MLP': 'orange',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    for ax, (tname, res) in zip(axes, targets_data):
        models_present = res['model'].unique()
        avgs = []
        names = []
        for m in models_present:
            sub = res[res['model'] == m]
            avgs.append(sub['RMSE'].mean())
            names.append(m)
        cols  = [colours_map.get(n, 'purple') for n in names]
        bars  = ax.bar(range(len(names)), avgs, color=cols)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        ax.set_title(f"Target {tname}\n(avg across 4 LOTO rounds)", fontsize=10)
        ax.set_ylabel('RMSE' + (' (m/s)' if 'v' in tname.lower() or 'Δv' in tname else ' (m)'))
        ax.grid(axis='y', alpha=0.4)
        for bar, v in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, v * 1.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Phase 8 — All Targets: Avg RMSE Across LOTO Rounds', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase8_target_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUT_DIR / 'phase8_target_comparison.png'}")

    # Per-round line chart for B and C (best model = XGBoost)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (res, tname, unit) in zip(axes2, [
            (res_B, "B: Δv (velocity change)", "m/s"),
            (res_C, "C: Δs_corr (displacement correction)", "m")]):
        rounds = [r[0] for r in LOTO_ROUNDS]
        for mname, col, ls in [
                ('Naive (zero)', 'gray', '--'),
                ('XGBoost',      'steelblue', '-o'),
                ('MLP',          'orange', '-s'),
                ('RandomForest', 'green', '-^')]:
            vals = []
            for r in rounds:
                sub = res[(res['round'] == r) & (res['model'] == mname)]
                vals.append(sub['RMSE'].values[0] if not sub.empty else np.nan)
            ax.plot(rounds, vals, ls, color=col, linewidth=2,
                    markersize=7, label=mname)
        ax.set_title(f"Target {tname}\nRMSE per LOTO round", fontsize=10)
        ax.set_ylabel(f"RMSE ({unit})", fontsize=10)
        ax.set_xlabel("Held-Out Trip", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase8_BC_per_round.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUT_DIR / 'phase8_BC_per_round.png'}")

    # ----------------------------------------------------------------
    # Decision summary
    # ----------------------------------------------------------------
    print("\n" + "="*72)
    print("PHASE 8 DECISION SUMMARY")
    print("="*72)

    # Extract best model avg RMSE for B and C
    def best_model_avg(res):
        models_ml = ['Ridge', 'RandomForest', 'XGBoost', 'MLP']
        best_rmse = 1e9
        best_name = ''
        for m in models_ml:
            sub = res[res['model'] == m]
            if sub.empty:
                continue
            avg = sub['RMSE'].mean()
            if avg < best_rmse:
                best_rmse = avg
                best_name = m
        return best_name, best_rmse

    naive_A = res_A[res_A['model'] == 'Naive (last)']['RMSE'].mean()
    naive_B = res_B[res_B['model'] == 'Naive (zero)']['RMSE'].mean()
    naive_C = res_C[res_C['model'] == 'Naive (zero)']['RMSE'].mean()
    bestA_n, bestA_r = best_model_avg(res_A)
    bestB_n, bestB_r = best_model_avg(res_B)
    bestC_n, bestC_r = best_model_avg(res_C)

    impA = (naive_A - bestA_r) / max(abs(naive_A), 1e-9) * 100
    impB = (naive_B - bestB_r) / max(abs(naive_B), 1e-9) * 100
    impC = (naive_C - bestC_r) / max(abs(naive_C), 1e-9) * 100

    print(f"""
  Target A (v_GNSS  — Phase 7 ref):
    Naive RMSE: {naive_A:.4f} m/s  |  Best model ({bestA_n}): {bestA_r:.4f} m/s
    Improvement: {impA:+.1f}%
    Status: Phase 7 finding confirmed on segment formulation.

  Target B (Δv — velocity change per segment):
    Naive RMSE (predict 0): {naive_B:.4f} m/s  |  Best model ({bestB_n}): {bestB_r:.4f} m/s
    Improvement: {impB:+.1f}%
    Vibration features dominant: {vib_B}

  Target C (Δs_corr — displacement correction per segment):
    Naive RMSE (predict 0): {naive_C:.4f} m    |  Best model ({bestC_n}): {bestC_r:.4f} m
    Improvement: {impC:+.1f}%
    Vibration features dominant: {vib_C}

  PHASE 9 TARGET RECOMMENDATION (do NOT decide yet — report only):
    - If impB > 0 and not vib_B: carry Target B into Phase 9 (correct Δv).
    - If impC > 0 and not vib_C: carry Target C into Phase 9 (correct Δs directly).
    - If vibration features dominate both: flag generalization risk prominently;
      consider adding road-type indicator features or segment-smoothing.
    - Phase 9 architecture decision deferred until user reviews this report.

  HEADING FOR PHASE 9:
    The current short-window ML formulation does not provide sufficiently
    generalizable heading estimates; therefore Phase 9 will retain
    physics-based gyro heading as the primary heading estimator.
""")

    print("="*72)
    print("PHASE 8 COMPLETE")
    print("="*72)
    print(f"  Results: {OUT_DIR.resolve()}")
