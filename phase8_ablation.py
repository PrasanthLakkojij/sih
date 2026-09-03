"""
Phase 8 — Target C Duration-Ablation Experiment
SIH26168 — AI-ML Dead Reckoning System

Tests whether the +55.4% improvement in Target C (Δs_correction) is genuine
IMU-derived kinematic signal or a duration proxy artifact.

C1 — All 58 features (baseline, reproduces Phase 8 result: avg RMSE 36.7m)
C2 — Duration-ablated: remove seg_n_samples and seg_dur_s (→ 56 features)

Interpretation:
  C2 RMSE << 82.3m (naive) and close to C1 36.7m → genuine kinematic signal → lock C as Phase 9 target
  C2 RMSE → 82.3m (naive)                          → duration proxy → do NOT use C as-is for Phase 9

Also saves dataset to disk so this script is re-runnable without full IMU re-integration.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

PROCESSED_DIR = Path("data") / "processed_sessions"
OUT_DIR       = Path("plots") / "phase8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR     = Path("data") / "phase8_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
LOTO_ROUNDS = [
    ("S-Vw4",   "driver W, 210 min, motorway"),
    ("S-Vtb5",  "driver B group, 107 min, suburban"),
    ("S-Vta1a", "driver A group, 43 min, 1 Hz GPS"),
    ("S-Vtb2",  "driver B group, 10 min, short urban"),
]

NAIVE_ZERO_AVG   = 82.3093   # from Phase 8 (predict-zero baseline for Δs_corr)
C1_PHASE8_AVG    = 36.7495   # from Phase 8 (all 58 features)

# ====================================================================
# Feature names (must match Phase 8 exactly)
# ====================================================================
SIGNAL_KEYS = [
    'acc_x', 'acc_y', 'acc_z',
    'acc_lin_x', 'acc_lin_y', 'acc_lin_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'acc_veh_fwd', 'gyro_veh_yaw_rate',
]
STATS = ['mean', 'std', 'min', 'max']
FEAT_NAMES = []
for k in SIGNAL_KEYS:
    for s in STATS:
        FEAT_NAMES.append(f"{k}_{s}")
for mag in ['a_mag', 'al_mag', 'w_mag']:
    for s in STATS:
        FEAT_NAMES.append(f"{mag}_{s}")
FEAT_NAMES += ['seg_dur_s', 'seg_n_samples']   # indices -2, -1

DURATION_FEATS = {'seg_dur_s', 'seg_n_samples'}
C2_FEAT_MASK   = [f not in DURATION_FEATS for f in FEAT_NAMES]   # True = keep
C2_FEAT_NAMES  = [f for f, keep in zip(FEAT_NAMES, C2_FEAT_MASK) if keep]

# ====================================================================
# Helpers (identical to Phase 8)
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

def imu_displacement_segment(df, seg_start, seg_end):
    a_fwd    = df['acc_veh_fwd'].values
    dt_arr   = df['dt_sec'].values
    gyro_yaw = df['gyro_veh_yaw_rate'].values
    az_col   = get_col(df, 'azimuth')
    psi_mag  = np.radians(df[az_col].values)
    bearing_col = get_col(df, 'gps_orientation') or get_col(df, 'gps_bearing')
    v0   = df['gps_speed_ms'].values[seg_start]
    if bearing_col:
        b0   = df[bearing_col].values[seg_start]
        psi0 = psi_mag[seg_start] if (np.isnan(b0) or b0 == 0) else np.radians(b0)
    else:
        psi0 = psi_mag[seg_start]
    v_dr   = v0; psi_dr = psi0; e_dr = 0.0; n_dr = 0.0
    for t in range(seg_start + 1, seg_end + 1):
        if t >= len(dt_arr): break
        dt      = dt_arr[t]
        v_dr    = max(0.0, v_dr + a_fwd[t] * dt)
        psi_dr += gyro_yaw[t] * dt
        e_dr   += v_dr * np.sin(psi_dr) * dt
        n_dr   += v_dr * np.cos(psi_dr) * dt
    return float(np.sqrt(e_dr**2 + n_dr**2))

def segment_features(df, seg_start, seg_end):
    def _s(k):
        c = get_col(df, k)
        return df[c].values[seg_start:seg_end+1] if c else np.zeros(seg_end-seg_start+1)
    feats = []
    for key in SIGNAL_KEYS:
        w = _s(key)
        feats += ([np.mean(w), np.std(w), np.min(w), np.max(w)] if len(w)
                  else [0.0, 0.0, 0.0, 0.0])
    for mag_fn in [
        lambda: np.sqrt(_s('acc_x')**2 + _s('acc_y')**2 + _s('acc_z')**2),
        lambda: np.sqrt(_s('acc_lin_x')**2 + _s('acc_lin_y')**2 + _s('acc_lin_z')**2),
        lambda: np.sqrt(_s('gyro_x')**2 + _s('gyro_y')**2 + _s('gyro_z')**2),
    ]:
        w = mag_fn()
        feats += ([np.mean(w), np.std(w), np.min(w), np.max(w)] if len(w)
                  else [0.0, 0.0, 0.0, 0.0])
    t_arr   = df['time_rel_sec'].values
    dur_s   = t_arr[seg_end] - t_arr[seg_start]
    n_samp  = seg_end - seg_start + 1
    feats  += [dur_s, float(n_samp)]
    return np.array(feats, dtype=np.float32)

# ====================================================================
# Build or load cached dataset
# ====================================================================
def build_or_load_dataset():
    X_path    = CACHE_DIR / "X_all.npy"
    yC_path   = CACHE_DIR / "yC_all.npy"
    meta_path = CACHE_DIR / "meta_all.parquet"

    if X_path.exists() and yC_path.exists() and meta_path.exists():
        print("  Loading cached dataset ...")
        X_all    = np.load(X_path)
        yC_all   = np.load(yC_path)
        meta_all = pd.read_parquet(meta_path)
        print(f"  Loaded: {len(yC_all):,} segments, {X_all.shape[1]} features")
        return X_all, yC_all, meta_all

    print("  Building dataset from scratch (caching for future runs) ...")
    SKIP = {'S-Vfa01.parquet', 'S-Vfa02.parquet'}
    all_X, all_yC, all_meta = [], [], []

    for sf in sorted(PROCESSED_DIR.glob("S-V*.parquet")):
        if sf.name in SKIP:
            continue
        try:
            df = pd.read_parquet(sf)
        except Exception as ex:
            print(f"    SKIP {sf.stem}: {ex}"); continue
        if 'acc_veh_fwd' not in df.columns:
            try:
                from phase4_orientation import transform_to_vehicle_frame
                df, _, _ = transform_to_vehicle_frame(df)
            except Exception as ex:
                print(f"    SKIP {sf.stem} (Phase4 fail): {ex}"); continue

        gnss_idx  = find_genuine_gnss_idx(df)
        v_gps     = df['gps_speed_ms'].values
        t_arr     = df['time_rel_sec'].values
        lat0, lon0 = df['gps_lat'].values[0], df['gps_lon'].values[0]
        east_all, north_all = latlon_to_enu(
            df['gps_lat'].values, df['gps_lon'].values, lat0, lon0)

        n_segs = 0
        for k in range(len(gnss_idx) - 1):
            i0, i1 = int(gnss_idx[k]), int(gnss_idx[k+1])
            if i1 - i0 < 2: continue
            dur = t_arr[i1] - t_arr[i0]
            if dur <= 0 or dur > 30: continue
            v_k, v_k1 = float(v_gps[i0]), float(v_gps[i1])
            if v_k < 0.5 and v_k1 < 0.5: continue
            de = east_all[i1] - east_all[i0]
            dn = north_all[i1] - north_all[i0]
            ds_gnss = float(np.sqrt(de**2 + dn**2))
            ds_imu  = imu_displacement_segment(df, i0, i1)
            yC      = ds_gnss - ds_imu
            feat    = segment_features(df, i0, i1)
            all_X.append(feat)
            all_yC.append(yC)
            all_meta.append({'session': sf.stem, 'seg_start': i0,
                             'seg_end': i1, 'dur_s': dur})
            n_segs += 1
        if n_segs > 0:
            print(f"    {sf.stem:<22}  {n_segs:>4} segs")

    X_all    = np.vstack(all_X)
    yC_all   = np.array(all_yC, dtype=np.float32)
    meta_all = pd.DataFrame(all_meta)
    np.save(X_path, X_all)
    np.save(yC_path, yC_all)
    meta_all.to_parquet(meta_path)
    print(f"  Cached: {len(yC_all):,} segments, {X_all.shape[1]} features")
    return X_all, yC_all, meta_all

# ====================================================================
# LOTO-CV for one feature set — Target C only
# ====================================================================
def loto_cv_C(X_all, yC_all, meta_all, feat_mask, tag):
    """
    Run 4-round LOTO-CV on Target C (Δs_corr).
    feat_mask: boolean array selecting which of the 58 columns to use.
    Returns dict: round→rmse, plus avg.
    """
    X_sub = X_all[:, feat_mask]
    results = {}
    for test_trip, _ in LOTO_ROUNDS:
        is_test  = meta_all['session'].str.startswith(test_trip)
        is_train = ~is_test
        if is_test.sum() == 0:
            print(f"    [{tag}] {test_trip}: no test windows, skipping")
            continue
        X_tr, y_tr = X_sub[is_train], yC_all[is_train]
        X_te, y_te = X_sub[is_test],  yC_all[is_test]
        model = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        results[test_trip] = (rmse, model.feature_importances_)
        print(f"    [{tag}] {test_trip:<12}  n_test={is_test.sum():>4}  RMSE = {rmse:.4f} m")
    return results

# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("PHASE 8 — TARGET C DURATION-ABLATION EXPERIMENT (C1 vs C2)")
    print("=" * 72)
    print(f"  C1: all {len(FEAT_NAMES)} features (baseline)")
    print(f"  C2: {sum(C2_FEAT_MASK)} features after removing: {sorted(DURATION_FEATS)}")
    print(f"  Naive-zero baseline: {NAIVE_ZERO_AVG:.4f} m (predict-zero for Δs_corr)")
    print(f"  C1 Phase 8 result:   {C1_PHASE8_AVG:.4f} m (all 58 features)\n")

    print("Building / loading dataset ...")
    X_all, yC_all, meta_all = build_or_load_dataset()

    # ------------------------------------------------------------------
    # C1 — All features (reproduce Phase 8 for verification)
    # ------------------------------------------------------------------
    print("\nRunning C1 (all features) ...")
    c1_mask = np.ones(len(FEAT_NAMES), dtype=bool)
    c1_res  = loto_cv_C(X_all, yC_all, meta_all, c1_mask, "C1")

    # ------------------------------------------------------------------
    # C2 — Duration-ablated
    # ------------------------------------------------------------------
    print("\nRunning C2 (duration-ablated) ...")
    c2_mask = np.array(C2_FEAT_MASK)
    c2_res  = loto_cv_C(X_all, yC_all, meta_all, c2_mask, "C2")

    # ------------------------------------------------------------------
    # Report comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("C1 vs C2 COMPARISON TABLE  —  Target C: Δs_corr (displacement correction)")
    print("=" * 72)
    rounds = [r[0] for r in LOTO_ROUNDS]
    header = f"  {'Round':<14}  {'Naive (0)':>10}  {'C1 (58 feat)':>13}  {'C2 (56 feat)':>13}  {'C1 imp':>8}  {'C2 imp':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    c1_rmses = []
    c2_rmses = []
    # Per-round naive baselines from Phase 8 raw CSV (or recompute inline)
    naive_per_round = {}
    for test_trip, _ in LOTO_ROUNDS:
        is_test = meta_all['session'].str.startswith(test_trip)
        if is_test.sum() == 0:
            naive_per_round[test_trip] = NAIVE_ZERO_AVG
            continue
        y_te = yC_all[is_test]
        naive_per_round[test_trip] = float(np.sqrt(np.mean(y_te**2)))  # RMSE of predict-0

    for rnd in rounds:
        naive = naive_per_round.get(rnd, NAIVE_ZERO_AVG)
        c1_r  = c1_res.get(rnd, (None, None))[0]
        c2_r  = c2_res.get(rnd, (None, None))[0]
        if c1_r is None or c2_r is None:
            continue
        c1_imp = (naive - c1_r) / abs(naive) * 100
        c2_imp = (naive - c2_r) / abs(naive) * 100
        c1_rmses.append(c1_r)
        c2_rmses.append(c2_r)
        print(f"  {rnd:<14}  {naive:>10.4f}  {c1_r:>13.4f}  {c2_r:>13.4f}  "
              f"{c1_imp:>+7.1f}%  {c2_imp:>+7.1f}%")

    c1_avg = float(np.mean(c1_rmses)) if c1_rmses else float('nan')
    c2_avg = float(np.mean(c2_rmses)) if c2_rmses else float('nan')
    naive_avg = NAIVE_ZERO_AVG
    c1_avg_imp = (naive_avg - c1_avg) / abs(naive_avg) * 100
    c2_avg_imp = (naive_avg - c2_avg) / abs(naive_avg) * 100

    print("  " + "-" * (len(header) - 2))
    print(f"  {'AVERAGE':<14}  {naive_avg:>10.4f}  {c1_avg:>13.4f}  {c2_avg:>13.4f}  "
          f"{c1_avg_imp:>+7.1f}%  {c2_avg_imp:>+7.1f}%")

    # ------------------------------------------------------------------
    # Top-5 feature importances for C2 (averaged across rounds)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("C2 — TOP-5 FEATURE IMPORTANCES (XGBoost, averaged across LOTO rounds)")
    print("=" * 72)
    imp_arrays = []
    for rnd in rounds:
        if rnd in c2_res and c2_res[rnd][1] is not None:
            imp_arrays.append(c2_res[rnd][1])
    if imp_arrays:
        imp_avg = np.mean(imp_arrays, axis=0)
        top5    = np.argsort(imp_avg)[::-1][:5]
        vib_set = {'acc_lin_z_std', 'acc_z_std', 'acc_lin_z_max', 'al_mag_std',
                   'a_mag_std', 'acc_lin_z_mean'}
        fwd_set = {'acc_veh_fwd_mean', 'acc_veh_fwd_std', 'acc_lin_x_mean',
                   'acc_lin_x_std', 'al_mag_mean'}
        any_vib = False
        print(f"  (Removed from C2: seg_n_samples, seg_dur_s)")
        print()
        for rank, i in enumerate(top5, 1):
            fname = C2_FEAT_NAMES[i] if i < len(C2_FEAT_NAMES) else f"feat_{i}"
            tag   = ''
            if fname in vib_set:
                tag = '  ⚠ VIBRATION'
                any_vib = True
            elif fname in fwd_set:
                tag = '  ✓ FWD-ACCEL'
            print(f"  {rank}. {fname:<35}  importance={imp_avg[i]:.4f}{tag}")
        print()
        if any_vib:
            print("  ⚠ RISK: Vibration features still present in C2 top-5.")
        else:
            print("  ✓ CLEAN: Vibration features absent from C2 top-5.")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    drop_vs_c1  = c2_avg - c1_avg       # positive = C2 is worse
    drop_pct    = drop_vs_c1 / abs(c1_avg) * 100
    gap_to_naive = naive_avg - c2_avg   # positive = C2 still beats naive

    print(f"""
  C1 (58 features, incl. duration)  avg RMSE = {c1_avg:.2f} m  ({c1_avg_imp:+.1f}% vs naive)
  C2 (56 features, no duration)     avg RMSE = {c2_avg:.2f} m  ({c2_avg_imp:+.1f}% vs naive)
  Naive-zero baseline                avg RMSE = {naive_avg:.2f} m

  RMSE increase from C1→C2: {drop_vs_c1:+.2f} m  ({drop_pct:+.1f}% relative degradation)
  C2 gap above naive:        {gap_to_naive:+.2f} m  (positive = C2 still better than naive)

  Physics-driven displacement error formula (for reference):
    Δp = (1/2) × ε_a × t²
  Stochastic sensor noise contributes additional non-deterministic uncertainty
  whose growth depends on the specific noise process and integration scheme.
""")

    if gap_to_naive > 20 and drop_pct < 50:
        verdict = "LOCK C AS PRIMARY"
        print(f"  VERDICT: *** {verdict} ***")
        print(f"  C2 improvement over naive ({c2_avg_imp:+.1f}%) is substantial and durable.")
        print(f"  The {drop_pct:+.1f}% RMSE rise from removing duration features is acceptable.")
        print(f"  The model is learning genuine IMU-derived kinematic correction, not")
        print(f"  merely a duration proxy. Target C (without duration features) can be")
        print(f"  carried into Phase 9 as the primary ML target.")
    elif gap_to_naive > 5 and drop_pct < 80:
        verdict = "CONDITIONAL — USE DURATION-NORMALIZED C"
        print(f"  VERDICT: *** {verdict} ***")
        print(f"  C2 still beats naive ({c2_avg_imp:+.1f}%), confirming partial genuine signal.")
        print(f"  But the {drop_pct:+.1f}% RMSE rise shows duration features carried significant")
        print(f"  load. Consider predicting correction-per-second (Δs_corr / seg_dur_s)")
        print(f"  as a duration-normalized variant before Phase 9.")
    else:
        verdict = "REJECT C — USE TARGET B OR DURATION-NORMALIZED C"
        print(f"  VERDICT: *** {verdict} ***")
        print(f"  C2 RMSE ({c2_avg:.2f} m) collapses toward naive ({naive_avg:.2f} m).")
        print(f"  The Phase 8 +55.4% improvement was predominantly duration-driven.")
        print(f"  Target C should NOT be used as-is for Phase 9.")
        print(f"  Options: (A) Target B (Δv, +6%, clean kinematics), or")
        print(f"           (B) Duration-normalized Target C: Δs_corr / seg_dur_s.")

    print("\n" + "=" * 72)
    print("PHASE 8 ABLATION COMPLETE")
    print("=" * 72)
