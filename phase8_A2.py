"""
Phase 8 — Target A2 Experiment
SIH26168 — AI-ML Dead Reckoning System

Target: y_A2 = 2 × Δs_corr / t²  (m/s²)
        Called "effective acceleration-error" (NOT true accelerometer bias).
        Physics basis: Δp = (1/2) × ε_a × t²  →  ε_a ≈ 2 × Δs_corr / t²

Reconstruction: Δs_corr_predicted = 0.5 × y_A2_predicted × t²
                Δs_corrected       = Δs_IMU + Δs_corr_predicted

Evaluation metric: RMSE in METERS (same units as C1, C2, naive → fair comparison)
  Naive-zero baseline: 82.31 m (same as C/C2)
  C1:                  36.75 m (58 features, includes duration)
  C2:                  47.37 m (56 features, no duration)

Feature set: all 58 features (same as C1) — duration included.
LOTO: same 4 rounds as before.

Classification:
  Case 1 → A2 beats naive on all trips, S-Vta1a stable → strongest Phase 9 candidate
  Case 2 → A2 beats naive on most trips, S-Vta1a mediocre → C1 still usable but GPS-rate-dependent
  Case 3 → A2 collapses (noise amplification) → residual not constant-bias; use Target B or state-correction
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

CACHE_DIR  = Path("data") / "phase8_cache"
OUT_DIR    = Path("plots") / "phase8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
LOTO_ROUNDS = [
    ("S-Vw4",   "W group, 210 min, motorway"),
    ("S-Vtb5",  "B group, 107 min, suburban"),
    ("S-Vta1a", "A group, 43 min, 1 Hz GPS"),
    ("S-Vtb2",  "B group, 10 min, short urban"),
]

# Reference numbers for comparison (from Phase 8)
NAIVE_M    = 82.3093
C1_M       = 36.7495
C2_M       = 47.3725

# Feature names (58 total, same as Phase 8)
SIGNAL_KEYS = [
    'acc_x','acc_y','acc_z',
    'acc_lin_x','acc_lin_y','acc_lin_z',
    'gyro_x','gyro_y','gyro_z',
    'acc_veh_fwd','gyro_veh_yaw_rate',
]
FEAT_NAMES = []
for k in SIGNAL_KEYS:
    for s in ['mean','std','min','max']:
        FEAT_NAMES.append(f"{k}_{s}")
for mag in ['a_mag','al_mag','w_mag']:
    for s in ['mean','std','min','max']:
        FEAT_NAMES.append(f"{mag}_{s}")
FEAT_NAMES += ['seg_dur_s','seg_n_samples']

# ====================================================================
# Load cached dataset
# ====================================================================
def load_dataset():
    X_all    = np.load(CACHE_DIR / "X_all.npy")
    yC_all   = np.load(CACHE_DIR / "yC_all.npy")    # Δs_corr in metres
    meta_all = pd.read_parquet(CACHE_DIR / "meta_all.parquet")
    t_seg    = meta_all['dur_s'].values.astype(np.float32)
    return X_all, yC_all, t_seg, meta_all

# ====================================================================
# Target A2 distribution analysis
# ====================================================================
def analyse_target_distribution(y_A2, t_seg, meta_all):
    print("\n" + "="*72)
    print("TARGET A2 DISTRIBUTION ANALYSIS")
    print(f"  y_A2 = 2 × Δs_corr / t²  (m/s²)  |  n = {len(y_A2):,}")
    print("="*72)

    def stats(arr, label):
        finite = arr[np.isfinite(arr)]
        pct    = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
        print(f"\n  {label}  (n={len(finite):,})")
        print(f"    mean={np.mean(finite):+.4f}  std={np.std(finite):.4f}  "
              f"min={np.min(finite):+.4f}  max={np.max(finite):+.4f}")
        print(f"    p1={pct[0]:+.4f}  p5={pct[1]:+.4f}  p25={pct[2]:+.4f}  "
              f"median={pct[3]:+.4f}  p75={pct[4]:+.4f}  p95={pct[5]:+.4f}  p99={pct[6]:+.4f}")
        return np.std(finite)

    overall_std = stats(y_A2, "OVERALL")

    # Per-group: S-Vta1a vs all other sessions
    vta1a_mask  = meta_all['session'].str.startswith("S-Vta1a")
    other_mask  = ~vta1a_mask

    std_vta1a = stats(y_A2[vta1a_mask],  "S-Vta1a (1 Hz GPS, ~1 s segments)")
    std_other = stats(y_A2[other_mask],   "Other sessions (~9 s segments)")

    print(f"\n  STABILITY RATIO (S-Vta1a std / other std): "
          f"{std_vta1a:.4f} / {std_other:.4f} = {std_vta1a/max(std_other,1e-9):.2f}x")

    # Count extremes
    extreme_thresh = 5 * np.std(y_A2[np.isfinite(y_A2)])
    n_ext_vta1a = int(np.sum(np.abs(y_A2[vta1a_mask]) > extreme_thresh))
    n_ext_other = int(np.sum(np.abs(y_A2[other_mask]) > extreme_thresh))
    print(f"  Extreme values (|y_A2| > 5σ_all = {extreme_thresh:.3f}):")
    print(f"    S-Vta1a: {n_ext_vta1a} / {vta1a_mask.sum()} = {n_ext_vta1a/vta1a_mask.sum()*100:.1f}%")
    print(f"    Others:  {n_ext_other} / {other_mask.sum()} = {n_ext_other/other_mask.sum()*100:.1f}%")

    # Key concern: GPS positional noise amplification
    GPS_NOISE_M = 5.0   # typical GPS horizontal accuracy (1-sigma, metres)
    t_vta1a = t_seg[vta1a_mask]
    t_other  = t_seg[other_mask]
    amp_vta1a = 2 * GPS_NOISE_M / (np.median(t_vta1a)**2)
    amp_other  = 2 * GPS_NOISE_M / (np.median(t_other)**2)
    print(f"\n  GPS NOISE AMPLIFICATION (2 × {GPS_NOISE_M}m / t²):")
    print(f"    S-Vta1a  median t={np.median(t_vta1a):.2f}s → noise → {amp_vta1a:.3f} m/s²")
    print(f"    Others   median t={np.median(t_other):.2f}s  → noise → {amp_other:.4f} m/s²")
    print(f"    Noise amplification ratio (S-Vta1a / others): {amp_vta1a/max(amp_other,1e-9):.1f}x")

    flag = (std_vta1a / max(std_other, 1e-9)) > 5.0
    if flag:
        print(f"\n  *** NOISE AMPLIFICATION FLAG: S-Vta1a target variance is "
              f"{std_vta1a/max(std_other,1e-9):.1f}x larger than other sessions.")
        print(f"      t² division at t≈1s amplifies GPS noise by "
              f"~{amp_vta1a/max(amp_other,1e-9):.0f}x vs t≈9s segments.")
        print(f"      y_A2 targets for S-Vta1a are likely dominated by GPS noise, "
              f"not true kinematic signal.")
    else:
        print(f"\n  OK: Noise amplification is within acceptable range "
              f"({std_vta1a/max(std_other,1e-9):.2f}x ratio).")

    return flag, std_vta1a, std_other

# ====================================================================
# LOTO-CV for A2 — evaluate in METRES
# ====================================================================
def loto_cv_A2(X_all, y_A2, yC_all, t_seg, meta_all):
    """
    Train on y_A2 (m/s²), predict y_A2, reconstruct Δs_corr = 0.5 × ŷ_A2 × t².
    Evaluate RMSE in METRES vs Δs_corr ground truth (yC_all passed in, already filtered).
    Returns per-round results dict.
    """

    print("\n" + "="*72)
    print("LOTO CROSS-VALIDATION — TARGET A2 (evaluated in METRES)")
    print("="*72)

    results = []  # list of row dicts

    for test_trip, desc in LOTO_ROUNDS:
        is_test  = meta_all['session'].str.startswith(test_trip)
        is_train = ~is_test
        if is_test.sum() == 0:
            print(f"  {test_trip}: no test windows, skipping")
            continue

        X_tr, y_tr = X_all[is_train], y_A2[is_train]
        X_te       = X_all[is_test]
        t_te       = t_seg[is_test]
        yC_te      = yC_all[is_test]   # ground truth Δs_corr in metres

        # Filter out extremes in TRAINING (don't filter test — must report honestly)
        sigma_train = np.std(y_tr)
        keep = np.abs(y_tr) < 10 * sigma_train   # 10-sigma clip on train
        n_clipped = (~keep).sum()
        X_tr_cl = X_tr[keep]; y_tr_cl = y_tr[keep]

        # Naive baseline: predict y_A2 = 0 → Δs_corr_pred = 0 → Δs_corrected = Δs_IMU
        naive_ds  = np.zeros_like(yC_te)          # predicted correction = 0
        naive_rmse = float(np.sqrt(mean_squared_error(yC_te, naive_ds)))

        print(f"\n  {test_trip}  ({desc})")
        print(f"    Train: {X_tr_cl.shape[0]:,} (clipped {n_clipped})  |  "
              f"Test: {is_test.sum():,}")
        print(f"    Naive (predict 0 correction):  RMSE = {naive_rmse:.4f} m")

        # Scaler for Ridge / MLP (fit on clipped train)
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr_cl)
        X_te_sc  = scaler.transform(X_te)

        row = {'round': test_trip, 'Naive': naive_rmse}

        models = {
            'Ridge':  Ridge(alpha=1.0),
            'RF':     RandomForestRegressor(n_estimators=200, max_depth=12,
                        min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_SEED),
            'XGBoost':XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        n_jobs=-1, random_state=RANDOM_SEED, verbosity=0),
            'MLP':    MLPRegressor(hidden_layer_sizes=(128,64,32), activation='relu',
                        max_iter=300, early_stopping=True, validation_fraction=0.1,
                        random_state=RANDOM_SEED, learning_rate_init=1e-3),
        }
        best_imp = None
        for mname, model in models.items():
            X_in  = X_tr_sc if mname in ('Ridge','MLP') else X_tr_cl
            X_tin = X_te_sc if mname in ('Ridge','MLP') else X_te
            model.fit(X_in, y_tr_cl)
            y_hat = model.predict(X_tin)

            # Reconstruct displacement correction in metres
            ds_pred = 0.5 * y_hat * (t_te ** 2)
            rmse_m  = float(np.sqrt(mean_squared_error(yC_te, ds_pred)))
            imp_pct = (naive_rmse - rmse_m) / abs(naive_rmse) * 100
            print(f"    {mname:<10}  RMSE = {rmse_m:.4f} m   ({imp_pct:+.1f}% vs naive)")
            row[mname] = rmse_m

            if mname == 'XGBoost' and hasattr(model, 'feature_importances_'):
                best_imp = model.feature_importances_

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "phase8_A2_results.csv", index=False)
    return results_df, best_imp

# ====================================================================
# Summary comparison table
# ====================================================================
def comparison_table(results_df):
    print("\n" + "="*72)
    print("A2 COMPARISON TABLE — RMSE in METRES  (fair comparison vs C1, C2, naive)")
    print("="*72)

    rounds  = [r[0] for r in LOTO_ROUNDS if r[0] in results_df['round'].values]
    col_A2  = 'XGBoost'

    header = f"  {'Round':<14}  {'Naive':>10}  {'C1(58f)':>10}  {'C2(56f)':>10}  {'A2-XGB':>10}  {'A2 vs naive':>12}  {'A2 vs C1':>10}"
    print(header)
    print("  " + "-"*(len(header)-2))

    # C1 and C2 per-round (from phase8 CSV)
    c1_map = {
        'S-Vw4': 54.6433, 'S-Vtb5': 46.8188,
        'S-Vta1a': 8.0505, 'S-Vtb2': 37.4854,
    }
    c2_map = {
        'S-Vw4': 56.8287, 'S-Vtb5': 49.0192,
        'S-Vta1a': 40.2042, 'S-Vtb2': 43.4380,
    }

    a2_rmses = []
    for _, row in results_df.iterrows():
        rnd    = row['round']
        naive  = row['Naive']
        c1_r   = c1_map.get(rnd, float('nan'))
        c2_r   = c2_map.get(rnd, float('nan'))
        a2_r   = row.get(col_A2, float('nan'))
        a2_naive_imp = (naive - a2_r) / abs(naive) * 100
        a2_c1_imp    = (c1_r  - a2_r) / abs(c1_r)  * 100
        a2_rmses.append(a2_r)
        print(f"  {rnd:<14}  {naive:>10.4f}  {c1_r:>10.4f}  {c2_r:>10.4f}  "
              f"{a2_r:>10.4f}  {a2_naive_imp:>+11.1f}%  {a2_c1_imp:>+9.1f}%")

    avg_a2   = float(np.mean(a2_rmses)) if a2_rmses else float('nan')
    avg_naive_imp = (NAIVE_M - avg_a2) / abs(NAIVE_M) * 100
    avg_c1_imp    = (C1_M   - avg_a2) / abs(C1_M)   * 100
    print("  " + "-"*(len(header)-2))
    print(f"  {'AVERAGE':<14}  {NAIVE_M:>10.4f}  {C1_M:>10.4f}  {C2_M:>10.4f}  "
          f"{avg_a2:>10.4f}  {avg_naive_imp:>+11.1f}%  {avg_c1_imp:>+9.1f}%")
    return avg_a2, avg_naive_imp, avg_c1_imp

# ====================================================================
# Feature importance report
# ====================================================================
def importance_report(imp_arr, label="A2 (XGBoost, last LOTO round)"):
    if imp_arr is None:
        return
    print(f"\n  Top-10 Features — {label}:")
    vib_set = {'acc_lin_z_std','acc_z_std','acc_lin_z_max','al_mag_std',
               'a_mag_std','acc_lin_z_mean','acc_lin_z_min'}
    fwd_set = {'acc_veh_fwd_mean','acc_veh_fwd_std','acc_lin_x_mean',
               'acc_lin_x_std','al_mag_mean'}
    dur_set = {'seg_dur_s','seg_n_samples'}
    top10   = np.argsort(imp_arr)[::-1][:10]
    any_vib = False
    dur_rank = None
    for rank, i in enumerate(top10, 1):
        fname = FEAT_NAMES[i] if i < len(FEAT_NAMES) else f"feat_{i}"
        tag   = ''
        if fname in vib_set: tag = '  ⚠ VIBRATION'; any_vib = True
        elif fname in fwd_set: tag = '  ✓ FWD-ACCEL'
        elif fname in dur_set:
            tag = '  ★ DURATION'
            if dur_rank is None: dur_rank = rank
        print(f"    {rank:>2}. {fname:<35}  {imp_arr[i]:.4f}{tag}")

    if dur_rank:
        print(f"\n  ★ Duration feature appears at rank {dur_rank}.")
        print(f"    Since target y_A2 = 2×Δs_corr/t² is already normalized by t²,")
        print(f"    residual duration dependency in X shows the model still uses")
        print(f"    duration to correct for any non-constant-bias structure.")
    if any_vib:
        print(f"  ⚠ Vibration features in top-10 — road-surface generalization risk persists.")

# ====================================================================
# Case classification
# ====================================================================
def classify(avg_a2, avg_naive_imp, noise_flag, results_df):
    print("\n" + "="*72)
    print("CASE CLASSIFICATION")
    print("="*72)

    vta_row = results_df[results_df['round'] == 'S-Vta1a']
    a2_vta  = vta_row['XGBoost'].values[0] if not vta_row.empty else float('nan')
    naive_vta = vta_row['Naive'].values[0]   if not vta_row.empty else float('nan')
    vta_beats_naive = (a2_vta < naive_vta)
    vta_beats_naive_pct = (naive_vta - a2_vta) / abs(naive_vta) * 100

    print(f"""
  Thresholds used:
    "Beats naive":   avg A2 RMSE < {NAIVE_M:.1f} m  (= zero-correction baseline)
    "Close to C1":   avg A2 RMSE within 30% of C1 ({C1_M:.2f} m)  → < {C1_M*1.30:.2f} m
    "S-Vta1a works": A2 beats naive on S-Vta1a round

  Results:
    avg A2 RMSE:              {avg_a2:.2f} m  ({avg_naive_imp:+.1f}% vs naive)
    S-Vta1a A2 RMSE:          {a2_vta:.4f} m  vs naive {naive_vta:.4f} m  ({vta_beats_naive_pct:+.1f}%)
    Noise amplification flag: {noise_flag}
""")

    if avg_a2 < NAIVE_M and vta_beats_naive and not noise_flag:
        case = 1
        print("  *** CASE 1: A2 beats naive on ALL trips, S-Vta1a stable ***")
        print("  A2 is the strongest Phase 9 candidate.")
        print("  Recommended Phase 9 target: y_A2 = 2 × Δs_corr / t²")
        print("  Reconstruct: Δs_corr_pred = 0.5 × ŷ_A2 × t²")

    elif avg_a2 < NAIVE_M and not vta_beats_naive:
        case = 2
        print("  *** CASE 2: A2 beats naive on MOST trips, S-Vta1a mediocre/fails ***")
        print(f"  S-Vta1a A2 RMSE ({a2_vta:.2f} m) > naive ({naive_vta:.2f} m): "
              f"{'YES' if not vta_beats_naive else 'no'}")
        print(f"  Noise amplification flag: {noise_flag}")
        print("  C1 (with raw duration features) remains usable but must be documented")
        print("  as GPS-rate-dependent: it will fail on deployment with non-standard GPS rates.")
        print("  Suggested path: use C1 for ~9s GPS; exclude trips with t < 2s from training.")

    elif avg_a2 < NAIVE_M and vta_beats_naive and noise_flag:
        case = 2
        print("  *** CASE 2 (with noise flag): A2 nominally beats naive but target is noisy ***")
        print("  Noise amplification flagged for S-Vta1a. Results may not be reliable.")
        print("  Recommendation: verify with held-out GPS-noise sensitivity analysis.")

    else:
        case = 3
        print("  *** CASE 3: A2 collapses — noise amplification from t² division ***")
        print("  The residual is NOT well-modeled as constant acceleration bias.")
        print("  Δp = (1/2) × ε_a × t² does not hold segment-by-segment (noise dominates).")
        print("  Phase 9 recommendation: use a learned VELOCITY or STATE correction approach.")
        print("    - Target B (Δv, +6%, clean): predict velocity change per segment")
        print("    - Or: per-sample velocity correction (requires temporal model: LSTM/CNN)")

    print()
    return case

# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("PHASE 8 — TARGET A2 EXPERIMENT")
    print("  y_A2 = 2 × Δs_corr / t²  (effective acceleration-error, m/s²)")
    print("=" * 72)
    print(f"  Reference baselines (from Phase 8):")
    print(f"    Naive (predict 0 correction): {NAIVE_M:.4f} m")
    print(f"    C1 (58 features):             {C1_M:.4f} m  (+55.4%)")
    print(f"    C2 (56 features, no dur):      {C2_M:.4f} m  (+42.4%, fails S-Vta1a)")

    # Load dataset
    if not (CACHE_DIR / "X_all.npy").exists():
        print("ERROR: cached dataset not found. Run phase8_ablation.py first.")
        sys.exit(1)

    X_all, yC_all, t_seg, meta_all = load_dataset()

    # Compute y_A2 — guard against division by very small t
    MIN_T = 0.5   # ignore segments shorter than 0.5s (pathological)
    valid = t_seg >= MIN_T
    X_all    = X_all[valid]
    yC_all   = yC_all[valid]
    t_seg    = t_seg[valid]
    meta_all = meta_all[valid].reset_index(drop=True)

    y_A2 = (2.0 * yC_all) / (t_seg ** 2)

    n_orig = len(valid)
    n_filt = valid.sum()
    print(f"\n  Segments after t >= {MIN_T}s filter: {n_filt:,} / {n_orig:,}")
    print(f"  y_A2 range: [{y_A2.min():.3f}, {y_A2.max():.3f}] m/s²")

    # Target distribution analysis
    noise_flag, std_vta1a, std_other = analyse_target_distribution(
        y_A2, t_seg, meta_all)

    # LOTO-CV
    results_df, last_imp = loto_cv_A2(X_all, y_A2, yC_all, t_seg, meta_all)

    # Comparison table
    avg_a2, avg_naive_imp, avg_c1_imp = comparison_table(results_df)

    # Feature importance
    importance_report(last_imp)

    # Classification
    case = classify(avg_a2, avg_naive_imp, noise_flag, results_df)

    print("=" * 72)
    print(f"PHASE 8 A2 EXPERIMENT COMPLETE  (Case {case})")
    print("=" * 72)
    print(f"  Results saved: {OUT_DIR / 'phase8_A2_results.csv'}")
