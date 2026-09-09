"""
Phase 7 — ML Velocity Estimation Experiment
SIH26168 — AI-ML Dead Reckoning System

TARGET: Instantaneous vehicle forward velocity (m/s) predicted from
        a 2-second sliding window of raw IMU signals.

DESIGN:
  - Window: 20 samples (2 s at 10 Hz), stride every IMU sample.
  - Labels: gps_speed_ms ONLY at genuine GNSS fix rows (is_gnss_update==True).
  - Features: mean, std, min, max per axis across the window
    (acc, acc_lin, gyro, mag + vehicle-frame acc_veh_fwd, gyro_veh_yaw_rate).
  - Validation: Leave-One-Trip-Out (LOTO) cross-validation over 4 held-out trips.
  - Models: Naive baseline | Ridge | RandomForest | XGBoost | MLP
  - Secondary diagnostic: heading (course) prediction with best velocity model arch.
  - NO LSTM/CNN/Transformer — escalation decision deferred to Phase 9.

Cross-validation rounds:
  Round 1  Test = S-Vw4      (driver W, 210 min, long motorway)
  Round 2  Test = S-Vtb5     (driver B, 107 min, suburban)
  Round 3  Test = S-Vta1a    (driver A, 43 min, 1 Hz GPS outlier)
  Round 4  Test = S-Vtb2     (driver B, 10 min, short urban)
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
OUT_DIR       = Path("plots") / "phase7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_LEN    = 20       # samples (2 s at 10 Hz)
MIN_SPEED_MS  = 0.5      # discard windows where GPS truth < 0.5 m/s (stationary noise)
LOTO_ROUNDS   = [
    ("S-Vw4",    "driver W, 210 min, motorway"),
    ("S-Vtb5",   "driver B, 107 min, suburban"),
    ("S-Vta1a",  "driver A,  43 min, 1 Hz GPS"),
    ("S-Vtb2",   "driver B,  10 min, short urban"),
]
RANDOM_SEED = 42

# ====================================================================
# Helpers
# ====================================================================
def get_col(df, keyword):
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None

def find_genuine_gnss_idx(df):
    """Return row indices of genuine GNSS fixes (from Phase 3 flag or lat-change detection)."""
    if 'is_gnss_update' in df.columns:
        idx = np.where(df['is_gnss_update'].values.astype(bool))[0]
        if len(idx) > 10:
            return idx
    lat = df['gps_lat'].values
    lon = df['gps_lon'].values
    changed = (np.abs(np.diff(lat)) > 1e-7) | (np.abs(np.diff(lon)) > 1e-7)
    return np.where(np.concatenate([[True], changed]))[0]

def gnss_course_rad(df, gnss_idx):
    """
    Compute course (heading) in radians from consecutive genuine GNSS fixes.
    Only valid where consecutive distance > 5 m.
    Returns array of shape (N,) with NaN where not computable.
    """
    lat0, lon0 = df['gps_lat'].values[0], df['gps_lon'].values[0]
    R = 6371000.0
    east  = R * np.radians(df['gps_lon'].values - lon0) * np.cos(np.radians(lat0))
    north = R * np.radians(df['gps_lat'].values - lat0)

    course = np.full(len(df), np.nan)
    for k in range(len(gnss_idx) - 1):
        i0, i1 = gnss_idx[k], gnss_idx[k+1]
        de = east[i1]  - east[i0]
        dn = north[i1] - north[i0]
        dist = np.sqrt(de**2 + dn**2)
        if dist > 5.0:
            b = np.arctan2(de, dn)
            b = b + 2*np.pi if b < 0 else b
            course[i1] = b   # assign to the FIX at i1 (end of segment)
    return course

# ====================================================================
# Feature extraction from one session
# ====================================================================
STAT_AXES = [
    'acc_x', 'acc_y', 'acc_z',
    'acc_lin_x', 'acc_lin_y', 'acc_lin_z',
    'gyro_x', 'gyro_y', 'gyro_z',
]

def _safe_col(df, name):
    """Return column values or zeros if column missing."""
    c = get_col(df, name)
    return df[c].values if c else np.zeros(len(df))

def extract_features_labels(df, session_name):
    """
    Returns (X, y_vel, y_course, meta_df) where rows correspond to
    windows ending at genuine GNSS fix rows with non-trivial speed.

    X          : (n_windows, n_features)  — windowed IMU stats
    y_vel      : (n_windows,)             — GPS speed at fix row (m/s)
    y_course   : (n_windows,)             — GNSS course at fix row (rad), NaN if not computable
    meta_df    : DataFrame with session, fix_row, t_fix, speed for traceability
    """
    if 'acc_veh_fwd' not in df.columns:
        from phase4_orientation import transform_to_vehicle_frame
        df, _, _ = transform_to_vehicle_frame(df)

    gnss_idx  = find_genuine_gnss_idx(df)
    course    = gnss_course_rad(df, gnss_idx)
    v_gps     = df['gps_speed_ms'].values
    t_arr     = df['time_rel_sec'].values

    # Build raw signal arrays for windowed stats
    signals   = {}
    for col in STAT_AXES:
        signals[col] = _safe_col(df, col)

    # Vehicle-frame (may exist after Phase 4 transform)
    signals['acc_veh_fwd']      = df['acc_veh_fwd'].values
    signals['gyro_veh_yaw_rate'] = df['gyro_veh_yaw_rate'].values

    # Magnitude features (derived)
    a_mag  = np.sqrt(signals['acc_x']**2 + signals['acc_y']**2 + signals['acc_z']**2)
    w_mag  = np.sqrt(signals['gyro_x']**2 + signals['gyro_y']**2 + signals['gyro_z']**2)
    al_mag = np.sqrt(signals['acc_lin_x']**2 + signals['acc_lin_y']**2 + signals['acc_lin_z']**2)
    signals['a_mag']  = a_mag
    signals['w_mag']  = w_mag
    signals['al_mag'] = al_mag

    stat_names = list(signals.keys())
    stats      = ['mean', 'std', 'min', 'max']
    feat_names = [f"{s}_{st}" for s in stat_names for st in stats]

    rows_X, rows_y_vel, rows_y_course, meta = [], [], [], []

    for fix_row in gnss_idx:
        if fix_row < WINDOW_LEN:
            continue
        v_target = v_gps[fix_row]
        if v_target < MIN_SPEED_MS:
            continue

        win_start = fix_row - WINDOW_LEN
        win_end   = fix_row          # inclusive

        feat_vec = []
        for sname in stat_names:
            window = signals[sname][win_start : win_end + 1]
            feat_vec.extend([
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
            ])

        rows_X.append(feat_vec)
        rows_y_vel.append(v_target)
        rows_y_course.append(course[fix_row])
        meta.append({
            'session':  session_name,
            'fix_row':  fix_row,
            't_fix_s':  t_arr[fix_row],
            'v_true':   v_target,
        })

    if not rows_X:
        return None, None, None, None

    X        = np.array(rows_X,     dtype=np.float32)
    y_vel    = np.array(rows_y_vel, dtype=np.float32)
    y_course = np.array(rows_y_course, dtype=np.float32)
    meta_df  = pd.DataFrame(meta)

    return X, y_vel, y_course, meta_df

# ====================================================================
# Build full dataset from all vehicle sessions
# ====================================================================
def build_dataset():
    print("\n" + "="*70)
    print("BUILDING WINDOWED IMU DATASET")
    print("="*70)

    all_X, all_y_vel, all_y_course, all_meta = [], [], [], []
    session_files = sorted(PROCESSED_DIR.glob("S-V*.parquet"))
    # Exclude malformed sessions noted in Phase 1
    SKIP = {'S-Vfa01.parquet', 'S-Vfa02.parquet'}

    for sf in session_files:
        if sf.name in SKIP:
            continue
        try:
            df = pd.read_parquet(sf)
            X, y_vel, y_course, meta_df = extract_features_labels(df, sf.stem)
            if X is None:
                continue
            all_X.append(X)
            all_y_vel.append(y_vel)
            all_y_course.append(y_course)
            all_meta.append(meta_df)
            print(f"  {sf.stem:<20}  {len(y_vel):>5} windows  "
                  f"v_mean={y_vel.mean():.2f} m/s  "
                  f"v_max={y_vel.max():.2f} m/s")
        except Exception as ex:
            print(f"  SKIP {sf.stem}: {ex}")

    X_all       = np.vstack(all_X)
    y_vel_all   = np.concatenate(all_y_vel)
    y_course_all= np.concatenate(all_y_course)
    meta_all    = pd.concat(all_meta, ignore_index=True)

    print(f"\n  Total dataset: {len(y_vel_all):,} windows, "
          f"{X_all.shape[1]} features, "
          f"v in [{y_vel_all.min():.2f}, {y_vel_all.max():.2f}] m/s")

    return X_all, y_vel_all, y_course_all, meta_all

# ====================================================================
# Model definitions
# ====================================================================
def build_models():
    return {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_SEED),
        'XGBoost': XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=RANDOM_SEED, verbosity=0),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_SEED, learning_rate_init=1e-3),
    }

# ====================================================================
# Naive baseline: last known velocity (persistence model)
# ====================================================================
def naive_last_known(meta_test_df, y_vel_test):
    """
    For each test window, predict the velocity of the PREVIOUS genuine fix
    in the same session. If no previous fix, predict the session mean.
    """
    preds = np.zeros_like(y_vel_test)
    meta_test_df = meta_test_df.reset_index(drop=True)
    for sess in meta_test_df['session'].unique():
        mask   = meta_test_df['session'] == sess
        idx    = meta_test_df[mask].index.tolist()
        speeds = y_vel_test[idx]
        prev   = np.concatenate([[speeds[0]], speeds[:-1]])  # shift by 1
        preds[idx] = prev
    return preds

# ====================================================================
# Leave-One-Trip-Out (LOTO) Cross-Validation
# ====================================================================
def run_loto_cv(X_all, y_vel_all, meta_all):
    print("\n" + "="*70)
    print("LEAVE-ONE-TRIP-OUT CROSS-VALIDATION — VELOCITY")
    print("="*70)

    all_results = []   # one row per (round, model)

    for test_trip, trip_desc in LOTO_ROUNDS:
        # Mask for test trip (handles sub-session names like S-Vtb1_sub01)
        is_test = meta_all['session'].str.startswith(test_trip)
        is_train = ~is_test

        n_test  = is_test.sum()
        n_train = is_train.sum()

        if n_test == 0:
            print(f"  [SKIP] {test_trip} — no windows found in dataset")
            continue

        X_train, y_train = X_all[is_train], y_vel_all[is_train]
        X_test,  y_test  = X_all[is_test],  y_vel_all[is_test]
        meta_test = meta_all[is_test].reset_index(drop=True)

        # Scale features (fit on train only — no leakage)
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        print(f"\n  Round: test={test_trip} ({trip_desc})")
        print(f"    Train: {n_train:,} windows  |  Test: {n_test:,} windows")

        # Naive baseline
        naive_pred = naive_last_known(meta_test, y_test)
        naive_rmse = float(np.sqrt(mean_squared_error(y_test, naive_pred)))
        print(f"    Naive (last-known velocity)  RMSE = {naive_rmse:.4f} m/s")

        all_results.append({
            'round': test_trip, 'model': 'Naive (last-known)',
            'n_test': n_test, 'RMSE_ms': naive_rmse,
            'pct_improvement': 0.0,
        })

        models = build_models()
        for mname, model in models.items():
            X_tr = X_tr_sc if mname in ('Ridge', 'MLP') else X_train
            X_te = X_te_sc if mname in ('Ridge', 'MLP') else X_test

            model.fit(X_tr, y_train)
            pred    = model.predict(X_te)
            pred    = np.clip(pred, 0.0, None)   # velocity >= 0
            rmse    = float(np.sqrt(mean_squared_error(y_test, pred)))
            pct_imp = (naive_rmse - rmse) / naive_rmse * 100

            print(f"    {mname:<18}             RMSE = {rmse:.4f} m/s  "
                  f"  ({pct_imp:+.1f}% vs naive)")
            all_results.append({
                'round': test_trip, 'model': mname,
                'n_test': n_test, 'RMSE_ms': rmse,
                'pct_improvement': pct_imp,
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "phase7_loto_results.csv", index=False)
    return results_df

# ====================================================================
# Summary pivot table
# ====================================================================
def print_summary_table(results_df):
    print("\n" + "="*70)
    print("PHASE 7 VELOCITY — CROSS-VALIDATION SUMMARY (RMSE in m/s)")
    print("="*70)

    models  = results_df['model'].unique()
    rounds  = [r[0] for r in LOTO_ROUNDS]
    rows    = []

    for m in models:
        row = {'Model': m}
        rmses = []
        for r in rounds:
            sub = results_df[(results_df['round']==r) & (results_df['model']==m)]
            if sub.empty:
                row[r] = '—'
            else:
                v = sub['RMSE_ms'].values[0]
                row[r] = round(v, 4)
                rmses.append(v)
        row['Avg RMSE'] = round(float(np.mean(rmses)), 4) if rmses else '—'
        rows.append(row)

    summ = pd.DataFrame(rows)
    print(summ.to_string(index=False))

    print("\n  % Improvement vs Naive baseline:")
    for m in models:
        if m == 'Naive (last-known)':
            continue
        sub = results_df[results_df['model'] == m]
        avg_imp = sub['pct_improvement'].mean()
        min_imp = sub['pct_improvement'].min()
        max_imp = sub['pct_improvement'].max()
        print(f"    {m:<18}  avg={avg_imp:+.1f}%  "
              f"min={min_imp:+.1f}%  max={max_imp:+.1f}%")

    return summ

# ====================================================================
# Secondary diagnostic: heading (course) prediction
# ====================================================================
def heading_diagnostic(X_all, y_course_all, meta_all):
    """
    Predict GNSS course (heading) from the same IMU window features.
    Heading is circular — represent as sin/cos targets, then reconstruct angle.
    Report MAE in degrees.
    Uses the same 4 LOTO rounds and best-architecture (XGBoost) for comparison.
    """
    print("\n" + "="*70)
    print("SECONDARY DIAGNOSTIC — HEADING (COURSE) PREDICTION")
    print("="*70)
    print("  Target: GNSS-computed course bearing (radians, reconstructed via sin/cos)")
    print("  Model:  XGBoost (best-performing velocity architecture)")
    print("  Circular representation: train sin(θ) and cos(θ) separately")

    # Only use rows where course is not NaN
    valid = ~np.isnan(y_course_all)
    X_h     = X_all[valid]
    y_h     = y_course_all[valid]
    meta_h  = meta_all[valid].reset_index(drop=True)
    sin_h   = np.sin(y_h)
    cos_h   = np.cos(y_h)

    print(f"  Valid heading samples: {len(y_h):,} "
          f"({len(y_h)/len(y_course_all)*100:.1f}% of total windows)")

    all_head_results = []

    for test_trip, _ in LOTO_ROUNDS:
        is_test  = meta_h['session'].str.startswith(test_trip)
        is_train = ~is_test
        if is_test.sum() == 0:
            continue

        X_tr, sin_tr, cos_tr = X_h[is_train], sin_h[is_train], cos_h[is_train]
        X_te, sin_te, cos_te = X_h[is_test],  sin_h[is_test],  cos_h[is_test]
        y_te_rad = y_h[is_test]

        # Train two XGBoost regressors (sin, cos)
        m_sin = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
        m_cos = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
        m_sin.fit(X_tr, sin_tr)
        m_cos.fit(X_tr, cos_tr)

        pred_sin = m_sin.predict(X_te)
        pred_cos = m_cos.predict(X_te)
        pred_rad = np.arctan2(pred_sin, pred_cos)
        pred_rad[pred_rad < 0] += 2 * np.pi

        # Circular error — minimum angular distance
        err_rad = np.abs(np.arctan2(np.sin(pred_rad - y_te_rad),
                                     np.cos(pred_rad - y_te_rad)))
        mae_deg = float(np.degrees(np.mean(err_rad)))

        # Naive baseline: use GPS orientation from the phase column if available
        naive_deg = 90.0   # a generous naive baseline (random guess MAE for uniform circular)

        print(f"  {test_trip:<12}  Heading MAE = {mae_deg:.2f}°  "
              f"  (n={is_test.sum()}, naive circular random ~90°)")
        all_head_results.append({
            'round': test_trip,
            'heading_MAE_deg': round(mae_deg, 2),
            'n_test': int(is_test.sum()),
        })

    head_df = pd.DataFrame(all_head_results)
    if not head_df.empty:
        print(f"\n  Average heading MAE across rounds: "
              f"{head_df['heading_MAE_deg'].mean():.2f}°")
        print(f"  Note: MAE << 90° = learnable signal from IMU alone.")
        print(f"        MAE ~90°   = no useful heading signal from IMU alone.")
        print(f"        Physics-based gyro heading (Phase 5) will be preferred")
        print(f"        in Phase 9 unless this MAE is substantially < 45°.")
    head_df.to_csv(OUT_DIR / "phase7_heading_diagnostic.csv", index=False)
    return head_df

# ====================================================================
# Feature importance (best model: XGBoost on full data)
# ====================================================================
def feature_importance_analysis(X_all, y_vel_all, meta_all):
    """Train XGBoost on all data, report top-20 features."""
    STAT_NAMES = []
    signal_names = (
        STAT_AXES + ['acc_veh_fwd', 'gyro_veh_yaw_rate', 'a_mag', 'w_mag', 'al_mag']
    )
    for s in signal_names:
        for st in ['mean', 'std', 'min', 'max']:
            STAT_NAMES.append(f"{s}_{st}")

    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
    model.fit(X_all, y_vel_all)
    imp = model.feature_importances_

    top20_idx = np.argsort(imp)[::-1][:20]
    print("\n" + "="*70)
    print("TOP-20 FEATURES (XGBoost trained on full dataset)")
    print("="*70)
    for rank, i in enumerate(top20_idx, 1):
        fname = STAT_NAMES[i] if i < len(STAT_NAMES) else f"feat_{i}"
        print(f"  {rank:>2}. {fname:<35}  importance={imp[i]:.4f}")

    # Save feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    top_names = [STAT_NAMES[i] if i < len(STAT_NAMES) else f"feat_{i}"
                 for i in top20_idx]
    top_vals  = imp[top20_idx]
    ax.barh(range(20), top_vals[::-1], color='steelblue')
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (gain)', fontsize=11)
    ax.set_title('Top-20 IMU Features for Velocity Prediction\n(XGBoost, full dataset)',
                 fontsize=11)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase7_feature_importance.png", dpi=150)
    plt.close()

# ====================================================================
# Prediction vs ground-truth scatter (best model, one test trip)
# ====================================================================
def scatter_plot(X_all, y_vel_all, meta_all, best_model_name):
    """
    For test=S-Vw4, train=rest; predict with best model and plot pred vs true.
    Also plot velocity over time for a 10-minute segment.
    """
    test_trip = "S-Vw4"
    is_test   = meta_all['session'].str.startswith(test_trip)
    is_train  = ~is_test

    X_tr, y_tr = X_all[is_train], y_vel_all[is_train]
    X_te, y_te = X_all[is_test],  y_vel_all[is_test]
    meta_te    = meta_all[is_test].reset_index(drop=True)

    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
    model.fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 0, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    axes[0].scatter(y_te, pred, s=4, alpha=0.4, color='steelblue')
    mx = max(y_te.max(), pred.max())
    axes[0].plot([0, mx], [0, mx], 'r--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_xlabel('True GPS Speed (m/s)', fontsize=11)
    axes[0].set_ylabel('Predicted Speed (m/s)', fontsize=11)
    axes[0].set_title(f'XGBoost Velocity Prediction — Test: {test_trip}\n'
                      f'RMSE = {np.sqrt(mean_squared_error(y_te, pred)):.3f} m/s',
                      fontsize=11)
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    # Time-series (first 10 min = first 600 windows approx or all if fewer)
    n_show = min(len(y_te), 70)   # ~10 min at 1 GNSS fix per 9s
    t_show = meta_te['t_fix_s'].values[:n_show]
    axes[1].plot(t_show / 60, y_te[:n_show], 'k-', linewidth=1.5,
                 label='GPS Truth', zorder=3)
    axes[1].plot(t_show / 60, pred[:n_show], 'b--', linewidth=1.5,
                 label='XGBoost Prediction')
    axes[1].set_xlabel('Time (minutes)', fontsize=11)
    axes[1].set_ylabel('Speed (m/s)', fontsize=11)
    axes[1].set_title(f'Velocity Over Time — First ~10 min of {test_trip}', fontsize=11)
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase7_velocity_scatter_S-Vw4.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {OUT_DIR / 'phase7_velocity_scatter_S-Vw4.png'}")

# ====================================================================
# RMSE summary bar chart across rounds
# ====================================================================
def results_bar_chart(results_df):
    rounds = [r[0] for r in LOTO_ROUNDS if r[0] in results_df['round'].values]
    models = results_df['model'].unique()
    colours = {
        'Naive (last-known)': 'gray',
        'Ridge':              'lightblue',
        'RandomForest':       'green',
        'XGBoost':            'steelblue',
        'MLP':                'orange',
    }

    fig, axes = plt.subplots(1, len(rounds), figsize=(5 * len(rounds), 5), sharey=True)
    if len(rounds) == 1:
        axes = [axes]

    for ax, rnd in zip(axes, rounds):
        sub = results_df[results_df['round'] == rnd]
        names  = sub['model'].tolist()
        rmses  = sub['RMSE_ms'].tolist()
        cols   = [colours.get(n, 'purple') for n in names]
        bars   = ax.bar(range(len(names)), rmses, color=cols)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_title(f"Test: {rnd}", fontsize=10)
        ax.set_ylabel('RMSE (m/s)' if ax == axes[0] else '')
        ax.grid(axis='y', alpha=0.4)
        for bar, v in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Phase 7 — Velocity RMSE by Model and Held-Out Trip', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase7_rmse_bar_chart.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUT_DIR / 'phase7_rmse_bar_chart.png'}")

# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 7: ML VELOCITY ESTIMATION EXPERIMENT")
    print("=" * 70)
    print(f"  Window: {WINDOW_LEN} samples ({WINDOW_LEN/10:.1f} s)")
    print(f"  Min GPS speed for label: {MIN_SPEED_MS} m/s")
    print(f"  LOTO rounds: {[r[0] for r in LOTO_ROUNDS]}")

    X_all, y_vel_all, y_course_all, meta_all = build_dataset()

    # Persist cache for downstream phases (Phase 10 reads this)
    np.save("data/phase7_cache_X.npy", X_all)
    np.save("data/phase7_cache_y.npy", y_vel_all)
    meta_all.to_parquet("data/phase7_cache_meta.parquet")
    print(f"  Cache written: data/phase7_cache_{{X,y,meta}}.* for Phase 10.")

    results_df = run_loto_cv(X_all, y_vel_all, meta_all)
    summ       = print_summary_table(results_df)

    head_df    = heading_diagnostic(X_all, y_course_all, meta_all)
    feature_importance_analysis(X_all, y_vel_all, meta_all)

    scatter_plot(X_all, y_vel_all, meta_all, 'XGBoost')
    results_bar_chart(results_df)

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE")
    print("=" * 70)
    print(f"  Results: {OUT_DIR / 'phase7_loto_results.csv'}")
    print(f"  Heading: {OUT_DIR / 'phase7_heading_diagnostic.csv'}")
    print(f"  Plots:   {OUT_DIR.resolve()}")
