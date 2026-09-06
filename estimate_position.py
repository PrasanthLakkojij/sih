"""
Position Estimator Module (Step 1 of App-Build Plan)
SIH26168 — AI-ML Dead Reckoning System

Combines:
  1. Phase 5B: ZUPT-enhanced physics dead reckoning.
  2. Phase 8: 58-feature extraction from windowed IMU data.
  3. Phase 9: C1 XGBoost displacement correction model.

Design:
  - Pure Python/NumPy logic for feature extraction and integration so it can easily
    be ported to Kotlin/Android in Step 2.
  - estimate_position() processes one ~9s IMU window and returns the updated state.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

# Phase 5B locked thresholds
A_TH = 5.389       # m/s^2 linear acceleration magnitude threshold
W_TH = 0.753       # rad/s angular rate magnitude threshold
WINDOW_LEN = 8     # samples for stationary filter confirmation

# 11 base signals used for feature statistics in Phase 8
SIGNAL_KEYS = [
    'acc_x', 'acc_y', 'acc_z',
    'acc_lin_x', 'acc_lin_y', 'acc_lin_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'acc_veh_fwd', 'gyro_veh_yaw_rate',
]

def extract_58_features(window_data: Dict[str, np.ndarray],
                        seg_dur_s: float = None,
                        seg_n_samples: int = None) -> np.ndarray:
    """
    Extract the exact 58 features from an IMU window matching Phase 8/9 training.
    Kept pure NumPy so it translates directly to Kotlin / Java.

    Features layout (58 total):
      - 11 signals x 4 stats (mean, std, min, max) = 44
      - 3 magnitudes (a_mag, al_mag, w_mag) x 4 stats = 12
      - 2 interval timing features (seg_dur_s, seg_n_samples) = 2
    """
    feats = []
    n_samp = len(window_data['acc_x']) if seg_n_samples is None else seg_n_samples
    if seg_dur_s is None:
        seg_dur_s = float(np.sum(window_data['dt_sec'])) if 'dt_sec' in window_data else float(n_samp * 0.1)

    # 1. 11 base signals (44 features)
    for key in SIGNAL_KEYS:
        arr = window_data[key]
        if len(arr) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
            ])

    # 2. 3 magnitude signals (12 features)
    ax = window_data['acc_x']
    ay = window_data['acc_y']
    az = window_data['acc_z']
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)

    lx = window_data['acc_lin_x']
    ly = window_data['acc_lin_y']
    lz = window_data['acc_lin_z']
    al_mag = np.sqrt(lx**2 + ly**2 + lz**2)

    gx = window_data['gyro_x']
    gy = window_data['gyro_y']
    gz = window_data['gyro_z']
    w_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    for mag in [a_mag, al_mag, w_mag]:
        if len(mag) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend([
                float(np.mean(mag)),
                float(np.std(mag)),
                float(np.min(mag)),
                float(np.max(mag)),
            ])

    # 3. Timing context (2 features)
    feats.extend([float(seg_dur_s), float(n_samp)])

    return np.array(feats, dtype=np.float32).reshape(1, -1)


def compute_zupt_mask(acc_lin_x: np.ndarray, acc_lin_y: np.ndarray, acc_lin_z: np.ndarray,
                      gyro_x: np.ndarray, gyro_y: np.ndarray, gyro_z: np.ndarray,
                      a_th: float = A_TH, w_th: float = W_TH, window_len: int = WINDOW_LEN) -> np.ndarray:
    """
    Compute ZUPT v1 mask using Phase 5B locked thresholds.
    """
    a_mag = np.sqrt(acc_lin_x**2 + acc_lin_y**2 + acc_lin_z**2)
    w_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    raw_stationary = (a_mag < a_th) & (w_mag < w_th)

    n = len(raw_stationary)
    mask = np.zeros(n, dtype=bool)
    consec = 0
    for i in range(n):
        if raw_stationary[i]:
            consec += 1
            if consec >= window_len:
                mask[i] = True
        else:
            consec = 0
    return mask


def estimate_position(imu_window: Union[pd.DataFrame, Dict[str, np.ndarray]],
                      last_known_state: Dict[str, float],
                      model: Any,
                      zupt_mask: np.ndarray = None,
                      seg_dur_s: float = None) -> Dict[str, float]:
    """
    Unified Phase 5B + Phase 9 Dead Reckoning Pipeline Function.

    Input:
        imu_window: ~9 seconds of raw/vehicle IMU data at 10Hz.
                    For a segment [ia, ib], contains the samples spanning from
                    ia to ib inclusive (shape: N samples). Integration advances
                    across the N-1 steps between sample 0 and sample N-1.
                    Must include: 'acc_x', 'acc_y', 'acc_z', 'acc_lin_x', 'acc_lin_y', 'acc_lin_z',
                                  'gyro_x', 'gyro_y', 'gyro_z', 'acc_veh_fwd', 'gyro_veh_yaw_rate',
                                  and 'dt_sec'.
        last_known_state: {'x': float, 'y': float, 'heading': float, 'velocity': float}
                          Last known state (from GPS or previous estimate).
                          x: Easting (meters)
                          y: Northing (meters)
                          heading: Navigation azimuth in radians (0=North, pi/2=East, clockwise)
                          velocity: Forward speed (m/s)
        model: Trained Phase 9 C1 XGBoost model.
        zupt_mask: Optional precomputed boolean array of stationary samples for imu_window.
        seg_dur_s: Optional segment duration in seconds.

    Output:
        {'x': float, 'y': float, 'heading': float, 'velocity': float, 'ds_imu': float, 'ds_corr': float}
        The new estimated state after integrating across this window.
    """
    # Convert DataFrame to dictionary of NumPy arrays if needed
    if isinstance(imu_window, pd.DataFrame):
        win_dict = {col: imu_window[col].values for col in imu_window.columns}
    else:
        win_dict = imu_window

    dt_arr = win_dict['dt_sec']
    a_fwd = win_dict['acc_veh_fwd']
    gyro_yaw = win_dict['gyro_veh_yaw_rate']
    n_samples = len(a_fwd)

    # Compute ZUPT mask if not provided
    if zupt_mask is None:
        zupt_mask = compute_zupt_mask(
            win_dict['acc_lin_x'], win_dict['acc_lin_y'], win_dict['acc_lin_z'],
            win_dict['gyro_x'], win_dict['gyro_y'], win_dict['gyro_z']
        )

    # 1. Physics DR Integration across the window
    # When imu_window contains samples [0..N-1] (corresponding to ia..ib in dataset),
    # the integration loop iterates through steps 1..N-1 (matching range(ia+1, ib+1))
    de_sub = 0.0
    dn_sub = 0.0
    v_step = float(last_known_state['velocity'])
    psi_step = float(last_known_state['heading'])

    start_k = 1 if n_samples > 1 else 0
    for t in range(start_k, n_samples):
        dt = float(dt_arr[t])
        if zupt_mask[t]:
            v_step = 0.0
        else:
            v_step = max(0.0, v_step + float(a_fwd[t]) * dt)
        psi_step += float(gyro_yaw[t]) * dt
        de_sub += v_step * np.sin(psi_step) * dt
        dn_sub += v_step * np.cos(psi_step) * dt

    ds_imu = float(np.sqrt(de_sub**2 + dn_sub**2))
    h_dir = float(np.arctan2(de_sub, dn_sub)) if ds_imu > 1e-3 else psi_step

    # 2. Extract 58 features from the window
    feats = extract_58_features(win_dict, seg_dur_s=seg_dur_s, seg_n_samples=n_samples)

    # 3. Predict correction with C1 XGBoost model
    ds_corr = float(model.predict(feats)[0])

    # 4. Corrected displacement
    ds_final = max(0.0, ds_imu + ds_corr)

    # 5. Update position coordinates (East = x, North = y)
    new_x = float(last_known_state['x']) + ds_final * np.sin(h_dir)
    new_y = float(last_known_state['y']) + ds_final * np.cos(h_dir)

    return {
        'x': new_x,
        'y': new_y,
        'heading': psi_step,
        'velocity': v_step,
        'ds_imu': ds_imu,
        'ds_corr': ds_corr
    }
