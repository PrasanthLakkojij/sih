"""
edge_engine_cli.py -- Hardware-agnostic edge dead-reckoning engine (Phase 8).

Demonstrates that estimate_position() (estimate_position.py) is not tied to
smartphone sensors or the IO-VNBD dataset format: it operates purely on a
documented intermediate schema, so any IMU source (phone, external/FOG unit,
different logging app) works as long as an adapter produces this schema once.

REQUIRED INPUT CSV SCHEMA (one row per 10Hz sample, already vehicle-frame
transformed -- this engine does not know or care what hardware produced it):

    acc_x, acc_y, acc_z          raw accelerometer (includes gravity), m/s^2
    acc_lin_x, acc_lin_y, acc_lin_z   linear acceleration (gravity removed), m/s^2
    gyro_x, gyro_y, gyro_z       raw gyroscope, rad/s
    acc_veh_fwd, gyro_veh_yaw_rate    vehicle-frame forward accel (m/s^2) and
                                      yaw rate (rad/s) -- from whatever
                                      orientation solution the adapter uses
    dt_sec                       time since previous sample, seconds

Usage:
    python edge_engine_cli.py --input <schema.csv> --model models/c1_correction_model.onnx \
        --window-samples 91 --output trajectory.csv

    # or, to run WITHOUT a correction model (physics + ZUPT only, no ML):
    python edge_engine_cli.py --input <schema.csv> --no-model --output trajectory.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from estimate_position import estimate_position

REQUIRED_COLUMNS = [
    "acc_x", "acc_y", "acc_z",
    "acc_lin_x", "acc_lin_y", "acc_lin_z",
    "gyro_x", "gyro_y", "gyro_z",
    "acc_veh_fwd", "gyro_veh_yaw_rate",
    "dt_sec",
]


class OnnxModelAdapter:
    """Wraps onnxruntime.InferenceSession behind XGBoost's .predict(X)[0]
    interface, so estimate_position() doesn't need to know or care whether
    the model is a live XGBoost object or an exported ONNX file -- the SAME
    .onnx file used on Android can be used here, edge-deployable in Python
    on any platform with onnxruntime (a Raspberry Pi, an industrial PC, a
    FOG-IMU logging box) without a phone or the training framework at all.
    """

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: np.ndarray) -> np.ndarray:
        result = self.session.run(None, {self.input_name: features.astype(np.float32)})
        return np.asarray(result[0]).reshape(-1)


class NoOpModel:
    """Physics + ZUPT only, zero ML correction -- for edge deployments with
    no trained model available yet, or as a sanity baseline."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(features.shape[0], dtype=np.float32)


def validate_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}\n"
            f"Required schema: {REQUIRED_COLUMNS}\n"
            f"See edge_engine_cli.py's module docstring for the full contract."
        )


def run(input_csv: str, model, window_samples: int, output_csv: str, init_state=None):
    df = pd.read_csv(input_csv)
    validate_schema(df)
    print(f"Loaded {len(df)} samples from {input_csv} ({len(df) / 10:.1f}s @ nominal 10Hz)")

    state = init_state or {"x": 0.0, "y": 0.0, "heading": 0.0, "velocity": 0.0}
    rows = [{"t_idx": 0, **state, "ds_imu": 0.0, "ds_corr": 0.0}]

    n = len(df)
    n_windows = 0
    for start in range(0, n - 1, window_samples - 1):
        end = min(start + window_samples, n)
        if end - start < 2:
            break
        window = df.iloc[start:end]
        win_dict = {col: window[col].values for col in REQUIRED_COLUMNS}
        seg_dur_s = float(win_dict["dt_sec"][1:].sum())

        result = estimate_position(win_dict, state, model, seg_dur_s=seg_dur_s)
        state = {"x": result["x"], "y": result["y"], "heading": result["heading"], "velocity": result["velocity"]}
        rows.append({"t_idx": end - 1, **state, "ds_imu": result["ds_imu"], "ds_corr": result["ds_corr"]})
        n_windows += 1

    traj = pd.DataFrame(rows)
    traj.to_csv(output_csv, index=False)
    total_dist = float(np.hypot(traj["x"].iloc[-1] - traj["x"].iloc[0], traj["y"].iloc[-1] - traj["y"].iloc[0]))
    print(f"Processed {n_windows} windows -> {output_csv}")
    print(f"Final state: x={state['x']:.2f}m y={state['y']:.2f}m heading={np.degrees(state['heading']):.1f}deg "
          f"velocity={state['velocity']:.2f}m/s")
    print(f"Net displacement (start->end): {total_dist:.2f}m")
    return traj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input CSV matching the documented schema")
    parser.add_argument("--model", help="Path to an ONNX correction model (e.g. models/c1_correction_model.onnx)")
    parser.add_argument("--no-model", action="store_true", help="Run physics+ZUPT only, no ML correction")
    parser.add_argument("--window-samples", type=int, default=91, help="Samples per window (default 91, ~9s @ 10Hz)")
    parser.add_argument("--output", default="trajectory.csv", help="Output trajectory CSV path")
    args = parser.parse_args()

    if args.no_model:
        model = NoOpModel()
        print("Running with NO-OP model (physics + ZUPT only, zero ML correction).")
    elif args.model:
        model = OnnxModelAdapter(args.model)
        print(f"Loaded ONNX model: {args.model}")
    else:
        print("Error: must specify --model <path> or --no-model", file=sys.stderr)
        sys.exit(1)

    run(args.input, model, args.window_samples, args.output)


if __name__ == "__main__":
    main()
