"""
adapt_sensorlogger_to_schema.py -- Hardware adapter, Phase 8 demonstration.

Converts raw "Sensor Logger" app CSVs (testing_real_data/) -- a completely
different raw format from IO-VNBD's (different column names, different
units convention, different logging app, different phone) -- into
edge_engine_cli.py's documented generic schema.

This is the point of the hardware-agnostic architecture: estimate_position.py
itself never changes. Only this ~60-line adapter is hardware-specific. A new
IMU source (an external FOG unit, a different phone app) needs an adapter
like this one, not a change to the core engine.

Usage:
    python adapt_sensorlogger_to_schema.py --sensor-dir testing_real_data/2026-09-04_07-20-44 \
        --output schema_real_device.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_sensor_csv(sensor_dir: Path, csv_name: str, out_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(sensor_dir / f"{csv_name}.csv")
    df["t_epoch"] = df["time"] / 1e9  # this app's raw `time` column is epoch nanoseconds
    return df[["t_epoch", "x", "y", "z"]].rename(
        columns={"x": f"{out_prefix}_x", "y": f"{out_prefix}_y", "z": f"{out_prefix}_z"}
    )


def adapt(sensor_dir: str, mounting_yaw_rad: float = 0.0) -> pd.DataFrame:
    sensor_dir = Path(sensor_dir)
    acc_lin = load_sensor_csv(sensor_dir, "Accelerometer", "acc_lin")
    grav = load_sensor_csv(sensor_dir, "Gravity", "grav")
    gyro = load_sensor_csv(sensor_dir, "Gyroscope", "gyro")
    acc_raw = load_sensor_csv(sensor_dir, "TotalAcceleration", "acc")

    t0 = max(acc_lin["t_epoch"].min(), grav["t_epoch"].min(), gyro["t_epoch"].min(), acc_raw["t_epoch"].min())
    t1 = min(acc_lin["t_epoch"].max(), grav["t_epoch"].max(), gyro["t_epoch"].max(), acc_raw["t_epoch"].max())
    grid = np.arange(t0, t1, 0.1)  # resample this app's ~100Hz native rate down to the engine's 10Hz convention

    df = pd.DataFrame({"t_epoch": grid})
    for name, src in [("acc", acc_raw), ("acc_lin", acc_lin), ("grav", grav), ("gyro", gyro)]:
        merged = pd.merge_asof(df.sort_values("t_epoch"), src.sort_values("t_epoch"), on="t_epoch", direction="nearest")
        for axis in "xyz":
            df[f"{name}_{axis}"] = merged[f"{name}_{axis}"]

    df["dt_sec"] = df["t_epoch"].diff().fillna(0.1)

    # Vehicle-frame projection (gravity-based tilt correction, same math as
    # IMUSensorCollector.kt / phase4_orientation.py -- hardware-independent).
    gx, gy, gz = df["grav_x"].values, df["grav_y"].values, df["grav_z"].values
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    up_x, up_y, up_z = -gx / g_norm, -gy / g_norm, -gz / g_norm
    axl, ayl, azl = df["acc_lin_x"].values, df["acc_lin_y"].values, df["acc_lin_z"].values
    a_vert = axl * up_x + ayl * up_y + azl * up_z
    ax_h = axl - a_vert * up_x
    ay_h = ayl - a_vert * up_y
    df["acc_veh_fwd"] = ax_h * np.cos(mounting_yaw_rad) + ay_h * np.sin(mounting_yaw_rad)
    wx, wy, wz = df["gyro_x"].values, df["gyro_y"].values, df["gyro_z"].values
    df["gyro_veh_yaw_rate"] = wx * up_x + wy * up_y + wz * up_z

    schema_cols = ["acc_x", "acc_y", "acc_z", "acc_lin_x", "acc_lin_y", "acc_lin_z",
                   "gyro_x", "gyro_y", "gyro_z", "acc_veh_fwd", "gyro_veh_yaw_rate", "dt_sec"]
    return df[schema_cols]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor-dir", required=True, help="Directory containing the Sensor Logger CSVs")
    parser.add_argument("--output", required=True, help="Output schema-conformant CSV path")
    parser.add_argument("--mounting-yaw-deg", type=float, default=0.0, help="Mounting yaw offset in degrees")
    args = parser.parse_args()

    df = adapt(args.sensor_dir, np.radians(args.mounting_yaw_deg))
    df.to_csv(args.output, index=False)
    print(f"Adapted {len(df)} samples ({len(df)/10:.1f}s @ 10Hz) -> {args.output}")


if __name__ == "__main__":
    main()
