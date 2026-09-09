"""
Phase 6: Real-device validation, session port work.

IMPORTANT CAVEAT (read before trusting any number this prints):
The only real-device recording in this repo (testing_real_data/) is a
~74s / 16m WALKING test (GPX-labeled "Activity = walking", max speed
4 km/h) -- not a vehicle drive. Every fix validated this session (NHC,
ZUPT-vs-ML veto gate, EKF process noise, the Phase 7 speed model) was
tuned against vehicle dynamics (1-10 m/s^2 accelerations, up to ~70 km/h
IO-VNBD driving). Running that pipeline against a person walking is a
severe domain mismatch on purpose -- there is no vehicle recording
available to test against instead. Treat this run as "does the pipeline
behave sanely at near-zero speed", NOT as a vehicle GNSS-outage validation.
"""
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from phase6_outage_sim import latlon_to_enu, find_genuine_gnss_updates
from phase4_orientation import compute_kinematic_alignment
from phase5c_physics_baseline import calibrate_zupt_v2_thresholds, get_zupt_v2_mask
from phase10_fusion_engine import simulate_ekf_outage, estimate_gyro_bias_preoutage, estimate_accel_bias_preoutage

DATA_DIR = Path("testing_real_data")
SENSOR_DIR = DATA_DIR / "2026-09-04_07-20-44"
GPX_PATH = DATA_DIR / "20260904-125057.gpx"
OUT_DIR = Path("plots") / "phase6_real_device"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sensor_csv(csv_name, out_prefix):
    df = pd.read_csv(SENSOR_DIR / f"{csv_name}.csv")
    df["t_epoch"] = df["time"] / 1e9  # nanoseconds -> seconds, epoch-aligned (verified vs Metadata.csv)
    return df[["t_epoch", "x", "y", "z"]].rename(columns={"x": f"{out_prefix}_x", "y": f"{out_prefix}_y", "z": f"{out_prefix}_z"})


def load_gpx(path):
    ns = {"g": "http://www.topografix.com/GPX/1/0"}
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for trkpt in root.iter("{http://www.topografix.com/GPX/1/0}trkpt"):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])
        t_el = trkpt.find("g:time", ns)
        spd_el = trkpt.find("g:speed", ns)
        t = datetime.strptime(t_el.text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        rows.append({"t_epoch": t.timestamp(), "lat": lat, "lon": lon, "speed": float(spd_el.text)})
    return pd.DataFrame(rows).sort_values("t_epoch").reset_index(drop=True)


def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def build_dataframe():
    print("[1] Loading sensor CSVs (epoch-nanosecond aligned)...")
    acc = load_sensor_csv("Accelerometer", "acc_lin")       # linear acceleration (gravity removed)
    grav = load_sensor_csv("Gravity", "grav")
    gyro = load_sensor_csv("Gyroscope", "gyro")
    total = load_sensor_csv("TotalAcceleration", "acc")      # raw accelerometer (includes gravity)

    t0 = max(acc["t_epoch"].min(), grav["t_epoch"].min(), gyro["t_epoch"].min(), total["t_epoch"].min())
    t1 = min(acc["t_epoch"].max(), grav["t_epoch"].max(), gyro["t_epoch"].max(), total["t_epoch"].max())
    print(f"    Overlapping sensor span: {t1 - t0:.1f}s")

    # Common 10Hz grid (pipeline convention throughout this project)
    grid = np.arange(t0, t1, 0.1)
    df = pd.DataFrame({"t_epoch": grid})
    for name, src in [("acc", total), ("acc_lin", acc), ("grav", grav), ("gyro", gyro)]:
        merged = pd.merge_asof(df.sort_values("t_epoch"), src.sort_values("t_epoch"), on="t_epoch", direction="nearest")
        for axis in "xyz":
            df[f"{name}_{axis}"] = merged[f"{name}_{axis}"]

    print("[2] Loading GPX ground truth...")
    gpx = load_gpx(GPX_PATH)
    print(f"    {len(gpx)} track points, {gpx['t_epoch'].max()-gpx['t_epoch'].min():.0f}s span, "
          f"max speed {gpx['speed'].max()*3.6:.1f} km/h, activity=walking (per GPX header)")

    gpx["bearing"] = 0.0
    for i in range(len(gpx) - 1):
        gpx.loc[i, "bearing"] = bearing_deg(gpx["lat"][i], gpx["lon"][i], gpx["lat"][i + 1], gpx["lon"][i + 1])
    gpx.loc[len(gpx) - 1, "bearing"] = gpx["bearing"].iloc[-2] if len(gpx) > 1 else 0.0

    gps_merged = pd.merge_asof(df.sort_values("t_epoch"), gpx.sort_values("t_epoch"), on="t_epoch", direction="backward")
    df["gps_lat"] = gps_merged["lat"]
    df["gps_lon"] = gps_merged["lon"]
    df["gps_speed_ms"] = gps_merged["speed"]
    df["gps_orientation"] = gps_merged["bearing"]
    df["azimuth"] = gps_merged["bearing"]
    df = df.dropna(subset=["gps_lat", "gps_lon"]).reset_index(drop=True)

    df["time_rel_sec"] = df["t_epoch"] - df["t_epoch"].iloc[0]
    df["dt_sec"] = df["time_rel_sec"].diff().fillna(0.1)

    print("[3] Vehicle-frame transform (mounting yaw)...")
    theta, corr = compute_kinematic_alignment(df.rename(columns={"acc_lin_x": "acc_lin_x", "acc_lin_y": "acc_lin_y"}))
    print(f"    compute_kinematic_alignment: theta={np.degrees(theta):.1f}deg corr={corr:.3f} "
          f"(expected to fail/fallback -- all speeds are below the 3.0 m/s minimum, this is a walking test)")

    gx, gy, gz = df["grav_x"].values, df["grav_y"].values, df["grav_z"].values
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    up_x, up_y, up_z = -gx / g_norm, -gy / g_norm, -gz / g_norm
    axl, ayl, azl = df["acc_lin_x"].values, df["acc_lin_y"].values, df["acc_lin_z"].values
    a_vert = axl * up_x + ayl * up_y + azl * up_z
    ax_h = axl - a_vert * up_x
    ay_h = ayl - a_vert * up_y
    df["acc_veh_fwd"] = ax_h * np.cos(theta) + ay_h * np.sin(theta)
    df["acc_veh_lat"] = -ax_h * np.sin(theta) + ay_h * np.cos(theta)
    wx, wy, wz = df["gyro_x"].values, df["gyro_y"].values, df["gyro_z"].values
    df["gyro_veh_yaw_rate"] = wx * up_x + wy * up_y + wz * up_z

    gnss_idx = find_genuine_gnss_updates(df)
    df["is_gnss_update"] = False
    df.loc[gnss_idx, "is_gnss_update"] = True
    print(f"    {len(gnss_idx)} genuine GNSS fixes detected across {len(df)} samples ({len(df)/10:.0f}s @ 10Hz)")

    return df


def main():
    print("=" * 80)
    print("PHASE 6: REAL-DEVICE VALIDATION (walking-pace test -- see caveat in file docstring)")
    print("=" * 80)

    df = build_dataframe()
    df.to_parquet(OUT_DIR / "real_device_processed.parquet")

    zupt_a_th, zupt_w_th, zupt_a_var_th, zupt_w_var_th = calibrate_zupt_v2_thresholds()
    zupt_mask = get_zupt_v2_mask(df, zupt_a_th, zupt_w_th, zupt_a_var_th, zupt_w_var_th)
    print(f"\n[4] ZUPT v2: {zupt_mask.sum()}/{len(df)} samples classified stationary ({100*zupt_mask.mean():.0f}%) -- "
          f"plausible given this is a slow walk with frequent pauses")

    gnss_idx = np.where(df["is_gnss_update"].values)[0]
    lat0, lon0 = df["gps_lat"].iloc[0], df["gps_lon"].iloc[0]
    east_all, north_all = latlon_to_enu(df["gps_lat"].values, df["gps_lon"].values, lat0, lon0)

    print("\n[5] Loading production Phase 7 velocity model (trained on ALL IO-VNBD sessions --")
    print("    i.e. UK vehicle driving 30-90 km/h. Testing against Indian pedestrian data")
    print("    0-4 km/h is an extreme, deliberate domain-shift stress test.)")
    X_p7 = np.load("data/phase7_cache_X.npy")
    y_p7 = np.load("data/phase7_cache_y.npy")
    model_p7 = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbosity=0)
    model_p7.fit(X_p7, y_p7)

    # Pre-outage bias calibration from the first ~15s (GPX's first point has speed=0.000)
    n_total = len(df)
    pre_outage_end = min(150, n_total // 3)
    t_arr = df["time_rel_sec"].values
    gyro_yaw = df["gyro_veh_yaw_rate"].values
    a_fwd_arr = df["acc_veh_fwd"].values
    a_lat_arr = df["acc_veh_lat"].values
    b_w0, calib_w = estimate_gyro_bias_preoutage(t_arr, gyro_yaw, zupt_mask, pre_outage_end, calib_window_s=15.0, min_samples=20)
    b_ax0, b_ay0, calib_a = estimate_accel_bias_preoutage(t_arr, a_fwd_arr, a_lat_arr, zupt_mask, pre_outage_end, calib_window_s=15.0, min_samples=20)
    print(f"\n[6] Pre-outage bias calibration (first {pre_outage_end/10:.0f}s): "
          f"gyro calibrated={calib_w} (b_w={b_w0:.4f}), accel calibrated={calib_a} (b_ax={b_ax0:.3f}, b_ay={b_ay0:.3f})")

    s_gnss = pre_outage_end
    e_gnss = n_total - 1
    outage_dur_s = t_arr[e_gnss] - t_arr[s_gnss]
    dist_true = float(np.hypot(east_all[e_gnss] - east_all[s_gnss], north_all[e_gnss] - north_all[s_gnss]))
    print(f"\n[7] Simulating outage: {outage_dur_s:.1f}s, ground-truth displacement {dist_true:.1f}m "
          f"(map-matching disabled -- no representative nearby-road assumption for a pedestrian path)")

    pos_err, vel_err, head_err, traj_e, traj_n = simulate_ekf_outage(
        df, zupt_mask, gnss_idx, s_gnss, e_gnss, model_p7, r_speed_ml=1.8,
        lat0=lat0, lon0=lon0, east_all=east_all, north_all=north_all,
        use_map_matching=False
    )

    drift_pct = 100.0 * pos_err / max(dist_true, 1e-3)
    print("\n" + "=" * 80)
    print("REAL-DEVICE (WALKING) RESULT -- NOT a vehicle validation, see caveat above")
    print("=" * 80)
    print(f"  Outage duration:        {outage_dur_s:.1f}s")
    print(f"  Ground-truth distance:  {dist_true:.1f}m")
    print(f"  Position error:         {pos_err:.1f}m")
    print(f"  Drift %%:                {drift_pct:.1f}%%")
    print(f"  Velocity error:         {vel_err:.2f} m/s")
    print(f"  Heading error:          {head_err:.1f} deg")

    result = {
        "outage_duration_s": outage_dur_s, "dist_true_m": dist_true, "pos_err_m": pos_err,
        "drift_pct": drift_pct, "vel_err_ms": vel_err, "head_err_deg": head_err,
        "caveat": "Walking-pace test (max 4 km/h, 16m), NOT a vehicle GNSS-outage validation. "
                  "Model trained on 30-90 km/h vehicle driving (IO-VNBD). Extreme domain mismatch by necessity -- "
                  "no vehicle recording exists in this repo.",
    }
    with open(OUT_DIR / "real_device_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {OUT_DIR / 'real_device_result.json'}")


if __name__ == "__main__":
    main()
