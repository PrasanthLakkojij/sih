# SIH26168 — AI-ML Dead Reckoning Using Smartphone Sensors
**Organisation: ISRO | Domain: Software / AI-ML**

A physics-first, progressively-refined dead reckoning system for vehicle navigation using smartphone IMU and GNSS sensors.

---

## Project Phases

| Phase | Script | Status | Description |
|-------|--------|--------|-------------|
| 1 | `phase1_inspect.py` | ✅ | Dataset audit: 72 CSVs, 1.07M samples, 10 Hz, schema discovery |
| 2 | `phase2_visualize.py` | ✅ | Sensor visualization, GPS trajectory, GPS step-function pattern |
| 3 | `phase3_preprocess.py` | ✅ | Preprocessing → 80 parquet sub-sessions, dedup, split, gravity removal |
| 4 | `phase4_orientation.py` | ✅ | Kinematic coordinate transformation (phone frame → vehicle frame) |
| 5A | `phase5_dead_reckoning.py` | ✅ | Open-loop physics DR baseline |
| 5.1 | `phase5_1_audit.py` | ✅ | Numerical sanity audit (dt, units, bias sources) |
| 5B | `phase5b_zupt.py` | ✅ | ZUPT v1 calibration from static sessions |
| 5B-inv | `phase5b_investigations.py` | ✅ | False-positive + heading drift investigations |
| 5C | `phase5c_physics_baseline.py` | ✅ | ZUPT v2 + circular complementary heading filter |
| 5C-diag | `phase5c_diagnostic_bundle.py` | ✅ | Full diagnostic bundle: trajectory, integration verification, path length |
| 6 | `phase6_outage_sim.py` | ✅ | GNSS outage simulation (30/60/120/300 s, genuine-fix-snapped init) |
| 7 | `phase7_ml_velocity.py` | ✅ | ML velocity estimation: LOTO-CV, Ridge/RF/XGBoost/MLP |

## Dataset
IO-VNBD S-Dataset (not included — commercial licence). Place raw CSVs in `data/raw/`.

## Setup
```bash
pip install numpy pandas pyarrow scikit-learn xgboost matplotlib scipy
python phase3_preprocess.py   # builds data/processed_sessions/
python phase4_orientation.py  # adds vehicle-frame columns
python phase7_ml_velocity.py  # runs Phase 7 ML experiment
```

## Key Findings (Physics Baseline)
- Pure open-loop DR position error at 300 s GNSS outage: **~7 km** (gyro-only)
- ZUPT v1 reduces this to **~4.7 km** at 300 s, but is **worse** at short outages (30–60 s)
- Dominant error source: residual accelerometer bias → Δp = (1/2) × ε_a × t²
- Heading error at 30 s driven by **road curvature** (23.7°), not secular drift (1.57°)

## Key Findings (Phase 7 ML)
See `plots/phase7/` for full results.
