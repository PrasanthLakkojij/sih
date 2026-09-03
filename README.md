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
| 8 | `phase8_ml_correction.py` | ✅ | Physics-error-correction ML formulation: Δv vs Δs_corr, duration-ablation (C1 vs C2) & A2 |
| 9 | `phase9_ai_hybrid.py` | ✅ | AI + Physics Hybrid system (ZUPT v1 + C1 XGBoost displacement correction) |

## Dataset
IO-VNBD S-Dataset (not included — commercial licence). Place raw CSVs in `data/raw/`.

## Setup
```bash
pip install numpy pandas pyarrow scikit-learn xgboost matplotlib scipy
python phase3_preprocess.py   # builds data/processed_sessions/
python phase4_orientation.py  # adds vehicle-frame columns
python phase9_ai_hybrid.py    # runs Phase 9 AI + Physics Hybrid benchmark
```

## Key Findings (Physics Baselines)
- Pure open-loop DR position error at 300 s GNSS outage: **~7.0 km** (gyro-only)
- ZUPT v1 reduces this to **~4.7 km** at 300 s, but is **worse** at short outages (30–60 s) due to false-positive stops during cruise
- Dominant error source: residual accelerometer bias → $\Delta p = (1/2) \times \varepsilon_a \times t^2$
- Heading error at 30 s driven by **road curvature** (23.7°), not secular drift (1.57°)

## Key Findings (Phase 9 AI + Physics Hybrid)
- **30s Outage:** AI-Hybrid position error **359.6m** (+11.5% vs Gyro, +47.1% vs ZUPT v1)
- **60s Outage:** AI-Hybrid position error **811.1m** (+9.1% vs Gyro, +38.4% vs ZUPT v1)
- **120s Outage:** AI-Hybrid position error **1,312.2m** (+22.7% vs Gyro, +40.8% vs ZUPT v1)
- **300s Outage:** AI-Hybrid position error **3,505.1m** (+50.3% vs Gyro, +26.0% vs ZUPT v1)
- Caps error growth at 3.5 km compared to 7.0 km open-loop explosion, outperforming all physics baselines across all evaluated durations.
- Generated presentation assets saved to `plots/phase9/`.
