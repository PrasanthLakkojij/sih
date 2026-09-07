package com.sih26168.deadreckoning.test

/**
 * Synthetic stationary window reference data (91 samples at 10Hz = 9.0s).
 * Simulates a device at rest on a desk or dashboard:
 *  - |a_lin| < 0.05 m/s^2 (locked A_TH = 5.389 m/s^2)
 *  - |omega| < 0.005 rad/s (locked W_TH = 0.753 rad/s)
 *
 * Used to verify ZUPT detection, velocity zeroing, and that ML correction is skipped.
 */
object SyntheticStationaryData {
    const val SEG_DUR_S = 9.0f
    const val N_SAMPLES = 91

    val acc_x = FloatArray(N_SAMPLES) { 0.01f }
    val acc_y = FloatArray(N_SAMPLES) { -0.01f }
    val acc_z = FloatArray(N_SAMPLES) { 9.81f }

    val acc_lin_x = FloatArray(N_SAMPLES) { 0.01f }
    val acc_lin_y = FloatArray(N_SAMPLES) { -0.01f }
    val acc_lin_z = FloatArray(N_SAMPLES) { 0.02f }

    val gyro_x = FloatArray(N_SAMPLES) { 0.001f }
    val gyro_y = FloatArray(N_SAMPLES) { -0.001f }
    val gyro_z = FloatArray(N_SAMPLES) { 0.002f }

    val acc_veh_fwd = FloatArray(N_SAMPLES) { 0.005f }
    val gyro_veh_yaw_rate = FloatArray(N_SAMPLES) { 0.001f }
    val dt_sec = FloatArray(N_SAMPLES) { 0.1f }

    const val INITIAL_X = 0.0f
    const val INITIAL_Y = 0.0f
    const val INITIAL_VELOCITY = 0.0f
    const val INITIAL_HEADING_RAD = 0.0f
}
