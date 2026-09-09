package com.sih26168.deadreckoning.ml

/**
 * Phase7FeatureExtractor: Translates phase7_ml_velocity.py's
 * extract_features_labels() / phase10_fusion_engine.py's
 * extract_p7_features_at_step() feature schema into Kotlin, for use with
 * VelocityModel (the instantaneous forward-speed ONNX model).
 *
 * 56 features = 14 signals x 4 stats (mean, std, min, max), in this EXACT
 * order (Python dict-insertion order, verified against both source scripts):
 *   1-9:   acc_x, acc_y, acc_z, acc_lin_x, acc_lin_y, acc_lin_z, gyro_x, gyro_y, gyro_z
 *   10-11: acc_veh_fwd, gyro_veh_yaw_rate
 *   12-14: a_mag, w_mag, al_mag   <- NOTE: different order from FeatureExtractor.kt's
 *          C1 schema (a_mag, al_mag, w_mag). Do not "harmonize" these -- they
 *          must match their respective Python training scripts exactly.
 *
 * Window is 20 samples (~2s at 10Hz), not FeatureExtractor.kt's 91 (~9s).
 */
object Phase7FeatureExtractor {

    const val FEATURE_COUNT = 56
    const val WINDOW_LEN = 20

    fun extractFeatures(
        accX: FloatArray,
        accY: FloatArray,
        accZ: FloatArray,
        accLinX: FloatArray,
        accLinY: FloatArray,
        accLinZ: FloatArray,
        gyroX: FloatArray,
        gyroY: FloatArray,
        gyroZ: FloatArray,
        accVehFwd: FloatArray,
        gyroVehYawRate: FloatArray
    ): FloatArray {
        val features = FloatArray(FEATURE_COUNT)
        var offset = 0

        val baseSignals = arrayOf(
            accX, accY, accZ,
            accLinX, accLinY, accLinZ,
            gyroX, gyroY, gyroZ,
            accVehFwd, gyroVehYawRate
        )
        for (signal in baseSignals) {
            val stats = FeatureExtractor.computeStats(signal)
            System.arraycopy(stats, 0, features, offset, 4)
            offset += 4
        }

        // Magnitude order: a_mag, w_mag, al_mag (Phase 7's own order, not C1's)
        val aMag = FeatureExtractor.compute3DMagnitude(accX, accY, accZ)
        val wMag = FeatureExtractor.compute3DMagnitude(gyroX, gyroY, gyroZ)
        val alMag = FeatureExtractor.compute3DMagnitude(accLinX, accLinY, accLinZ)
        for (mag in arrayOf(aMag, wMag, alMag)) {
            val stats = FeatureExtractor.computeStats(mag)
            System.arraycopy(stats, 0, features, offset, 4)
            offset += 4
        }

        return features
    }
}
