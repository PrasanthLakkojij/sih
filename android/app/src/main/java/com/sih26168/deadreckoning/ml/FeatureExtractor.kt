package com.sih26168.deadreckoning.ml

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * FeatureExtractor: Translates the Python extract_58_features() logic into Kotlin.
 *
 * Exactly computes 58 features in identical order matching the trained XGBoost / ONNX model:
 * 1. 11 base signals x 4 stats (mean, std, min, max) = 44
 *    Order: acc_x, acc_y, acc_z, acc_lin_x, acc_lin_y, acc_lin_z,
 *           gyro_x, gyro_y, gyro_z, acc_veh_fwd, gyro_veh_yaw_rate
 * 2. 3 magnitude signals x 4 stats (mean, std, min, max) = 12
 *    Order: a_mag (sqrt(ax^2+ay^2+az^2)),
 *           al_mag (sqrt(lx^2+ly^2+lz^2)),
 *           w_mag (sqrt(gx^2+gy^2+gz^2))
 * 3. Timing context = 2 features:
 *    seg_dur_s, seg_n_samples
 * Total: 58 features
 */
object FeatureExtractor {

    /**
     * Compute summary statistics (mean, population std, min, max) for an array of floats.
     * Matches NumPy np.mean, np.std (ddof=0), np.min, np.max.
     */
    fun computeStats(arr: FloatArray): FloatArray {
        val n = arr.size
        if (n == 0) return floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)

        var sum = 0.0
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE

        for (v in arr) {
            sum += v
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }

        val mean = (sum / n).toFloat()

        var sumSqDiff = 0.0
        for (v in arr) {
            val diff = v - mean
            sumSqDiff += diff * diff
        }
        val std = sqrt(sumSqDiff / n).toFloat()

        return floatArrayOf(mean, std, minVal, maxVal)
    }

    /**
     * Compute element-wise 3D vector Euclidean magnitude: sqrt(x^2 + y^2 + z^2).
     */
    fun compute3DMagnitude(x: FloatArray, y: FloatArray, z: FloatArray): FloatArray {
        val n = x.size
        val mag = FloatArray(n)
        for (i in 0 until n) {
            mag[i] = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])
        }
        return mag
    }

    /**
     * Extracts the complete 58-feature vector from raw + vehicle frame arrays.
     */
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
        gyroVehYawRate: FloatArray,
        segDurS: Float,
        nSamp: Int
    ): FloatArray {
        val features = FloatArray(58)
        var offset = 0

        // 1. 11 base signals (44 features)
        val baseSignals = arrayOf(
            accX, accY, accZ,
            accLinX, accLinY, accLinZ,
            gyroX, gyroY, gyroZ,
            accVehFwd, gyroVehYawRate
        )

        for (signal in baseSignals) {
            val stats = computeStats(signal)
            System.arraycopy(stats, 0, features, offset, 4)
            offset += 4
        }

        // 2. 3 magnitude signals (12 features)
        val aMag = compute3DMagnitude(accX, accY, accZ)
        val alMag = compute3DMagnitude(accLinX, accLinY, accLinZ)
        val wMag = compute3DMagnitude(gyroX, gyroY, gyroZ)

        val magSignals = arrayOf(aMag, alMag, wMag)
        for (mag in magSignals) {
            val stats = computeStats(mag)
            System.arraycopy(stats, 0, features, offset, 4)
            offset += 4
        }

        // 3. Timing context (2 features)
        features[56] = segDurS
        features[57] = nSamp.toFloat()

        return features
    }
}
