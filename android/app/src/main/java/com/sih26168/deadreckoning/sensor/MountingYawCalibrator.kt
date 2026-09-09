package com.sih26168.deadreckoning.sensor

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * MountingYawCalibrator: direct Kotlin port of phase4_orientation.py's
 * compute_kinematic_alignment() -- determines the phone's mounting yaw
 * offset (theta) relative to the vehicle's forward axis, using GPS-speed-
 * derived longitudinal acceleration as ground truth to correlate against
 * candidate rotations of the phone's horizontal linear-acceleration axes.
 *
 * This replaces the app's previous hardcoded `mountingYawRad = 0.0f`
 * (dashboard-flat-mount assumption) with an online estimate from real
 * driving dynamics, matching this session's PS-gap item (dynamic in-vehicle
 * alignment). Requires GPS to be active (correlates against GPS-derived
 * acceleration), so it runs continuously during GPS-active mode -- same
 * pattern as EkfPositionEstimator's pre-outage bias calibration -- and the
 * resulting theta is what's used during the subsequent outage.
 */
object MountingYawCalibrator {

    data class Result(val thetaRad: Double, val correlation: Double, val calibrated: Boolean)

    private const val MIN_MOVING_SAMPLES = 100
    private const val MIN_ACCEL_SAMPLES = 50
    private const val ACCEL_EVENT_THRESHOLD = 0.3 // m/s^2, matches Python's |a_gps| > 0.3 gate
    private const val ANGLE_STEPS = 360 // -> 361 candidate angles, matches np.linspace(-pi, pi, 361)

    /**
     * @param tSec Timestamps in seconds (monotonic, ascending)
     * @param gpsSpeedMs GPS speed at each sample (held/interpolated between fixes, m/s)
     * @param accLinX Phone-frame linear acceleration X at each sample
     * @param accLinY Phone-frame linear acceleration Y at each sample
     * @param minSpeedMs Minimum GPS speed to consider "moving" (default 3.0 m/s, matches Python)
     */
    fun calibrate(
        tSec: DoubleArray,
        gpsSpeedMs: DoubleArray,
        accLinX: DoubleArray,
        accLinY: DoubleArray,
        minSpeedMs: Double = 3.0
    ): Result {
        val n = tSec.size
        require(gpsSpeedMs.size == n && accLinX.size == n && accLinY.size == n) {
            "MountingYawCalibrator: all input arrays must have equal length"
        }
        if (n == 0) return Result(0.0, 0.0, false)

        val movingMask = BooleanArray(n) { gpsSpeedMs[it] > minSpeedMs }
        if (movingMask.count { it } < MIN_MOVING_SAMPLES) return Result(0.0, 0.0, false)

        val aGps = centralDiffGradient(gpsSpeedMs, tSec)

        var accelMask = BooleanArray(n) { movingMask[it] && abs(aGps[it]) > ACCEL_EVENT_THRESHOLD }
        if (accelMask.count { it } < MIN_ACCEL_SAMPLES) accelMask = movingMask

        val subCount = accelMask.count { it }
        val aGpsSub = DoubleArray(subCount)
        val axSub = DoubleArray(subCount)
        val aySub = DoubleArray(subCount)
        var idx = 0
        for (i in 0 until n) {
            if (accelMask[i]) {
                aGpsSub[idx] = aGps[i]; axSub[idx] = accLinX[i].toDouble(); aySub[idx] = accLinY[i].toDouble()
                idx++
            }
        }

        var bestTheta = 0.0
        var bestCorr = -Double.MAX_VALUE
        for (k in 0..ANGLE_STEPS) {
            val theta = -PI + k * (2.0 * PI / ANGLE_STEPS)
            val aFwdCand = DoubleArray(subCount) { axSub[it] * cos(theta) + aySub[it] * sin(theta) }
            val corr = pearsonCorrelation(aFwdCand, aGpsSub)
            if (corr > bestCorr) {
                bestCorr = corr
                bestTheta = theta
            }
        }
        return Result(bestTheta, bestCorr, true)
    }

    /** Matches numpy's np.gradient() central-difference formula exactly (including edges). */
    private fun centralDiffGradient(y: DoubleArray, x: DoubleArray): DoubleArray {
        val n = y.size
        val out = DoubleArray(n)
        if (n == 1) return out
        out[0] = (y[1] - y[0]) / (x[1] - x[0])
        out[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
        for (i in 1 until n - 1) {
            out[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        }
        return out
    }

    /** Population Pearson correlation (ddof=0, matches numpy's corrcoef/std defaults). */
    private fun pearsonCorrelation(a: DoubleArray, b: DoubleArray): Double {
        val n = a.size
        val meanA = a.average()
        val meanB = b.average()
        var num = 0.0; var denA = 0.0; var denB = 0.0
        for (i in 0 until n) {
            val da = a[i] - meanA
            val db = b[i] - meanB
            num += da * db
            denA += da * da
            denB += db * db
        }
        val stdA = sqrt(denA / n)
        val stdB = sqrt(denB / n)
        if (stdA <= 1e-4 || stdB <= 1e-4) return 0.0
        return num / sqrt(denA * denB)
    }
}
