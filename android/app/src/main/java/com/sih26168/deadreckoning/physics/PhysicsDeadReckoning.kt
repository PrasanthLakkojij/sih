package com.sih26168.deadreckoning.physics

import com.sih26168.deadreckoning.sensor.IMUSensorCollector
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Result of physics dead reckoning over an IMU window (~9 seconds).
 *
 * @param deltaS_imu Physics-only integrated displacement in meters.
 * @param newHeading Integrated navigation heading in radians (0=North, pi/2=East, clockwise).
 * @param newHeadingDeg Integrated navigation heading in degrees [0, 360).
 * @param newVelocity Integrated forward velocity at end of window in m/s (>= 0).
 * @param headingDir Direction of displacement vector over the window in radians (atan2(de, dn)).
 * @param headingDirDeg Direction of displacement vector in degrees.
 * @param newX Updated Easting coordinate in local meters.
 * @param newY Updated Northing coordinate in local meters.
 * @param de Delta East displacement in meters.
 * @param dn Delta North displacement in meters.
 * @param stationarySamples Count of samples classified as stationary in this window.
 */
data class PhysicsResult(
    val deltaS_imu: Float,
    val newHeading: Float,
    val newHeadingDeg: Float,
    val newVelocity: Float,
    val headingDir: Float,
    val headingDirDeg: Float,
    val newX: Float,
    val newY: Float,
    val de: Float,
    val dn: Float,
    val stationarySamples: Int
)

/**
 * PhysicsDeadReckoning:
 *
 * Port of the Phase 5B ZUPT-enhanced physics dead reckoning baseline and Step 1's
 * estimate_position() physics integration engine to Kotlin.
 *
 * Locked Phase 5B Parameters:
 *  - a_th = 5.389 m/s^2 (linear acceleration magnitude threshold)
 *  - w_th = 0.753 rad/s (angular rate magnitude threshold)
 *  - stationary window = 0.8 s (8 consecutive samples at 10Hz)
 *
 * Internal State:
 *  - currentX: Local Easting in meters
 *  - currentY: Local Northing in meters
 *  - currentHeading: Navigation azimuth in radians (0=North, pi/2=East, clockwise)
 *  - currentVelocity: Forward velocity in m/s (>= 0)
 *  - consecutiveStationaryCount: Counter of consecutive stationary samples
 */
class PhysicsDeadReckoning(
    var currentX: Float = 0.0f,
    var currentY: Float = 0.0f,
    var currentHeading: Float = 0.0f,
    var currentVelocity: Float = 0.0f
) {
    companion object {
        const val A_TH: Float = 5.389f
        const val W_TH: Float = 0.753f
        const val WINDOW_LEN: Int = 8
        private const val TWO_PI: Float = (2.0 * Math.PI).toFloat()
        private const val RAD_TO_DEG: Float = (180.0 / Math.PI).toFloat()

        /**
         * Normalizes an angle in radians to [0, 2*pi).
         */
        fun normalizeAngleRad(rad: Float): Float {
            var angle = rad % TWO_PI
            if (angle < 0f) angle += TWO_PI
            return angle
        }

        /**
         * Computes the ZUPT stationary mask over arrays of linear acceleration and gyro readings.
         *
         * @param accLinX Linear acceleration X (m/s^2)
         * @param accLinY Linear acceleration Y (m/s^2)
         * @param accLinZ Linear acceleration Z (m/s^2)
         * @param gyroX Gyroscope rate X (rad/s)
         * @param gyroY Gyroscope rate Y (rad/s)
         * @param gyroZ Gyroscope rate Z (rad/s)
         * @param initialConsecutive Starting consecutive count carried from preceding window (default 0)
         * @return BooleanArray indicating whether each sample is confirmed stationary
         */
        fun computeZuptMask(
            accLinX: FloatArray,
            accLinY: FloatArray,
            accLinZ: FloatArray,
            gyroX: FloatArray,
            gyroY: FloatArray,
            gyroZ: FloatArray,
            initialConsecutive: Int = 0
        ): BooleanArray {
            val n = accLinX.size
            val mask = BooleanArray(n)
            var consec = initialConsecutive
            for (i in 0 until n) {
                val aMag = sqrt(accLinX[i] * accLinX[i] + accLinY[i] * accLinY[i] + accLinZ[i] * accLinZ[i])
                val wMag = sqrt(gyroX[i] * gyroX[i] + gyroY[i] * gyroY[i] + gyroZ[i] * gyroZ[i])
                if (aMag < A_TH && wMag < W_TH) {
                    consec++
                    if (consec >= WINDOW_LEN) {
                        mask[i] = true
                    }
                } else {
                    consec = 0
                }
            }
            return mask
        }
    }

    var consecutiveStationaryCount: Int = 0

    /**
     * Resets or sets the internal navigation state (e.g. from genuine GNSS fix).
     */
    fun resetState(x: Float, y: Float, heading: Float, velocity: Float) {
        currentX = x
        currentY = y
        currentHeading = heading
        currentVelocity = max(0.0f, velocity)
        consecutiveStationaryCount = 0
    }

    /**
     * Processes an IMU window given as parallel signal arrays.
     * Matches estimate_position() physics integration step-by-step.
     *
     * @param accLinX Linear acceleration X array
     * @param accLinY Linear acceleration Y array
     * @param accLinZ Linear acceleration Z array
     * @param gyroX Gyroscope X array
     * @param gyroY Gyroscope Y array
     * @param gyroZ Gyroscope Z array
     * @param accVehFwd Forward vehicle acceleration array
     * @param gyroVehYawRate Vehicle yaw rate array
     * @param dtArr Optional per-sample time step array in seconds. If null, segDurS / (N-1) is used.
     * @param segDurS Optional total segment duration in seconds.
     * @param zuptMask Optional precomputed stationary mask. If null, computed via computeZuptMask().
     * @param updateInternalState Whether to update internal state (currentX, currentY, currentHeading, currentVelocity).
     */
    fun processWindow(
        accLinX: FloatArray,
        accLinY: FloatArray,
        accLinZ: FloatArray,
        gyroX: FloatArray,
        gyroY: FloatArray,
        gyroZ: FloatArray,
        accVehFwd: FloatArray,
        gyroVehYawRate: FloatArray,
        dtArr: FloatArray? = null,
        segDurS: Float? = null,
        zuptMask: BooleanArray? = null,
        updateInternalState: Boolean = true
    ): PhysicsResult {
        val nSamples = accVehFwd.size
        require(nSamples > 0) { "imuWindow cannot be empty" }

        // Compute or reuse ZUPT mask
        val mask = zuptMask ?: computeZuptMask(
            accLinX, accLinY, accLinZ,
            gyroX, gyroY, gyroZ,
            initialConsecutive = consecutiveStationaryCount
        )

        // Default dt if dtArr is not supplied
        val defaultDt = if (segDurS != null && nSamples > 1) {
            segDurS / (nSamples - 1)
        } else {
            0.1f // Nominal 10Hz
        }

        var vStep = currentVelocity
        var psiStep = currentHeading
        var deSub = 0.0f
        var dnSub = 0.0f
        var stationaryCount = 0

        val startK = if (nSamples > 1) 1 else 0
        for (t in startK until nSamples) {
            val dt = dtArr?.getOrNull(t) ?: defaultDt

            if (mask[t]) {
                vStep = 0.0f
                stationaryCount++
            } else {
                vStep = max(0.0f, vStep + accVehFwd[t] * dt)
            }

            psiStep += gyroVehYawRate[t] * dt
            deSub += vStep * sin(psiStep) * dt
            dnSub += vStep * cos(psiStep) * dt
        }

        val dsImu = sqrt(deSub * deSub + dnSub * dnSub)
        val hDir = if (dsImu > 1e-3f) {
            atan2(deSub, dnSub)
        } else {
            psiStep
        }

        val newX = currentX + deSub
        val newY = currentY + dnSub

        if (updateInternalState) {
            currentX = newX
            currentY = newY
            currentHeading = psiStep
            currentVelocity = vStep

            // Track stationary streak at end of window
            var streak = 0
            for (i in (nSamples - 1) downTo 0) {
                val aMag = sqrt(accLinX[i] * accLinX[i] + accLinY[i] * accLinY[i] + accLinZ[i] * accLinZ[i])
                val wMag = sqrt(gyroX[i] * gyroX[i] + gyroY[i] * gyroY[i] + gyroZ[i] * gyroZ[i])
                if (aMag < A_TH && wMag < W_TH) {
                    streak++
                } else {
                    break
                }
            }
            consecutiveStationaryCount = streak
        }

        return PhysicsResult(
            deltaS_imu = dsImu,
            newHeading = psiStep,
            newHeadingDeg = normalizeAngleRad(psiStep) * RAD_TO_DEG,
            newVelocity = vStep,
            headingDir = hDir,
            headingDirDeg = normalizeAngleRad(hDir) * RAD_TO_DEG,
            newX = newX,
            newY = newY,
            de = deSub,
            dn = dnSub,
            stationarySamples = stationaryCount
        )
    }

    /**
     * Convenience overload for processing buffered samples from IMUSensorCollector.
     */
    fun processWindow(
        samples: List<IMUSensorCollector.Sample>,
        updateInternalState: Boolean = true
    ): PhysicsResult {
        val n = samples.size
        require(n > 0) { "samples cannot be empty" }

        val lx = FloatArray(n) { samples[it].accLinX }
        val ly = FloatArray(n) { samples[it].accLinY }
        val lz = FloatArray(n) { samples[it].accLinZ }
        val gx = FloatArray(n) { samples[it].gyroX }
        val gy = FloatArray(n) { samples[it].gyroY }
        val gz = FloatArray(n) { samples[it].gyroZ }
        val fwd = FloatArray(n) { samples[it].accVehFwd }
        val yaw = FloatArray(n) { samples[it].gyroVehYawRate }

        val dtArr = FloatArray(n)
        for (i in 1 until n) {
            val dtNs = samples[i].timestampNs - samples[i - 1].timestampNs
            dtArr[i] = if (dtNs in 10_000_000L..1_000_000_000L) {
                dtNs / 1_000_000_000f
            } else {
                0.1f
            }
        }

        val totalDurS = (samples.last().timestampNs - samples.first().timestampNs) / 1_000_000_000f

        return processWindow(
            accLinX = lx, accLinY = ly, accLinZ = lz,
            gyroX = gx, gyroY = gy, gyroZ = gz,
            accVehFwd = fwd, gyroVehYawRate = yaw,
            dtArr = dtArr, segDurS = totalDurS,
            updateInternalState = updateInternalState
        )
    }
}
