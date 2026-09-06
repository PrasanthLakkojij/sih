package com.sih26168.deadreckoning.engine

import com.sih26168.deadreckoning.ml.FeatureExtractor
import com.sih26168.deadreckoning.ml.ICorrectionModel
import com.sih26168.deadreckoning.physics.PhysicsDeadReckoning
import com.sih26168.deadreckoning.sensor.IMUSensorCollector
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Navigation state output from PositionEstimator.
 *
 * @param x Easting coordinate in local meters.
 * @param y Northing coordinate in local meters.
 * @param heading Navigation azimuth in radians (0=North, pi/2=East, clockwise).
 * @param headingDeg Navigation azimuth in degrees [0, 360).
 * @param velocity Forward speed in m/s (>= 0).
 * @param deltaS_imu Uncorrected physics displacement over the window in meters.
 * @param deltaS_corr ML correction predicted by ONNX model in meters.
 * @param deltaS_final Total corrected displacement: max(0.0, deltaS_imu + deltaS_corr) in meters.
 * @param headingDir Motion displacement vector azimuth in radians (atan2(de, dn)).
 * @param headingDirDeg Motion displacement vector azimuth in degrees.
 */
data class NavigationState(
    val x: Float,
    val y: Float,
    val heading: Float,
    val headingDeg: Float,
    val velocity: Float,
    val deltaS_imu: Float,
    val deltaS_corr: Float,
    val deltaS_final: Float,
    val headingDir: Float,
    val headingDirDeg: Float
)

/**
 * PositionEstimator:
 *
 * Full unified dead reckoning pipeline matching estimate_position.py in Step 1.
 * Wires together:
 *  1. FeatureExtractor (Step 3a): 58-feature extraction from ~9-second IMU window.
 *  2. PhysicsDeadReckoning (Step 3b): Phase 5B ZUPT-enhanced forward velocity and heading integration.
 *  3. CorrectionModel (Step 2b): Phase 9 C1 XGBoost displacement correction via ONNX Runtime.
 *
 * Combines physics displacement and ML correction:
 *  - correctedDeltaS = max(0.0f, deltaS_imu + deltaS_corr)
 *  - newX = currentX + correctedDeltaS * sin(hDir)
 *  - newY = currentY + correctedDeltaS * cos(hDir)
 *  - Updates state: (x, y, heading, velocity)
 */
class PositionEstimator(
    val correctionModel: ICorrectionModel,
    val physicsDeadReckoning: PhysicsDeadReckoning = PhysicsDeadReckoning()
) {
    // Current estimated position state
    var currentX: Float = 0.0f
    var currentY: Float = 0.0f
    var currentHeading: Float = 0.0f
    var currentVelocity: Float = 0.0f

    /**
     * Resets or sets the navigation state (e.g. from genuine GNSS fix or start of outage).
     */
    fun resetState(x: Float, y: Float, heading: Float, velocity: Float) {
        currentX = x
        currentY = y
        currentHeading = heading
        currentVelocity = max(0.0f, velocity)
        physicsDeadReckoning.resetState(x, y, heading, velocity)
    }

    /**
     * Processes an IMU window given as raw arrays matching estimate_position.py exactly.
     *
     * @param accX Raw accelerometer X array
     * @param accY Raw accelerometer Y array
     * @param accZ Raw accelerometer Z array
     * @param accLinX Linear acceleration X array
     * @param accLinY Linear acceleration Y array
     * @param accLinZ Linear acceleration Z array
     * @param gyroX Gyroscope X array
     * @param gyroY Gyroscope Y array
     * @param gyroZ Gyroscope Z array
     * @param accVehFwd Forward vehicle acceleration array
     * @param gyroVehYawRate Vehicle yaw rate array
     * @param dtArr Optional per-sample time step array in seconds
     * @param segDurS Optional segment duration in seconds
     * @param zuptMask Optional precomputed stationary mask
     * @return Updated NavigationState after processing this ~9-second segment
     */
    fun estimatePosition(
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
        dtArr: FloatArray? = null,
        segDurS: Float? = null,
        zuptMask: BooleanArray? = null
    ): NavigationState {
        val nSamples = accVehFwd.size
        require(nSamples > 0) { "imuWindow cannot be empty" }

        val duration = segDurS ?: if (dtArr != null) dtArr.sum() else (nSamples * 0.1f)

        // 1. Run PhysicsDeadReckoning across window
        val physResult = physicsDeadReckoning.processWindow(
            accLinX = accLinX,
            accLinY = accLinY,
            accLinZ = accLinZ,
            gyroX = gyroX,
            gyroY = gyroY,
            gyroZ = gyroZ,
            accVehFwd = accVehFwd,
            gyroVehYawRate = gyroVehYawRate,
            dtArr = dtArr,
            segDurS = duration,
            zuptMask = zuptMask,
            updateInternalState = true
        )

        // 2. Extract 58 features matching Phase 8/9 training
        val features = FeatureExtractor.extractFeatures(
            accX = accX,
            accY = accY,
            accZ = accZ,
            accLinX = accLinX,
            accLinY = accLinY,
            accLinZ = accLinZ,
            gyroX = gyroX,
            gyroY = gyroY,
            gyroZ = gyroZ,
            accVehFwd = accVehFwd,
            gyroVehYawRate = gyroVehYawRate,
            segDurS = duration,
            nSamp = nSamples
        )

        // 3. Predict displacement correction with ONNX model
        val deltaS_corr = correctionModel.predict(features)

        // 4. Corrected displacement (clamped to >= 0, matching estimate_position.py)
        val correctedDeltaS = max(0.0f, physResult.deltaS_imu + deltaS_corr)

        // 5. Update position coordinates (East = x, North = y) using motion direction hDir
        val newX = currentX + correctedDeltaS * sin(physResult.headingDir)
        val newY = currentY + correctedDeltaS * cos(physResult.headingDir)

        currentX = newX
        currentY = newY
        currentHeading = physResult.newHeading
        currentVelocity = physResult.newVelocity

        return NavigationState(
            x = newX,
            y = newY,
            heading = physResult.newHeading,
            headingDeg = physResult.newHeadingDeg,
            velocity = physResult.newVelocity,
            deltaS_imu = physResult.deltaS_imu,
            deltaS_corr = deltaS_corr,
            deltaS_final = correctedDeltaS,
            headingDir = physResult.headingDir,
            headingDirDeg = physResult.headingDirDeg
        )
    }

    /**
     * Convenience overload for processing buffered samples from IMUSensorCollector.
     */
    fun estimatePosition(samples: List<IMUSensorCollector.Sample>): NavigationState {
        val n = samples.size
        require(n > 0) { "samples cannot be empty" }

        val ax = FloatArray(n) { samples[it].accX }
        val ay = FloatArray(n) { samples[it].accY }
        val az = FloatArray(n) { samples[it].accZ }
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

        return estimatePosition(
            accX = ax, accY = ay, accZ = az,
            accLinX = lx, accLinY = ly, accLinZ = lz,
            gyroX = gx, gyroY = gy, gyroZ = gz,
            accVehFwd = fwd, gyroVehYawRate = yaw,
            dtArr = dtArr,
            segDurS = totalDurS
        )
    }
}
