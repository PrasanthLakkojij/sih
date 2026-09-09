package com.sih26168.deadreckoning.engine

import com.sih26168.deadreckoning.ml.IVelocityModel
import com.sih26168.deadreckoning.ml.Phase7FeatureExtractor
import com.sih26168.deadreckoning.physics.PhysicsDeadReckoning
import com.sih26168.deadreckoning.sensor.IMUSensorCollector
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * EkfPositionEstimator: continuous (per-10Hz-sample) fusion estimator for use
 * during a GNSS outage. Ports every fix validated in this session's Python
 * research (phase10_fusion_engine.py) into the live per-sample Android path:
 *
 *  - ExtendedKalmanFilter with NHC every step (session Phase 1)
 *  - Pre-outage gyro/accel bias calibration from stationary history (session
 *    fixes #2, #4) -- see calibrateBiasesFromHistory()
 *  - Map-matching heading feedback (session fix #3) -- via the roadMatcher hook
 *  - ZUPT-vs-ML veto gate (session fix #5, the single biggest win: ~99%->~81%
 *    drift in the Python validation) -- do NOT apply a ZUPT correction if the
 *    ML velocity model is confident the vehicle is moving, even when the raw
 *    IMU-threshold ZUPT mask says "stationary". Both threshold-only and
 *    variance-gated ZUPT masks were measured to false-positive during smooth
 *    constant-speed cruise; only an independent speed source (the ML model)
 *    can catch that.
 *  - ML speed bias correction (session fix #6) -- baked into VelocityModel's
 *    own predict(), not duplicated here.
 *
 * NOTE ON PARITY: the pre-outage/continuous ZUPT classification here uses
 * PhysicsDeadReckoning's existing v1 (magnitude-threshold-only) detector, not
 * the variance-gated v2 detector the *final* Python validation run used
 * (get_zupt_v2_mask, phase5c_physics_baseline.py -- never ported to Kotlin).
 * v1 has a known, measured false-positive blind spot during smooth cruise;
 * the ZUPT-vs-ML veto gate below substantially mitigates it but a v2 port
 * would be a further improvement.
 */
class EkfPositionEstimator(
    private val velocityModel: IVelocityModel,
    /** Returns a matched road heading (radians) for the given ENU position and
     * current heading estimate, or null if no confident match. Wire this to
     * LightweightMapMatcher's existing heading-gated snap logic. */
    private val roadMatcher: ((px: Double, py: Double, psiCurrent: Double) -> Double?)? = null
) {
    companion object {
        private const val PHASE7_WINDOW = Phase7FeatureExtractor.WINDOW_LEN // 20 samples (~2s)
        private const val ML_UPDATE_CADENCE = 10   // every ~1.0s at 10Hz
        private const val MAP_MATCH_CADENCE = 5    // every ~0.5s at 10Hz (matches Python's tightened 2Hz)
        private const val ZUPT_ML_VETO_MS = 1.0
        private const val NHC_R_LAT = 0.2
        private const val ZUPT_R = 0.05
        private const val HEADING_R_RAD = 0.13962634 // 8 degrees
        private const val ML_SPEED_R = 1.8           // default; caller may override via mlSpeedSigma
        private const val ZUPT_A_TH = PhysicsDeadReckoning.A_TH.toDouble()
        private const val ZUPT_W_TH = PhysicsDeadReckoning.W_TH.toDouble()
        private const val ZUPT_WINDOW_LEN = PhysicsDeadReckoning.WINDOW_LEN
        private const val BIAS_CALIB_MIN_SAMPLES = 50
        private const val BIAS_CALIB_MAX_BW = 0.1     // rad/s clamp
        private const val BIAS_CALIB_MAX_BA = 1.0      // m/s^2 clamp
    }

    data class BiasCalibration(
        val bAx: Double, val bAy: Double, val bW: Double,
        val accelCalibrated: Boolean, val gyroCalibrated: Boolean
    )

    data class EkfSnapshot(
        val x: Double, val y: Double, val heading: Double, val headingDeg: Double,
        val speed: Double, val isMapMatched: Boolean
    )

    var mlSpeedSigma: Double = ML_SPEED_R

    private var ekf: ExtendedKalmanFilter? = null
    private val phase7Buffer = ArrayDeque<IMUSensorCollector.Sample>()
    private var lastVMl: Double = 0.0
    private var sampleCountSinceStart = 0
    private var consecutiveStationaryCount = 0

    /**
     * Calibrate gyro/accel bias from a rolling pre-outage history buffer of
     * stationary periods. Caller maintains `history` continuously during
     * GPS-active mode (e.g. last ~60s / ~600 samples at 10Hz).
     */
    fun calibrateBiasesFromHistory(history: List<IMUSensorCollector.Sample>): BiasCalibration {
        if (history.isEmpty()) return BiasCalibration(0.0, 0.0, 0.0, false, false)

        val stationaryIdx = mutableListOf<Int>()
        var consec = 0
        for (i in history.indices) {
            val s = history[i]
            val aMag = sqrt((s.accLinX * s.accLinX + s.accLinY * s.accLinY + s.accLinZ * s.accLinZ).toDouble())
            val wMag = sqrt((s.gyroX * s.gyroX + s.gyroY * s.gyroY + s.gyroZ * s.gyroZ).toDouble())
            if (aMag < ZUPT_A_TH && wMag < ZUPT_W_TH) {
                consec++
                if (consec >= ZUPT_WINDOW_LEN) stationaryIdx.add(i)
            } else {
                consec = 0
            }
        }

        if (stationaryIdx.size < BIAS_CALIB_MIN_SAMPLES) {
            return BiasCalibration(0.0, 0.0, 0.0, false, false)
        }

        var sumFwd = 0.0; var sumLat = 0.0; var sumYaw = 0.0
        for (i in stationaryIdx) {
            sumFwd += history[i].accVehFwd
            sumLat += history[i].accVehLat
            sumYaw += history[i].gyroVehYawRate
        }
        val n = stationaryIdx.size
        val bAx = (sumFwd / n).coerceIn(-BIAS_CALIB_MAX_BA, BIAS_CALIB_MAX_BA)
        val bAy = (sumLat / n).coerceIn(-BIAS_CALIB_MAX_BA, BIAS_CALIB_MAX_BA)
        val bW = (sumYaw / n).coerceIn(-BIAS_CALIB_MAX_BW, BIAS_CALIB_MAX_BW)
        return BiasCalibration(bAx, bAy, bW, true, true)
    }

    /** Start (or restart) continuous EKF tracking at a genuine GNSS fix boundary. */
    fun startOutage(
        initPosEnu: DoubleArray,
        initHeading: Double,
        initSpeed: Double,
        biasCalibration: BiasCalibration
    ) {
        val vx0 = initSpeed * kotlin.math.sin(initHeading)
        val vy0 = initSpeed * kotlin.math.cos(initHeading)
        val newEkf = ExtendedKalmanFilter(
            initPos = initPosEnu,
            initVel = doubleArrayOf(vx0, vy0),
            initHeading = initHeading,
            sigmaAProc = 4.0,
            sigmaWProc = Math.toRadians(2.0),
            sigmaBaProc = 1e-3,
            sigmaBwProc = 1e-4
        )
        newEkf.seedBias(5, biasCalibration.bAx, if (biasCalibration.accelCalibrated) 0.15 else null)
        newEkf.seedBias(6, biasCalibration.bAy, if (biasCalibration.accelCalibrated) 0.15 else null)
        newEkf.seedBias(7, biasCalibration.bW, if (biasCalibration.gyroCalibrated) Math.toRadians(0.2) else null)
        ekf = newEkf

        phase7Buffer.clear()
        lastVMl = initSpeed
        sampleCountSinceStart = 0
        consecutiveStationaryCount = 0
    }

    fun stopOutage() {
        ekf = null
    }

    /**
     * Process one 10Hz IMU sample during an active outage. Returns the
     * updated EKF snapshot, or null if startOutage() hasn't been called.
     */
    fun onSample(sample: IMUSensorCollector.Sample, dtSeconds: Double): EkfSnapshot? {
        val e = ekf ?: return null
        sampleCountSinceStart++

        e.predict(sample.accVehFwd.toDouble(), sample.accVehLat.toDouble(), sample.gyroVehYawRate.toDouble(), dtSeconds)
        e.updateNhc(NHC_R_LAT)

        var mapMatched = false
        if (roadMatcher != null && sampleCountSinceStart % MAP_MATCH_CADENCE == 0) {
            val matchedHeading = roadMatcher.invoke(e.x[0], e.x[1], e.x[4])
            if (matchedHeading != null) {
                e.updateHeading(matchedHeading, HEADING_R_RAD)
                mapMatched = true
            }
        }

        // Streaming ZUPT classification (v1-equivalent, see class doc for the
        // known false-positive limitation this inherits).
        val aMag = sqrt((sample.accLinX * sample.accLinX + sample.accLinY * sample.accLinY + sample.accLinZ * sample.accLinZ).toDouble())
        val wMag = sqrt((sample.gyroX * sample.gyroX + sample.gyroY * sample.gyroY + sample.gyroZ * sample.gyroZ).toDouble())
        val isStationaryRaw = if (aMag < ZUPT_A_TH && wMag < ZUPT_W_TH) {
            consecutiveStationaryCount++
            consecutiveStationaryCount >= ZUPT_WINDOW_LEN
        } else {
            consecutiveStationaryCount = 0
            false
        }

        // ZUPT-vs-ML veto gate: this is the fix that took Python's drift from
        // ~99% to ~81% alone. Do not zero velocity if the ML model's last
        // known estimate says the vehicle is moving.
        if (isStationaryRaw && lastVMl < ZUPT_ML_VETO_MS) {
            e.updateZupt(ZUPT_R)
        }

        phase7Buffer.addLast(sample)
        if (phase7Buffer.size > PHASE7_WINDOW) phase7Buffer.removeFirst()
        if (sampleCountSinceStart % ML_UPDATE_CADENCE == 0 && phase7Buffer.size == PHASE7_WINDOW) {
            val feats = buildPhase7Features(phase7Buffer)
            val vMl = velocityModel.predict(feats)
            if (vMl != null) {
                e.updateMlSpeed(vMl.toDouble(), mlSpeedSigma)
                lastVMl = vMl.toDouble()
            }
            // else: inference failed (see velocityModel.lastFailure) -- skip
            // this cycle's ML update, keep prior lastVMl rather than resetting
            // to a value that would silently look like "not moving".
        }

        val speed = hypot(e.x[2], e.x[3])
        val headingDeg = (Math.toDegrees(e.x[4]) + 360.0) % 360.0
        return EkfSnapshot(e.x[0], e.x[1], e.x[4], headingDeg, speed, mapMatched)
    }

    private fun buildPhase7Features(buffer: ArrayDeque<IMUSensorCollector.Sample>): FloatArray {
        val n = buffer.size
        val accX = FloatArray(n); val accY = FloatArray(n); val accZ = FloatArray(n)
        val accLinX = FloatArray(n); val accLinY = FloatArray(n); val accLinZ = FloatArray(n)
        val gyroX = FloatArray(n); val gyroY = FloatArray(n); val gyroZ = FloatArray(n)
        val accVehFwd = FloatArray(n); val gyroVehYawRate = FloatArray(n)
        for (i in 0 until n) {
            val s = buffer[i]
            accX[i] = s.accX; accY[i] = s.accY; accZ[i] = s.accZ
            accLinX[i] = s.accLinX; accLinY[i] = s.accLinY; accLinZ[i] = s.accLinZ
            gyroX[i] = s.gyroX; gyroY[i] = s.gyroY; gyroZ[i] = s.gyroZ
            accVehFwd[i] = s.accVehFwd; gyroVehYawRate[i] = s.gyroVehYawRate
        }
        return Phase7FeatureExtractor.extractFeatures(
            accX, accY, accZ, accLinX, accLinY, accLinZ, gyroX, gyroY, gyroZ, accVehFwd, gyroVehYawRate
        )
    }
}
