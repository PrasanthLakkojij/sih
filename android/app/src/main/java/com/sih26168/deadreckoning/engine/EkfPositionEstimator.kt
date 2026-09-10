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

        // --- Pre-motion hard stationary lock (fixes: red dot runs away when the
        // phone is stationary, e.g. handheld/standing, at outage start) ---
        //
        // Root cause (verified against the shipped c1_velocity_model.onnx):
        // Phase 7 training dropped all GPS-truth < 0.5 m/s as "stationary noise"
        // (phase7_ml_velocity.py MIN_SPEED_MS), so the model has never seen a
        // true-zero example and outputs ~3.2 m/s on all-zero input -- confirmed
        // by direct ONNX inference. That phantom speed exceeds ZUPT_ML_VETO_MS,
        // so the existing veto gate (correctly, by its own design) keeps trusting
        // the ML model and never zeros velocity -- the EKF integrates ~3.2 m/s
        // forever and the marker runs away.
        //
        // Cannot fix this by simply making the ZUPT mask always win over ML:
        // a stationary phone and a car cruising at constant speed produce the
        // *same* low-jerk IMU signature (that's precisely why the ML veto gate
        // was added -- see zuptVetoGate_doesNotZeroVelocity_whenMlModelReportsMotion
        // in EkfPositionEstimatorTest, which validates that exact override for
        // real cruise). Pure IMU variance cannot tell "standing still" apart from
        // "smooth highway cruise".
        //
        // Fix: was originally a ONE-SHOT latch (armed only pre-motion, disarmed
        // forever after the first confirmed-moving sample) -- real-device
        // testing showed that's wrong: after moving then genuinely stopping,
        // the marker kept drifting exactly like the original bug, because the
        // lock had permanently switched itself off. Changed to ALWAYS
        // re-evaluate every sample: any time IMU variance drops below these
        // (tight, hand/table-stillness-level) thresholds for a full window,
        // hard lock re-engages, moving or not, pre-motion or mid-outage.
        //
        // This does re-open the theoretical cruise-false-positive risk the
        // comment above describes -- but real vehicle engine/road vibration is
        // an order of magnitude above these thresholds (>0.2 vs 0.04), so in
        // practice a genuinely moving vehicle keeps failing the hard-lock
        // variance check and falls through to the ML veto gate as before; this
        // only fires for actual near-total stillness. See
        // zuptVetoGate_doesNotZeroVelocity_whenMlModelReportsMotion in
        // EkfPositionEstimatorTest, updated to inject realistic vehicle-level
        // vibration noise instead of clean synthetic zero, confirming the
        // cruise case still isn't caught by this lock.
        private const val HARD_LOCK_WINDOW = 10          // 1.0s @ 10Hz
        private const val HARD_LOCK_VAR_A_TH = 0.04      // m^2/s^4 (hand/table stillness; vehicle idle typically >0.2)
        private const val HARD_LOCK_VAR_W_TH = 0.005     // rad^2/s^2
        private const val HARD_ZUPT_R = 0.02             // tighter than the normal veto-gated ZUPT_R

        // Reject sub-noise-floor accel/gyro before they reach the physics
        // integrator (predict() only -- NOT the ML feature buffer, which needs
        // the raw signal to match its training distribution). Absorbs natural
        // hand tremor / phone micro-rotation without needing full stillness;
        // does not by itself fix a deliberate hand rotation (see class doc).
        private const val PREDICT_ACCEL_DEADBAND = 0.15  // m/s^2
        private const val PREDICT_YAW_DEADBAND = 0.02    // rad/s
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

    private val hardLockAMagWindow = ArrayDeque<Double>()
    private val hardLockWMagWindow = ArrayDeque<Double>()

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
        hardLockAMagWindow.clear()
        hardLockWMagWindow.clear()
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

        val dAccFwd = deadband(sample.accVehFwd.toDouble(), PREDICT_ACCEL_DEADBAND)
        val dAccLat = deadband(sample.accVehLat.toDouble(), PREDICT_ACCEL_DEADBAND)
        val dYawRate = deadband(sample.gyroVehYawRate.toDouble(), PREDICT_YAW_DEADBAND)
        e.predict(dAccFwd, dAccLat, dYawRate, dtSeconds)
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

        // Hard lock: re-evaluated every sample, always armed (see companion
        // doc comment -- was a one-shot pre-motion-only latch, real testing
        // showed it must stay live after real motion too, to catch genuine
        // stops).
        hardLockAMagWindow.addLast(aMag)
        if (hardLockAMagWindow.size > HARD_LOCK_WINDOW) hardLockAMagWindow.removeFirst()
        hardLockWMagWindow.addLast(wMag)
        if (hardLockWMagWindow.size > HARD_LOCK_WINDOW) hardLockWMagWindow.removeFirst()

        var isHardLocked = false
        if (hardLockAMagWindow.size == HARD_LOCK_WINDOW) {
            // Variance alone is not enough: a large but perfectly constant
            // acceleration (e.g. steady real acceleration ramp) also has zero
            // variance and would wrongly hard-lock. Require the window's mean
            // magnitude to also be small (same coarse thresholds as the v1
            // ZUPT mask) -- this makes the hard lock a strict refinement of
            // isStationaryRaw, adding the variance check on top rather than
            // replacing the magnitude check.
            val meanA = hardLockAMagWindow.sum() / hardLockAMagWindow.size
            val meanW = hardLockWMagWindow.sum() / hardLockWMagWindow.size
            val varA = variance(hardLockAMagWindow)
            val varW = variance(hardLockWMagWindow)
            isHardLocked = meanA < ZUPT_A_TH && meanW < ZUPT_W_TH &&
                varA < HARD_LOCK_VAR_A_TH && varW < HARD_LOCK_VAR_W_TH
        }

        if (isHardLocked) {
            e.updateZupt(HARD_ZUPT_R)
            lastVMl = 0.0
        } else {
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
        }

        val speed = hypot(e.x[2], e.x[3])
        val headingDeg = (Math.toDegrees(e.x[4]) + 360.0) % 360.0
        return EkfSnapshot(e.x[0], e.x[1], e.x[4], headingDeg, speed, mapMatched)
    }

    private fun variance(values: ArrayDeque<Double>): Double {
        val mean = values.sum() / values.size
        return values.sumOf { (it - mean) * (it - mean) } / values.size
    }

    private fun deadband(value: Double, threshold: Double): Double =
        if (kotlin.math.abs(value) < threshold) 0.0 else value

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
