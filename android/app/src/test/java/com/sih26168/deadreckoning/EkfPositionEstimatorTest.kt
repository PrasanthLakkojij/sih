package com.sih26168.deadreckoning

import com.sih26168.deadreckoning.engine.EkfPositionEstimator
import com.sih26168.deadreckoning.ml.IVelocityModel
import com.sih26168.deadreckoning.sensor.IMUSensorCollector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EkfPositionEstimatorTest {

    private fun sample(
        accLinX: Float = 0f, accLinY: Float = 0f, accLinZ: Float = 0f,
        gyroX: Float = 0f, gyroY: Float = 0f, gyroZ: Float = 0f,
        accVehFwd: Float = 0f, accVehLat: Float = 0f, gyroVehYawRate: Float = 0f,
        tNs: Long = 0L
    ) = IMUSensorCollector.Sample(
        accX = 0f, accY = 0f, accZ = 9.81f,
        accLinX = accLinX, accLinY = accLinY, accLinZ = accLinZ,
        gyroX = gyroX, gyroY = gyroY, gyroZ = gyroZ,
        accVehFwd = accVehFwd, accVehLat = accVehLat, gyroVehYawRate = gyroVehYawRate,
        magX = 0f, magY = 0f, magZ = 0f,
        timestampNs = tNs
    )

    @Test
    fun calibrateBiasesFromHistory_recoversKnownBias() {
        val mockModel = IVelocityModel { 0.0f }
        val estimator = EkfPositionEstimator(mockModel)

        // 100 samples, all below ZUPT thresholds (A_TH=5.389, W_TH=0.753),
        // with a consistent known "bias" on the vehicle-frame channels.
        val history = (0 until 100).map {
            sample(
                accLinX = 0.3f, accLinY = 0f, accLinZ = 0f,
                gyroX = 0f, gyroY = 0f, gyroZ = 0f,
                accVehFwd = 0.30f, accVehLat = -0.10f, gyroVehYawRate = 0.02f,
                tNs = it * 100_000_000L
            )
        }

        val calib = estimator.calibrateBiasesFromHistory(history)
        assertTrue("Should have enough stationary samples to calibrate", calib.accelCalibrated)
        assertTrue(calib.gyroCalibrated)
        assertEquals(0.30, calib.bAx, 1e-6)
        assertEquals(-0.10, calib.bAy, 1e-6)
        assertEquals(0.02, calib.bW, 1e-6)
    }

    @Test
    fun calibrateBiasesFromHistory_fallsBackToZero_whenInsufficientStationaryData() {
        val mockModel = IVelocityModel { 0.0f }
        val estimator = EkfPositionEstimator(mockModel)

        // All samples clearly moving (above A_TH) -- no stationary periods at all.
        val history = (0 until 100).map {
            sample(accLinX = 6.0f, accVehFwd = 1.0f, tNs = it * 100_000_000L)
        }

        val calib = estimator.calibrateBiasesFromHistory(history)
        assertEquals(false, calib.accelCalibrated)
        assertEquals(false, calib.gyroCalibrated)
        assertEquals(0.0, calib.bAx, 1e-9)
    }

    @Test
    fun zuptVetoGate_doesNotZeroVelocity_whenMlModelReportsMotion() {
        // ML model always reports 5.0 m/s -- confident the vehicle is moving.
        // IMU noise here alternates at real vehicle engine/road-vibration
        // magnitude (variance ~0.16 m^2/s^4 accel, ~0.02 rad^2/s^2 gyro) --
        // an order of magnitude above the hard lock's tight thresholds
        // (0.04 / 0.005), but still below the coarse ZUPT_A_TH/W_TH mask
        // thresholds. This is what actually separates "smooth cruise" from
        // "genuinely stationary" now that the hard lock re-evaluates every
        // sample instead of disarming after the first confirmed-moving sample
        // (see EkfPositionEstimator doc comment for why that one-shot design
        // was replaced -- real-device testing showed it failed to catch a
        // genuine stop after real motion).
        val mockModel = IVelocityModel { 5.0f }
        val estimator = EkfPositionEstimator(mockModel)
        estimator.startOutage(
            initPosEnu = doubleArrayOf(0.0, 0.0),
            initHeading = 0.0, // North
            initSpeed = 5.0,
            biasCalibration = EkfPositionEstimator.BiasCalibration(0.0, 0.0, 0.0, false, false)
        )

        var lastSnapshot: EkfPositionEstimator.EkfSnapshot? = null
        for (i in 0 until 30) {
            val accLin = if (i % 2 == 0) 0.2f else 1.0f   // mean 0.6, var 0.16
            val gyro = if (i % 2 == 0) 0.05f else 0.35f   // mean 0.2, var 0.0225
            val s = sample(accLinX = accLin, accLinY = 0.01f, gyroX = gyro, tNs = i * 100_000_000L)
            lastSnapshot = estimator.onSample(s, dtSeconds = 0.1)
        }

        // After the ML update kicks in (sample 10), velocity should be pulled
        // toward ~5 m/s and NOT get zeroed -- neither by the raw ZUPT mask
        // (veto gate) nor by the hard lock (vibration keeps it above threshold).
        assertTrue(
            "Speed should reflect ML model's motion estimate during real cruise-level vibration (was ${lastSnapshot?.speed})",
            (lastSnapshot?.speed ?: 0.0) > 1.0
        )
    }

    @Test
    fun genuineStopAfterRealMotion_zerosVelocity_insteadOfDriftingForever() {
        // Reproduces the newest reported bug: "i moved little bit and i
        // stopped but it moving without any motion of my phone". With the
        // old one-shot latch, hard lock disarmed permanently on the first
        // confirmed-moving sample and never protected a later genuine stop.
        val mockModel = IVelocityModel { 5.0f } // ML stays wrong/high even after the real stop
        val estimator = EkfPositionEstimator(mockModel)
        estimator.startOutage(
            initPosEnu = doubleArrayOf(0.0, 0.0),
            initHeading = 0.0,
            initSpeed = 0.0,
            biasCalibration = EkfPositionEstimator.BiasCalibration(0.0, 0.0, 0.0, false, false)
        )

        var lastSnapshot: EkfPositionEstimator.EkfSnapshot? = null
        var i = 0
        // Phase 1: 3s of real driving-level motion (well above hard-lock AND
        // ZUPT thresholds) -- speed should build up toward the ML estimate.
        for (k in 0 until 30) {
            lastSnapshot = estimator.onSample(
                sample(accLinX = 6.0f, gyroX = 0.1f, accVehFwd = 2.0f, gyroVehYawRate = 0.05f, tNs = (i++) * 100_000_000L),
                0.1
            )
        }
        assertTrue(
            "Should show real motion during the driving phase (was ${lastSnapshot?.speed})",
            (lastSnapshot?.speed ?: 0.0) > 1.0
        )

        // Phase 2: genuine stop -- 10s of near-zero IMU. ML model (fixed mock)
        // still reports 5.0 m/s, exactly like the real shipped model's
        // never-seen-true-zero bias -- hard lock must catch this anyway.
        for (k in 0 until 100) {
            lastSnapshot = estimator.onSample(sample(accLinX = 0.01f, accLinY = 0.01f, gyroX = 0.001f, tNs = (i++) * 100_000_000L), 0.1)
        }

        assertTrue(
            "Speed should zero out after a genuine stop, even mid-outage after real motion (was ${lastSnapshot?.speed})",
            (lastSnapshot?.speed ?: 99.0) < 0.5
        )
    }

    @Test
    fun zuptGate_doesZeroVelocity_whenMlModelAgreesStationary() {
        // ML model always reports 0.0 m/s -- agrees the vehicle is stationary.
        val mockModel = IVelocityModel { 0.0f }
        val estimator = EkfPositionEstimator(mockModel)
        estimator.startOutage(
            initPosEnu = doubleArrayOf(0.0, 0.0),
            initHeading = 0.0,
            initSpeed = 2.0, // starts with some residual velocity to be corrected away
            biasCalibration = EkfPositionEstimator.BiasCalibration(0.0, 0.0, 0.0, false, false)
        )

        var lastSnapshot: EkfPositionEstimator.EkfSnapshot? = null
        for (i in 0 until 30) {
            val s = sample(accLinX = 0.01f, accLinY = 0.01f, gyroX = 0.001f, tNs = i * 100_000_000L)
            lastSnapshot = estimator.onSample(s, dtSeconds = 0.1)
        }

        assertTrue(
            "Speed should be driven toward 0 when both ZUPT and ML agree stationary (was ${lastSnapshot?.speed})",
            (lastSnapshot?.speed ?: 99.0) < 0.5
        )
    }

    @Test
    fun hardStationaryLock_preventsDriftAway_whenOutageStartsWhileStanding() {
        // Reproduces the reported bug: c1_velocity_model.onnx outputs ~3.23 m/s
        // on all-zero input (verified via direct ONNX inference -- the model was
        // trained with GPS-truth < 0.5 m/s dropped, so it has no true-zero
        // baseline). Outage starts from a stationary phone (initSpeed=0), IMU is
        // near-zero noise throughout -- exactly the "standing still, red dot
        // runs away" scenario.
        val mockModel = IVelocityModel { 3.23f }
        val estimator = EkfPositionEstimator(mockModel)
        estimator.startOutage(
            initPosEnu = doubleArrayOf(0.0, 0.0),
            initHeading = 0.0,
            initSpeed = 0.0,
            biasCalibration = EkfPositionEstimator.BiasCalibration(0.0, 0.0, 0.0, false, false)
        )

        var lastSnapshot: EkfPositionEstimator.EkfSnapshot? = null
        // 30s @ 10Hz -- long enough that, pre-fix, 3.23 m/s integrated would
        // have carried the marker ~97m away.
        for (i in 0 until 300) {
            val s = sample(accLinX = 0.01f, accLinY = 0.01f, gyroX = 0.001f, tNs = i * 100_000_000L)
            lastSnapshot = estimator.onSample(s, dtSeconds = 0.1)
        }

        val speed = lastSnapshot?.speed ?: 99.0
        val displacement = hypotDist(lastSnapshot)
        assertTrue("Speed should stay near-zero while genuinely stationary (was $speed)", speed < 0.5)
        assertTrue("Position should not drift away while stationary (was ${displacement}m)", displacement < 2.0)
    }

    private fun hypotDist(s: EkfPositionEstimator.EkfSnapshot?): Double {
        if (s == null) return 0.0
        return kotlin.math.hypot(s.x, s.y)
    }

    @Test
    fun handheldRotation_doesNotCauseRunawayDrift_whenPhoneStaysInPlace() {
        // Reproduces: "moved slightly and rotated, kept phone stable -- but it
        // moved so far". A brief handheld tilt/rotation (not real vehicle
        // travel) injects real gyro + gravity-leaked accel into the vehicle
        // frame (see onSample() doc comment on the rigid-mount assumption).
        // With the sustained-confirmation latch + predict-path deadband, this
        // should settle back near the origin instead of running away.
        val mockModel = IVelocityModel { 3.23f } // worst case: same phantom-speed artifact
        val estimator = EkfPositionEstimator(mockModel)
        estimator.startOutage(
            initPosEnu = doubleArrayOf(0.0, 0.0),
            initHeading = 0.0,
            initSpeed = 0.0,
            biasCalibration = EkfPositionEstimator.BiasCalibration(0.0, 0.0, 0.0, false, false)
        )

        var lastSnapshot: EkfPositionEstimator.EkfSnapshot? = null
        var i = 0
        // 3s held still
        for (k in 0 until 30) {
            lastSnapshot = estimator.onSample(sample(accLinX = 0.01f, accLinY = 0.01f, gyroX = 0.001f, tNs = (i++) * 100_000_000L), 0.1)
        }
        // ~0.5s brief rotation/tilt event: real gyro + gravity-leaked accel
        for (k in 0 until 5) {
            lastSnapshot = estimator.onSample(
                sample(
                    accLinX = 0.4f, accLinY = 0.2f, gyroX = 0.3f,
                    accVehFwd = 0.35f, accVehLat = 0.15f, gyroVehYawRate = 0.25f,
                    tNs = (i++) * 100_000_000L
                ), 0.1
            )
        }
        // held still again for the rest -- 10s
        for (k in 0 until 100) {
            lastSnapshot = estimator.onSample(sample(accLinX = 0.01f, accLinY = 0.01f, gyroX = 0.001f, tNs = (i++) * 100_000_000L), 0.1)
        }

        val speed = lastSnapshot?.speed ?: 99.0
        val displacement = hypotDist(lastSnapshot)
        assertTrue("Speed should settle back near-zero after the rotation event (was $speed)", speed < 0.5)
        assertTrue("Displacement should stay small (same spot), not run away (was ${displacement}m)", displacement < 5.0)
    }
}
