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
        val mockModel = IVelocityModel { 5.0f }
        val estimator = EkfPositionEstimator(mockModel)
        estimator.startOutage(
            initPosEnu = doubleArrayOf(0.0, 0.0),
            initHeading = 0.0, // North
            initSpeed = 0.0,
            biasCalibration = EkfPositionEstimator.BiasCalibration(0.0, 0.0, 0.0, false, false)
        )

        // 30 samples of near-zero IMU signal (raw ZUPT mask would say "stationary"
        // from sample 8 onward) -- but the ML model insists the vehicle is moving.
        var lastSnapshot: EkfPositionEstimator.EkfSnapshot? = null
        for (i in 0 until 30) {
            val s = sample(accLinX = 0.01f, accLinY = 0.01f, gyroX = 0.001f, tNs = i * 100_000_000L)
            lastSnapshot = estimator.onSample(s, dtSeconds = 0.1)
        }

        // After the ML update kicks in (sample 10), velocity should be pulled
        // toward ~5 m/s and NOT get zeroed by the (raw-triggered) ZUPT mask.
        assertTrue(
            "Speed should reflect ML model's motion estimate, not be zeroed by ZUPT veto gate (was ${lastSnapshot?.speed})",
            (lastSnapshot?.speed ?: 0.0) > 1.0
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
}
