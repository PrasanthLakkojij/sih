package com.sih26168.deadreckoning

import com.sih26168.deadreckoning.engine.NavigationState
import com.sih26168.deadreckoning.engine.PositionEstimator
import com.sih26168.deadreckoning.ml.ICorrectionModel
import com.sih26168.deadreckoning.physics.PhysicsDeadReckoning
import com.sih26168.deadreckoning.sensor.IMUSensorCollector
import com.sih26168.deadreckoning.test.SyntheticStationaryData
import com.sih26168.deadreckoning.test.TestWindow0Data
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class PositionEstimatorTest {

    // Step 1 estimate_position.py Python ground truth values for Window 0
    companion object {
        const val GT_X = -101899.566f
        const val GT_Y = 38157.767f
        const val GT_HEADING_RAD = 5.3622912f
        const val GT_HEADING_DEG = 307.23666f
        const val GT_VELOCITY = 0.0f
        const val GT_DS_IMU = 2.6890445f
        const val GT_DS_CORR = 106.241974f
        const val GT_DS_FINAL = 108.93102f
    }

    @Test
    fun testWindow0PositionEstimator() {
        println("================================================================================")
        println("STEP 3c UNIT TEST: PositionEstimator Verification on Window 0 of S-Vw4")
        println("================================================================================")

        // Mock correction model returning the exact C1 model prediction
        val mockModel = ICorrectionModel { features ->
            // Verify feature count
            assertEquals(58, features.size)
            GT_DS_CORR
        }

        val estimator = PositionEstimator(mockModel)
        estimator.resetState(
            x = TestWindow0Data.INITIAL_X,
            y = TestWindow0Data.INITIAL_Y,
            heading = TestWindow0Data.INITIAL_HEADING_RAD,
            velocity = TestWindow0Data.INITIAL_VELOCITY
        )

        val state: NavigationState = estimator.estimatePosition(
            accX = TestWindow0Data.acc_x,
            accY = TestWindow0Data.acc_y,
            accZ = TestWindow0Data.acc_z,
            accLinX = TestWindow0Data.acc_lin_x,
            accLinY = TestWindow0Data.acc_lin_y,
            accLinZ = TestWindow0Data.acc_lin_z,
            gyroX = TestWindow0Data.gyro_x,
            gyroY = TestWindow0Data.gyro_y,
            gyroZ = TestWindow0Data.gyro_z,
            accVehFwd = TestWindow0Data.acc_veh_fwd,
            gyroVehYawRate = TestWindow0Data.gyro_veh_yaw_rate,
            dtArr = TestWindow0Data.dt_sec,
            segDurS = TestWindow0Data.SEG_DUR_S
        )

        val diffX = abs(state.x - GT_X)
        val diffY = abs(state.y - GT_Y)
        val diffHeadingDeg = abs(state.headingDeg - GT_HEADING_DEG)
        val diffVelocity = abs(state.velocity - GT_VELOCITY)
        val diffDsImu = abs(state.deltaS_imu - GT_DS_IMU)
        val diffDsFinal = abs(state.deltaS_final - GT_DS_FINAL)

        println("Python Step 1 GT vs Kotlin PositionEstimator:")
        println("  X (Easting)  : Python = $GT_X m | Kotlin = ${state.x} m | Diff = $diffX m")
        println("  Y (Northing) : Python = $GT_Y m | Kotlin = ${state.y} m | Diff = $diffY m")
        println("  Heading      : Python = $GT_HEADING_DEG deg | Kotlin = ${state.headingDeg} deg | Diff = $diffHeadingDeg deg")
        println("  Velocity     : Python = $GT_VELOCITY m/s | Kotlin = ${state.velocity} m/s | Diff = $diffVelocity m/s")
        println("  deltaS_imu   : Python = $GT_DS_IMU m | Kotlin = ${state.deltaS_imu} m | Diff = $diffDsImu m")
        println("  deltaS_corr  : Python = $GT_DS_CORR m | Kotlin = ${state.deltaS_corr} m")
        println("  deltaS_final : Python = $GT_DS_FINAL m | Kotlin = ${state.deltaS_final} m | Diff = $diffDsFinal m")

        assertTrue("X error $diffX m exceeds tolerance 0.05m", diffX < 0.05f)
        assertTrue("Y error $diffY m exceeds tolerance 0.05m", diffY < 0.05f)
        assertTrue("Heading error $diffHeadingDeg deg exceeds tolerance 0.01 deg", diffHeadingDeg < 0.01f)
        assertTrue("Velocity error $diffVelocity m/s exceeds tolerance 0.01 m/s", diffVelocity < 0.01f)
        assertTrue("deltaS_imu error $diffDsImu m exceeds tolerance 0.01m", diffDsImu < 0.01f)
        assertTrue("deltaS_final error $diffDsFinal m exceeds tolerance 0.05m", diffDsFinal < 0.05f)

        println(">>> STEP 3c POSITION ESTIMATOR BIT-EXACT VERIFICATION PASSED! <<<")
    }

    @Test
    fun testStationaryWindowZeroDisplacement() {
        println("\n================================================================================")
        println("VERIFICATION: Stationary Window ZUPT Gating (Zero Drift & Correction Skipped)")
        println("================================================================================")

        var modelPredictCalled = false
        val mockModel = ICorrectionModel {
            modelPredictCalled = true
            55.0f // Should NOT be called when stationary
        }
        val estimator = PositionEstimator(mockModel)
        estimator.resetState(x = 0.0f, y = 0.0f, heading = 0.0f, velocity = 0.0f)

        val state = estimator.estimatePosition(
            accX = SyntheticStationaryData.acc_x,
            accY = SyntheticStationaryData.acc_y,
            accZ = SyntheticStationaryData.acc_z,
            accLinX = SyntheticStationaryData.acc_lin_x,
            accLinY = SyntheticStationaryData.acc_lin_y,
            accLinZ = SyntheticStationaryData.acc_lin_z,
            gyroX = SyntheticStationaryData.gyro_x,
            gyroY = SyntheticStationaryData.gyro_y,
            gyroZ = SyntheticStationaryData.gyro_z,
            accVehFwd = SyntheticStationaryData.acc_veh_fwd,
            gyroVehYawRate = SyntheticStationaryData.gyro_veh_yaw_rate,
            dtArr = SyntheticStationaryData.dt_sec,
            segDurS = SyntheticStationaryData.SEG_DUR_S
        )

        println("Stationary Window Results:")
        println("  isStationary  : ${state.isStationary}")
        println("  deltaS_imu    : ${state.deltaS_imu} m")
        println("  deltaS_corr   : ${state.deltaS_corr} m")
        println("  deltaS_final  : ${state.deltaS_final} m")
        println("  effectiveSpeed: ${state.effectiveSpeed * 3.6f} km/h")
        println("  Model called  : $modelPredictCalled")

        assertTrue("ZUPT should flag window as stationary", state.isStationary)
        assertFalse("ML model predict must NOT be called when stationary", modelPredictCalled)
        assertEquals(0.0f, state.deltaS_corr, 0.0f)
        assertTrue("deltaS_imu must be near zero (< 0.01m)", state.deltaS_imu < 0.01f)
        assertTrue("deltaS_final must be near zero (< 0.01m)", state.deltaS_final < 0.01f)
        assertEquals(0.0f, state.x, 0.01f)
        assertEquals(0.0f, state.y, 0.01f)
        assertEquals(0.0f, state.effectiveSpeed, 0.01f)
    }

    @Test
    fun testConsecutiveStationaryWindows() {
        println("\n================================================================================")
        println("VERIFICATION: 3 Consecutive Stationary Windows (Simulated Standing Still 27s)")
        println("================================================================================")

        var modelCallCount = 0
        val mockModel = ICorrectionModel {
            modelCallCount++
            12.6f
        }
        val estimator = PositionEstimator(mockModel)
        estimator.resetState(x = 0.0f, y = 0.0f, heading = 0.0f, velocity = 0.0f)

        val nSamples = 91
        val dtNs = 100_000_000L // 10Hz = 100ms
        var currentTimeNs = 1_000_000_000L

        for (w in 1..3) {
            val samples = mutableListOf<IMUSensorCollector.Sample>()
            for (i in 0 until nSamples) {
                samples.add(
                    IMUSensorCollector.Sample(
                        accX = 0.01f, accY = -0.01f, accZ = 9.81f,
                        accLinX = 0.01f, accLinY = -0.01f, accLinZ = 0.02f,
                        gyroX = 0.001f, gyroY = -0.001f, gyroZ = 0.002f,
                        accVehFwd = 0.005f, accVehLat = 0.0f, gyroVehYawRate = 0.001f,
                        magX = 0f, magY = 0f, magZ = 0f,
                        timestampNs = currentTimeNs
                    )
                )
                currentTimeNs += dtNs
            }

            val state = estimator.estimatePosition(samples)
            println("Stationary Window #$w (t=${w * 9}s): x=${String.format("%.2f", state.x)}m, y=${String.format("%.2f", state.y)}m, " +
                    "isStationary=${state.isStationary}, v=${String.format("%.2f", state.effectiveSpeed * 3.6f)}km/h, " +
                    "Δs_imu=${String.format("%.4f", state.deltaS_imu)}m, Δs_corr=${String.format("%.2f", state.deltaS_corr)}m, " +
                    "Δs_final=${String.format("%.4f", state.deltaS_final)}m")

            assertTrue("Window #$w must be flagged stationary", state.isStationary)
            assertEquals(0.0f, state.deltaS_corr, 0.0f)
            assertTrue("Window #$w Δs_final must be < 0.01m", state.deltaS_final < 0.01f)
            assertEquals(0.0f, state.x, 0.01f)
            assertEquals(0.0f, state.y, 0.01f)
            assertEquals(0.0f, state.effectiveSpeed, 0.01f)
        }

        assertEquals("Model predict should never be called across all stationary windows", 0, modelCallCount)
        println(">>> CONSECUTIVE STATIONARY TRACKING SIMULATION PASSED! ZERO DRIFT MAINTAINED! <<<")
    }

    @Test
    fun testConsecutiveMovingWindows() {
        println("\n================================================================================")
        println("STEP 3c SIMULATION: Consecutive Moving Windows (Simulated Driving/Moving)")
        println("================================================================================")

        // Mock model that predicts 12.6m displacement per 9-second segment
        val mockModel = ICorrectionModel { 12.6f }
        val estimator = PositionEstimator(mockModel)
        estimator.resetState(x = 0.0f, y = 0.0f, heading = 0.0f, velocity = 1.4f)

        val nSamples = 91
        val dtNs = 100_000_000L // 10Hz = 100ms
        var currentTimeNs = 1_000_000_000L

        for (w in 1..3) {
            val samples = mutableListOf<IMUSensorCollector.Sample>()
            for (i in 0 until nSamples) {
                samples.add(
                    IMUSensorCollector.Sample(
                        accX = 0f, accY = 0f, accZ = 9.81f,
                        accLinX = 6.0f, accLinY = 0.5f, accLinZ = 0.2f, // Linear acceleration above locked A_TH = 5.389 m/s^2
                        gyroX = 0.1f, gyroY = 0.1f, gyroZ = 0.0f,
                        accVehFwd = 0.5f, accVehLat = 0.0f, gyroVehYawRate = 0f, // Active forward acceleration
                        magX = 0f, magY = 0f, magZ = 0f,
                        timestampNs = currentTimeNs
                    )
                )
                currentTimeNs += dtNs
            }

            val state = estimator.estimatePosition(samples)
            println("Moving Window #$w (t=${w * 9}s): x=${String.format("%.2f", state.x)}m, y=${String.format("%.2f", state.y)}m, " +
                    "heading=${String.format("%.1f", state.headingDeg)}°, v=${String.format("%.2f", state.effectiveSpeed * 3.6f)}km/h, " +
                    "Δs_corr=${String.format("%.2f", state.deltaS_corr)}m, Δs_final=${String.format("%.2f", state.deltaS_final)}m")

            assertFalse("Moving window must NOT be stationary", state.isStationary)
            assertEquals(12.6f, state.deltaS_corr, 0.01f)
            assertTrue("Δs_final must include physics + ML correction (> 12.6m)", state.deltaS_final > 12.6f)
            assertTrue("Y coordinate must advance North", state.y > 0.0f)
        }
        println(">>> CONSECUTIVE MOVING TRACKING SIMULATION PASSED! <<<")
    }
}
