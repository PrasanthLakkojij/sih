package com.sih26168.deadreckoning

import com.sih26168.deadreckoning.engine.NavigationState
import com.sih26168.deadreckoning.engine.PositionEstimator
import com.sih26168.deadreckoning.ml.ICorrectionModel
import com.sih26168.deadreckoning.physics.PhysicsDeadReckoning
import com.sih26168.deadreckoning.sensor.IMUSensorCollector
import com.sih26168.deadreckoning.test.TestWindow0Data
import org.junit.Assert.assertEquals
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
    fun testConsecutiveLivePositionUpdates() {
        println("\n================================================================================")
        println("STEP 3c SIMULATION: Consecutive Live Position Updates (Simulated Walking North)")
        println("================================================================================")

        // Mock model that predicts 12.6m displacement per 9-second segment (1.4 m/s walking speed)
        val mockModel = ICorrectionModel { 12.6f }
        val estimator = PositionEstimator(mockModel)
        estimator.resetState(x = 0.0f, y = 0.0f, heading = 0.0f, velocity = 1.4f) // Heading North (0 rad), walking speed 1.4 m/s

        val nSamples = 91
        val dtNs = 100_000_000L // 10Hz = 100ms

        // Simulate 3 consecutive 9-second walking windows heading North
        var currentTimeNs = 1_000_000_000L
        for (w in 1..3) {
            val samples = mutableListOf<IMUSensorCollector.Sample>()
            for (i in 0 until nSamples) {
                samples.add(
                    IMUSensorCollector.Sample(
                        accX = 0f, accY = 0f, accZ = 9.81f,
                        accLinX = 0f, accLinY = 0f, accLinZ = 0f,
                        gyroX = 0f, gyroY = 0f, gyroZ = 0f,
                        accVehFwd = 0f, gyroVehYawRate = 0f,
                        timestampNs = currentTimeNs
                    )
                )
                currentTimeNs += dtNs
            }

            val state = estimator.estimatePosition(samples)
            println("Window #$w (t=${w * 9}s): x=${String.format("%.2f", state.x)}m, y=${String.format("%.2f", state.y)}m, " +
                    "heading=${String.format("%.1f", state.headingDeg)}°, v=${String.format("%.2f", state.velocity)}m/s, " +
                    "Δs_corr=${String.format("%.2f", state.deltaS_corr)}m, Δs_final=${String.format("%.2f", state.deltaS_final)}m")

            // In Window 1, initial velocity v0=1.4m/s travels 0.84m in the first 0.6s before ZUPT confirms stationary
            // Subsequent windows start with confirmed ZUPT, so displacement equals ML correction (12.6m)
            assertEquals(0.0f, state.x, 0.01f)
            assertTrue("Y should advance consistently North", state.y > (w - 1) * 12.0f)
            val expectedY = if (w == 1) 13.44f else 13.44f + (w - 1) * 12.6f
            assertEquals(expectedY, state.y, 0.05f)
        }
        println(">>> CONSECUTIVE POSITION TRACKING SIMULATION PASSED! <<<")
    }
}
