package com.sih26168.deadreckoning

import com.sih26168.deadreckoning.physics.PhysicsDeadReckoning
import com.sih26168.deadreckoning.test.TestWindow0Data
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class PhysicsDeadReckoningTest {

    @Test
    fun testWindow0PhysicsDeadReckoning() {
        println("================================================================================")
        println("STEP 3b UNIT TEST: PhysicsDeadReckoning Verification on Window 0 of S-Vw4")
        println("================================================================================")

        val pdr = PhysicsDeadReckoning()
        pdr.resetState(
            x = TestWindow0Data.INITIAL_X,
            y = TestWindow0Data.INITIAL_Y,
            heading = TestWindow0Data.INITIAL_HEADING_RAD,
            velocity = TestWindow0Data.INITIAL_VELOCITY
        )

        val result = pdr.processWindow(
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

        // Differences against Python ground truth
        val diffDs = abs(result.deltaS_imu - TestWindow0Data.GT_STANDALONE_DS_IMU)
        val diffVel = abs(result.newVelocity - TestWindow0Data.GT_NEW_VELOCITY)
        val diffHeadingDeg = abs(result.newHeadingDeg - TestWindow0Data.GT_NEW_HEADING_DEG)

        println("Python GT vs Kotlin PhysicsResult:")
        println("  deltaS_imu: Python = ${TestWindow0Data.GT_STANDALONE_DS_IMU} m | Kotlin = ${result.deltaS_imu} m | Diff = $diffDs m")
        println("  newVelocity: Python = ${TestWindow0Data.GT_NEW_VELOCITY} m/s | Kotlin = ${result.newVelocity} m/s | Diff = $diffVel m/s")
        println("  newHeading:  Python = ${TestWindow0Data.GT_NEW_HEADING_DEG} deg | Kotlin = ${result.newHeadingDeg} deg | Diff = $diffHeadingDeg deg")
        println("  Stationary samples: ${result.stationarySamples} / ${TestWindow0Data.N_SAMPLES}")
        println("  headingDir: ${result.headingDirDeg} deg (Python GT: ${TestWindow0Data.GT_HEADING_DIR_DEG} deg)")

        // Acceptance Criteria:
        // 1. deltaS_imu within 1% or within 0.1m
        val maxDsAllowed = maxOf(0.1f, TestWindow0Data.GT_STANDALONE_DS_IMU * 0.01f)
        assertTrue(
            "deltaS_imu error $diffDs exceeds allowed threshold $maxDsAllowed",
            diffDs <= maxDsAllowed
        )

        // 2. newHeading within 0.5 degrees
        assertTrue(
            "newHeading error $diffHeadingDeg deg exceeds allowed 0.5 deg",
            diffHeadingDeg <= 0.5f
        )

        // 3. newVelocity within 0.05 m/s
        assertTrue(
            "newVelocity error $diffVel m/s exceeds allowed 0.05 m/s",
            diffVel <= 0.05f
        )

        println(">>> ALL 3 ACCEPTANCE CRITERIA PASSED! <<<")
    }

    @Test
    fun testWindow0NominalConstantDt() {
        // Test with nominal constant dt (SEG_DUR_S / 90)
        val pdr = PhysicsDeadReckoning()
        pdr.resetState(
            x = TestWindow0Data.INITIAL_X,
            y = TestWindow0Data.INITIAL_Y,
            heading = TestWindow0Data.INITIAL_HEADING_RAD,
            velocity = TestWindow0Data.INITIAL_VELOCITY
        )

        val result = pdr.processWindow(
            accLinX = TestWindow0Data.acc_lin_x,
            accLinY = TestWindow0Data.acc_lin_y,
            accLinZ = TestWindow0Data.acc_lin_z,
            gyroX = TestWindow0Data.gyro_x,
            gyroY = TestWindow0Data.gyro_y,
            gyroZ = TestWindow0Data.gyro_z,
            accVehFwd = TestWindow0Data.acc_veh_fwd,
            gyroVehYawRate = TestWindow0Data.gyro_veh_yaw_rate,
            dtArr = null, // uses segDurS / (N-1)
            segDurS = TestWindow0Data.SEG_DUR_S
        )

        val diffDs = abs(result.deltaS_imu - TestWindow0Data.GT_STANDALONE_DS_IMU)
        val diffVel = abs(result.newVelocity - TestWindow0Data.GT_NEW_VELOCITY)
        val diffHeadingDeg = abs(result.newHeadingDeg - TestWindow0Data.GT_NEW_HEADING_DEG)

        println("Nominal Constant dt Results:")
        println("  deltaS_imu: ${result.deltaS_imu} m (diff: $diffDs m)")
        println("  newVelocity: ${result.newVelocity} m/s (diff: $diffVel m/s)")
        println("  newHeading: ${result.newHeadingDeg} deg (diff: $diffHeadingDeg deg)")

        assertTrue(diffDs <= 0.1f)
        assertTrue(diffHeadingDeg <= 0.5f)
        assertTrue(diffVel <= 0.05f)
    }
}
