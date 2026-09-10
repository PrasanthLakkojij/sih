package com.sih26168.deadreckoning

import com.sih26168.deadreckoning.util.GpsDisplaySmoother
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GpsDisplaySmootherTest {

    @Test
    fun firstFix_setsAnchorToRawPosition() {
        val state = GpsDisplaySmoother.update(rawEast = 10.0, rawNorth = 20.0, accuracyM = 5.0, state = null)
        assertEquals(10.0, state.anchor.east, 1e-9)
        assertEquals(20.0, state.anchor.north, 1e-9)
    }

    @Test
    fun jitterWithinDeadband_holdsAnchorExactly() {
        val state0 = GpsDisplaySmoother.State(GpsDisplaySmoother.Point(0.0, 0.0))
        val state1 = GpsDisplaySmoother.update(rawEast = 1.0, rawNorth = 0.5, accuracyM = 5.0, state = state0)
        assertEquals(0.0, state1.anchor.east, 1e-9)
        assertEquals(0.0, state1.anchor.north, 1e-9)
    }

    @Test
    fun repeatedJitterFixes_neverAccumulateDrift() {
        var state = GpsDisplaySmoother.State(GpsDisplaySmoother.Point(0.0, 0.0))
        for (i in 0 until 200) {
            val jitterE = 1.0 + 1.5 * (i % 5).toDouble() / 5.0 // 1.0..1.9m from anchor at 0
            state = GpsDisplaySmoother.update(rawEast = jitterE, rawNorth = 0.0, accuracyM = 5.0, state = state)
        }
        assertEquals("Anchor must not drift under pure in-deadband jitter", 0.0, state.anchor.east, 1e-9)
    }

    @Test
    fun singleOutlierFix_doesNotMoveAnchor_thenClearsOnGoodFixReturning() {
        // Reproduces the reported "jumped crazy, then came back": one bad fix
        // (e.g. multipath spike) must not move the displayed marker at all --
        // it needs a second, agreeing fix before being trusted.
        var state = GpsDisplaySmoother.State(GpsDisplaySmoother.Point(0.0, 0.0))
        state = GpsDisplaySmoother.update(rawEast = 80.0, rawNorth = 0.0, accuracyM = 5.0, state = state) // spike
        assertEquals("A lone outlier must not move the anchor", 0.0, state.anchor.east, 1e-9)

        state = GpsDisplaySmoother.update(rawEast = 1.0, rawNorth = 0.0, accuracyM = 5.0, state = state) // back to normal
        assertEquals("Anchor should still be settled near the original position", 0.0, state.anchor.east, 1e-9)
    }

    @Test
    fun threeConsecutiveAgreeingFixes_stillNotEnough_underFourConfirmationRequirement() {
        // Reproduces the actual reported case: real device logcat showed raw
        // GPS drifting ~70m over 3 *consecutive, mutually-agreeing* fixes
        // (correlated multipath error, not one outlier) -- the old
        // REQUIRED_CONFIRMATIONS=2 committed to that drift. With 4 required,
        // 3 agreeing fixes must still hold the anchor.
        // Steps of 10m -- within the confirm radius (deadband 5m x 3 = 15m)
        // so each fix agrees with the last, same as real correlated drift.
        var state = GpsDisplaySmoother.State(GpsDisplaySmoother.Point(0.0, 0.0))
        state = GpsDisplaySmoother.update(rawEast = 25.0, rawNorth = 0.0, accuracyM = 5.0, state = state)
        state = GpsDisplaySmoother.update(rawEast = 35.0, rawNorth = 0.0, accuracyM = 5.0, state = state)
        state = GpsDisplaySmoother.update(rawEast = 45.0, rawNorth = 0.0, accuracyM = 5.0, state = state)
        assertEquals("3 agreeing fixes must not be enough to commit (was ${state.anchor.east})", 0.0, state.anchor.east, 1e-9)
    }

    @Test
    fun fourConsecutiveAgreeingFixes_confirmGenuineMove_committingToTheirAverage() {
        var state = GpsDisplaySmoother.State(GpsDisplaySmoother.Point(0.0, 0.0))
        state = GpsDisplaySmoother.update(rawEast = 25.0, rawNorth = 0.0, accuracyM = 5.0, state = state)
        state = GpsDisplaySmoother.update(rawEast = 35.0, rawNorth = 0.0, accuracyM = 5.0, state = state)
        state = GpsDisplaySmoother.update(rawEast = 45.0, rawNorth = 0.0, accuracyM = 5.0, state = state)
        state = GpsDisplaySmoother.update(rawEast = 55.0, rawNorth = 0.0, accuracyM = 5.0, state = state)
        // Commits to the average of the 4-fix streak, not just the last one.
        val expectedAvg = (25.0 + 35.0 + 45.0 + 55.0) / 4.0
        assertEquals("4th agreeing fix should confirm, committing to the streak average", expectedAvg, state.anchor.east, 1e-9)
    }

    @Test
    fun sustainedRealMotion_tracksContinuouslyWithFourFixLatency() {
        var state = GpsDisplaySmoother.State(GpsDisplaySmoother.Point(0.0, 0.0))
        // Each step is a genuine ~10m move in the same direction -- should
        // keep confirming and advancing, not get stuck, just with more delay
        // now (~4 fixes instead of ~2).
        for (i in 1..12) {
            state = GpsDisplaySmoother.update(rawEast = i * 10.0, rawNorth = 0.0, accuracyM = 5.0, state = state)
        }
        assertTrue("Should have advanced well into the real move (was ${state.anchor.east})", state.anchor.east > 50.0)
    }

    @Test
    fun poorlyReportedAccuracy_stillGetsMinimumDeadbandFloor() {
        val state0 = GpsDisplaySmoother.State(GpsDisplaySmoother.Point(0.0, 0.0))
        val state1 = GpsDisplaySmoother.update(rawEast = 2.0, rawNorth = 0.0, accuracyM = 0.5, state = state0)
        assertEquals("2m jitter should be held by the 4m floor even with a 0.5m reported accuracy", 0.0, state1.anchor.east, 1e-9)
    }
}
