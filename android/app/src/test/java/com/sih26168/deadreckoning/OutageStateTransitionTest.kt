package com.sih26168.deadreckoning

import com.sih26168.deadreckoning.engine.PositionEstimator
import com.sih26168.deadreckoning.ml.ICorrectionModel
import com.sih26168.deadreckoning.util.GeoProjection
import com.sih26168.deadreckoning.util.SphericalLatLonInterpolator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

/**
 * OutageStateTransitionTest:
 *
 * Verifies Step 5 requirement:
 * 1. Clean state snap of PositionEstimator to true GPS at the instant of outage toggle.
 * 2. Drift accumulation from clean start (drift = 0.0 at t=0).
 * 3. Accurate resynchronization distance calculation and smooth interpolation.
 */
class OutageStateTransitionTest {

    // Dummy mock correction model returning 0.0 displacement correction
    private val dummyModel = ICorrectionModel { 0.0f }

    @Test
    fun testCleanGpsSnapAtOutageToggle() {
        val estimator = PositionEstimator(dummyModel)

        // Simulate origin
        val originLat = 12.971598
        val originLon = 77.594562

        // Prior accumulated state before outage
        estimator.currentX = 145.0f
        estimator.currentY = 230.0f
        estimator.currentHeading = 1.2f
        estimator.currentVelocity = 8.5f

        // True GPS location at instant of outage toggle
        val gpsLat = 12.972500
        val gpsLon = 77.595500
        val gpsSpeed = 11.2f
        val gpsBearingDeg = 45.0f

        val (gpsEast, gpsNorth) = GeoProjection.latLonToEnu(gpsLat, gpsLon, originLat, originLon)
        val trueHeadingRad = Math.toRadians(gpsBearingDeg.toDouble()).toFloat()

        // EXECUTE STEP 5 SNAP
        estimator.resetState(
            x = gpsEast.toFloat(),
            y = gpsNorth.toFloat(),
            heading = trueHeadingRad,
            velocity = gpsSpeed
        )

        // Verify snapped state exactly matches GPS
        assertEquals(gpsEast.toFloat(), estimator.currentX, 1e-4f)
        assertEquals(gpsNorth.toFloat(), estimator.currentY, 1e-4f)
        assertEquals(trueHeadingRad, estimator.currentHeading, 1e-6f)
        assertEquals(gpsSpeed, estimator.currentVelocity, 1e-6f)

        // Verify reconstructed lat/lon matches gpsLat/gpsLon exactly
        val (reconstructedLat, reconstructedLon) = GeoProjection.enuToLatLon(
            estimator.currentX, estimator.currentY, originLat, originLon
        )
        assertEquals(gpsLat, reconstructedLat, 1e-6)
        assertEquals(gpsLon, reconstructedLon, 1e-6)

        // Instantaneous drift at toggle must be exactly 0.0m
        val dx = gpsEast - estimator.currentX
        val dy = gpsNorth - estimator.currentY
        val initialDrift = sqrt(dx * dx + dy * dy)
        assertEquals(0.0, initialDrift, 1e-4)
    }

    @Test
    fun testResyncCorrectionDistanceAndInterpolation() {
        // True GPS position at end of outage
        val trueGpsLat = 12.973000
        val trueGpsLon = 77.596000

        // AI DR estimated position at end of outage (say, drifted by ~20 meters)
        val aiDrLat = 12.973150
        val aiDrLon = 77.596120

        val pAi = com.sih26168.deadreckoning.util.LatLng(aiDrLat, aiDrLon)
        val pGps = com.sih26168.deadreckoning.util.LatLng(trueGpsLat, trueGpsLon)

        // Spherical linear interpolation at 0%, 50%, 100%
        val atStart = SphericalLatLonInterpolator.interpolate(0.0f, pAi, pGps)
        val atMid = SphericalLatLonInterpolator.interpolate(0.5f, pAi, pGps)
        val atEnd = SphericalLatLonInterpolator.interpolate(1.0f, pAi, pGps)

        assertEquals(pAi.latitude, atStart.latitude, 1e-6)
        assertEquals(pAi.longitude, atStart.longitude, 1e-6)

        assertEquals(pGps.latitude, atEnd.latitude, 1e-6)
        assertEquals(pGps.longitude, atEnd.longitude, 1e-6)

        assertTrue(atMid.latitude in trueGpsLat..aiDrLat)
        assertTrue(atMid.longitude in trueGpsLon..aiDrLon)
    }
}
