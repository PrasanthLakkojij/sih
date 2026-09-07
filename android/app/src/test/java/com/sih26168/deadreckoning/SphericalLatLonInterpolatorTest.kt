package com.sih26168.deadreckoning

import com.google.android.gms.maps.model.LatLng
import com.sih26168.deadreckoning.util.SphericalLatLonInterpolator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class SphericalLatLonInterpolatorTest {

    @Test
    fun testFractionBoundaries() {
        val p1 = LatLng(12.971598, 77.594562)
        val p2 = LatLng(12.972598, 77.595562)

        val at0 = SphericalLatLonInterpolator.interpolate(0.0f, p1, p2)
        assertEquals(p1.latitude, at0.latitude, 1e-6)
        assertEquals(p1.longitude, at0.longitude, 1e-6)

        val at1 = SphericalLatLonInterpolator.interpolate(1.0f, p1, p2)
        assertEquals(p2.latitude, at1.latitude, 1e-6)
        assertEquals(p2.longitude, at1.longitude, 1e-6)
    }

    @Test
    fun testMidpointInterpolation() {
        val p1 = LatLng(12.970000, 77.590000)
        val p2 = LatLng(12.972000, 77.592000)

        val mid = SphericalLatLonInterpolator.interpolate(0.5f, p1, p2)
        val expectedLat = 12.971000
        val expectedLon = 77.591000

        assertTrue("Midpoint latitude should be close to 12.971000", abs(mid.latitude - expectedLat) < 1e-4)
        assertTrue("Midpoint longitude should be close to 77.591000", abs(mid.longitude - expectedLon) < 1e-4)
    }

    @Test
    fun testMonotonicity() {
        val p1 = LatLng(12.970000, 77.590000)
        val p2 = LatLng(12.980000, 77.600000)

        var prevLat = p1.latitude
        for (i in 1..10) {
            val frac = i / 10.0f
            val pt = SphericalLatLonInterpolator.interpolate(frac, p1, p2)
            assertTrue("Latitude must monotonically increase", pt.latitude > prevLat)
            prevLat = pt.latitude
        }
    }
}
