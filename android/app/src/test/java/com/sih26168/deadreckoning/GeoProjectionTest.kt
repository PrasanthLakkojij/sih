package com.sih26168.deadreckoning

import com.sih26168.deadreckoning.util.GeoProjection
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.abs

class GeoProjectionTest {

    @Test
    fun testRoundTripCoordinateConversion() {
        val lat0 = 48.1351 // Munich reference origin
        val lon0 = 11.5820

        val testLat = 48.1450
        val testLon = 11.5950

        val (east, north) = GeoProjection.latLonToEnu(testLat, testLon, lat0, lon0)
        val (recLat, recLon) = GeoProjection.enuToLatLon(east, north, lat0, lon0)

        val diffLat = abs(testLat - recLat)
        val diffLon = abs(testLon - recLon)

        println("GeoProjection Round-Trip Test:")
        println("  Original:   ($testLat, $testLon)")
        println("  Local ENU:  East=${east}m, North=${north}m")
        println("  Recovered:  ($recLat, $recLon)")
        println("  Diff Lat:   $diffLat deg")
        println("  Diff Lon:   $diffLon deg")

        assertEquals(testLat, recLat, 1e-9)
        assertEquals(testLon, recLon, 1e-9)
    }

    @Test
    fun testPositionEstimatorOffsetConversion() {
        // Test typical walk offsets: 100m East, 50m North from Bangalore ISRO HQ coords (13.033, 77.564)
        val lat0 = 13.0333
        val lon0 = 77.5644

        val eastOffset = 100.0f
        val northOffset = 50.0f

        val (lat, lon) = GeoProjection.enuToLatLon(eastOffset, northOffset, lat0, lon0)
        val (recEast, recNorth) = GeoProjection.latLonToEnu(lat, lon, lat0, lon0)

        assertEquals(eastOffset.toDouble(), recEast, 1e-4)
        assertEquals(northOffset.toDouble(), recNorth, 1e-4)
    }
}
