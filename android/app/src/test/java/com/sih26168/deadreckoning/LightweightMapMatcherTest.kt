package com.sih26168.deadreckoning

import com.sih26168.deadreckoning.engine.PositionEstimator
import com.sih26168.deadreckoning.mapmatching.LightweightMapMatcher
import com.sih26168.deadreckoning.mapmatching.RoadSegment
import com.sih26168.deadreckoning.ml.ICorrectionModel
import com.sih26168.deadreckoning.util.GeoProjection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlin.math.atan2
import kotlin.math.sqrt

class LightweightMapMatcherTest {

    private val originLat = 17.718000
    private val originLon = 83.171000

    private lateinit var mapMatcher: LightweightMapMatcher

    // Helper to create an ENU-projected RoadSegment
    private fun createSegment(
        id: Long,
        name: String,
        lat1: Double,
        lon1: Double,
        lat2: Double,
        lon2: Double
    ): RoadSegment {
        val (x1, y1) = GeoProjection.latLonToEnu(lat1, lon1, originLat, originLon)
        val (x2, y2) = GeoProjection.latLonToEnu(lat2, lon2, originLat, originLon)
        val dx = x2 - x1
        val dy = y2 - y1
        val len = sqrt(dx * dx + dy * dy)
        var heading = atan2(dx, dy)
        if (heading < 0.0) heading += 2.0 * Math.PI
        return RoadSegment(
            id = id,
            name = name,
            highwayType = "residential",
            startLat = lat1,
            startLon = lon1,
            endLat = lat2,
            endLon = lon2,
            x1 = x1,
            y1 = y1,
            x2 = x2,
            y2 = y2,
            headingRad = heading,
            lengthM = len
        )
    }

    @Before
    fun setUp() {
        // Create a North-South road ("Lakeside Road") along lon 83.171000
        val northSouthRoad = createSegment(
            id = 101L,
            name = "Lakeside Road",
            lat1 = 17.717000,
            lon1 = 83.171000,
            lat2 = 17.720000,
            lon2 = 83.171000
        )

        // Create an East-West cross street ("Cross Street") along lat 17.718500
        val eastWestRoad = createSegment(
            id = 102L,
            name = "Cross Street",
            lat1 = 17.718500,
            lon1 = 83.170000,
            lat2 = 17.718500,
            lon2 = 83.173000
        )

        mapMatcher = LightweightMapMatcher(
            roadSegments = listOf(northSouthRoad, eastWestRoad),
            maxSnapDistanceM = 50.0,
            maxHeadingDiffRad = Math.toRadians(45.0)
        )
    }

    @Test
    fun testParallelRoadSnapping() {
        // Point is 15m East of Lakeside Road at lat 17.718500
        val (roadEast, roadNorth) = GeoProjection.latLonToEnu(17.718500, 83.171000, originLat, originLon)
        val rawEast = roadEast + 15.0
        val rawNorth = roadNorth
        val (rawLat, rawLon) = GeoProjection.enuToLatLon(rawEast, rawNorth, originLat, originLon)

        // Vehicle is driving North (heading 5° ~ 0.087 rad)
        val headingRad = Math.toRadians(5.0)

        val result = mapMatcher.snap(rawLat, rawLon, headingRad, originLat, originLon)

        assertTrue("Expected point to be snapped to Lakeside Road", result.isSnapped)
        assertEquals("Lakeside Road", result.matchedRoadName)
        assertEquals(15.0, result.snapDistanceM, 0.5)

        // Snapped coordinate should match road centerline (lon ~ 83.171000)
        assertEquals(83.171000, result.displayLon, 1e-5)
        assertEquals(17.718500, result.displayLat, 1e-5)
    }

    @Test
    fun testPerpendicularRoadRejection() {
        // Point is near the East-West road (which is perpendicular to vehicle heading)
        val (roadEast, roadNorth) = GeoProjection.latLonToEnu(17.718500, 83.172000, originLat, originLon)
        val rawEast = roadEast
        val rawNorth = roadNorth + 10.0 // 10m off
        val (rawLat, rawLon) = GeoProjection.enuToLatLon(rawEast, rawNorth, originLat, originLon)

        // Vehicle heading is North (0°), but this road runs East-West (90° -> diff 90° > 45°)
        val headingRad = Math.toRadians(0.0)

        val result = mapMatcher.snap(rawLat, rawLon, headingRad, originLat, originLon)

        // East-West road must be rejected due to heading mismatch!
        // (Lakeside road is > 50m away here, so no snap should occur)
        assertFalse("Cross street should be rejected due to >45° heading deviation", result.isSnapped)
        assertEquals(rawLat, result.displayLat, 1e-7)
        assertEquals(rawLon, result.displayLon, 1e-7)
    }

    @Test
    fun testDistanceToleranceRejection() {
        // Point is 65m West of Lakeside Road (> 50m tolerance)
        val (roadEast, roadNorth) = GeoProjection.latLonToEnu(17.718500, 83.171000, originLat, originLon)
        val rawEast = roadEast - 65.0
        val rawNorth = roadNorth
        val (rawLat, rawLon) = GeoProjection.enuToLatLon(rawEast, rawNorth, originLat, originLon)

        val headingRad = Math.toRadians(0.0)

        val result = mapMatcher.snap(rawLat, rawLon, headingRad, originLat, originLon)

        assertFalse("Point farther than 50m should not be snapped", result.isSnapped)
        assertEquals(rawLat, result.displayLat, 1e-7)
        assertEquals(rawLon, result.displayLon, 1e-7)
    }

    @Test
    fun testLakeObstacleAvoidance() {
        // Scenario: A lake is located on the West side of Lakeside Road (offset by -20m to -100m).
        // Raw AI-DR drifts 22m West directly into the lake.
        val (roadEast, roadNorth) = GeoProjection.latLonToEnu(17.718200, 83.171000, originLat, originLon)
        val lakeEast = roadEast - 22.0 // Inside the lake!
        val lakeNorth = roadNorth + 5.0
        val (lakeLat, lakeLon) = GeoProjection.enuToLatLon(lakeEast, lakeNorth, originLat, originLon)

        val headingRad = Math.toRadians(10.0) // Moving along road

        val result = mapMatcher.snap(lakeLat, lakeLon, headingRad, originLat, originLon)

        assertTrue("Expected lake-drifting point to snap back onto road", result.isSnapped)
        assertEquals("Lakeside Road", result.matchedRoadName)
        assertEquals(22.0, result.snapDistanceM, 1.0)

        // Snapped coordinate is pulled out of the lake and onto the road
        assertEquals(83.171000, result.displayLon, 1e-5)
    }

    @Test
    fun testOfflineEmptyRoadsFallback() {
        val emptyMatcher = LightweightMapMatcher(roadSegments = emptyList())
        val rawLat = 17.718500
        val rawLon = 83.171000

        val result = emptyMatcher.snap(rawLat, rawLon, 0.0, originLat, originLon)

        assertFalse(result.isSnapped)
        assertEquals(rawLat, result.displayLat, 1e-7)
        assertEquals(rawLon, result.displayLon, 1e-7)
    }

    @Test
    fun testMatchHeading_returnsRoadDirectionForCompatiblePoint() {
        // Same scenario as testParallelRoadSnapping: point 15m East of a
        // North-South road, vehicle heading North.
        val (roadEast, roadNorth) = GeoProjection.latLonToEnu(17.718500, 83.171000, originLat, originLon)
        val px = roadEast + 15.0
        val py = roadNorth
        val headingRad = Math.toRadians(5.0)

        val matched = mapMatcher.matchHeading(px, py, headingRad)

        org.junit.Assert.assertNotNull("Should match Lakeside Road's heading", matched)
        // Lakeside Road runs from lat 17.717 to 17.720 (due North) -> ~0 rad
        assertEquals(0.0, matched!!, Math.toRadians(2.0))
    }

    @Test
    fun testMatchHeading_returnsNull_whenPerpendicular() {
        val (roadEast, roadNorth) = GeoProjection.latLonToEnu(17.718500, 83.172000, originLat, originLon)
        val px = roadEast
        val py = roadNorth + 10.0
        val headingRad = Math.toRadians(0.0)

        assertEquals(null, mapMatcher.matchHeading(px, py, headingRad))
    }

    @Test
    fun testMatchHeading_returnsNull_whenNoRoadsLoaded() {
        val emptyMatcher = LightweightMapMatcher(roadSegments = emptyList())
        assertEquals(null, emptyMatcher.matchHeading(0.0, 0.0, 0.0))
    }

    @Test
    fun testPositionEstimatorInvariance() {
        // Verify Requirement 2d: Map matching does NOT touch or mutate PositionEstimator state
        val dummyModel = ICorrectionModel { 2.5f }
        val estimator = PositionEstimator(dummyModel)
        estimator.resetState(100.0f, 200.0f, 1.57f, 10.0f)

        val initialX = estimator.currentX
        val initialY = estimator.currentY
        val initialHeading = estimator.currentHeading
        val initialVelocity = estimator.currentVelocity

        // Perform map-matching snap on reconstructed coordinates
        val (lat, lon) = GeoProjection.enuToLatLon(initialX.toDouble(), initialY.toDouble(), originLat, originLon)
        val snapResult = mapMatcher.snap(lat, lon, initialHeading.toDouble(), originLat, originLon)
        org.junit.Assert.assertNotNull(snapResult)

        // Assert that PositionEstimator internal state is 100% UNTOUCHED
        assertEquals(initialX, estimator.currentX, 1e-6f)
        assertEquals(initialY, estimator.currentY, 1e-6f)
        assertEquals(initialHeading, estimator.currentHeading, 1e-6f)
        assertEquals(initialVelocity, estimator.currentVelocity, 1e-6f)
    }
}
