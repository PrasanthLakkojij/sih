package com.sih26168.deadreckoning.mapmatching

/**
 * Represents a directed or bidirectional road segment in geographic and local ENU space.
 *
 * @param id Unique identifier of the OSM way or segment
 * @param name Road name if available from OSM tags (e.g. "Main Street", "Duvvada Road")
 * @param highwayType OSM highway tag value (e.g. "residential", "primary", "service")
 * @param startLat Latitude of start node in degrees
 * @param startLon Longitude of start node in degrees
 * @param endLat Latitude of end node in degrees
 * @param endLon Longitude of end node in degrees
 * @param x1 Local Easting coordinate of start node in meters
 * @param y1 Local Northing coordinate of start node in meters
 * @param x2 Local Easting coordinate of end node in meters
 * @param y2 Local Northing coordinate of end node in meters
 * @param headingRad Azimuth of the road segment vector in radians [0, 2*PI), clockwise from North
 * @param lengthM Length of the road segment in meters
 */
data class RoadSegment(
    val id: Long,
    val name: String? = null,
    val highwayType: String? = null,
    val startLat: Double,
    val startLon: Double,
    val endLat: Double,
    val endLon: Double,
    val x1: Double,
    val y1: Double,
    val x2: Double,
    val y2: Double,
    val headingRad: Double,
    val lengthM: Double
)

/**
 * Result of a map-matching snapping evaluation.
 *
 * @param displayLat Latitude to render on the map in degrees
 * @param displayLon Longitude to render on the map in degrees
 * @param isSnapped True if point was snapped to a candidate road; false if raw point was preserved
 * @param snapDistanceM Distance in meters from the raw point to the snapped road point
 * @param headingDiffDeg Angular deviation in degrees between vehicle heading and road segment orientation
 * @param matchedRoadName Name or type of the matched road, or null if unsnapped
 */
data class SnapResult(
    val displayLat: Double,
    val displayLon: Double,
    val isSnapped: Boolean,
    val snapDistanceM: Double,
    val headingDiffDeg: Double = 0.0,
    val matchedRoadName: String? = null
)
