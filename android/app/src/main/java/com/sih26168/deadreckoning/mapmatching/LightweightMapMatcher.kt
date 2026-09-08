package com.sih26168.deadreckoning.mapmatching

import com.sih26168.deadreckoning.util.GeoProjection
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.min
import kotlin.math.sqrt

/**
 * LightweightMapMatcher:
 *
 * Demo-quality display-only map matching layer for the AI-DR trajectory during GPS outage mode.
 * Snaps the displayed AI-DR coordinates onto the nearest heading-compatible road segment.
 *
 * Core Guarantees:
 * 1. DISPLAY ONLY: Does not modify PositionEstimator internal state or benchmark drift calculations.
 * 2. HEADING GATED: Only considers candidate road segments within 45° of current vehicle heading,
 *    preventing spurious snapping onto perpendicular cross-streets.
 * 3. DISTANCE TOLERANCE: Rejects candidates farther than 50m to avoid forced snaps on distant roads.
 * 4. FAIL-SAFE: Gracefully falls back to raw coordinates if no compatible road is nearby or if
 *    road data is unavailable.
 */
class LightweightMapMatcher(
    var roadSegments: List<RoadSegment> = emptyList(),
    val maxSnapDistanceM: Double = 50.0,
    val maxHeadingDiffRad: Double = Math.toRadians(45.0)
) {
    companion object {
        private const val TWO_PI = 2.0 * PI

        /**
         * Normalizes an angle in radians to [0, 2*PI).
         */
        fun normalizeAngleRad(rad: Double): Double {
            var a = rad % TWO_PI
            if (a < 0.0) a += TWO_PI
            return a
        }

        /**
         * Computes the minimal angular deviation between a heading and a bidirectional road segment.
         * Accounts for travel in both directions of the segment.
         */
        fun computeBidirectionalAngleDiff(vehicleHeadingRad: Double, roadHeadingRad: Double): Double {
            val vH = normalizeAngleRad(vehicleHeadingRad)
            val rH = normalizeAngleRad(roadHeadingRad)

            // Forward direction difference
            val diff1 = abs(vH - rH)
            val angle1 = min(diff1, TWO_PI - diff1)

            // Reverse direction difference
            val rHRev = normalizeAngleRad(rH + PI)
            val diff2 = abs(vH - rHRev)
            val angle2 = min(diff2, TWO_PI - diff2)

            return min(angle1, angle2)
        }
    }

    /**
     * Evaluates candidate road segments and snaps the raw AI-DR coordinate.
     *
     * @param rawLat Raw estimated latitude from PositionEstimator in degrees
     * @param rawLon Raw estimated longitude from PositionEstimator in degrees
     * @param headingRad Estimated vehicle heading in radians (0=North, PI/2=East, clockwise)
     * @param originLat Reference origin latitude in degrees
     * @param originLon Reference origin longitude in degrees
     * @return SnapResult containing display coordinates (snapped or raw) and metadata
     */
    fun snap(
        rawLat: Double,
        rawLon: Double,
        headingRad: Double,
        originLat: Double,
        originLon: Double
    ): SnapResult {
        val segments = roadSegments
        if (segments.isEmpty()) {
            return SnapResult(
                displayLat = rawLat,
                displayLon = rawLon,
                isSnapped = false,
                snapDistanceM = 0.0,
                headingDiffDeg = 0.0,
                matchedRoadName = null
            )
        }

        // Convert raw point to local ENU meters
        val (px, py) = GeoProjection.latLonToEnu(rawLat, rawLon, originLat, originLon)

        var bestSegment: RoadSegment? = null
        var bestSnappedX = px
        var bestSnappedY = py
        var minScore = Double.MAX_VALUE
        var bestDistance = 0.0
        var bestAngleDiff = 0.0

        for (seg in segments) {
            val dx = seg.x2 - seg.x1
            val dy = seg.y2 - seg.y1
            val lenSq = dx * dx + dy * dy
            if (lenSq < 1e-4) continue

            // 1. Check heading compatibility first (within 45°)
            val segHeading = atan2(dx, dy) // Azimuth from North in radians
            val angleDiff = computeBidirectionalAngleDiff(headingRad, segHeading)
            if (angleDiff > maxHeadingDiffRad) {
                continue // Perpendicular or misaligned road segment
            }

            // 2. Orthogonal projection onto segment, clamped to [0, 1]
            val t = ((px - seg.x1) * dx + (py - seg.y1) * dy) / lenSq
            val clampedT = t.coerceIn(0.0, 1.0)
            val qx = seg.x1 + clampedT * dx
            val qy = seg.y1 + clampedT * dy

            // 3. Perpendicular distance to projected point
            val dist = sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy))
            if (dist > maxSnapDistanceM) {
                continue // Beyond distance tolerance
            }

            // 4. Combined score prioritizing proximity and heading alignment
            val score = dist * (1.0 + (angleDiff / maxHeadingDiffRad))
            if (score < minScore) {
                minScore = score
                bestSegment = seg
                bestSnappedX = qx
                bestSnappedY = qy
                bestDistance = dist
                bestAngleDiff = angleDiff
            }
        }

        return if (bestSegment != null) {
            val (snappedLat, snappedLon) = GeoProjection.enuToLatLon(bestSnappedX, bestSnappedY, originLat, originLon)
                val roadLabel = bestSegment.name?.takeIf { it.isNotBlank() }
                    ?: bestSegment.highwayType?.takeIf { it.isNotBlank() }?.let { "${it.replaceFirstChar { c -> c.uppercase() }} Road" }
                    ?: "Road #${bestSegment.id}"

                SnapResult(
                    displayLat = snappedLat,
                    displayLon = snappedLon,
                    isSnapped = true,
                    snapDistanceM = bestDistance,
                    headingDiffDeg = Math.toDegrees(bestAngleDiff),
                    matchedRoadName = roadLabel
                )
        } else {
            SnapResult(
                displayLat = rawLat,
                displayLon = rawLon,
                isSnapped = false,
                snapDistanceM = 0.0,
                headingDiffDeg = 0.0,
                matchedRoadName = null
            )
        }
    }
}
