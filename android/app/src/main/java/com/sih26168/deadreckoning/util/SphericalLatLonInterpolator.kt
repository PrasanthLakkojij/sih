package com.sih26168.deadreckoning.util

import com.google.android.gms.maps.model.LatLng
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * SphericalLatLonInterpolator:
 *
 * Implements great-circle spherical linear interpolation (slerp) between two LatLng points.
 * Used for smooth marker transitions during regular ~9-second updates and seamless resynchronization
 * when transitioning from simulated GNSS outage mode back to active GPS mode.
 */
object SphericalLatLonInterpolator {

    /**
     * Interpolates between [from] and [to] by [fraction] (0.0 .. 1.0) along the spherical geodesic.
     *
     * @param fraction Completion ratio between 0.0 (returns [from]) and 1.0 (returns [to]).
     * @param from Starting geographic point.
     * @param to Destination geographic point.
     * @return Interpolated LatLng on the spherical surface.
     */
    fun interpolate(fraction: Float, from: LatLng, to: LatLng): LatLng {
        if (fraction <= 0.0f) return from
        if (fraction >= 1.0f) return to

        val fromLat = Math.toRadians(from.latitude)
        val fromLon = Math.toRadians(from.longitude)
        val toLat = Math.toRadians(to.latitude)
        val toLon = Math.toRadians(to.longitude)

        val cosFromLat = cos(fromLat)
        val cosToLat = cos(toLat)

        val d = computeAngleBetween(fromLat, fromLon, toLat, toLon)
        if (d < 1e-6) {
            // Points are virtually identical; use linear approximation to avoid 0 division
            val lat = from.latitude + fraction * (to.latitude - from.latitude)
            val lon = from.longitude + fraction * (to.longitude - from.longitude)
            return LatLng(lat, lon)
        }

        val sinD = sin(d)
        val a = sin((1.0 - fraction) * d) / sinD
        val b = sin(fraction * d) / sinD

        // Convert spherical coords to 3D Cartesian coordinates
        val x = a * cosFromLat * cos(fromLon) + b * cosToLat * cos(toLon)
        val y = a * cosFromLat * sin(fromLon) + b * cosToLat * sin(toLon)
        val z = a * sin(fromLat) + b * sin(toLat)

        // Convert back to latitude / longitude in degrees
        val lat = atan2(z, sqrt(x * x + y * y))
        val lon = atan2(y, x)

        return LatLng(Math.toDegrees(lat), Math.toDegrees(lon))
    }

    private fun computeAngleBetween(fromLat: Double, fromLon: Double, toLat: Double, toLon: Double): Double {
        val dLat = fromLat - toLat
        val dLon = fromLon - toLon
        return 2.0 * asin(
            sqrt(
                sin(dLat / 2.0).pow(2) +
                        cos(fromLat) * cos(toLat) * sin(dLon / 2.0).pow(2)
            )
        )
    }
}
