package com.sih26168.deadreckoning.util

import kotlin.math.PI
import kotlin.math.cos

/**
 * Geographic projection utilities using the local tangent plane / equirectangular approximation.
 * Matches Phase 1-9 (phase6_outage_sim.py / phase4_orientation.py) with R_EARTH = 6371000.0 m.
 */
object GeoProjection {

    const val R_EARTH = 6371000.0 // Mean Earth radius in meters matching Python baseline

    /**
     * Converts WGS-84 latitude and longitude (in degrees) to local ENU (East, North in meters)
     * relative to a reference origin (lat0, lon0).
     *
     * @param lat Target latitude in degrees
     * @param lon Target longitude in degrees
     * @param lat0 Reference origin latitude in degrees
     * @param lon0 Reference origin longitude in degrees
     * @return Pair(east, north) in meters
     */
    fun latLonToEnu(
        lat: Double,
        lon: Double,
        lat0: Double,
        lon0: Double
    ): Pair<Double, Double> {
        val lat0Rad = lat0 * (PI / 180.0)
        val dLonRad = (lon - lon0) * (PI / 180.0)
        val dLatRad = (lat - lat0) * (PI / 180.0)

        val east = R_EARTH * dLonRad * cos(lat0Rad)
        val north = R_EARTH * dLatRad
        return Pair(east, north)
    }

    /**
     * Converts local ENU offsets (East, North in meters) back to WGS-84 latitude and longitude
     * relative to the reference origin (lat0, lon0).
     * Exact inverse of latLonToEnu.
     *
     * @param east Easting offset in meters
     * @param north Northing offset in meters
     * @param lat0 Reference origin latitude in degrees
     * @param lon0 Reference origin longitude in degrees
     * @return Pair(latitude, longitude) in degrees
     */
    fun enuToLatLon(
        east: Double,
        north: Double,
        lat0: Double,
        lon0: Double
    ): Pair<Double, Double> {
        val lat0Rad = lat0 * (PI / 180.0)
        val lat = lat0 + (north / R_EARTH) * (180.0 / PI)
        val lon = lon0 + (east / (R_EARTH * cos(lat0Rad))) * (180.0 / PI)
        return Pair(lat, lon)
    }

    /**
     * Float overload for converting PositionEstimator's x,y (Easting/Northing in meters).
     */
    fun enuToLatLon(
        east: Float,
        north: Float,
        lat0: Double,
        lon0: Double
    ): Pair<Double, Double> = enuToLatLon(east.toDouble(), north.toDouble(), lat0, lon0)
}
