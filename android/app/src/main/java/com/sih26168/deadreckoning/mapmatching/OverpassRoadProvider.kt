package com.sih26168.deadreckoning.mapmatching

import android.content.Context
import android.util.Log
import com.sih26168.deadreckoning.util.GeoProjection
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Provides and manages road network data from local bundled assets and Overpass API.
 *
 * Designed to ensure the demo NEVER fails:
 * 1. Offline First: Loads embedded road segments from assets (e.g. Duvvada test area).
 * 2. Asynchronous Live Fetching: Queries Overpass API in the background without blocking the UI or DR window loop.
 * 3. Session Bounding-Box Cache: Minimizes network usage by caching road geometry.
 * 4. Coordinate Alignment: Maintains ENU-projected segments matching MainActivity's origin (originLat, originLon).
 */
class OverpassRoadProvider(
    private val context: Context? = null
) {
    companion object {
        private const val TAG = "OverpassRoadProvider"
        private const val OVERPASS_URL_PRIMARY = "https://lz4.overpass-api.de/api/interpreter"
        private const val OVERPASS_URL_BACKUP = "https://overpass-api.de/api/interpreter"
    }

    // Internal representation of an OSM polyline way with geographic coordinates
    data class RawWay(
        val id: Long,
        val name: String?,
        val highway: String?,
        val coordinates: List<Pair<Double, Double>> // Pair(lat, lon)
    )

    private val rawWays = CopyOnWriteArrayList<RawWay>()
    private val cachedSegments = CopyOnWriteArrayList<RoadSegment>()

    private val executor = Executors.newSingleThreadExecutor()

    private var cachedCenterLat: Double? = null
    private var cachedCenterLon: Double? = null
    private var currentOriginLat: Double? = null
    private var currentOriginLon: Double? = null

    init {
        // Automatically load embedded roads if Context is available
        context?.let {
            loadEmbeddedRoads(it, "roads/duvvada_roads.json")
        }
    }

    /**
     * Loads pre-bundled OSM road data from Android assets.
     */
    fun loadEmbeddedRoads(ctx: Context, assetPath: String) {
        try {
            val jsonString = ctx.assets.open(assetPath).bufferedReader().use { it.readText() }
            val parsedWays = parseOsmJson(jsonString)
            rawWays.addAll(parsedWays)
            Log.i(TAG, "Loaded ${parsedWays.size} embedded road ways from asset '$assetPath'")
            reprojectCurrentOrigin()
        } catch (e: Exception) {
            Log.w(TAG, "Could not load embedded road asset '$assetPath': ${e.message}")
        }
    }

    /**
     * Updates the reference origin used to compute local ENU coordinates.
     */
    fun updateOrigin(originLat: Double, originLon: Double) {
        currentOriginLat = originLat
        currentOriginLon = originLon
        reprojectCurrentOrigin()
    }

    /**
     * Returns the active list of ENU-projected RoadSegments for map matching.
     */
    fun getActiveSegments(): List<RoadSegment> = cachedSegments

    /**
     * Reprojects all raw ways into local ENU coordinates relative to the current origin.
     */
    private fun reprojectCurrentOrigin() {
        val oLat = currentOriginLat ?: return
        val oLon = currentOriginLon ?: return

        val newSegments = mutableListOf<RoadSegment>()

        for (way in rawWays) {
            val coords = way.coordinates
            if (coords.size < 2) continue

            for (i in 0 until coords.size - 1) {
                val (lat1, lon1) = coords[i]
                val (lat2, lon2) = coords[i + 1]

                val (x1, y1) = GeoProjection.latLonToEnu(lat1, lon1, oLat, oLon)
                val (x2, y2) = GeoProjection.latLonToEnu(lat2, lon2, oLat, oLon)

                val dx = x2 - x1
                val dy = y2 - y1
                val len = sqrt(dx * dx + dy * dy)
                if (len < 0.5) continue // Skip micro-segments < 0.5m

                var heading = atan2(dx, dy)
                if (heading < 0.0) heading += 2.0 * Math.PI

                newSegments.add(
                    RoadSegment(
                        id = way.id,
                        name = way.name,
                        highwayType = way.highway,
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
                )
            }
        }

        cachedSegments.clear()
        cachedSegments.addAll(newSegments)
        Log.i(TAG, "Reprojected ${cachedSegments.size} road segments with origin ($oLat, $oLon)")
    }

    /**
     * Asynchronously queries Overpass API around (lat, lon) within radiusM (~300m).
     * If the query fails, network is unavailable, or times out, it gracefully retains existing roads.
     */
    fun fetchRoadsAroundAsync(
        lat: Double,
        lon: Double,
        radiusM: Double = 300.0,
        onComplete: ((Boolean) -> Unit)? = null
    ) {
        val lastCenterLat = cachedCenterLat
        val lastCenterLon = cachedCenterLon

        // Check if we are already within cached bounding box
        if (lastCenterLat != null && lastCenterLon != null) {
            val (dx, dy) = GeoProjection.latLonToEnu(lat, lon, lastCenterLat, lastCenterLon)
            val distFromCenter = sqrt(dx * dx + dy * dy)
            if (distFromCenter < (radiusM * 0.6)) {
                // Inside cached zone, no need to re-query
                onComplete?.invoke(true)
                return
            }
        }

        executor.execute {
            var success = false
            try {
                // Compute ~300m bounding box in degrees
                val dLat = (radiusM / GeoProjection.R_EARTH) * (180.0 / Math.PI)
                val dLon = (radiusM / (GeoProjection.R_EARTH * kotlin.math.cos(Math.toRadians(lat)))) * (180.0 / Math.PI)

                val south = lat - dLat
                val north = lat + dLat
                val west = lon - dLon
                val east = lon + dLon

                val query = """[out:json][timeout:8];
way["highway"]($south,$west,$north,$east);
(._;>;);
out body;
"""
                val jsonResponse = executeOverpassQuery(query)
                if (jsonResponse != null) {
                    val ways = parseOsmJson(jsonResponse)
                    if (ways.isNotEmpty()) {
                        // Merge ways avoiding duplicate IDs
                        val existingIds = rawWays.map { it.id }.toSet()
                        val newWays = ways.filter { it.id !in existingIds }
                        rawWays.addAll(newWays)
                        cachedCenterLat = lat
                        cachedCenterLon = lon
                        reprojectCurrentOrigin()
                        Log.i(TAG, "Successfully fetched ${newWays.size} new ways from Overpass")
                        success = true
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Overpass query failed (offline or timeout): ${e.message}")
            }
            onComplete?.invoke(success)
        }
    }

    /**
     * Executes an HTTP POST query against Overpass API with failover.
     */
    private fun executeOverpassQuery(query: String): String? {
        val endpoints = listOf(OVERPASS_URL_PRIMARY, OVERPASS_URL_BACKUP)
        for (endpoint in endpoints) {
            var conn: HttpURLConnection? = null
            try {
                val url = URL(endpoint)
                conn = url.openConnection() as HttpURLConnection
                conn.requestMethod = "POST"
                conn.doOutput = true
                conn.connectTimeout = 6000
                conn.readTimeout = 8000
                conn.setRequestProperty("User-Agent", "DeadReckoningApp/1.0 (SIH26168)")
                conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")

                val postData = "data=" + URLEncoder.encode(query, "UTF-8")
                conn.outputStream.use { os ->
                    os.write(postData.toByteArray(Charsets.UTF_8))
                    os.flush()
                }

                if (conn.responseCode == HttpURLConnection.HTTP_OK) {
                    val response = conn.inputStream.bufferedReader().use(BufferedReader::readText)
                    return response
                }
            } catch (e: Exception) {
                Log.w(TAG, "Endpoint $endpoint failed: ${e.message}")
            } finally {
                conn?.disconnect()
            }
        }
        return null
    }

    /**
     * Parses OSM JSON format containing node and way elements.
     */
    fun parseOsmJson(jsonString: String): List<RawWay> {
        val ways = mutableListOf<RawWay>()
        try {
            val root = JSONObject(jsonString)
            val elements = root.getJSONArray("elements")

            val nodeMap = mutableMapOf<Long, Pair<Double, Double>>()

            // First pass: collect nodes
            for (i in 0 until elements.length()) {
                val elem = elements.getJSONObject(i)
                if (elem.optString("type") == "node") {
                    val id = elem.getLong("id")
                    val lat = elem.getDouble("lat")
                    val lon = elem.getDouble("lon")
                    nodeMap[id] = Pair(lat, lon)
                }
            }

            // Second pass: collect ways
            for (i in 0 until elements.length()) {
                val elem = elements.getJSONObject(i)
                if (elem.optString("type") == "way") {
                    val id = elem.getLong("id")
                    val tags = elem.optJSONObject("tags")
                    val highway = tags?.optString("highway")?.takeIf { it.isNotBlank() }
                    val name = tags?.optString("name")?.takeIf { it.isNotBlank() }

                    val nodeIds = elem.optJSONArray("nodes") ?: continue
                    val wayCoords = mutableListOf<Pair<Double, Double>>()
                    for (j in 0 until nodeIds.length()) {
                        val nId = nodeIds.getLong(j)
                        val coord = nodeMap[nId]
                        if (coord != null) {
                            wayCoords.add(coord)
                        }
                    }

                    if (wayCoords.size >= 2) {
                        ways.add(
                            RawWay(
                                id = id,
                                name = name,
                                highway = highway,
                                coordinates = wayCoords
                            )
                        )
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse OSM JSON: ${e.message}", e)
        }
        return ways
    }
}
