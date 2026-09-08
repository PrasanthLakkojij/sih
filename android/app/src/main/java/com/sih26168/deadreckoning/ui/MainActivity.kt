package com.sih26168.deadreckoning.ui

import android.Manifest
import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.Drawable
import android.location.Location
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.view.animation.LinearInterpolator
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.material.button.MaterialButton
import com.sih26168.deadreckoning.R
import com.sih26168.deadreckoning.engine.NavigationState
import com.sih26168.deadreckoning.engine.PositionEstimator
import com.sih26168.deadreckoning.ml.CorrectionModel
import com.sih26168.deadreckoning.sensor.IMUSensorCollector
import com.sih26168.deadreckoning.test.OnnxVerificationActivity
import com.sih26168.deadreckoning.mapmatching.LightweightMapMatcher
import com.sih26168.deadreckoning.mapmatching.OverpassRoadProvider
import com.sih26168.deadreckoning.util.GeoProjection
import com.sih26168.deadreckoning.util.LatLng
import com.sih26168.deadreckoning.util.SphericalLatLonInterpolator
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import kotlin.math.sqrt

/**
 * MainActivity: Live Dual-Marker Dead Reckoning Map Activity powered by osmdroid (OpenStreetMap).
 *
 * Replaces Google Maps SDK completely (no API keys, no billing requirements).
 *
 * Core Features:
 *  1. Dual-marker tracking:
 *     - Blue marker / GPS polyline: Genuine GNSS fix from phone GPS.
 *     - Orange marker / AI polyline: Physics + AI ML dead reckoning from PositionEstimator.
 *  2. Simulated GPS Loss Toggle:
 *     - Snaps PositionEstimator state to true GPS at instant of toggle.
 *     - AI-DR becomes the authoritative primary position displayed to the user.
 *     - Real GPS is tracked faintly in background as ground truth comparison.
 *     - HUD tracks live outage duration, accumulated distance, and drift.
 *  3. Resync to GPS Active:
 *     - Smooth spherical interpolation (1.8s) transitions marker without teleportation.
 *     - Logs correction distance and outage metrics to SIH_POSITION_TEST.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG_POSITION = "SIH_POSITION_TEST"
        private const val LOCATION_PERMISSION_REQ_CODE = 1001
        private const val DEFAULT_ZOOM = 18.5
    }

    private lateinit var mapView: MapView
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback

    // Dead Reckoning Engine & Sensor Collector
    private lateinit var correctionModel: CorrectionModel
    private lateinit var positionEstimator: PositionEstimator
    private lateinit var sensorCollector: IMUSensorCollector

    // Reference Origin Point for ENU <-> LatLon conversion
    private var originLat: Double? = null
    private var originLon: Double? = null
    private var lastGpsLocation: Location? = null
    private var lastAiState: NavigationState? = null
    private var totalDistanceTraveledM: Float = 0.0f
    private var windowCount = 0

    // Outage Simulation State
    private var isGpsOutageMode = false
    private var outageStartTimeMs: Long = 0L
    private var outageDistanceTraveledM: Float = 0.0f
    private val uiHandler = Handler(Looper.getMainLooper())
    private var outageTimerRunnable: Runnable? = null

    // osmdroid Markers & Polylines
    private var gpsMarker: Marker? = null
    private var aiMarker: Marker? = null
    private var gpsPolyline: Polyline? = null
    private var aiPolyline: Polyline? = null

    // Lightweight Map-Matching Layer (Display-only, isolated & revertable)
    private var isMapMatchingEnabled = true
    private lateinit var roadProvider: OverpassRoadProvider
    private lateinit var mapMatcher: LightweightMapMatcher

    // UI Views
    private lateinit var tvGpsStatusBadge: TextView
    private lateinit var llOutageBanner: LinearLayout
    private lateinit var tvOutageBannerText: TextView
    private lateinit var tvOutageTimer: TextView
    private lateinit var tvGpsCoords: TextView
    private lateinit var tvGpsSpeed: TextView
    private lateinit var tvAiCoords: TextView
    private lateinit var tvAiMotion: TextView
    private lateinit var tvDriftDistance: TextView
    private lateinit var tvWindowCount: TextView
    private lateinit var btnToggleGpsOutage: MaterialButton
    private lateinit var btnResetOrigin: Button
    private lateinit var btnVerifyTests: Button
    private lateinit var llGpsColumn: LinearLayout
    private lateinit var llAiColumn: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1. Initialize osmdroid configuration and user-agent BEFORE loading layout
        Configuration.getInstance().load(
            applicationContext,
            getSharedPreferences("osmdroid_prefs", MODE_PRIVATE)
        )
        Configuration.getInstance().userAgentValue = packageName

        setContentView(R.layout.activity_main)

        // 2. Bind UI Views
        tvGpsStatusBadge = findViewById(R.id.tvGpsStatusBadge)
        llOutageBanner = findViewById(R.id.llOutageBanner)
        tvOutageBannerText = findViewById(R.id.tvOutageBannerText)
        tvOutageTimer = findViewById(R.id.tvOutageTimer)
        tvGpsCoords = findViewById(R.id.tvGpsCoords)
        tvGpsSpeed = findViewById(R.id.tvGpsSpeed)
        tvAiCoords = findViewById(R.id.tvAiCoords)
        tvAiMotion = findViewById(R.id.tvAiMotion)
        tvDriftDistance = findViewById(R.id.tvDriftDistance)
        tvWindowCount = findViewById(R.id.tvWindowCount)
        btnToggleGpsOutage = findViewById(R.id.btnToggleGpsOutage)
        btnResetOrigin = findViewById(R.id.btnResetOrigin)
        btnVerifyTests = findViewById(R.id.btnVerifyTests)
        llGpsColumn = findViewById(R.id.llGpsColumn)
        llAiColumn = findViewById(R.id.llAiColumn)

        // Initialize HUD to clean GPS-active standby state
        llAiColumn.alpha = 0.6f
        tvAiCoords.text = "Standby (GPS Active)"
        tvAiMotion.text = "Standby (Synced to GPS)"
        tvDriftDistance.text = "Drift: 0.0 m (GPS Active)"

        btnToggleGpsOutage.setOnClickListener {
            toggleGpsOutageMode()
        }

        btnResetOrigin.setOnClickListener {
            resetOriginToCurrentLocation()
        }

        btnVerifyTests.setOnClickListener {
            startActivity(Intent(this, OnnxVerificationActivity::class.java))
        }

        // 3. Setup osmdroid MapView
        setupOsmMapView()

        // 4. Initialize ONNX Model & Position Estimator
        correctionModel = CorrectionModel(this)
        positionEstimator = PositionEstimator(correctionModel)

        // 5. Initialize IMU Sensor Collector (10Hz target rate, SENSOR_DELAY_GAME)
        sensorCollector = IMUSensorCollector(
            context = this,
            onWindowSamplesReady = { samples ->
                onNewImuWindowReceived(samples)
            }
        )

        // 6. Initialize GPS Location Services
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        setupLocationCallback()
        checkLocationPermissionsAndStart()

        // 7. Initialize Lightweight Map-Matching Layer (Display only)
        roadProvider = OverpassRoadProvider(this)
        mapMatcher = LightweightMapMatcher(roadSegments = roadProvider.getActiveSegments())
    }

    private fun setupOsmMapView() {
        mapView = findViewById(R.id.mapView)
        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(DEFAULT_ZOOM)

        // Initialize Polylines
        val gpsLine = Polyline(mapView).apply {
            outlinePaint.color = Color.parseColor("#1E88E5") // Blue for GPS
            outlinePaint.strokeWidth = 8f
            outlinePaint.strokeCap = Paint.Cap.ROUND
        }
        gpsPolyline = gpsLine
        mapView.overlays.add(gpsLine)

        val aiLine = Polyline(mapView).apply {
            outlinePaint.color = Color.parseColor("#FF5722") // Deep Orange for AI Dead Reckoning
            outlinePaint.strokeWidth = 8f
            outlinePaint.strokeCap = Paint.Cap.ROUND
            isEnabled = false // Hidden during GPS-active mode; only shown during simulated outage
        }
        aiPolyline = aiLine
        mapView.overlays.add(aiLine)

        // Initialize Markers
        val gpsMark = Marker(mapView).apply {
            title = "Real GPS Location"
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
            icon = createMarkerDrawable(fillColor = Color.parseColor("#1E88E5"), strokeColor = Color.WHITE)
        }
        gpsMarker = gpsMark
        mapView.overlays.add(gpsMark)

        val aiMark = Marker(mapView).apply {
            title = "Physics + AI Estimated"
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
            icon = createMarkerDrawable(fillColor = Color.parseColor("#FF5722"), strokeColor = Color.WHITE)
            alpha = 0.0f
            isEnabled = false // Hidden during GPS-active mode; only shown during simulated outage
        }
        aiMarker = aiMark
        mapView.overlays.add(aiMark)
    }

    private fun createMarkerDrawable(fillColor: Int, strokeColor: Int, sizeDp: Int = 22): Drawable {
        val density = resources.displayMetrics.density
        val sizePx = (sizeDp * density).toInt()
        val strokePx = (3 * density).toInt()

        val bitmap = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = fillColor
            style = Paint.Style.FILL
        }
        val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = strokeColor
            style = Paint.Style.STROKE
            strokeWidth = strokePx.toFloat()
        }

        val radius = (sizePx - strokePx) / 2f
        val center = sizePx / 2f
        canvas.drawCircle(center, center, radius, fillPaint)
        canvas.drawCircle(center, center, radius, strokePaint)

        return BitmapDrawable(resources, bitmap)
    }

    private fun setupLocationCallback() {
        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                for (location in result.locations) {
                    onGpsLocationUpdated(location)
                }
            }
        }
    }

    private fun checkLocationPermissionsAndStart() {
        val fineLocation = ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
        val coarseLocation = ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)

        if (fineLocation != PackageManager.PERMISSION_GRANTED || coarseLocation != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
                LOCATION_PERMISSION_REQ_CODE
            )
        } else {
            startLocationUpdates()
            sensorCollector.start()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == LOCATION_PERMISSION_REQ_CODE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startLocationUpdates()
            sensorCollector.start()
        }
    }

    private fun startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) return

        val locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
            .setMinUpdateIntervalMillis(500L)
            .build()

        fusedLocationClient.requestLocationUpdates(locationRequest, locationCallback, Looper.getMainLooper())
    }

    /**
     * Handles each incoming GPS location fix.
     */
    private fun onGpsLocationUpdated(location: Location) {
        lastGpsLocation = location
        val geoPoint = GeoPoint(location.latitude, location.longitude)

        // 1. If origin reference point is not set, initialize it from this first GPS fix
        if (originLat == null || originLon == null) {
            originLat = location.latitude
            originLon = location.longitude

            val initialBearingRad = Math.toRadians(location.bearing.toDouble()).toFloat()
            positionEstimator.resetState(
                x = 0.0f,
                y = 0.0f,
                heading = initialBearingRad,
                velocity = location.speed
            )

            roadProvider.updateOrigin(location.latitude, location.longitude)
            mapMatcher.roadSegments = roadProvider.getActiveSegments()

            gpsMarker?.position = geoPoint
            aiMarker?.position = geoPoint

            mapView.controller.setCenter(geoPoint)
        } else {
            gpsMarker?.position = geoPoint
            if (!isGpsOutageMode) {
                // Keep PositionEstimator continuously snapped/synced to real GPS fix (Requirement 1)
                val oLat = originLat ?: return
                val oLon = originLon ?: return
                val (gpsEast, gpsNorth) = GeoProjection.latLonToEnu(location.latitude, location.longitude, oLat, oLon)
                val trueBearingRad = Math.toRadians(location.bearing.toDouble()).toFloat()
                val trueVelocity = location.speed
                positionEstimator.resetState(
                    x = gpsEast.toFloat(),
                    y = gpsNorth.toFloat(),
                    heading = trueBearingRad,
                    velocity = trueVelocity
                )
                aiMarker?.position = geoPoint

                mapView.controller.animateTo(geoPoint)
            }
        }

        // 2. Append to GPS trail
        gpsPolyline?.addPoint(geoPoint)
        mapView.invalidate()

        // 3. Update HUD Display
        val speedKmh = location.speed * 3.6f
        tvGpsCoords.text = "Lat: ${String.format("%.5f", location.latitude)}\nLon: ${String.format("%.5f", location.longitude)}"
        tvGpsSpeed.text = "Speed: ${String.format("%.1f", speedKmh)} km/h"

        updateSeparationAndDrift()

        // 4. Asynchronously refresh nearby road network within ~300m in background
        if (isMapMatchingEnabled) {
            roadProvider.fetchRoadsAroundAsync(location.latitude, location.longitude, radiusM = 300.0) { success ->
                if (success) {
                    mapMatcher.roadSegments = roadProvider.getActiveSegments()
                }
            }
        }
    }

    /**
     * Handles each completed ~9-second IMU window from IMUSensorCollector.
     */
    private fun onNewImuWindowReceived(samples: List<IMUSensorCollector.Sample>) {
        val oLat = originLat ?: return
        val oLon = originLon ?: return

        windowCount++

        if (isGpsOutageMode) {
            // =========================================================================
            // OUTAGE MODE: AI-DR is authoritative, estimate & display position (Requirement 2)
            // =========================================================================
            val navState = positionEstimator.estimatePosition(samples)
            lastAiState = navState
            totalDistanceTraveledM += navState.deltaS_final
            outageDistanceTraveledM += navState.deltaS_final

            val (aiLat, aiLon) = GeoProjection.enuToLatLon(navState.x, navState.y, oLat, oLon)
            val rawGeoPoint = GeoPoint(aiLat, aiLon)

            // Lightweight Map-Matching (DISPLAY LAYER ONLY - leaves navState and drift telemetry 100% untouched)
            val displayGeoPoint = if (isMapMatchingEnabled) {
                val snapResult = mapMatcher.snap(aiLat, aiLon, Math.toRadians(navState.headingDeg.toDouble()), oLat, oLon)
                if (snapResult.isSnapped) {
                    val mapMatchLog = "[MAP_MATCH] Window #$windowCount: SNAPPED by ${String.format("%.1f", snapResult.snapDistanceM)}m onto '${snapResult.matchedRoadName}' (heading diff: ${String.format("%.1f", snapResult.headingDiffDeg)}°) | Raw: (${String.format("%.6f", aiLat)}, ${String.format("%.6f", aiLon)}) -> Display: (${String.format("%.6f", snapResult.displayLat)}, ${String.format("%.6f", snapResult.displayLon)})"
                    Log.i(TAG_POSITION, mapMatchLog)
                    GeoPoint(snapResult.displayLat, snapResult.displayLon)
                } else {
                    Log.i(TAG_POSITION, "[MAP_MATCH] Window #$windowCount: UNSNAPPED (Raw point preserved)")
                    rawGeoPoint
                }
            } else {
                rawGeoPoint
            }

            val gps = lastGpsLocation
            val (driftM, driftPct) = computeDriftMetrics()

            val corrLogStr = if (navState.isStationary) {
                "0.00m (skipped -- ZUPT active)"
            } else {
                "${String.format("%.2f", navState.deltaS_corr)}m"
            }

            val elapsedS = (System.currentTimeMillis() - outageStartTimeMs) / 1000f
            val logMsg = "[GPS_LOST_MODE] Window #$windowCount (Outage: ${String.format("%.1f", elapsedS)}s) | " +
                    "True GPS: (${String.format("%.6f", gps?.latitude ?: 0.0)}, ${String.format("%.6f", gps?.longitude ?: 0.0)}) | " +
                    "AI-DR: (${String.format("%.6f", aiLat)}, ${String.format("%.6f", aiLon)}) | " +
                    "Outage Drift: ${String.format("%.1f", driftM)}m (${String.format("%.2f", driftPct)}%) | " +
                    "dS_imu: ${String.format("%.2f", navState.deltaS_imu)}m | dS_corr: $corrLogStr | " +
                    "dS_final: ${String.format("%.2f", navState.deltaS_final)}m"
            Log.i(TAG_POSITION, logMsg)

            // Compute signal metrics across this window for diagnostic inspection
            var sumLinMag = 0f
            var minLinMag = Float.MAX_VALUE
            var maxLinMag = 0f
            var sumRawMag = 0f
            var sumGyroMag = 0f
            var maxGyroMag = 0f
            for (s in samples) {
                val lMag = kotlin.math.sqrt(s.accLinX * s.accLinX + s.accLinY * s.accLinY + s.accLinZ * s.accLinZ)
                val rMag = kotlin.math.sqrt(s.accX * s.accX + s.accY * s.accY + s.accZ * s.accZ)
                val gMag = kotlin.math.sqrt(s.gyroX * s.gyroX + s.gyroY * s.gyroY + s.gyroZ * s.gyroZ)
                sumLinMag += lMag
                if (lMag < minLinMag) minLinMag = lMag
                if (lMag > maxLinMag) maxLinMag = lMag
                sumRawMag += rMag
                sumGyroMag += gMag
                if (gMag > maxGyroMag) maxGyroMag = gMag
            }
            val avgLinMag = sumLinMag / samples.size
            val avgRawMag = sumRawMag / samples.size
            val avgGyroMag = sumGyroMag / samples.size
            val s0 = samples.first()

            val diagLog = "Window #$windowCount | isStationary=${navState.isStationary} | " +
                    "a_mag(lin): avg=${String.format("%.3f", avgLinMag)}, max=${String.format("%.3f", maxLinMag)} m/s² (A_TH=5.389) | " +
                    "a_raw(grav): avg=${String.format("%.3f", avgRawMag)} m/s² | " +
                    "gyro_mag: avg=${String.format("%.3f", avgGyroMag)}, max=${String.format("%.3f", maxGyroMag)} rad/s (W_TH=0.753) | " +
                    "dS_imu=${String.format("%.3f", navState.deltaS_imu)}m, dS_corr=$corrLogStr, dS_final=${String.format("%.3f", navState.deltaS_final)}m, v_eff=${String.format("%.2f", navState.effectiveSpeed * 3.6f)} km/h | " +
                    "Sample[0] raw_acc=(${String.format("%.2f", s0.accX)}, ${String.format("%.2f", s0.accY)}, ${String.format("%.2f", s0.accZ)}), lin_acc=(${String.format("%.2f", s0.accLinX)}, ${String.format("%.2f", s0.accLinY)}, ${String.format("%.2f", s0.accLinZ)})"
            Log.i(TAG_POSITION, diagLog)

            runOnUiThread {
                animateMarkerTo(aiMarker, displayGeoPoint, durationMs = 1000L)
                mapView.controller.animateTo(displayGeoPoint)
                aiPolyline?.addPoint(displayGeoPoint)
                mapView.invalidate()

                val aiSpeedKmh = navState.effectiveSpeed * 3.6f
                tvAiCoords.text = "Lat: ${String.format("%.5f", aiLat)}\nLon: ${String.format("%.5f", aiLon)}"
                tvAiMotion.text = "Speed: ${String.format("%.1f", aiSpeedKmh)} km/h | ψ: ${String.format("%.1f", navState.headingDeg)}°"
                tvWindowCount.text = "Window #$windowCount (Δs=${String.format("%.1f", navState.deltaS_final)}m)"

                updateSeparationAndDrift()
            }
        } else {
            // =========================================================================
            // GPS ACTIVE MODE: Do NOT draw AI-DR marker or accumulate polyline (Requirement 1)
            // PositionEstimator stays continuously synced to real GPS
            // =========================================================================
            val gps = lastGpsLocation
            val logMsg = "[GPS_ACTIVE_MODE] Window #$windowCount | GPS: (${String.format("%.6f", gps?.latitude ?: 0.0)}, ${String.format("%.6f", gps?.longitude ?: 0.0)}) | AI-DR: Standby (Synced to GPS)"
            Log.i(TAG_POSITION, logMsg)

            runOnUiThread {
                tvWindowCount.text = "Window #$windowCount (GPS Active)"
                updateSeparationAndDrift()
            }
        }
    }

    /**
     * Toggles between GPS Active and GPS Lost (Simulated) Dead Reckoning mode.
     */
    private fun toggleGpsOutageMode() {
        val gps = lastGpsLocation
        val oLat = originLat
        val oLon = originLon

        if (gps == null || oLat == null || oLon == null) {
            Toast.makeText(this, "Waiting for initial GPS fix...", Toast.LENGTH_SHORT).show()
            return
        }

        if (!isGpsOutageMode) {
            // =========================================================================
            // TRANSITION TO: GPS LOST (SIMULATED OUTAGE)
            // =========================================================================
            isGpsOutageMode = true
            outageStartTimeMs = System.currentTimeMillis()
            outageDistanceTraveledM = 0.0f

            // 1. Clean snap of PositionEstimator state to true current GPS values
            val (gpsEast, gpsNorth) = GeoProjection.latLonToEnu(gps.latitude, gps.longitude, oLat, oLon)
            val trueBearingRad = Math.toRadians(gps.bearing.toDouble()).toFloat()
            val trueVelocity = gps.speed

            positionEstimator.resetState(
                x = gpsEast.toFloat(),
                y = gpsNorth.toFloat(),
                heading = trueBearingRad,
                velocity = trueVelocity
            )

            // Snap AI marker exactly to current GPS position at start of outage
            val currentGpsGeoPoint = GeoPoint(gps.latitude, gps.longitude)
            aiMarker?.position = currentGpsGeoPoint
            aiMarker?.isEnabled = true
            aiMarker?.alpha = 1.0f
            aiMarker?.title = "PRIMARY: AI Dead Reckoning (Authoritative)"

            // Start accumulating AI Polyline fresh from current GPS location (Requirement 2)
            aiPolyline?.isEnabled = true
            aiPolyline?.setPoints(mutableListOf(currentGpsGeoPoint))

            // 2. Visual Differentiation: AI Marker becomes primary; GPS marker becomes faint background ground truth
            gpsMarker?.alpha = 0.35f
            gpsMarker?.title = "Ground Truth Reference (GPS - Inactive)"

            // 3. Update Controls & HUD
            btnToggleGpsOutage.text = "RESTORE GPS SIGNAL"
            btnToggleGpsOutage.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#2E7D32")) // Green
            btnToggleGpsOutage.setIconResource(android.R.drawable.ic_menu_compass)

            tvGpsStatusBadge.text = "MODE: GPS LOST - AI DR"
            tvGpsStatusBadge.setBackgroundColor(Color.parseColor("#D32F2F")) // Red

            llOutageBanner.visibility = View.VISIBLE
            tvOutageBannerText.text = "⚠️ GPS LOST: AI ESTIMATING"
            tvOutageTimer.text = "Outage: 0.0s"

            llAiColumn.alpha = 1.0f
            llGpsColumn.alpha = 0.5f

            tvAiCoords.text = "Lat: ${String.format("%.5f", gps.latitude)}\nLon: ${String.format("%.5f", gps.longitude)}"
            tvAiMotion.text = "Speed: ${String.format("%.1f", gps.speed * 3.6f)} km/h | ψ: ${String.format("%.1f", gps.bearing)}°"

            // Start 1Hz UI ticker to update outage elapsed duration in real-time
            startOutageTimerTicker()
            mapView.invalidate()

            Log.i(TAG_POSITION, "GPS OUTAGE SIMULATED: State cleanly snapped to GPS (lat=${gps.latitude}, lon=${gps.longitude}, speed=${trueVelocity}m/s, heading=${gps.bearing}°)")
            Toast.makeText(this, "GNSS Outage Simulated: AI Dead Reckoning is now Authoritative", Toast.LENGTH_SHORT).show()

        } else {
            // =========================================================================
            // TRANSITION TO: GPS ACTIVE (SMOOTH RESYNCHRONIZATION)
            // =========================================================================
            stopOutageTimerTicker()

            val aiCurrentGeo = aiMarker?.position ?: GeoPoint(gps.latitude, gps.longitude)
            val realGpsGeo = GeoPoint(gps.latitude, gps.longitude)
            val elapsedOutageS = (System.currentTimeMillis() - outageStartTimeMs) / 1000f

            // Compute Euclidean correction distance in meters
            val results = FloatArray(1)
            Location.distanceBetween(
                aiCurrentGeo.latitude, aiCurrentGeo.longitude,
                realGpsGeo.latitude, realGpsGeo.longitude,
                results
            )
            val correctionDistanceM = results[0]

            // Log transition metrics immediately as required by Step 5
            Log.i(
                TAG_POSITION,
                "RESYNC: AI-DR pos: (${String.format("%.6f", aiCurrentGeo.latitude)}, ${String.format("%.6f", aiCurrentGeo.longitude)}) | " +
                        "Real GPS: (${String.format("%.6f", realGpsGeo.latitude)}, ${String.format("%.6f", realGpsGeo.longitude)}) | " +
                        "Correction Distance: ${String.format("%.2f", correctionDistanceM)}m | " +
                        "Outage Duration: ${String.format("%.1f", elapsedOutageS)}s"
            )

            // Update UI to indicate resynchronization in progress
            tvGpsStatusBadge.text = "RESYNCING..."
            tvGpsStatusBadge.setBackgroundColor(Color.parseColor("#F57C00")) // Orange
            tvOutageBannerText.text = "🔄 RESYNCING: Correcting ${String.format("%.1f", correctionDistanceM)}m drift..."

            // Smoothly animate the AI marker from its DR position to the true GPS position over 1.8 seconds
            animateResync(aiMarker, aiCurrentGeo, realGpsGeo, durationMs = 1800L) {
                isGpsOutageMode = false

                // 1. Hide AI-DR marker and polyline (Requirements 1 & 3)
                aiMarker?.isEnabled = false
                aiMarker?.alpha = 0.0f
                aiPolyline?.isEnabled = false
                aiPolyline?.setPoints(emptyList())

                // 2. Restore primary/secondary marker styling
                gpsMarker?.alpha = 1.0f
                gpsMarker?.title = "Primary: Real GPS Location"

                // 3. Re-align PositionEstimator to current GPS position for clean continuous tracking
                val (gpsEast, gpsNorth) = GeoProjection.latLonToEnu(gps.latitude, gps.longitude, oLat, oLon)
                positionEstimator.resetState(
                    x = gpsEast.toFloat(),
                    y = gpsNorth.toFloat(),
                    heading = Math.toRadians(gps.bearing.toDouble()).toFloat(),
                    velocity = gps.speed
                )

                // 4. Restore controls & HUD
                btnToggleGpsOutage.text = "SIMULATE GPS LOSS"
                btnToggleGpsOutage.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#D32F2F")) // Red
                btnToggleGpsOutage.setIconResource(android.R.drawable.ic_dialog_alert)

                tvGpsStatusBadge.text = "MODE: LIVE GPS"
                tvGpsStatusBadge.setBackgroundColor(Color.parseColor("#2E7D32")) // Green
                llOutageBanner.visibility = View.GONE

                llAiColumn.alpha = 0.6f
                llGpsColumn.alpha = 1.0f

                tvAiCoords.text = "Standby (GPS Active)"
                tvAiMotion.text = "Standby (Synced to GPS)"
                tvDriftDistance.text = "Drift: 0.0 m (GPS Active)"

                mapView.controller.animateTo(realGpsGeo)
                mapView.invalidate()
                Toast.makeText(this@MainActivity, "GPS Restored: Smoothly resynced (${String.format("%.1f", correctionDistanceM)}m corrected)", Toast.LENGTH_SHORT).show()
            }
        }
    }

    /**
     * Smoothly animates marker between GeoPoint coordinates using spherical linear interpolation.
     */
    private fun animateMarkerTo(marker: Marker?, targetGeoPoint: GeoPoint, durationMs: Long) {
        if (marker == null) return
        val startGeo = marker.position ?: targetGeoPoint
        val startLatLng = LatLng(startGeo.latitude, startGeo.longitude)
        val targetLatLng = LatLng(targetGeoPoint.latitude, targetGeoPoint.longitude)

        ValueAnimator.ofFloat(0f, 1f).apply {
            duration = durationMs
            interpolator = LinearInterpolator()
            addUpdateListener { animation ->
                val fraction = animation.animatedFraction
                val interpolated = SphericalLatLonInterpolator.interpolate(fraction, startLatLng, targetLatLng)
                marker.position = GeoPoint(interpolated.latitude, interpolated.longitude)
                mapView.invalidate()
            }
            start()
        }
    }

    /**
     * Smoothly animates marker for resynchronization over 1.8s with ease-in-out easing.
     */
    private fun animateResync(
        marker: Marker?,
        fromGeo: GeoPoint,
        toGeo: GeoPoint,
        durationMs: Long,
        onComplete: () -> Unit
    ) {
        if (marker == null) {
            onComplete()
            return
        }

        val fromLatLng = LatLng(fromGeo.latitude, fromGeo.longitude)
        val toLatLng = LatLng(toGeo.latitude, toGeo.longitude)

        ValueAnimator.ofFloat(0f, 1f).apply {
            duration = durationMs
            interpolator = AccelerateDecelerateInterpolator()
            addUpdateListener { animation ->
                val fraction = animation.animatedFraction
                val interpolated = SphericalLatLonInterpolator.interpolate(fraction, fromLatLng, toLatLng)
                marker.position = GeoPoint(interpolated.latitude, interpolated.longitude)
                mapView.invalidate()
            }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    marker.position = toGeo
                    mapView.invalidate()
                    onComplete()
                }
            })
            start()
        }
    }

    /**
     * Starts 1Hz ticker to update outage elapsed duration and drift live on the HUD.
     */
    private fun startOutageTimerTicker() {
        stopOutageTimerTicker()
        outageTimerRunnable = object : Runnable {
            override fun run() {
                if (isGpsOutageMode) {
                    val elapsedS = (System.currentTimeMillis() - outageStartTimeMs) / 1000f
                    tvOutageTimer.text = "Outage: ${String.format("%.1f", elapsedS)}s"
                    updateSeparationAndDrift()
                    uiHandler.postDelayed(this, 500L)
                }
            }
        }
        uiHandler.post(outageTimerRunnable!!)
    }

    private fun stopOutageTimerTicker() {
        outageTimerRunnable?.let { uiHandler.removeCallbacks(it) }
        outageTimerRunnable = null
    }

    /**
     * Computes the current separation/drift distance and percentage.
     */
    private fun computeDriftMetrics(): Pair<Float, Float> {
        val oLat = originLat ?: return Pair(0.0f, 0.0f)
        val oLon = originLon ?: return Pair(0.0f, 0.0f)
        val gps = lastGpsLocation ?: return Pair(0.0f, 0.0f)
        val ai = lastAiState ?: return Pair(0.0f, 0.0f)

        val (eGps, nGps) = GeoProjection.latLonToEnu(gps.latitude, gps.longitude, oLat, oLon)
        val dx = (eGps - ai.x).toDouble()
        val dy = (nGps - ai.y).toDouble()
        val separationM = sqrt(dx * dx + dy * dy).toFloat()

        val distanceBasis = if (isGpsOutageMode) outageDistanceTraveledM else totalDistanceTraveledM
        val driftPct = if (distanceBasis > 1.0f) {
            (separationM / distanceBasis) * 100.0f
        } else {
            0.0f
        }

        return Pair(separationM, driftPct)
    }

    /**
     * Updates HUD display with current drift distance and percentage.
     */
    private fun updateSeparationAndDrift() {
        if (isGpsOutageMode) {
            val (separationM, driftPct) = computeDriftMetrics()
            tvDriftDistance.text = "Outage Drift: ${String.format("%.1f", separationM)} m (${String.format("%.2f", driftPct)}%)"
        } else {
            tvDriftDistance.text = "Drift: 0.0 m (GPS Active)"
        }
    }

    private fun resetOriginToCurrentLocation() {
        val gps = lastGpsLocation ?: return
        originLat = gps.latitude
        originLon = gps.longitude
        roadProvider.updateOrigin(gps.latitude, gps.longitude)
        mapMatcher.roadSegments = roadProvider.getActiveSegments()

        val initialBearingRad = Math.toRadians(gps.bearing.toDouble()).toFloat()
        positionEstimator.resetState(0.0f, 0.0f, initialBearingRad, gps.speed)

        val currentGeo = GeoPoint(gps.latitude, gps.longitude)
        gpsMarker?.position = currentGeo
        gpsPolyline?.setPoints(mutableListOf(currentGeo))

        if (isGpsOutageMode) {
            aiMarker?.position = currentGeo
            aiMarker?.isEnabled = true
            aiMarker?.alpha = 1.0f
            aiPolyline?.isEnabled = true
            aiPolyline?.setPoints(mutableListOf(currentGeo))
        } else {
            aiMarker?.position = currentGeo
            aiMarker?.isEnabled = false
            aiMarker?.alpha = 0.0f
            aiPolyline?.isEnabled = false
            aiPolyline?.setPoints(emptyList())
        }

        totalDistanceTraveledM = 0.0f
        outageDistanceTraveledM = 0.0f
        windowCount = 0

        tvGpsCoords.text = "Lat: ${String.format("%.5f", gps.latitude)}\nLon: ${String.format("%.5f", gps.longitude)}"
        tvAiCoords.text = if (isGpsOutageMode) "Lat: ${String.format("%.5f", gps.latitude)}\nLon: ${String.format("%.5f", gps.longitude)}" else "Standby (GPS Active)"
        tvAiMotion.text = if (isGpsOutageMode) "Speed: 0.0 km/h | ψ: ${String.format("%.1f", gps.bearing)}°" else "Standby (Synced to GPS)"
        tvDriftDistance.text = if (isGpsOutageMode) "Outage Drift: 0.0 m (0.00%)" else "Drift: 0.0 m (GPS Active)"
        tvWindowCount.text = "Window #0 (Reset)"

        mapView.controller.animateTo(currentGeo)
        mapView.invalidate()
        Log.i(TAG_POSITION, "Origin reset to current GPS: ($originLat, $originLon)")
    }

    override fun onResume() {
        super.onResume()
        mapView.onResume()
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopOutageTimerTicker()
        fusedLocationClient.removeLocationUpdates(locationCallback)
        sensorCollector.stop()
        correctionModel.close()
    }
}
