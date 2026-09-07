package com.sih26168.deadreckoning.ui

import android.Manifest
import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
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
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.BitmapDescriptorFactory
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.Marker
import com.google.android.gms.maps.model.MarkerOptions
import com.google.android.gms.maps.model.Polyline
import com.google.android.gms.maps.model.PolylineOptions
import com.google.android.material.button.MaterialButton
import com.sih26168.deadreckoning.R
import com.sih26168.deadreckoning.engine.NavigationState
import com.sih26168.deadreckoning.engine.PositionEstimator
import com.sih26168.deadreckoning.ml.CorrectionModel
import com.sih26168.deadreckoning.sensor.IMUSensorCollector
import com.sih26168.deadreckoning.test.OnnxVerificationActivity
import com.sih26168.deadreckoning.util.GeoProjection
import com.sih26168.deadreckoning.util.SphericalLatLonInterpolator
import kotlin.math.sqrt

/**
 * MainActivity: Live Dual-Marker Dead Reckoning Map Activity with Simulated GPS Outage Mode.
 *
 * Implements SIH26168 core Level 1 capability:
 *  1. Normal GPS Active Mode: Genuine GPS drives primary position; AI-DR estimates in background.
 *  2. Simulated GPS Loss Mode:
 *     - PositionEstimator state is cleanly snapped to true GPS at instant of toggle.
 *     - AI-DR becomes the authoritative primary position displayed to the user.
 *     - Real GPS is tracked faintly in background as ground truth comparison.
 *     - HUD tracks live outage duration, accumulated distance, and drift.
 *  3. Resync to GPS Active:
 *     - Smooth spherical interpolation (1.8s) transitions marker without teleportation.
 *     - Logs correction distance and outage metrics to SIH_POSITION_TEST.
 */
class MainActivity : AppCompatActivity(), OnMapReadyCallback {

    companion object {
        private const val TAG_POSITION = "SIH_POSITION_TEST"
        private const val LOCATION_PERMISSION_REQ_CODE = 1001
        private const val DEFAULT_ZOOM = 18f
    }

    private var googleMap: GoogleMap? = null
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

    // Map Markers & Polylines
    private var gpsMarker: Marker? = null
    private var aiMarker: Marker? = null
    private var gpsPolyline: Polyline? = null
    private var aiPolyline: Polyline? = null
    private val gpsTrail = mutableListOf<LatLng>()
    private val aiTrail = mutableListOf<LatLng>()

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Bind UI Views
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

        btnToggleGpsOutage.setOnClickListener {
            toggleGpsOutageMode()
        }

        btnResetOrigin.setOnClickListener {
            resetOriginToCurrentLocation()
        }

        btnVerifyTests.setOnClickListener {
            startActivity(Intent(this, OnnxVerificationActivity::class.java))
        }

        // Initialize Map
        val mapFragment = supportFragmentManager.findFragmentById(R.id.mapFragment) as SupportMapFragment
        mapFragment.getMapAsync(this)

        // Initialize ONNX Model & Position Estimator
        correctionModel = CorrectionModel(this)
        positionEstimator = PositionEstimator(correctionModel)

        // Initialize IMU Sensor Collector
        sensorCollector = IMUSensorCollector(
            context = this,
            onWindowSamplesReady = { samples ->
                onNewImuWindowReceived(samples)
            }
        )

        // Initialize GPS Location Services
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        setupLocationCallback()
    }

    override fun onMapReady(map: GoogleMap) {
        googleMap = map
        map.uiSettings.isZoomControlsEnabled = true
        map.uiSettings.isCompassEnabled = true

        // Create Polylines
        gpsPolyline = map.addPolyline(
            PolylineOptions()
                .color(Color.parseColor("#1E88E5")) // Blue for GPS
                .width(8f)
                .geodesic(true)
        )

        aiPolyline = map.addPolyline(
            PolylineOptions()
                .color(Color.parseColor("#FF5722")) // Deep Orange for AI Dead Reckoning
                .width(8f)
                .geodesic(true)
        )

        checkLocationPermissionsAndStart()
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
        val latLng = LatLng(location.latitude, location.longitude)

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

            // Setup GPS Marker (Primary initially)
            gpsMarker = googleMap?.addMarker(
                MarkerOptions()
                    .position(latLng)
                    .title("Real GPS Location")
                    .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_AZURE))
            )

            // Setup AI Predicted Marker (Secondary initially)
            aiMarker = googleMap?.addMarker(
                MarkerOptions()
                    .position(latLng)
                    .title("Physics + AI Estimated")
                    .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_ORANGE))
                    .alpha(0.85f)
            )

            googleMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(latLng, DEFAULT_ZOOM))
        } else {
            // Update GPS marker position
            gpsMarker?.position = latLng
            if (!isGpsOutageMode) {
                // In GPS Active mode, camera follows genuine GPS
                googleMap?.animateCamera(CameraUpdateFactory.newLatLng(latLng))
            }
        }

        // 2. Append to GPS trail
        gpsTrail.add(latLng)
        gpsPolyline?.points = gpsTrail

        // 3. Update HUD Display
        val speedKmh = location.speed * 3.6f
        tvGpsCoords.text = "Lat: ${String.format("%.5f", location.latitude)}\nLon: ${String.format("%.5f", location.longitude)}"
        tvGpsSpeed.text = "Speed: ${String.format("%.1f", speedKmh)} km/h"

        updateSeparationAndDrift()
    }

    /**
     * Handles each completed ~9-second IMU window from IMUSensorCollector.
     */
    private fun onNewImuWindowReceived(samples: List<IMUSensorCollector.Sample>) {
        val oLat = originLat ?: return
        val oLon = originLon ?: return

        windowCount++

        // 1. Run PositionEstimator full pipeline
        val navState = positionEstimator.estimatePosition(samples)
        lastAiState = navState
        totalDistanceTraveledM += navState.deltaS_final
        if (isGpsOutageMode) {
            outageDistanceTraveledM += navState.deltaS_final
        }

        // 2. Convert local ENU (meters) back to LatLng using exact inverse projection
        val (aiLat, aiLon) = GeoProjection.enuToLatLon(navState.x, navState.y, oLat, oLon)
        val targetLatLng = LatLng(aiLat, aiLon)

        // 3. Log to Logcat with tag SIH_POSITION_TEST
        val gps = lastGpsLocation
        val (driftM, driftPct) = computeDriftMetrics()

        if (isGpsOutageMode) {
            val elapsedS = (System.currentTimeMillis() - outageStartTimeMs) / 1000f
            val logMsg = "[GPS_LOST_MODE] Window #$windowCount (Outage: ${String.format("%.1f", elapsedS)}s) | " +
                    "True GPS: (${String.format("%.6f", gps?.latitude ?: 0.0)}, ${String.format("%.6f", gps?.longitude ?: 0.0)}) | " +
                    "AI-DR: (${String.format("%.6f", aiLat)}, ${String.format("%.6f", aiLon)}) | " +
                    "Outage Drift: ${String.format("%.1f", driftM)}m (${String.format("%.2f", driftPct)}%) | " +
                    "dS_imu: ${String.format("%.2f", navState.deltaS_imu)}m | dS_corr: ${String.format("%.2f", navState.deltaS_corr)}m | " +
                    "dS_final: ${String.format("%.2f", navState.deltaS_final)}m"
            Log.i(TAG_POSITION, logMsg)
        } else {
            val logMsg = "GPS: (${String.format("%.6f", gps?.latitude ?: 0.0)}, ${String.format("%.6f", gps?.longitude ?: 0.0)}) | " +
                    "AI: (${String.format("%.6f", aiLat)}, ${String.format("%.6f", aiLon)}) | " +
                    "Drift: ${String.format("%.1f", driftM)}m (${String.format("%.2f", driftPct)}%) | " +
                    "dS_imu: ${String.format("%.2f", navState.deltaS_imu)}m | dS_corr: ${String.format("%.2f", navState.deltaS_corr)}m"
            Log.i(TAG_POSITION, logMsg)
        }

        // 4. Smoothly animate AI Marker to new position over 1.0 second
        runOnUiThread {
            animateMarkerTo(aiMarker, targetLatLng, durationMs = 1000L)

            // If in GPS Lost mode, AI DR is authoritative; camera centers on AI position
            if (isGpsOutageMode) {
                googleMap?.animateCamera(CameraUpdateFactory.newLatLng(targetLatLng))
            }

            // Append to AI Trail
            aiTrail.add(targetLatLng)
            aiPolyline?.points = aiTrail

            // Update HUD text
            val aiSpeedKmh = navState.velocity * 3.6f
            tvAiCoords.text = "Lat: ${String.format("%.5f", aiLat)}\nLon: ${String.format("%.5f", aiLon)}"
            tvAiMotion.text = "Speed: ${String.format("%.1f", aiSpeedKmh)} km/h | ψ: ${String.format("%.1f", navState.headingDeg)}°"
            tvWindowCount.text = "Window #$windowCount (Δs=${String.format("%.1f", navState.deltaS_final)}m)"

            updateSeparationAndDrift()
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

            // 1. CRITICAL: Clean snap of PositionEstimator state to true current GPS values
            //    (Replicates exact Phase 6/Phase 9 Python outage initialization)
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
            val currentGpsLatLng = LatLng(gps.latitude, gps.longitude)
            aiMarker?.position = currentGpsLatLng

            // 2. Visual Differentiation: AI Marker becomes primary; GPS marker becomes faint background ground truth
            aiMarker?.alpha = 1.0f
            aiMarker?.title = "PRIMARY: AI Dead Reckoning (Authoritative)"
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

            // Start 1Hz UI ticker to update outage elapsed duration in real-time
            startOutageTimerTicker()

            Log.i(TAG_POSITION, "GPS OUTAGE SIMULATED: State cleanly snapped to GPS (lat=${gps.latitude}, lon=${gps.longitude}, speed=${trueVelocity}m/s, heading=${gps.bearing}°)")
            Toast.makeText(this, "GNSS Outage Simulated: AI Dead Reckoning is now Authoritative", Toast.LENGTH_SHORT).show()

        } else {
            // =========================================================================
            // TRANSITION TO: GPS ACTIVE (SMOOTH RESYNCHRONIZATION)
            // =========================================================================
            stopOutageTimerTicker()

            val aiCurrentLatLng = aiMarker?.position ?: LatLng(gps.latitude, gps.longitude)
            val realGpsLatLng = LatLng(gps.latitude, gps.longitude)
            val elapsedOutageS = (System.currentTimeMillis() - outageStartTimeMs) / 1000f

            // Compute Euclidean correction distance in meters
            val results = FloatArray(1)
            Location.distanceBetween(
                aiCurrentLatLng.latitude, aiCurrentLatLng.longitude,
                realGpsLatLng.latitude, realGpsLatLng.longitude,
                results
            )
            val correctionDistanceM = results[0]

            // Log transition metrics immediately as required by Step 5
            Log.i(
                TAG_POSITION,
                "RESYNC: AI-DR pos: (${String.format("%.6f", aiCurrentLatLng.latitude)}, ${String.format("%.6f", aiCurrentLatLng.longitude)}) | " +
                        "Real GPS: (${String.format("%.6f", realGpsLatLng.latitude)}, ${String.format("%.6f", realGpsLatLng.longitude)}) | " +
                        "Correction Distance: ${String.format("%.2f", correctionDistanceM)}m | " +
                        "Outage Duration: ${String.format("%.1f", elapsedOutageS)}s"
            )

            // Update UI to indicate resynchronization in progress
            tvGpsStatusBadge.text = "RESYNCING..."
            tvGpsStatusBadge.setBackgroundColor(Color.parseColor("#F57C00")) // Orange
            tvOutageBannerText.text = "🔄 RESYNCING: Correcting ${String.format("%.1f", correctionDistanceM)}m drift..."

            // Smoothly animate the AI marker from its DR position to the true GPS position over 1.8 seconds
            animateResync(aiMarker, aiCurrentLatLng, realGpsLatLng, durationMs = 1800L) {
                // On animation completion:
                isGpsOutageMode = false

                // Restore primary/secondary marker styling
                gpsMarker?.alpha = 1.0f
                gpsMarker?.title = "Primary: Real GPS Location"
                aiMarker?.alpha = 0.85f
                aiMarker?.title = "Secondary: AI DR Estimate"

                // Re-align PositionEstimator to current GPS position for clean continuous tracking
                val (gpsEast, gpsNorth) = GeoProjection.latLonToEnu(gps.latitude, gps.longitude, oLat, oLon)
                positionEstimator.resetState(
                    x = gpsEast.toFloat(),
                    y = gpsNorth.toFloat(),
                    heading = Math.toRadians(gps.bearing.toDouble()).toFloat(),
                    velocity = gps.speed
                )

                // Restore controls & HUD
                btnToggleGpsOutage.text = "SIMULATE GPS LOSS"
                btnToggleGpsOutage.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#D32F2F")) // Red
                btnToggleGpsOutage.setIconResource(android.R.drawable.ic_dialog_alert)

                tvGpsStatusBadge.text = "MODE: LIVE GPS"
                tvGpsStatusBadge.setBackgroundColor(Color.parseColor("#2E7D32")) // Green
                llOutageBanner.visibility = View.GONE

                googleMap?.animateCamera(CameraUpdateFactory.newLatLng(realGpsLatLng))
                Toast.makeText(this@MainActivity, "GPS Restored: Smoothly resynced (${String.format("%.1f", correctionDistanceM)}m corrected)", Toast.LENGTH_SHORT).show()
            }
        }
    }

    /**
     * Smoothly animates marker between LatLng coordinates using spherical linear interpolation.
     */
    private fun animateMarkerTo(marker: Marker?, targetLatLng: LatLng, durationMs: Long) {
        if (marker == null) return
        val startPos = marker.position

        ValueAnimator.ofFloat(0f, 1f).apply {
            duration = durationMs
            interpolator = LinearInterpolator()
            addUpdateListener { animation ->
                val fraction = animation.animatedFraction
                marker.position = SphericalLatLonInterpolator.interpolate(fraction, startPos, targetLatLng)
            }
            start()
        }
    }

    /**
     * Smoothly animates marker for resynchronization over 1.8s with ease-in-out easing.
     */
    private fun animateResync(
        marker: Marker?,
        fromLatLng: LatLng,
        toLatLng: LatLng,
        durationMs: Long,
        onComplete: () -> Unit
    ) {
        if (marker == null) {
            onComplete()
            return
        }

        ValueAnimator.ofFloat(0f, 1f).apply {
            duration = durationMs
            interpolator = AccelerateDecelerateInterpolator()
            addUpdateListener { animation ->
                val fraction = animation.animatedFraction
                marker.position = SphericalLatLonInterpolator.interpolate(fraction, fromLatLng, toLatLng)
            }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    marker.position = toLatLng
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
        val (separationM, driftPct) = computeDriftMetrics()
        val label = if (isGpsOutageMode) "Outage Drift" else "Separation"
        tvDriftDistance.text = "$label: ${String.format("%.1f", separationM)} m (${String.format("%.2f", driftPct)}%)"
    }

    private fun resetOriginToCurrentLocation() {
        val gps = lastGpsLocation ?: return
        originLat = gps.latitude
        originLon = gps.longitude

        val initialBearingRad = Math.toRadians(gps.bearing.toDouble()).toFloat()
        positionEstimator.resetState(0.0f, 0.0f, initialBearingRad, gps.speed)

        val currentLatLng = LatLng(gps.latitude, gps.longitude)
        gpsMarker?.position = currentLatLng
        aiMarker?.position = currentLatLng

        gpsTrail.clear()
        aiTrail.clear()
        gpsTrail.add(currentLatLng)
        aiTrail.add(currentLatLng)
        gpsPolyline?.points = gpsTrail
        aiPolyline?.points = aiTrail

        totalDistanceTraveledM = 0.0f
        outageDistanceTraveledM = 0.0f
        windowCount = 0

        tvAiCoords.text = "Lat: ${String.format("%.5f", gps.latitude)}\nLon: ${String.format("%.5f", gps.longitude)}"
        tvDriftDistance.text = "Separation: 0.0 m (0.00%)"
        tvWindowCount.text = "Window #0 (Reset)"

        googleMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(currentLatLng, DEFAULT_ZOOM))
        Log.i(TAG_POSITION, "Origin reset to current GPS: ($originLat, $originLon)")
    }

    override fun onDestroy() {
        super.onDestroy()
        stopOutageTimerTicker()
        fusedLocationClient.removeLocationUpdates(locationCallback)
        sensorCollector.stop()
        correctionModel.close()
    }
}
