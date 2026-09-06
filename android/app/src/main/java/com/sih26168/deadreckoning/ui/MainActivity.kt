package com.sih26168.deadreckoning.ui

import android.Manifest
import android.animation.ValueAnimator
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.location.Location
import android.os.Bundle
import android.os.Looper
import android.util.Log
import android.view.animation.LinearInterpolator
import android.widget.Button
import android.widget.TextView
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
import com.sih26168.deadreckoning.R
import com.sih26168.deadreckoning.engine.NavigationState
import com.sih26168.deadreckoning.engine.PositionEstimator
import com.sih26168.deadreckoning.ml.CorrectionModel
import com.sih26168.deadreckoning.sensor.IMUSensorCollector
import com.sih26168.deadreckoning.test.OnnxVerificationActivity
import com.sih26168.deadreckoning.util.GeoProjection
import kotlin.math.sqrt

/**
 * MainActivity: Live Dual-Marker Dead Reckoning Map Activity.
 *
 * Features:
 * 1. Dual-marker tracking:
 *    - Blue marker / GPS path: Genuine GNSS fix from phone GPS.
 *    - Orange marker / AI path: Physics + AI ML dead reckoning from PositionEstimator.
 * 2. Origin Alignment:
 *    - Initial GPS fix sets origin reference point (lat0, lon0).
 *    - PositionEstimator state starts at (x=0, y=0) relative to this origin.
 * 3. Smooth Animation:
 *    - Marker position animates smoothly over ~1.0s between 9-second window updates.
 * 4. Real-time Path History:
 *    - Dual polylines (Blue GPS vs Orange AI) display trajectory agreement and drift.
 * 5. Telemetry HUD:
 *    - Real-time display of coordinates, speed, heading, and GPS-AI separation.
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

    // Map Markers & Polylines
    private var gpsMarker: Marker? = null
    private var aiMarker: Marker? = null
    private var gpsPolyline: Polyline? = null
    private var aiPolyline: Polyline? = null
    private val gpsTrail = mutableListOf<LatLng>()
    private val aiTrail = mutableListOf<LatLng>()

    // UI Views
    private lateinit var tvGpsCoords: TextView
    private lateinit var tvGpsSpeed: TextView
    private lateinit var tvAiCoords: TextView
    private lateinit var tvAiMotion: TextView
    private lateinit var tvDriftDistance: TextView
    private lateinit var tvWindowCount: TextView
    private lateinit var btnResetOrigin: Button
    private lateinit var btnVerifyTests: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Bind UI Views
        tvGpsCoords = findViewById(R.id.tvGpsCoords)
        tvGpsSpeed = findViewById(R.id.tvGpsSpeed)
        tvAiCoords = findViewById(R.id.tvAiCoords)
        tvAiMotion = findViewById(R.id.tvAiMotion)
        tvDriftDistance = findViewById(R.id.tvDriftDistance)
        tvWindowCount = findViewById(R.id.tvWindowCount)
        btnResetOrigin = findViewById(R.id.btnResetOrigin)
        btnVerifyTests = findViewById(R.id.btnVerifyTests)

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

            // Setup GPS Marker
            gpsMarker = googleMap?.addMarker(
                MarkerOptions()
                    .position(latLng)
                    .title("Real GPS Location")
                    .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_AZURE))
            )

            // Setup AI Predicted Marker at same initial origin
            aiMarker = googleMap?.addMarker(
                MarkerOptions()
                    .position(latLng)
                    .title("Physics + AI Estimated")
                    .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_ORANGE))
            )

            googleMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(latLng, DEFAULT_ZOOM))
        } else {
            // Update GPS marker position
            gpsMarker?.position = latLng
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

        // 2. Convert local ENU (meters) back to LatLng using exact inverse projection
        val (aiLat, aiLon) = GeoProjection.enuToLatLon(navState.x, navState.y, oLat, oLon)
        val targetLatLng = LatLng(aiLat, aiLon)

        // 3. Log to Logcat with tag SIH_POSITION_TEST
        val logMsg = "Window #$windowCount: x=${String.format("%.2f", navState.x)}m, y=${String.format("%.2f", navState.y)}m, " +
                "lat=${String.format("%.6f", aiLat)}, lon=${String.format("%.6f", aiLon)}, " +
                "heading=${String.format("%.1f", navState.headingDeg)}°, v=${String.format("%.2f", navState.velocity)}m/s, " +
                "Δs_imu=${String.format("%.2f", navState.deltaS_imu)}m, Δs_corr=${String.format("%.2f", navState.deltaS_corr)}m, " +
                "Δs_final=${String.format("%.2f", navState.deltaS_final)}m"
        Log.i(TAG_POSITION, logMsg)

        // 4. Smoothly animate AI Marker to new position over 1.0 second
        runOnUiThread {
            animateMarkerTo(aiMarker, targetLatLng, durationMs = 1000L)

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
     * Animates marker smoothly between coordinates over durationMs.
     */
    private fun animateMarkerTo(marker: Marker?, targetLatLng: LatLng, durationMs: Long) {
        if (marker == null) return
        val startPos = marker.position

        ValueAnimator.ofFloat(0f, 1f).apply {
            duration = durationMs
            interpolator = LinearInterpolator()
            addUpdateListener { animation ->
                val fraction = animation.animatedFraction
                val lat = startPos.latitude + fraction * (targetLatLng.latitude - startPos.latitude)
                val lon = startPos.longitude + fraction * (targetLatLng.longitude - startPos.longitude)
                marker.position = LatLng(lat, lon)
            }
            start()
        }
    }

    /**
     * Computes the current separation distance between GPS and AI estimate in meters.
     */
    private fun updateSeparationAndDrift() {
        val oLat = originLat ?: return
        val oLon = originLon ?: return
        val gps = lastGpsLocation ?: return
        val ai = lastAiState ?: return

        // Convert current GPS location to local ENU relative to origin
        val (eGps, nGps) = GeoProjection.latLonToEnu(gps.latitude, gps.longitude, oLat, oLon)

        val dx = (eGps - ai.x).toDouble()
        val dy = (nGps - ai.y).toDouble()
        val separationM = sqrt(dx * dx + dy * dy).toFloat()

        val driftPct = if (totalDistanceTraveledM > 1.0f) {
            (separationM / totalDistanceTraveledM) * 100.0f
        } else {
            0.0f
        }

        tvDriftDistance.text = "Separation: ${String.format("%.1f", separationM)} m (${String.format("%.2f", driftPct)}%)"
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
        windowCount = 0

        tvAiCoords.text = "Lat: ${String.format("%.5f", gps.latitude)}\nLon: ${String.format("%.5f", gps.longitude)}"
        tvDriftDistance.text = "Separation: 0.0 m (0.00%)"
        tvWindowCount.text = "Window #0 (Reset)"

        googleMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(currentLatLng, DEFAULT_ZOOM))
        Log.i(TAG_POSITION, "Origin reset to current GPS: ($originLat, $originLon)")
    }

    override fun onDestroy() {
        super.onDestroy()
        fusedLocationClient.removeLocationUpdates(locationCallback)
        sensorCollector.stop()
        correctionModel.close()
    }
}
