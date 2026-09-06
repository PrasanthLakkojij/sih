package com.sih26168.deadreckoning.test

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.sih26168.deadreckoning.ml.CorrectionModel
import com.sih26168.deadreckoning.ml.FeatureExtractor
import com.sih26168.deadreckoning.physics.PhysicsDeadReckoning
import com.sih26168.deadreckoning.sensor.IMUSensorCollector

class OnnxVerificationActivity : AppCompatActivity() {

    companion object {
        private const val TAG_ONNX = "SIH_ONNX_TEST"
        private const val TAG_FEATURE = "SIH_FEATURE_TEST"
        private const val TAG_PHYSICS = "SIH_PHYSICS_TEST"

        // Expected 58 features from Window 0
        val EXPECTED_FEATURES_WINDOW_0 = floatArrayOf(
            0.576851f, 2.196496f, -7.459200f, 5.660200f, -0.632342f, 1.151108f, -3.528800f, 2.867200f,
            9.937638f, 0.704170f, 7.471200f, 11.656800f, 0.590827f, 2.201351f, -7.409600f, 5.735500f,
            -0.630366f, 1.150243f, -3.537400f, 2.862400f, 0.131123f, 0.704176f, -2.335200f, 1.850400f,
            -0.026637f, 0.074361f, -0.227500f, 0.145600f, -0.022525f, 0.223659f, -0.714500f, 0.833500f,
            0.043442f, 0.157361f, -0.544600f, 0.555600f, 0.279238f, 1.679649f, -4.092244f, 5.067989f,
            -0.043803f, 0.157457f, -0.556512f, 0.542041f, 10.272852f, 0.776717f, 8.076717f, 12.923003f,
            2.347225f, 1.385292f, 0.560172f, 7.648473f, 0.223625f, 0.182787f, 0.013356f, 0.995727f,
            9.001000f, 91.000000f
        )

        const val EXPECTED_PREDICTION_METERS = 106.241997f
        const val TOLERANCE_METERS = 0.01f
        const val TOLERANCE_FEATURE = 1e-4f
    }

    private var sensorCollector: IMUSensorCollector? = null
    private var model: CorrectionModel? = null
    private lateinit var logTextView: TextView
    private var liveWindowCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Setup UI dynamically for live monitoring
        val layout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(32, 48, 32, 32)
        }

        val titleView = TextView(this).apply {
            text = "SIH26168 Dead Reckoning Test Dashboard"
            textSize = 18f
            setTypeface(null, android.graphics.Typeface.BOLD)
            setTextColor(android.graphics.Color.DKGRAY)
        }
        layout.addView(titleView)

        val btnVerify = Button(this).apply {
            text = "Run Step 3a Verification Test"
            setOnClickListener { runVerificationTests() }
        }
        layout.addView(btnVerify)

        val btnStartLive = Button(this).apply {
            text = "Toggle Live IMU Sensor Logging"
            setOnClickListener { toggleLiveSensors() }
        }
        layout.addView(btnStartLive)

        val scrollView = ScrollView(this).apply {
            layoutParams = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT
            )
        }

        logTextView = TextView(this).apply {
            textSize = 12f
            typeface = android.graphics.Typeface.MONOSPACE
            setPadding(0, 16, 0, 0)
        }
        scrollView.addView(logTextView)
        layout.addView(scrollView)

        setContentView(layout)

        // Run automated verification tests on startup
        runVerificationTests()

        // Initialize model and sensor listener for live tests
        model = CorrectionModel(this)
        sensorCollector = IMUSensorCollector(this) { features, dtSeconds, sampleCount ->
            onLiveWindowReceived(features, dtSeconds, sampleCount)
        }
    }

    private fun appendLog(msg: String) {
        runOnUiThread {
            logTextView.append(msg + "\n")
        }
    }

    private fun runVerificationTests() {
        appendLog("==================================================")
        appendLog("STEP 3a: RUNNING VERIFICATION TEST ROUTINE")
        appendLog("==================================================")

        // 1. Test FeatureExtractor against hardcoded raw Window 0 sensor data
        try {
            val computedFeatures = FeatureExtractor.extractFeatures(
                TestWindow0Data.acc_x,
                TestWindow0Data.acc_y,
                TestWindow0Data.acc_z,
                TestWindow0Data.acc_lin_x,
                TestWindow0Data.acc_lin_y,
                TestWindow0Data.acc_lin_z,
                TestWindow0Data.gyro_x,
                TestWindow0Data.gyro_y,
                TestWindow0Data.gyro_z,
                TestWindow0Data.acc_veh_fwd,
                TestWindow0Data.gyro_veh_yaw_rate,
                TestWindow0Data.SEG_DUR_S,
                TestWindow0Data.N_SAMPLES
            )

            var maxFeatureDiff = 0.0f
            for (i in 0 until 58) {
                val diff = Math.abs(computedFeatures[i] - EXPECTED_FEATURES_WINDOW_0[i])
                if (diff > maxFeatureDiff) {
                    maxFeatureDiff = diff
                }
            }

            Log.i(TAG_FEATURE, "Feature Extraction Max Absolute Difference: $maxFeatureDiff")
            appendLog("Feature Extraction Verification:")
            appendLog("  Computed features: ${computedFeatures.size}")
            appendLog("  Max absolute diff: $maxFeatureDiff")

            if (maxFeatureDiff < TOLERANCE_FEATURE) {
                Log.i(TAG_FEATURE, ">>> FEATURE EXTRACTION TEST: PASSED <<<")
                appendLog("  Result: PASSED (diff < 1e-4)")
            } else {
                Log.e(TAG_FEATURE, ">>> FEATURE EXTRACTION TEST: FAILED <<<")
                appendLog("  Result: FAILED (exceeds tolerance)")
            }

            // 2. Feed computed features into ONNX CorrectionModel
            val correctionModel = CorrectionModel(this)
            val prediction = correctionModel.predict(computedFeatures)
            val predDiff = Math.abs(prediction - EXPECTED_PREDICTION_METERS)

            Log.i(TAG_ONNX, "ONNX Prediction from extracted features: $prediction meters")
            Log.i(TAG_ONNX, "Expected prediction: $EXPECTED_PREDICTION_METERS meters (diff=$predDiff)")
            appendLog("\nONNX Model Inference Verification:")
            appendLog("  Prediction: $prediction meters")
            appendLog("  Expected  : $EXPECTED_PREDICTION_METERS meters")
            appendLog("  Diff      : $predDiff meters")

            if (predDiff < TOLERANCE_METERS) {
                Log.i(TAG_ONNX, ">>> ONNX PREDICTION ON EXTRACTED FEATURES: PASSED <<<")
                appendLog("  Result: PASSED (tolerance < 0.01m)")
            } else {
                Log.e(TAG_ONNX, ">>> ONNX PREDICTION ON EXTRACTED FEATURES: FAILED <<<")
                appendLog("  Result: FAILED")
            }
            correctionModel.close()

            // 3. Step 3b: Test PhysicsDeadReckoning on Window 0
            appendLog("\n==================================================")
            appendLog("STEP 3b: PHYSICS DEAD RECKONING VERIFICATION")
            appendLog("==================================================")

            val pdr = PhysicsDeadReckoning()
            pdr.resetState(
                x = TestWindow0Data.INITIAL_X,
                y = TestWindow0Data.INITIAL_Y,
                heading = TestWindow0Data.INITIAL_HEADING_RAD,
                velocity = TestWindow0Data.INITIAL_VELOCITY
            )

            val pResult = pdr.processWindow(
                accLinX = TestWindow0Data.acc_lin_x,
                accLinY = TestWindow0Data.acc_lin_y,
                accLinZ = TestWindow0Data.acc_lin_z,
                gyroX = TestWindow0Data.gyro_x,
                gyroY = TestWindow0Data.gyro_y,
                gyroZ = TestWindow0Data.gyro_z,
                accVehFwd = TestWindow0Data.acc_veh_fwd,
                gyroVehYawRate = TestWindow0Data.gyro_veh_yaw_rate,
                dtArr = TestWindow0Data.dt_sec,
                segDurS = TestWindow0Data.SEG_DUR_S
            )

            val diffDs = Math.abs(pResult.deltaS_imu - TestWindow0Data.GT_STANDALONE_DS_IMU)
            val diffVel = Math.abs(pResult.newVelocity - TestWindow0Data.GT_NEW_VELOCITY)
            val diffHeadingDeg = Math.abs(pResult.newHeadingDeg - TestWindow0Data.GT_NEW_HEADING_DEG)

            appendLog("Physics DR Output vs Python Ground Truth:")
            appendLog("  deltaS_imu : Python=${TestWindow0Data.GT_STANDALONE_DS_IMU} m | Kotlin=${pResult.deltaS_imu} m | Diff=${String.format("%.7f", diffDs)} m")
            appendLog("  newVelocity: Python=${TestWindow0Data.GT_NEW_VELOCITY} m/s | Kotlin=${pResult.newVelocity} m/s | Diff=${String.format("%.7f", diffVel)} m/s")
            appendLog("  newHeading : Python=${TestWindow0Data.GT_NEW_HEADING_DEG}° | Kotlin=${String.format("%.5f", pResult.newHeadingDeg)}° | Diff=${String.format("%.7f", diffHeadingDeg)}°")
            appendLog("  Stationary : ${pResult.stationarySamples}/${TestWindow0Data.N_SAMPLES} samples")
            appendLog("  headingDir : Kotlin=${String.format("%.4f", pResult.headingDirDeg)}° (Python GT: ${TestWindow0Data.GT_HEADING_DIR_DEG}°)")

            val passDs = diffDs <= 0.1f
            val passVel = diffVel <= 0.05f
            val passHeading = diffHeadingDeg <= 0.5f

            if (passDs && passVel && passHeading) {
                Log.i(TAG_PHYSICS, ">>> PHYSICS DEAD RECKONING TEST: PASSED <<<")
                appendLog("  Result: ALL 3 CRITERIA PASSED (Δs<0.1m, v<0.05m/s, ψ<0.5°)")
            } else {
                Log.e(TAG_PHYSICS, ">>> PHYSICS DEAD RECKONING TEST: FAILED <<<")
                appendLog("  Result: FAILED")
            }

        } catch (e: Exception) {
            Log.e(TAG_FEATURE, "Verification error: ${e.message}", e)
            appendLog("ERROR: ${e.message}")
        }
    }

    private fun toggleLiveSensors() {
        val collector = sensorCollector ?: return
        collector.start()
        appendLog("\n[Live Sensors] Started listening at 10Hz. Move phone around...")
    }

    private fun onLiveWindowReceived(features: FloatArray, dtSeconds: Float, sampleCount: Int) {
        liveWindowCount++
        val pred = model?.predict(features) ?: 0.0f

        val logMsg = "Window #$liveWindowCount (dt=${String.format("%.2f", dtSeconds)}s, N=$sampleCount): " +
                "a_mean=${String.format("%.2f", features[0])}, a_std=${String.format("%.2f", features[1])}, " +
                "al_mag_mean=${String.format("%.2f", features[48])}, w_mag_mean=${String.format("%.3f", features[52])} " +
                "-> Predicted Δs_corr = ${String.format("%.2f", pred)}m"

        Log.i(TAG_FEATURE, logMsg)
        appendLog(logMsg)
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorCollector?.stop()
        model?.close()
    }
}
