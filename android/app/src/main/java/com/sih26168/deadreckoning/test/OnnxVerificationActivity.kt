package com.sih26168.deadreckoning.test

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.sih26168.deadreckoning.ml.CorrectionModel

class OnnxVerificationActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "SIH_ONNX_TEST"

        // Window 0 feature vector from S-Vw4 300s outage test (58 features)
        val TEST_FEATURES_WINDOW_0 = floatArrayOf(
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
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.i(TAG, "============================================================")
        Log.i(TAG, "STEP 2b: STARTING ON-DEVICE ONNX RUNTIME VERIFICATION")
        Log.i(TAG, "============================================================")

        try {
            val model = CorrectionModel(this)
            val prediction = model.predict(TEST_FEATURES_WINDOW_0)
            val diff = Math.abs(prediction - EXPECTED_PREDICTION_METERS)

            Log.i(TAG, "Predicted displacement correction: $prediction meters")
            Log.i(TAG, "Expected Python ONNX prediction  : $EXPECTED_PREDICTION_METERS meters")
            Log.i(TAG, "Absolute difference              : $diff meters")

            if (diff < TOLERANCE_METERS) {
                Log.i(TAG, ">>> ONNX RUNTIME ON-DEVICE TEST: PASSED (tolerance < 0.01m) <<<")
            } else {
                Log.e(TAG, ">>> ONNX RUNTIME ON-DEVICE TEST: FAILED (diff = $diff) <<<")
            }
            model.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error executing ONNX inference: ${e.message}", e)
        }
    }
}
