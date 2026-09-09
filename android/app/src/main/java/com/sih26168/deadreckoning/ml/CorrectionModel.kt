package com.sih26168.deadreckoning.ml

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.FloatBuffer

/**
 * Functional interface for dead-reckoning correction prediction.
 *
 * Returns null on inference failure -- NOT 0.0f. A silent 0.0f is
 * indistinguishable from a genuine "no correction needed" prediction, so a
 * broken model would look like it agrees with physics instead of visibly
 * failing (bug found and fixed this session; same pattern as
 * VelocityModel/IVelocityModel).
 */
fun interface ICorrectionModel {
    fun predict(features: FloatArray): Float?
}

/**
 * CorrectionModel: Wrapper around the Phase 9 C1 XGBoost ONNX model.
 *
 * Requirements:
 * 1. Takes 58 features as FloatArray.
 * 2. Explicitly constructs a 2D batch tensor of shape [1, 58] (batch size = 1),
 *    matching the FloatTensorType([None, 58]) expected by the converted XGBoost model.
 * 3. Gracefully handles model loading and execution errors -- by reporting
 *    them via the return value and [lastFailure], not by hiding them.
 */
class CorrectionModel : ICorrectionModel, AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val modelAssetPath = "c1_correction_model.onnx"

    var lastFailure: Exception? = null
        private set

    constructor(context: Context) {
        try {
            val assetManager = context.assets
            val modelBytes = assetManager.open(modelAssetPath).use { input: InputStream ->
                input.readBytes()
            }
            session = env.createSession(modelBytes, OrtSession.SessionOptions())
        } catch (e: Exception) {
            e.printStackTrace()
            lastFailure = e
            session = null
        }
    }

    constructor(modelBytes: ByteArray) {
        try {
            session = env.createSession(modelBytes, OrtSession.SessionOptions())
        } catch (e: Exception) {
            e.printStackTrace()
            lastFailure = e
            session = null
        }
    }

    /**
     * Run inference on 58 windowed IMU features.
     *
     * @param features FloatArray of size 58.
     * @return Displacement correction in meters (deltaS_corr), or null if
     * inference failed -- check [lastFailure] for the cause.
     */
    override fun predict(features: FloatArray): Float? {
        val currentSession = session ?: run {
            lastFailure = IllegalStateException("CorrectionModel: ONNX session is not initialized.")
            return null
        }

        require(features.size == 58) {
            "CorrectionModel expects exactly 58 features, but received ${features.size}"
        }

        return try {
            // Requirement 1: Explicitly specify shape [1, 58] for 2D batch tensor
            val shape = longArrayOf(1L, 58L)
            val buffer = FloatBuffer.wrap(features)
            val inputTensor = OnnxTensor.createTensor(env, buffer, shape)

            val inputName = currentSession.inputNames.iterator().next()
            val output = currentSession.run(mapOf(inputName to inputTensor))

            val resultValue = output.use { result ->
                val rawOutput = result[0].value
                when (rawOutput) {
                    is Array<*> -> {
                        // 2D output array: Array<FloatArray> or similar
                        val firstRow = rawOutput[0]
                        when (firstRow) {
                            is FloatArray -> firstRow[0]
                            is Number -> firstRow.toFloat()
                            else -> null
                        }
                    }
                    is FloatArray -> rawOutput[0]
                    else -> null
                }
            }
            inputTensor.close()
            lastFailure = null
            resultValue
        } catch (e: Exception) {
            e.printStackTrace()
            lastFailure = e
            null
        }
    }

    override fun close() {
        try {
            session?.close()
            session = null
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
