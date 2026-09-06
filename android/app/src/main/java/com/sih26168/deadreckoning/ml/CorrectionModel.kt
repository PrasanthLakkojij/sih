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
 */
fun interface ICorrectionModel {
    fun predict(features: FloatArray): Float
}

/**
 * CorrectionModel: Wrapper around the Phase 9 C1 XGBoost ONNX model.
 *
 * Requirements:
 * 1. Takes 58 features as FloatArray.
 * 2. Explicitly constructs a 2D batch tensor of shape [1, 58] (batch size = 1),
 *    matching the FloatTensorType([None, 58]) expected by the converted XGBoost model.
 * 3. Gracefully handles model loading and execution errors.
 */
class CorrectionModel : ICorrectionModel, AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val modelAssetPath = "c1_correction_model.onnx"

    constructor(context: Context) {
        try {
            val assetManager = context.assets
            val modelBytes = assetManager.open(modelAssetPath).use { input: InputStream ->
                input.readBytes()
            }
            session = env.createSession(modelBytes, OrtSession.SessionOptions())
        } catch (e: Exception) {
            e.printStackTrace()
            session = null
        }
    }

    constructor(modelBytes: ByteArray) {
        try {
            session = env.createSession(modelBytes, OrtSession.SessionOptions())
        } catch (e: Exception) {
            e.printStackTrace()
            session = null
        }
    }

    /**
     * Run inference on 58 windowed IMU features.
     *
     * @param features FloatArray of size 58.
     * @return Displacement correction in meters (Δs_corr). Returns 0.0f if inference fails.
     */
    override fun predict(features: FloatArray): Float {
        val currentSession = session ?: run {
            System.err.println("CorrectionModel: ONNX session is not initialized.")
            return 0.0f
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
                            else -> 0.0f
                        }
                    }
                    is FloatArray -> rawOutput[0]
                    else -> 0.0f
                }
            }
            inputTensor.close()
            resultValue
        } catch (e: Exception) {
            e.printStackTrace()
            0.0f
        }
    }

    override fun close() {
        try {
            session?.close()
            env.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
