package com.sih26168.deadreckoning.ml

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.InputStream
import java.nio.FloatBuffer

/**
 * Functional interface for instantaneous forward-speed prediction, mirroring
 * ICorrectionModel's pattern (CorrectionModel.kt) so callers can mock the
 * model in JVM unit tests without loading real ONNX Runtime native libs.
 * Returns null on failure -- see VelocityModel's own doc for why 0.0f would
 * be actively dangerous here.
 */
fun interface IVelocityModel {
    fun predict(features: FloatArray): Float?
}

/**
 * VelocityModel: Wrapper around the Phase 7 XGBoost instantaneous forward-speed
 * ONNX model (56 features, ~2s/20-sample window).
 *
 * This is the continuous measurement source the Python EKF (phase10_fusion_engine.py)
 * was validated against -- a different model from CorrectionModel.kt's Phase 9 C1
 * displacement-correction model (58 features, ~9s window). Both exist on-device;
 * EKF fusion uses this one for its updateMlSpeed() measurement, matching the
 * Python architecture that produced the ~74-78% drift result.
 *
 * Unlike CorrectionModel.kt, predict() returns null on failure instead of
 * silently defaulting to a magic value -- a silent 0.0f here would look like
 * a valid "stationary" measurement and get fed straight into the EKF as if it
 * were real data. Callers must handle the null case explicitly (e.g. skip the
 * ML-speed update for that cycle).
 */
class VelocityModel : IVelocityModel, AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val modelAssetPath = "c1_velocity_model.onnx"

    /** Measured signed bias of this model's raw predictions (v_ml - v_true),
     * averaged across 11,570 validated LOTO instants. Applied at prediction
     * time rather than baked into the exported weights, so it stays visible
     * and adjustable. See phase10_fusion_engine.py's V_ML_BIAS_CORRECTION. */
    companion object {
        const val BIAS_CORRECTION_MS = 0.957f
        private const val FEATURE_COUNT = 56
    }

    var lastFailure: Exception? = null
        private set

    constructor(context: Context) {
        try {
            val modelBytes = context.assets.open(modelAssetPath).use { input: InputStream ->
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
     * Run inference on 56 windowed IMU features, applying the measured bias
     * correction. Returns null (not a default value) if inference fails --
     * check [lastFailure] for the cause. Result is clamped to >= 0.
     */
    override fun predict(features: FloatArray): Float? {
        val currentSession = session ?: run {
            lastFailure = IllegalStateException("VelocityModel: ONNX session is not initialized.")
            return null
        }
        require(features.size == FEATURE_COUNT) {
            "VelocityModel expects exactly $FEATURE_COUNT features, but received ${features.size}"
        }

        return try {
            val shape = longArrayOf(1L, FEATURE_COUNT.toLong())
            val buffer = FloatBuffer.wrap(features)
            val inputTensor = OnnxTensor.createTensor(env, buffer, shape)

            val inputName = currentSession.inputNames.iterator().next()
            val output = currentSession.run(mapOf(inputName to inputTensor))

            val rawValue = output.use { result ->
                when (val rawOutput = result[0].value) {
                    is Array<*> -> {
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
            rawValue?.let { maxOf(0.0f, it + BIAS_CORRECTION_MS) }
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
