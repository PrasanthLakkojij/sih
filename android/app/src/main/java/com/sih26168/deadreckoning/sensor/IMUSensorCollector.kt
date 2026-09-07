package com.sih26168.deadreckoning.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.sih26168.deadreckoning.ml.FeatureExtractor
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * IMUSensorCollector:
 * 1. Collects live phone sensor streams (Accelerometer, Linear Acceleration, Gyroscope, Gravity).
 * 2. Buffers readings in a rolling ~9.0-second window downsampled/resampled to nominal 10Hz.
 * 3. Transforms phone body linear acceleration and gyro into vehicle coordinate frame
 *    (using vertical projection via gravity vector).
 * 4. Extracts the 58 features using FeatureExtractor and delivers them to a callback.
 */
class IMUSensorCollector(
    context: Context,
    var onWindowSamplesReady: ((samples: List<Sample>) -> Unit)? = null,
    private val onWindowReady: ((features: FloatArray, dtSeconds: Float, sampleCount: Int) -> Unit)? = null
) : SensorEventListener {

    constructor(
        context: Context,
        onWindowReady: (features: FloatArray, dtSeconds: Float, sampleCount: Int) -> Unit
    ) : this(context, onWindowSamplesReady = null, onWindowReady = onWindowReady)

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val linAccSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private val gyroSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val gravSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)

    // Current latest raw sample readings
    @Volatile private var latestAccX = 0f
    @Volatile private var latestAccY = 0f
    @Volatile private var latestAccZ = 9.81f

    @Volatile private var latestLinX = 0f
    @Volatile private var latestLinY = 0f
    @Volatile private var latestLinZ = 0f

    @Volatile private var latestGyroX = 0f
    @Volatile private var latestGyroY = 0f
    @Volatile private var latestGyroZ = 0f

    @Volatile private var latestGravX = 0f
    @Volatile private var latestGravY = 0f
    @Volatile private var latestGravZ = -9.81f

    // Buffer of synchronized 10Hz samples
    data class Sample(
        val accX: Float, val accY: Float, val accZ: Float,
        val accLinX: Float, val accLinY: Float, val accLinZ: Float,
        val gyroX: Float, val gyroY: Float, val gyroZ: Float,
        val accVehFwd: Float, val gyroVehYawRate: Float,
        val timestampNs: Long
    )

    private val sampleBuffer = CopyOnWriteArrayList<Sample>()
    private var isCollecting = false
    private var lastSampleTimeNs: Long = 0L

    // Sampling period: 100ms = 10Hz (100_000_000 ns)
    private val targetSampleIntervalNs = 100_000_000L
    // Window length: ~90 samples (~9 seconds)
    private val windowSamples = 91

    // Mounting alignment angle theta (default 0 for flat dashboard mount)
    var mountingYawRad: Float = 0.0f

    fun start() {
        if (isCollecting) return
        sampleBuffer.clear()
        lastSampleTimeNs = 0L
        isCollecting = true

        val rate = SensorManager.SENSOR_DELAY_GAME
        accSensor?.let { sensorManager.registerListener(this, it, rate) }
        linAccSensor?.let { sensorManager.registerListener(this, it, rate) }
        gyroSensor?.let { sensorManager.registerListener(this, it, rate) }
        gravSensor?.let { sensorManager.registerListener(this, it, rate) }
    }

    fun stop() {
        if (!isCollecting) return
        isCollecting = false
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || !isCollecting) return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAccX = event.values[0]
                latestAccY = event.values[1]
                latestAccZ = event.values[2]
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                latestLinX = event.values[0]
                latestLinY = event.values[1]
                latestLinZ = event.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                latestGyroX = event.values[0]
                latestGyroY = event.values[1]
                latestGyroZ = event.values[2]
            }
            Sensor.TYPE_GRAVITY -> {
                latestGravX = event.values[0]
                latestGravY = event.values[1]
                latestGravZ = event.values[2]
            }
        }

        val nowNs = event.timestamp
        if (lastSampleTimeNs == 0L || (nowNs - lastSampleTimeNs) >= targetSampleIntervalNs) {
            lastSampleTimeNs = nowNs

            // Compute vehicle frame projection:
            // 1. Gravity unit up vector
            val gNorm = sqrt(latestGravX * latestGravX + latestGravY * latestGravY + latestGravZ * latestGravZ)
            val upX = if (gNorm > 1e-4f) -latestGravX / gNorm else 0f
            val upY = if (gNorm > 1e-4f) -latestGravY / gNorm else 0f
            val upZ = if (gNorm > 1e-4f) -latestGravZ / gNorm else 1f

            // 2. Vertical linear acceleration
            val aVert = latestLinX * upX + latestLinY * upY + latestLinZ * upZ

            // 3. Horizontal linear acceleration
            val axHoriz = latestLinX - aVert * upX
            val ayHoriz = latestLinY - aVert * upY

            // 4. Vehicle forward & lateral acceleration
            val aFwd = axHoriz * cos(mountingYawRad) + ayHoriz * sin(mountingYawRad)

            // 5. Yaw rate around vertical axis
            val gyroYawRate = latestGyroX * upX + latestGyroY * upY + latestGyroZ * upZ

            val sample = Sample(
                accX = latestAccX, accY = latestAccY, accZ = latestAccZ,
                accLinX = latestLinX, accLinY = latestLinY, accLinZ = latestLinZ,
                gyroX = latestGyroX, gyroY = latestGyroY, gyroZ = latestGyroZ,
                accVehFwd = aFwd, gyroVehYawRate = gyroYawRate,
                timestampNs = nowNs
            )
            sampleBuffer.add(sample)

            // Check if buffer reached ~9.0 seconds (91 samples)
            if (sampleBuffer.size >= windowSamples) {
                val window = ArrayList(sampleBuffer.subList(0, windowSamples))
                // Slide window by 90 samples
                sampleBuffer.removeAll(window)

                onWindowSamplesReady?.invoke(window)

                if (onWindowReady != null) {
                    val n = window.size
                    val axArr = FloatArray(n) { window[it].accX }
                    val ayArr = FloatArray(n) { window[it].accY }
                    val azArr = FloatArray(n) { window[it].accZ }
                    val lxArr = FloatArray(n) { window[it].accLinX }
                    val lyArr = FloatArray(n) { window[it].accLinY }
                    val lzArr = FloatArray(n) { window[it].accLinZ }
                    val gxArr = FloatArray(n) { window[it].gyroX }
                    val gyArr = FloatArray(n) { window[it].gyroY }
                    val gzArr = FloatArray(n) { window[it].gyroZ }
                    val fwdArr = FloatArray(n) { window[it].accVehFwd }
                    val yawArr = FloatArray(n) { window[it].gyroVehYawRate }

                    val dtTotal = (window.last().timestampNs - window.first().timestampNs) / 1_000_000_000f

                    val features = FeatureExtractor.extractFeatures(
                        axArr, ayArr, azArr,
                        lxArr, lyArr, lzArr,
                        gxArr, gyArr, gzArr,
                        fwdArr, yawArr,
                        dtTotal, n
                    )

                    onWindowReady.invoke(features, dtTotal, n)
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
