package com.sih26168.deadreckoning.engine

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.PI

/**
 * ExtendedKalmanFilter (VehicleEKF):
 *
 * Direct Kotlin port of the Python EKF validated in phase10_fusion_engine.py,
 * including every fix found during that session's diagnosis work:
 *  - Non-Holonomic Constraint (NHC) pseudo-measurement (updateNhc)
 *  - Direct heading measurement update (updateHeading) for map-matching feedback
 *  - ZUPT measurement update (updateZupt) -- caller must apply the ZUPT-vs-ML
 *    veto gate found this session (see PositionEstimator's outage loop): do NOT
 *    call updateZupt if a trusted speed measurement says the vehicle is moving,
 *    even when the raw ZUPT mask says stationary. Both zupt v1 and v2 masks were
 *    measured to false-positive during smooth constant-speed cruise.
 *  - ML forward-speed measurement update (updateMlSpeed)
 *  - GNSS position+velocity measurement update (updateGnss)
 *
 * State vector x = [px, py, vx, vy, psi, b_ax, b_ay, b_w] (dim=8):
 *  - px, py : Local East, North position (meters)
 *  - vx, vy : Local East, North velocity (m/s)
 *  - psi    : Navigation heading/azimuth (radians, 0=North, pi/2=East, clockwise)
 *  - b_ax   : Accelerometer bias in vehicle forward axis (m/s^2)
 *  - b_ay   : Accelerometer bias in vehicle lateral axis (m/s^2)
 *  - b_w    : Gyroscope bias around vertical axis (rad/s)
 *
 * Matrix ops are hand-rolled (8x8 max, and only up to 4x4 needs inverting for
 * GNSS updates) rather than pulling in a linear-algebra dependency.
 */
class ExtendedKalmanFilter(
    initPos: DoubleArray,
    initVel: DoubleArray,
    initHeading: Double,
    private val sigmaAProc: Double = 4.0,
    private val sigmaWProc: Double = Math.toRadians(2.0),
    private val sigmaBaProc: Double = 1e-3,
    private val sigmaBwProc: Double = 1e-4
) {
    val x = DoubleArray(8)
    val P = Array(8) { DoubleArray(8) }

    companion object {
        private const val N = 8
        private const val TWO_PI = 2.0 * PI
    }

    init {
        x[0] = initPos[0]; x[1] = initPos[1]
        x[2] = initVel[0]; x[3] = initVel[1]
        x[4] = initHeading
        // x[5..7] (biases) default 0.0; caller seeds via setBias() if calibrated

        val initStd = doubleArrayOf(
            5.0, 5.0,                          // px, py (5m)
            0.5, 0.5,                          // vx, vy (0.5 m/s)
            Math.toRadians(5.0),               // psi (5 deg)
            0.1, 0.1,                          // b_ax, b_ay (0.1 m/s^2)
            Math.toRadians(0.5)                // b_w (0.5 deg/s)
        )
        for (i in 0 until N) P[i][i] = initStd[i] * initStd[i]
    }

    /** Seed a bias state (5=b_ax, 6=b_ay, 7=b_w) with a calibrated value and tighten its uncertainty. */
    fun seedBias(index: Int, value: Double, tightenedStdRad: Double? = null) {
        x[index] = value
        if (tightenedStdRad != null) P[index][index] = tightenedStdRad * tightenedStdRad
    }

    private fun normalizeAngle(rad: Double): Double {
        var a = (rad + PI) % TWO_PI - PI
        if (a < -PI) a += TWO_PI
        return a
    }

    /**
     * Propagate state and covariance over dt using vehicle-frame IMU inputs.
     * Identical math to VehicleEKF.predict() in phase10_fusion_engine.py.
     */
    fun predict(aFwdRaw: Double, aLatRaw: Double, gyroWRaw: Double, dt: Double) {
        val px = x[0]; val py = x[1]; val vx = x[2]; val vy = x[3]
        val psi = x[4]; val bAx = x[5]; val bAy = x[6]; val bW = x[7]

        val aFwd = aFwdRaw - bAx
        val aLat = aLatRaw - bAy
        val w = gyroWRaw - bW

        val sinP = sin(psi)
        val cosP = cos(psi)

        val ae = aFwd * sinP - aLat * cosP
        val an = aFwd * cosP + aLat * sinP

        x[0] = px + vx * dt + 0.5 * ae * dt * dt
        x[1] = py + vy * dt + 0.5 * an * dt * dt
        x[2] = vx + ae * dt
        x[3] = vy + an * dt
        x[4] = normalizeAngle(psi + w * dt)
        // biases: random walk, mean unchanged

        // State transition Jacobian F = df/dx (8x8), identity + populated terms
        val F = Array(N) { i -> DoubleArray(N) { j -> if (i == j) 1.0 else 0.0 } }
        F[0][2] = dt
        F[1][3] = dt
        F[0][4] = 0.5 * dt * dt * (aFwd * cosP + aLat * sinP)
        F[1][4] = 0.5 * dt * dt * (-aFwd * sinP + aLat * cosP)
        F[2][4] = dt * (aFwd * cosP + aLat * sinP)
        F[3][4] = dt * (-aFwd * sinP + aLat * cosP)
        F[0][5] = -0.5 * dt * dt * sinP
        F[0][6] = 0.5 * dt * dt * cosP
        F[1][5] = -0.5 * dt * dt * cosP
        F[1][6] = -0.5 * dt * dt * sinP
        F[2][5] = -dt * sinP
        F[2][6] = dt * cosP
        F[3][5] = -dt * cosP
        F[3][6] = -dt * sinP
        F[4][7] = -dt

        val qPos = (0.5 * sigmaAProc * dt * dt).let { it * it }
        val qVel = (sigmaAProc * dt).let { it * it }
        val qPsi = (sigmaWProc * dt).let { it * it }
        val qBa = (sigmaBaProc * dt).let { it * it }
        val qBw = (sigmaBwProc * dt).let { it * it }
        val Q = Array(N) { DoubleArray(N) }
        Q[0][0] = qPos; Q[1][1] = qPos
        Q[2][2] = qVel; Q[3][3] = qVel
        Q[4][4] = qPsi
        Q[5][5] = qBa; Q[6][6] = qBa
        Q[7][7] = qBw

        // P = F P F^T + Q
        val FP = matMul(F, P)
        val FPFt = matMul(FP, transpose(F))
        for (i in 0 until N) for (j in 0 until N) P[i][j] = FPFt[i][j] + Q[i][j]
    }

    /** Non-Holonomic Constraint: vehicle lateral velocity (vehicle frame) ~= 0. */
    fun updateNhc(rLat: Double = 0.2) {
        val psi = x[4]; val vx = x[2]; val vy = x[3]
        val sinP = sin(psi); val cosP = cos(psi)
        val zPred = -vx * cosP + vy * sinP
        val H = DoubleArray(N)
        H[2] = -cosP; H[3] = sinP; H[4] = vx * sinP + vy * cosP
        kfUpdateScalar(H, 0.0 - zPred, rLat * rLat)
    }

    /** Direct heading measurement (e.g. confident map-match onto a known road). */
    fun updateHeading(psiMeas: Double, rHeading: Double) {
        val H = DoubleArray(N); H[4] = 1.0
        val y = normalizeAngle(psiMeas - x[4])
        kfUpdateScalar(H, y, rHeading * rHeading)
    }

    /** Zero-Velocity Update: vx = 0, vy = 0. Caller applies the ZUPT-vs-ML veto gate. */
    fun updateZupt(rZupt: Double = 0.05) {
        val H1 = DoubleArray(N); H1[2] = 1.0
        val H2 = DoubleArray(N); H2[3] = 1.0
        val H = arrayOf(H1, H2)
        val y = doubleArrayOf(0.0 - x[2], 0.0 - x[3])
        val R = arrayOf(doubleArrayOf(rZupt * rZupt, 0.0), doubleArrayOf(0.0, rZupt * rZupt))
        kfUpdate(H, y, R)
    }

    /** ML forward-speed measurement: h(x) = vx*sin(psi) + vy*cos(psi). */
    fun updateMlSpeed(vPredFwd: Double, rSpeed: Double) {
        val psi = x[4]; val vx = x[2]; val vy = x[3]
        val sinP = sin(psi); val cosP = cos(psi)
        val zPred = vx * sinP + vy * cosP
        val H = DoubleArray(N)
        H[2] = sinP; H[3] = cosP; H[4] = vx * cosP - vy * sinP
        kfUpdateScalar(H, vPredFwd - zPred, rSpeed * rSpeed)
    }

    /** GNSS position + velocity measurement: z = [e, n, ve, vn]. */
    fun updateGnss(posGnss: DoubleArray, velGnss: DoubleArray, rPos: Double = 3.0, rVel: Double = 0.3) {
        val H = Array(4) { DoubleArray(N) }
        H[0][0] = 1.0; H[1][1] = 1.0; H[2][2] = 1.0; H[3][3] = 1.0
        val zPred = doubleArrayOf(x[0], x[1], x[2], x[3])
        val z = doubleArrayOf(posGnss[0], posGnss[1], velGnss[0], velGnss[1])
        val y = DoubleArray(4) { z[it] - zPred[it] }
        val R = Array(4) { i -> DoubleArray(4) { j -> if (i == j) (if (i < 2) rPos * rPos else rVel * rVel) else 0.0 } }
        kfUpdate(H, y, R)
    }

    // ---- internal: scalar (1-dim) update convenience wrapper. Caller must
    // pass an already-correct residual `y` -- e.g. wrap it through
    // normalizeAngle() first if it's an angle (see updateHeading); a
    // velocity residual (NHC, ML speed) must NOT be wrapped. ----
    private fun kfUpdateScalar(H: DoubleArray, y: Double, rVar: Double) {
        kfUpdate(arrayOf(H), doubleArrayOf(y), arrayOf(doubleArrayOf(rVar)))
    }

    /** Standard Kalman update with Joseph-form covariance update for numerical stability. */
    private fun kfUpdate(H: Array<DoubleArray>, y: DoubleArray, R: Array<DoubleArray>) {
        val m = H.size
        val Ht = transpose(H)                      // N x m
        val PHt = matMul(P, Ht)                     // N x m
        val HP = matMul(H, P)                       // m x N
        val HPHt = matMul(HP, Ht)                   // m x m
        val S = Array(m) { i -> DoubleArray(m) { j -> HPHt[i][j] + R[i][j] } }
        val Sinv = invert(S)
        val K = matMul(PHt, Sinv)                   // N x m

        val dx = matVec(K, y)                       // N
        for (i in 0 until N) x[i] += dx[i]
        x[4] = normalizeAngle(x[4])

        // Joseph form: P = (I-KH) P (I-KH)^T + K R K^T
        val KH = matMul(K, H)                        // N x N
        val IKH = Array(N) { i -> DoubleArray(N) { j -> (if (i == j) 1.0 else 0.0) - KH[i][j] } }
        val term1 = matMul(matMul(IKH, P), transpose(IKH))
        val KR = matMul(K, R)
        val term2 = matMul(KR, transpose(K))
        for (i in 0 until N) for (j in 0 until N) P[i][j] = term1[i][j] + term2[i][j]
    }

    // ---- minimal matrix helpers ----
    private fun matMul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val rows = a.size; val inner = b.size; val cols = b[0].size
        val out = Array(rows) { DoubleArray(cols) }
        for (i in 0 until rows) for (k in 0 until inner) {
            val aik = a[i][k]
            if (aik == 0.0) continue
            for (j in 0 until cols) out[i][j] += aik * b[k][j]
        }
        return out
    }

    private fun matVec(a: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        val out = DoubleArray(a.size)
        for (i in a.indices) {
            var s = 0.0
            for (j in v.indices) s += a[i][j] * v[j]
            out[i] = s
        }
        return out
    }

    private fun transpose(a: Array<DoubleArray>): Array<DoubleArray> {
        val rows = a.size; val cols = a[0].size
        return Array(cols) { j -> DoubleArray(rows) { i -> a[i][j] } }
    }

    /** Gauss-Jordan inverse for small (<=4x4) matrices -- all that's needed here. */
    private fun invert(m: Array<DoubleArray>): Array<DoubleArray> {
        val n = m.size
        val aug = Array(n) { i -> DoubleArray(2 * n).also { row ->
            for (j in 0 until n) row[j] = m[i][j]
            row[n + i] = 1.0
        } }
        for (col in 0 until n) {
            var pivotRow = col
            var maxAbs = kotlin.math.abs(aug[col][col])
            for (r in col + 1 until n) {
                val v = kotlin.math.abs(aug[r][col])
                if (v > maxAbs) { maxAbs = v; pivotRow = r }
            }
            if (pivotRow != col) { val tmp = aug[col]; aug[col] = aug[pivotRow]; aug[pivotRow] = tmp }
            val pivot = aug[col][col]
            val safePivot = if (kotlin.math.abs(pivot) < 1e-12) 1e-12 else pivot
            for (j in 0 until 2 * n) aug[col][j] /= safePivot
            for (r in 0 until n) {
                if (r == col) continue
                val factor = aug[r][col]
                if (factor == 0.0) continue
                for (j in 0 until 2 * n) aug[r][j] -= factor * aug[col][j]
            }
        }
        return Array(n) { i -> DoubleArray(n) { j -> aug[i][n + j] } }
    }
}
