package com.sih26168.deadreckoning.util

import kotlin.math.hypot

/**
 * Smooths the *displayed* raw-GPS marker position only. Consumer GPS chipsets
 * report a new fix that random-walks within roughly its own accuracy radius
 * even when the phone is genuinely stationary (multipath, satellite geometry
 * changes) -- that's normal GPS behavior, not a position-estimation bug. This
 * class only steadies what's drawn on screen; it must never be used for
 * anything that feeds position-estimation math (origin, PositionEstimator,
 * EkfPositionEstimator all keep consuming the raw fix directly).
 */
object GpsDisplaySmoother {

    // Below this radius from the current anchor, treat movement as pure
    // jitter and hold the anchor exactly -- never move it. Floor of 4m so a
    // too-optimistic reported accuracy (common near buildings/campus
    // multipath) doesn't shrink the dead-band to nothing.
    private const val MIN_DEADBAND_M = 4.0

    // Fallback when a fix reports zero/negative accuracy (shouldn't normally
    // happen, but Location.getAccuracy() carries no hard guarantee).
    private const val DEFAULT_ACCURACY_M = 5.0

    // A candidate outside the dead-band must be confirmed by several more
    // mutually-close fixes before the anchor actually moves -- see State doc.
    // Scaled off the dead-band (not a fixed meter value): a fixed radius
    // either blocks real movement once fixes land farther apart than it (a
    // faster walk/drive, or a sparser fix interval), or is too loose for a
    // tight dead-band.
    private const val CONFIRM_RADIUS_MULTIPLIER = 3.0
    // Raised from 2 -> 4: a single bad fix is filtered by 2, but a genuinely
    // *drifting* raw GPS solution (correlated multipath error across several
    // consecutive fixes, not one outlier -- confirmed via real device logcat
    // showing ~70m of raw-fix drift over ~15s while stationary on a campus
    // near buildings/rail lines) still confirms itself under a 2-fix
    // requirement, because each drifting fix legitimately agrees with the
    // last one. Requiring 4 consecutive agreeing fixes (~3-4s) makes a
    // committed move much more likely to be real, at the cost of the
    // displayed dot lagging further behind genuine fast movement. This is a
    // deliberate accuracy/responsiveness tradeoff, not a bug fix -- raw GPS
    // quality at this location is the underlying limitation.
    private const val REQUIRED_CONFIRMATIONS = 4

    data class Point(val east: Double, val north: Double)

    /**
     * @param anchor the currently displayed/settled position.
     * @param pendingCandidate the most recent out-of-deadband fix, used to
     *   check the next fix agrees with the trend (not just with the anchor).
     *   Null when nothing is pending.
     * @param pendingCount how many consecutive mutually-agreeing fixes seen
     *   so far, including the one that set pendingCandidate.
     * @param pendingSumEast/pendingSumNorth running sum of the pending
     *   streak's raw fixes, so a confirmed move commits to their *average*
     *   rather than just the latest sample -- extra stability against any
     *   one noisy fix inside an otherwise-real move.
     */
    data class State(
        val anchor: Point,
        val pendingCandidate: Point? = null,
        val pendingCount: Int = 0,
        val pendingSumEast: Double = 0.0,
        val pendingSumNorth: Double = 0.0
    )

    /**
     * Hysteresis / sticky-anchor filter with multi-fix confirmation.
     *
     * `anchor` only moves once a fix lands outside its dead-band AND
     * REQUIRED_CONFIRMATIONS consecutive fixes all mutually agree (each
     * within CONFIRM_RADIUS_MULTIPLIER x deadband of the previous one). A
     * single bad fix -- or even a short run of correlated GPS-multipath
     * drift shorter than the requirement -- looks identical to the start of
     * real movement from too few samples; requiring several agreeing
     * samples before committing filters that out while still accepting
     * sustained genuine movement (with a delay of ~REQUIRED_CONFIRMATIONS
     * fixes, by design -- see class doc on the tradeoff).
     *
     * @param rawEast raw GPS fix, local ENU meters (East)
     * @param rawNorth raw GPS fix, local ENU meters (North)
     * @param accuracyM reported horizontal accuracy of the raw fix, meters
     * @param state previous smoother state, or null on the first fix
     * @return updated state; read `.anchor` for the position to display
     */
    fun update(rawEast: Double, rawNorth: Double, accuracyM: Double, state: State?): State {
        val raw = Point(rawEast, rawNorth)
        if (state == null) return State(raw)

        val safeAccuracy = if (accuracyM > 0.0) accuracyM else DEFAULT_ACCURACY_M
        val deadband = maxOf(safeAccuracy, MIN_DEADBAND_M)
        val distFromAnchor = hypot(rawEast - state.anchor.east, rawNorth - state.anchor.north)

        if (distFromAnchor < deadband) {
            // Back within the settled zone -- any pending streak didn't
            // hold up, drop it.
            return State(state.anchor)
        }

        val candidate = state.pendingCandidate
        if (candidate != null) {
            val distFromCandidate = hypot(rawEast - candidate.east, rawNorth - candidate.north)
            if (distFromCandidate < deadband * CONFIRM_RADIUS_MULTIPLIER) {
                val count = state.pendingCount + 1
                val sumEast = state.pendingSumEast + rawEast
                val sumNorth = state.pendingSumNorth + rawNorth
                return if (count >= REQUIRED_CONFIRMATIONS) {
                    // Confirmed -- commit to the streak's average, not just
                    // this last sample.
                    State(Point(sumEast / count, sumNorth / count))
                } else {
                    State(state.anchor, raw, count, sumEast, sumNorth)
                }
            }
        }
        // First outlier, or one that doesn't agree with the trend so far --
        // start a fresh pending streak, anchor unchanged.
        return State(state.anchor, raw, 1, rawEast, rawNorth)
    }
}
