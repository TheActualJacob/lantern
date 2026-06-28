package com.lantern.recorder.scanning

import kotlin.math.sqrt

/**
 * Detects when the scanned object has been **disturbed** (picked up to be flipped) and
 * when it has **settled** again into a new resting pose — purely from the world-space
 * point the camera's center ray hits each tick, i.e. `cameraPos + forward * centerDepth`.
 *
 * The physical insight (no ML needed): while the object sits still and the user orbits,
 * that center point stays near the object's locked centroid even as the camera moves.
 * Lifting/flipping the object (or a hand reaching in) makes the point diverge from the
 * centroid persistently, or drops valid depth entirely. Setting the object back down makes
 * the point hold steady at a *new* location. That disturb → settle pattern is exactly the
 * signal the two-pass "flip the object" flow needs, and it doubles as an "object moved
 * during the scan" guard.
 *
 * Pure logic with hysteresis + dwell timers so momentary depth dropouts or fast camera
 * moves don't cause flicker. Fed on the GL thread; holds no Android dependencies.
 */
class FlipDetector {

    /** Result of one [update] tick. */
    data class Reading(
        val disturbed: Boolean,
        val settled: Boolean,
        val settleX: Float,
        val settleY: Float,
        val settleZ: Float,
        /** Distance from the reference point, or -1 when unavailable. */
        val deviation: Float,
    )

    private var refX = 0f
    private var refY = 0f
    private var refZ = 0f
    private var hasRef = false

    private var disturbed = false
    private var farSince = 0L
    private var farPending = false
    private var invalidSince = 0L

    // Running short mean of recent valid points while waiting for the object to settle.
    private var sX = 0f
    private var sY = 0f
    private var sZ = 0f
    private var sCount = 0
    private var settleSince = 0L
    private var settled = false

    /** Locks the reference point (the object's current centroid) to compare against. */
    fun setReference(x: Float, y: Float, z: Float) {
        refX = x; refY = y; refZ = z
        hasRef = true
        resetTransient()
    }

    /** Fully clears state (between sessions / when leaving the flip flow). */
    fun clear() {
        hasRef = false
        resetTransient()
    }

    private fun resetTransient() {
        disturbed = false
        farPending = false
        farSince = 0L
        invalidSince = 0L
        sCount = 0
        settleSince = 0L
        settled = false
    }

    /**
     * Feeds one tick.
     *
     * @param hasPoint whether a valid in-range center depth produced a world point
     * @param x,y,z the back-projected center world point (ignored when [hasPoint] is false)
     * @param nowMs monotonic time in milliseconds
     */
    fun update(hasPoint: Boolean, x: Float, y: Float, z: Float, nowMs: Long): Reading {
        val deviation = if (hasRef && hasPoint) dist(x, y, z, refX, refY, refZ) else -1f

        // --- Disturbance: point far from the reference for a sustained time, or depth
        //     invalid for a sustained time (a hand/occlusion over the object). ---
        if (hasPoint) {
            invalidSince = 0L
            val far = hasRef && deviation > DISTURB_M
            if (far) {
                if (!farPending) { farPending = true; farSince = nowMs }
                if (nowMs - farSince >= DISTURB_HOLD_MS) disturbed = true
            } else {
                farPending = false
            }
        } else {
            farPending = false
            if (invalidSince == 0L) invalidSince = nowMs
            if (nowMs - invalidSince >= INVALID_HOLD_MS) disturbed = true
        }

        // --- Settling: only meaningful once disturbed. Track a short running mean of
        //     recent valid points; once they hold within a small radius for the dwell
        //     time, the object has come to rest at a new pose. ---
        if (disturbed) {
            if (!hasPoint) {
                // Lost depth while waiting to settle — drop the candidate.
                sCount = 0
                settled = false
            } else if (sCount == 0) {
                sX = x; sY = y; sZ = z; sCount = 1
                settleSince = nowMs
                settled = false
            } else {
                val d = dist(x, y, z, sX, sY, sZ)
                if (d <= SETTLE_RADIUS_M) {
                    val n = sCount
                    sX = (sX * n + x) / (n + 1)
                    sY = (sY * n + y) / (n + 1)
                    sZ = (sZ * n + z) / (n + 1)
                    sCount = n + 1
                    if (nowMs - settleSince >= SETTLE_HOLD_MS) settled = true
                } else {
                    // Moved again — restart the settle candidate at the new point.
                    sX = x; sY = y; sZ = z; sCount = 1
                    settleSince = nowMs
                    settled = false
                }
            }
        }

        return Reading(disturbed, settled, sX, sY, sZ, deviation)
    }

    private fun dist(ax: Float, ay: Float, az: Float, bx: Float, by: Float, bz: Float): Float {
        val dx = ax - bx; val dy = ay - by; val dz = az - bz
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    companion object {
        /** Center point must sit this far (m) from the reference to look disturbed. */
        const val DISTURB_M = 0.18f

        /** Sustained time (ms) far from reference before declaring a disturbance. */
        const val DISTURB_HOLD_MS = 350L

        /** Sustained time (ms) of invalid depth (occlusion) before declaring disturbance. */
        const val INVALID_HOLD_MS = 600L

        /** Recent points must cluster within this radius (m) to count as settling. */
        const val SETTLE_RADIUS_M = 0.04f

        /** Dwell time (ms) the object must hold still before it's considered settled. */
        const val SETTLE_HOLD_MS = 1100L
    }
}
