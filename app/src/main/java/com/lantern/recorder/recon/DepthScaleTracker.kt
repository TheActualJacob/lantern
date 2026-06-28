package com.lantern.recorder.recon

/**
 * Maintains a **single, temporally-stable** affine `(s, t)` mapping DA3 relative disparity to
 * metric disparity for the whole scan.
 *
 * Why this exists: DA3 predicts only *relative* depth, so per frame we'd otherwise re-fit `(s, t)`
 * against ARCore's sparse-and-noisy depth (see [AffineScaleSolver]). On smooth scenes ARCore depth
 * is garbage, so that fit jumps frame-to-frame — placing the object at a slightly different size and
 * distance each view, which smears the fused mesh into a warped shell. Instead we treat each frame's
 * fit as a *candidate*, reject outliers (ARCore depth glitches), median-seed from a short warmup so a
 * single bad frame can't anchor us, smooth the rest with an EMA, and **lock** after a few good frames
 * so ARCore depth stops influencing geometry entirely. After lock, all views share one scale and
 * register into a single coherent surface.
 *
 * Touched only on the fusion worker; the volatile fields let the debug overlay read the live state.
 */
class DepthScaleTracker {

    @Volatile var s = 0f
        private set

    @Volatile var t = 0f
        private set

    @Volatile var locked = false
        private set

    private val warmupS = ArrayList<Float>(WARMUP)
    private val warmupT = ArrayList<Float>(WARMUP)
    private var accepted = 0

    /** True once a usable global scale has been established (warmup complete). */
    val ready: Boolean get() = accepted > 0

    fun reset() {
        warmupS.clear()
        warmupT.clear()
        accepted = 0
        s = 0f
        t = 0f
        locked = false
    }

    /**
     * Fold one per-frame [candidate] affine into the global scale. Returns the current value as an
     * [AffineScaleSolver.Affine] once [ready], else null (caller should skip fusion until then).
     */
    fun update(candidate: AffineScaleSolver.Affine?): AffineScaleSolver.Affine? {
        if (locked) return current()
        if (candidate == null || !candidate.s.isFinite() || !candidate.t.isFinite() || candidate.s <= 0f) {
            return current()
        }

        if (accepted == 0) {
            // Warmup: collect a few candidates and seed from their median, so one outlier frame
            // (e.g. an ARCore depth glitch) can't anchor the whole scan to a wrong scale.
            warmupS.add(candidate.s)
            warmupT.add(candidate.t)
            if (warmupS.size >= WARMUP) {
                warmupS.sort()
                warmupT.sort()
                s = warmupS[warmupS.size / 2]
                t = warmupT[warmupT.size / 2]
                accepted = warmupS.size
            }
            return current()
        }

        // Steady state: reject candidates whose scale jumps too far from the running estimate
        // (the noisy frames), and EMA-smooth the ones we trust.
        val ratio = candidate.s / s
        if (ratio in (1f / MAX_JUMP)..MAX_JUMP) {
            s = (1f - ALPHA) * s + ALPHA * candidate.s
            t = (1f - ALPHA) * t + ALPHA * candidate.t
            accepted++
            if (accepted >= LOCK_AFTER) locked = true
        }
        return current()
    }

    private fun current(): AffineScaleSolver.Affine? =
        if (ready) AffineScaleSolver.Affine(s, t) else null

    companion object {
        /** Candidates collected before seeding the global scale from their median. */
        const val WARMUP = 5

        /** Reject a candidate whose scale is >MAX_JUMP× or <1/MAX_JUMP× the running estimate. */
        const val MAX_JUMP = 1.5f

        /** EMA weight toward each accepted candidate (lower = steadier). */
        const val ALPHA = 0.25f

        /** Freeze the scale after this many accepted updates; ARCore depth then stops mattering. */
        const val LOCK_AFTER = 20
    }
}
