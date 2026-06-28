package com.lantern.recorder.recon

import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Verifies [AffineScaleSolver.fitObjectAffine] — the per-frame, object-anchored scale used by the
 * directed/batch reconstructor. The contract that matters there: recover the true
 * `metric_disparity = s*pred_disp + t` line from the masked object pixels alone, ignoring
 * background depth, and refuse (null) when the object region is too sparse to fit.
 */
class AffineScaleSolverTest {

    private val W = 8
    private val H = 8

    /**
     * Build a (disparity, depth) pair where masked object pixels follow `1/depth = s*disp + t`
     * exactly and background pixels carry corrupt depth/disp that must NOT influence the fit.
     */
    private fun frame(s: Float, t: Float, maskRadius: Int): Pair<DisparityMap, DepthMap> {
        val disp = FloatArray(W * H)
        val depth = FloatArray(W * H)
        val mask = FloatArray(W * H)
        var k = 0
        val lo = (H / 2) - maskRadius
        val hi = (H / 2) + maskRadius
        for (y in 0 until H) for (x in 0 until W) {
            val i = y * W + x
            val masked = x in lo until hi && y in lo until hi
            if (masked) {
                val d = 0.20f + 0.10f * (k.toFloat() / (maskRadius * maskRadius * 4))
                disp[i] = d
                depth[i] = 1f / (s * d + t) // ~0.9..1.1 m for s=2,t=0.5
                mask[i] = 1f
                k++
            } else {
                disp[i] = 5f       // wildly different background disparity
                depth[i] = 0.1f    // and depth — would wreck the fit if not masked out
                mask[i] = 0f
            }
        }
        return DisparityMap(W, H, disp) to DepthMap(W, H, depth)
    }

    private fun maskOf(maskRadius: Int): FloatArray {
        val mask = FloatArray(W * H)
        val lo = (H / 2) - maskRadius
        val hi = (H / 2) + maskRadius
        for (y in 0 until H) for (x in 0 until W) {
            if (x in lo until hi && y in lo until hi) mask[y * W + x] = 1f
        }
        return mask
    }

    @Test
    fun recoversObjectScaleIgnoringBackground() {
        val s = 2.0f
        val t = 0.5f
        val (disp, depth) = frame(s, t, maskRadius = 3) // 6x6 = 36 object pixels (>= MIN_SAMPLES)
        val mask = maskOf(3)
        val aff = AffineScaleSolver.fitObjectAffine(disp, depth, focusDepthM = 1.0f, mask = mask)
        assertNotNull("object fit should succeed with enough masked samples", aff)
        aff!!
        assertTrue("recovered s=${aff.s} should be ~$s", abs(aff.s - s) < 0.05f)
        assertTrue("recovered t=${aff.t} should be ~$t", abs(aff.t - t) < 0.05f)
    }

    @Test
    fun returnsNullWhenObjectRegionTooSparse() {
        // 2x2 masked region = 4 samples, below MIN_SAMPLES (16): no full-frame fallback, so null.
        val (disp, depth) = frame(2.0f, 0.5f, maskRadius = 1)
        val mask = maskOf(1)
        assertNull(
            "must refuse rather than fit on too few object pixels",
            AffineScaleSolver.fitObjectAffine(disp, depth, focusDepthM = 1.0f, mask = mask),
        )
    }
}
