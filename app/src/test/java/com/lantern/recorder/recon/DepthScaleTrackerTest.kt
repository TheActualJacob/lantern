package com.lantern.recorder.recon

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Verifies [DepthScaleTracker] establishes one stable DA3 disparity->metric scale: median-seeded
 * from a short warmup (robust to a bad first frame), outlier-rejecting, EMA-smoothed, and frozen
 * after enough good frames so ARCore depth noise stops moving the geometry.
 */
class DepthScaleTrackerTest {

    private fun aff(s: Float, t: Float = 0f) = AffineScaleSolver.Affine(s, t)

    @Test
    fun notReadyDuringWarmup() {
        val tr = DepthScaleTracker()
        repeat(DepthScaleTracker.WARMUP - 1) {
            assertNull("scale must not be usable mid-warmup", tr.update(aff(2f)))
        }
        assertFalse(tr.ready)
    }

    @Test
    fun seedsFromWarmupMedian() {
        val tr = DepthScaleTracker()
        // Five candidates; the median scale is 2.0 even with one wild outlier in the set.
        listOf(1.9f, 2.0f, 2.1f, 2.0f, 9.0f).forEach { tr.update(aff(it)) }
        assertTrue(tr.ready)
        assertEquals(2.0f, tr.s, 1e-4f)
    }

    @Test
    fun rejectsOutlierAfterLock() {
        val tr = DepthScaleTracker()
        repeat(DepthScaleTracker.WARMUP) { tr.update(aff(2.0f)) }
        val before = tr.s
        // A 5x scale spike (e.g. ARCore depth glitch) must be ignored, not blended in.
        tr.update(aff(10.0f))
        assertEquals("outlier candidate must not move the scale", before, tr.s, 1e-4f)
    }

    @Test
    fun emaDriftsTowardSustainedChange() {
        val tr = DepthScaleTracker()
        repeat(DepthScaleTracker.WARMUP) { tr.update(aff(2.0f)) }
        // A modest, in-band shift (within MAX_JUMP) is followed gradually.
        tr.update(aff(2.4f))
        assertTrue("EMA should move toward the new value", tr.s > 2.0f)
        assertTrue("but not jump all the way in one step", tr.s < 2.4f)
    }

    @Test
    fun locksAndFreezesAfterEnoughGoodFrames() {
        val tr = DepthScaleTracker()
        repeat(DepthScaleTracker.LOCK_AFTER + DepthScaleTracker.WARMUP) { tr.update(aff(2.0f)) }
        assertTrue("scale should lock after enough accepted frames", tr.locked)
        val locked = tr.s
        // Post-lock, even an in-band candidate is ignored — the scale is frozen.
        val out = tr.update(aff(2.3f))
        assertNotNull(out)
        assertEquals(locked, tr.s, 1e-6f)
    }

    @Test
    fun resetClearsState() {
        val tr = DepthScaleTracker()
        repeat(DepthScaleTracker.LOCK_AFTER + DepthScaleTracker.WARMUP) { tr.update(aff(2.0f)) }
        tr.reset()
        assertFalse(tr.ready)
        assertFalse(tr.locked)
        assertEquals(0f, tr.s, 0f)
    }
}
