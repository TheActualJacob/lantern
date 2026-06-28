package com.lantern.recorder.recon

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Verifies the asymmetric temporal behaviour of [MaskStabilizer]: slow to accept a newly-reported
 * region (so transient background noise never fuses) but quick to drop a region SAM stops reporting.
 */
class MaskStabilizerTest {

    private val w = 4
    private val h = 4
    private val n = w * h

    /** A mask that is fully "on" everywhere. */
    private fun full() = FloatArray(n) { 1f }

    private fun onCount(mask: FloatArray?): Int = mask?.count { it >= 0.5f } ?: 0

    @Test
    fun singleFrameNoiseNeverTurnsOn() {
        // Defaults: addRate 0.34 < onThreshold 0.5, so one frame of "object" stays below threshold.
        val s = MaskStabilizer()
        val out = s.update(full(), w, h)
        assertNull("a one-frame detection must not lock on (slow add)", out)
        assertEquals(0, s.lastOnCount)
    }

    @Test
    fun sustainedRegionLocksOnWithinAFewFrames() {
        val s = MaskStabilizer()
        s.update(full(), w, h)            // 0.34
        val out = s.update(full(), w, h)  // 0.68 >= 0.5 -> on
        assertNotNull("two consistent frames should lock the region on", out)
        assertEquals(n, onCount(out))
    }

    @Test
    fun lockedRegionDropsFasterThanItBuilt() {
        val s = MaskStabilizer()
        // Build a solid lock (saturate confidence well above threshold).
        repeat(5) { s.update(full(), w, h) }
        assertEquals(n, onCount(s.update(full(), w, h)))

        // Removal: confidence falls by removeRate (0.5) per empty frame. From ~1.0 it clears in two.
        val afterOne = s.update(null, w, h)
        val afterTwo = s.update(null, w, h)
        assertNull("a sustained dropout clears the lock quickly", afterTwo)
        // And it cleared at least as fast as the ~2 frames it took to build.
        assertEquals(0, onCount(afterTwo))
        // (afterOne may still be partially on — that single-frame grace prevents object flicker.)
        assertNotNull("one missing frame alone shouldn't necessarily wipe a saturated lock", afterOne)
    }

    @Test
    fun resetClearsConfidence() {
        val s = MaskStabilizer()
        repeat(5) { s.update(full(), w, h) }
        s.reset()
        // After reset, one frame again isn't enough to re-lock (back to slow-add from zero).
        assertNull(s.update(full(), w, h))
    }
}
