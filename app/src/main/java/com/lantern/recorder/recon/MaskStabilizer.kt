package com.lantern.recorder.recon

/**
 * Temporal hysteresis for the per-frame FastSAM object mask.
 *
 * The raw mask is noisy frame-to-frame: the chosen center anchor occasionally jumps to a background
 * region, edges jitter, and spurious blobs blink in and out. Fusing that directly lets background
 * noise leak into the TSDF. This stabilizer folds each raw mask into a per-pixel **confidence
 * field** with an intentionally *asymmetric* update:
 *
 *  - **Slow to turn on** (`addRate` small): a pixel must be reported as object for a couple of
 *    consecutive frames before it counts, so a one-frame background detection never fuses.
 *  - **Quick to turn off** (`removeRate` larger): when SAM stops reporting a region — the background
 *    was removed, the object moved, or a *different* real object came into view — the mask drops
 *    that region within a frame or two and re-locks onto whatever is now consistently present.
 *
 * Image-space and single-threaded (driven from the live-mesh preview worker, ~5 Hz). At that
 * cadence a hand-orbited object moves only ~1 px/frame, so the field tracks it with negligible edge
 * lag while still killing transient false positives.
 */
class MaskStabilizer(
    /** Per-frame confidence gained while a pixel is reported as object. ~1/addRate frames to lock. */
    private val addRate: Float = 0.34f,
    /** Per-frame confidence lost while a pixel is not reported. Larger => quicker to forget. */
    private val removeRate: Float = 0.5f,
    /** A pixel is "on" (object) once its confidence reaches this. */
    private val onThreshold: Float = 0.5f,
) {
    private var conf: FloatArray? = null
    private var w = 0
    private var h = 0

    /** Stable (on) pixel count produced by the last [update]; 0 when nothing is locked. */
    var lastOnCount = 0
        private set

    fun reset() {
        conf?.let { java.util.Arrays.fill(it, 0f) }
        lastOnCount = 0
    }

    /**
     * Fold one raw mask (>=0.5f = object) into the confidence field and return the stabilized binary
     * mask at the same [width]x[height], or null when no pixel is confidently on.
     *
     * A null [raw] means "SAM detected nothing this frame": every pixel decays (quick-remove), so a
     * brief dropout fades over a frame or two rather than vanishing — and a sustained absence clears
     * the lock entirely.
     */
    fun update(raw: FloatArray?, width: Int, height: Int): FloatArray? {
        val n = width * height
        var c = conf
        if (c == null || width != w || height != h || c.size != n) {
            c = FloatArray(n)
            conf = c
            w = width
            h = height
        }
        val out = FloatArray(n)
        var on = 0
        for (i in 0 until n) {
            val fg = raw != null && i < raw.size && raw[i] >= 0.5f
            var v = if (fg) c[i] + addRate else c[i] - removeRate
            if (v < 0f) v = 0f else if (v > 1f) v = 1f
            c[i] = v
            if (v >= onThreshold) {
                out[i] = 1f
                on++
            }
        }
        lastOnCount = on
        return if (on > 0) out else null
    }
}
