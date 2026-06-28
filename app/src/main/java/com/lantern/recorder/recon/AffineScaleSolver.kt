package com.lantern.recorder.recon

import kotlin.math.abs

/**
 * Fits an affine map from relative disparity to metric disparity:
 *
 *     metric_disparity ( = 1/depth_m )  ~=  s * pred_disp + t
 *
 * This is the on-device port of `scale_solver.py::solve_affine` (Huber-robust least
 * squares, with the same f_scale=0.1). Solving **both** s and t (not scale-only) is the
 * crux from roadmap Decision 3: a scale-only fit bends flat surfaces because the shift is
 * left unsolved.
 *
 * The reference signal is ARCore's sparse-but-metric raw depth; the dense signal is DA3's
 * relative disparity. The output is a dense metric depth map ready for TSDF fusion.
 */
object AffineScaleSolver {

    data class Affine(val s: Float, val t: Float)

    private const val HUBER_DELTA = 0.1
    private const val SATURATED_DEPTH_M = 65.0
    private const val MIN_SAMPLES = 16

    /** Fractional depth band around the object distance used for the object-focused scale fit. */
    private const val FOCUS_BAND = 0.4f

    /**
     * Combine a DA3 relative-disparity map with ARCore metric depth into a **dense metric
     * depth map** (meters, 0 = invalid) at the ARCore depth resolution.
     *
     * Returns null if the affine fit can't be solved (too few overlapping samples); the
     * caller then falls back to using [metric] (ARCore depth) directly.
     */
    fun buildMetricDepth(
        disp: DisparityMap,
        metric: DepthMap,
        confThreshold: Float = 0.5f,
        focusDepthM: Float? = null,
    ): DepthMap? {
        val w = metric.width
        val h = metric.height
        val dispAtMetric = resizeNearest(disp, w, h)
        // Prefer fitting the scale on the object's depth shell (so the background doesn't dominate
        // the fit and shift the object frame-to-frame); fall back to a full-frame fit if the
        // banded set is too small to solve.
        val affine = (focusDepthM?.let { solve(dispAtMetric, metric, confThreshold, it) }
            ?: solve(dispAtMetric, metric, confThreshold))
            ?: solve(dispAtMetric, metric, confThreshold)
            ?: return null

        val out = FloatArray(w * h)
        for (i in out.indices) {
            val md = affine.s * dispAtMetric[i] + affine.t
            out[i] = if (md > 1e-4f && md.isFinite()) 1f / md else 0f
        }
        return DepthMap(w, h, out)
    }

    /**
     * Fit (s, t) over confident, valid, overlapping samples. Null if too few.
     * If [focusDepthM] is given, only samples whose metric depth is within [FOCUS_BAND] of it are
     * used, so the fit tracks the object rather than the background.
     */
    fun solve(
        dispValues: FloatArray,
        metric: DepthMap,
        confThreshold: Float,
        focusDepthM: Float? = null,
    ): Affine? {
        val depth = metric.depthMeters
        val conf = metric.confidence
        val n = depth.size
        val xs = ArrayList<Double>(n / 4)
        val ys = ArrayList<Double>(n / 4)
        val near = focusDepthM?.let { it * (1f - FOCUS_BAND) }
        val far = focusDepthM?.let { it * (1f + FOCUS_BAND) }
        for (i in 0 until n) {
            val dm = depth[i]
            if (dm <= 0.0f || dm.toDouble() >= SATURATED_DEPTH_M || !dm.isFinite()) continue
            if (near != null && (dm < near || dm > far!!)) continue
            if (conf != null && conf[i] < confThreshold) continue
            val x = dispValues[i].toDouble()
            if (!x.isFinite()) continue
            xs.add(x)
            ys.add(1.0 / dm.toDouble())
        }
        if (xs.size < MIN_SAMPLES) return null

        val initial = linearFit(xs, ys) ?: return null
        var s = initial.s.toDouble()
        var t = initial.t.toDouble()
        // Huber IRLS: a few reweighting passes to reject ARCore edge-bleed outliers.
        repeat(5) {
            var sxx = 0.0
            var sx = 0.0
            var sw = 0.0
            var sxy = 0.0
            var sy = 0.0
            for (k in xs.indices) {
                val x = xs[k]
                val y = ys[k]
                val r = abs(s * x + t - y)
                val wgt = if (r <= HUBER_DELTA) 1.0 else HUBER_DELTA / r
                sxx += wgt * x * x
                sx += wgt * x
                sw += wgt
                sxy += wgt * x * y
                sy += wgt * y
            }
            val det = sxx * sw - sx * sx
            if (abs(det) < 1e-12) return@repeat
            s = (sxy * sw - sx * sy) / det
            t = (sxx * sy - sx * sxy) / det
        }
        if (!s.isFinite() || !t.isFinite()) return null
        return Affine(s.toFloat(), t.toFloat())
    }

    /** Ordinary least squares y ~= s*x + t via 2x2 normal equations. */
    private fun linearFit(xs: List<Double>, ys: List<Double>): Affine? {
        var sxx = 0.0
        var sx = 0.0
        var sxy = 0.0
        var sy = 0.0
        val nn = xs.size.toDouble()
        for (k in xs.indices) {
            val x = xs[k]
            val y = ys[k]
            sxx += x * x
            sx += x
            sxy += x * y
            sy += y
        }
        val det = sxx * nn - sx * sx
        if (abs(det) < 1e-12) return null
        val s = (sxy * nn - sx * sy) / det
        val t = (sxx * sy - sx * sxy) / det
        return Affine(s.toFloat(), t.toFloat())
    }

    private fun resizeNearest(disp: DisparityMap, w: Int, h: Int): FloatArray {
        if (disp.width == w && disp.height == h) return disp.values
        val out = FloatArray(w * h)
        val sx = disp.width.toFloat() / w
        val sy = disp.height.toFloat() / h
        for (y in 0 until h) {
            val srcY = (y * sy).toInt().coerceIn(0, disp.height - 1)
            for (x in 0 until w) {
                val srcX = (x * sx).toInt().coerceIn(0, disp.width - 1)
                out[y * w + x] = disp.values[srcY * disp.width + srcX]
            }
        }
        return out
    }
}
