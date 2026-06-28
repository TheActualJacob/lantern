package com.lantern.recorder.recon

import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module

private const val TAG = "LANTERN"

/**
 * Depth-Anything-3 **multi-view** via ExecuTorch — the directed/guided *build* backend.
 *
 * Unlike [ExecuTorchDepthModel] (one image -> relative depth, used live), this loads the
 * multi-view `.pte` exported by `export_da3_mv_executorch.py`: it takes a **fixed** [numViews]
 * images in one forward pass and returns, in **one shared, mutually-consistent coordinate frame**:
 *   - per-view depth `(N, resH, resW)` — distance (z), DA3 units, consistent across views,
 *   - per-view confidence `(N, resH, resW)`,
 *   - DA3's own estimated **extrinsics** `(N, 3, 4)` world->camera `[R|t]` (OpenCV),
 *   - DA3's own estimated **intrinsics** `(N, 3, 3)` at `resW x resH`.
 *
 * This is what makes a scanned object reconstruct coherently instead of the bent, convex "banana"
 * the mono per-frame + affine path produces (verified host-side). DA3 supplies poses + intrinsics,
 * so the caller needs no ARCore poses for the *shape*.
 *
 * The input is the camera's **native aspect** (e.g. 504x280 for 16:9), NOT a square — a square
 * letterbox pads the image with gray bars, which measurably degrades DA3's geometry. [numViews],
 * [resW], [resH] are frozen at export and read from the `_nN_wW_hH` filename; the caller must pass
 * exactly [numViews] views (pad by repetition only if fewer were captured). Heavy; call off-thread.
 */
class Da3MultiViewModel private constructor(
    private val module: Module,
    val numViews: Int,
    val resW: Int,
    val resH: Int,
) {
    /** Letterbox placement of one source image inside the [resW]x[resH] input (for mask alignment). */
    class ViewGeom(val padX: Int, val padY: Int, val newW: Int, val newH: Int)

    /**
     * Multi-view result. [depth]/[conf] are flat `N*resH*resW` (per view, row index `y*resW+x`);
     * [extrinsics] flat `N*12` (3x4 row-major, world->cam); [intrinsics] flat `N*9` (3x3 row-major).
     * [geom] gives each input view's letterbox so masks/colors map into the same grid as [depth].
     */
    class Result(
        val n: Int,
        val resW: Int,
        val resH: Int,
        val depth: FloatArray,
        val conf: FloatArray,
        val extrinsics: FloatArray,
        val intrinsics: FloatArray,
        val geom: List<ViewGeom>,
    )

    /** Run the model on exactly [numViews] color images. Returns null on shape mismatch / failure. */
    fun infer(argbs: List<ImageUtils.Argb>): Result? {
        if (argbs.size != numViews) {
            Log.w(TAG, "Da3MultiView: need $numViews views, got ${argbs.size}")
            return null
        }
        return try {
            val plane = resW * resH
            val input = FloatArray(numViews * 3 * plane)
            val geom = ArrayList<ViewGeom>(numViews)
            for (n in 0 until numViews) {
                val argb = argbs[n]
                val lb = Letterbox(argb.width, argb.height, resW, resH)
                geom.add(ViewGeom(lb.padX, lb.padY, lb.newW, lb.newH))
                writeView(input, n, argb, lb)
            }
            val tensor = org.pytorch.executorch.Tensor.fromBlob(
                input, longArrayOf(1, numViews.toLong(), 3, resH.toLong(), resW.toLong()),
            )
            val outs = module.forward(EValue.from(tensor))
            if (outs.size < 4) {
                Log.w(TAG, "Da3MultiView: expected 4 outputs, got ${outs.size}")
                return null
            }
            Result(
                n = numViews,
                resW = resW,
                resH = resH,
                depth = outs[0].toTensor().dataAsFloatArray,
                conf = outs[1].toTensor().dataAsFloatArray,
                extrinsics = outs[2].toTensor().dataAsFloatArray,
                intrinsics = outs[3].toTensor().dataAsFloatArray,
                geom = geom,
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Da3MultiView inference failed", t)
            null
        }
    }

    /** Write one letterboxed, ImageNet-normalized view into [input] at `[n][c][y][x]` (NCHW per view). */
    private fun writeView(input: FloatArray, n: Int, argb: ImageUtils.Argb, lb: Letterbox) {
        val plane = resW * resH
        val base = n * 3 * plane
        // Pad value = ImageNet-mean-normalized 0 (neutral gray), matching the mono export.
        java.util.Arrays.fill(input, base, base + 3 * plane, 0f)
        val sx = argb.width.toFloat() / lb.newW
        val sy = argb.height.toFloat() / lb.newH
        for (oy in 0 until lb.newH) {
            val srcY = (oy * sy).toInt().coerceIn(0, argb.height - 1)
            val rowBase = srcY * argb.width
            val dstY = lb.padY + oy
            for (ox in 0 until lb.newW) {
                val srcX = (ox * sx).toInt().coerceIn(0, argb.width - 1)
                val p = argb.pixels[rowBase + srcX]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                val o = dstY * resW + (lb.padX + ox)
                input[base + o] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                input[base + plane + o] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                input[base + 2 * plane + o] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
            }
        }
    }

    /** Aspect-preserving fit of [srcW]x[srcH] into [targetW]x[targetH] with centered padding. */
    class Letterbox(srcW: Int, srcH: Int, targetW: Int, targetH: Int) {
        val scale = minOf(targetW.toFloat() / srcW, targetH.toFloat() / srcH)
        val newW = (srcW * scale).toInt().coerceIn(1, targetW)
        val newH = (srcH * scale).toInt().coerceIn(1, targetH)
        val padX = (targetW - newW) / 2
        val padY = (targetH - newH) / 2
    }

    fun close() {
        try {
            module.destroy()
        } catch (t: Throwable) {
            Log.w(TAG, "Da3MultiView destroy failed", t)
        }
    }

    companion object {
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        /** Parse `_n<N>_w<W>_h<H>` (preferred) or legacy `_n<N>_r<RES>` (square) from the filename. */
        private fun parseShape(path: String): Triple<Int, Int, Int>? {
            Regex("_n(\\d+)_w(\\d+)_h(\\d+)\\.pte$").find(path)?.let {
                return Triple(it.groupValues[1].toInt(), it.groupValues[2].toInt(), it.groupValues[3].toInt())
            }
            Regex("_n(\\d+)_r(\\d+)\\.pte$").find(path)?.let {
                val r = it.groupValues[2].toInt()
                return Triple(it.groupValues[1].toInt(), r, r)
            }
            return null
        }

        /**
         * Load a multi-view DA3 `.pte`. [numViews]/[resW]/[resH] are read from the filename suffix
         * (they must match the export). Returns null if the file is missing, the suffix can't be
         * parsed, or the runtime can't load it (caller falls back to mono).
         */
        fun loadOrNull(modelPath: String): Da3MultiViewModel? {
            val shape = parseShape(modelPath)
            if (shape == null) {
                Log.w(TAG, "Da3MultiView: filename lacks _n<N>_w<W>_h<H> suffix ($modelPath); skipping")
                return null
            }
            return try {
                val module = Module.load(modelPath)
                Log.i(TAG, "Loaded multi-view DA3: $modelPath (N=${shape.first}, ${shape.second}x${shape.third})")
                Da3MultiViewModel(module, shape.first, shape.second, shape.third)
            } catch (t: Throwable) {
                Log.w(TAG, "Da3MultiView .pte not loaded ($modelPath); mono fallback. ${t.message}")
                null
            }
        }
    }
}
