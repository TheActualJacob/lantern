package com.lantern.recorder.recon

import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

private const val TAG = "LANTERN"

/**
 * Depth-Anything-3 Small monocular depth via the **ExecuTorch** runtime ([DepthBackend]).
 *
 * Loads the `.pte` exported by `export_da3_executorch.py` (XNNPACK CPU or QNN/SM8750 NPU
 * delegate) and returns a **relative disparity** map (`1 / da3_depth`) — the exact quantity the
 * host `pipeline_float.py` affine solver consumes. [AffineScaleSolver] turns it metric against
 * ARCore depth before fusion. Returns null (and logs) if the model is missing or fails, so the
 * pipeline degrades to ARCore-only depth instead of crashing.
 *
 * Input contract (must match the export): 1x1x3x[res]x[res] float32, **ImageNet-normalized**
 * RGB (B,N,C,H,W with N=1; DA3's net is multi-view). Output: depth [1,1,[res],[res]].
 *
 * This differs from [QnnDlcDepthModel], which consumes the native `.dlc` with NHWC `[0,1]`
 * input — hence each backend keeps its own [preprocess].
 */
class ExecuTorchDepthModel private constructor(
    private val module: Module,
    override val res: Int,
) : DepthBackend {
    override val kind = DepthBackendKind.EXECUTORCH

    /**
     * Run DA3 on a packed-ARGB color image; returns relative disparity at [res]x[res].
     * CPU/NPU heavy — call off the GL thread.
     */
    override fun inferDisparity(argb: ImageUtils.Argb): DisparityMap? {
        return try {
            val input = preprocess(argb)
            // DA3's net is multi-view: forward expects (B, N, 3, H, W) with N=1 for a single
            // image. ExecuTorch input ranks are immutable, so a 4D tensor fails with
            // "Error resizing tensor at input 0" (rank 5 -> 4). Same NCHW data, 5D shape.
            val tensor = Tensor.fromBlob(input, longArrayOf(1, 1, 3, res.toLong(), res.toLong()))
            val outputs = module.forward(EValue.from(tensor))
            if (outputs.isEmpty()) {
                Log.w(TAG, "DA3 forward returned no outputs")
                return null
            }
            val depth = outputs[0].toTensor().dataAsFloatArray
            if (depth.size < res * res) {
                Log.w(TAG, "DA3 output ${depth.size} < ${res * res}")
                return null
            }
            DisparityMap(res, res, depthToDisparity(depth, res))
        } catch (t: Throwable) {
            Log.e(TAG, "DA3 inference failed", t)
            null
        }
    }

    /** ARGB -> normalized NCHW float32 RGB at [res]x[res] (nearest resize, ImageNet norm). */
    private fun preprocess(argb: ImageUtils.Argb): FloatArray {
        val out = FloatArray(3 * res * res)
        val planeStride = res * res
        val sx = argb.width.toFloat() / res
        val sy = argb.height.toFloat() / res
        for (oy in 0 until res) {
            val srcY = (oy * sy).toInt().coerceIn(0, argb.height - 1)
            for (ox in 0 until res) {
                val srcX = (ox * sx).toInt().coerceIn(0, argb.width - 1)
                val p = argb.pixels[srcY * argb.width + srcX]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                val o = oy * res + ox
                out[o] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                out[planeStride + o] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                out[2 * planeStride + o] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
            }
        }
        return out
    }

    override fun close() {
        try {
            module.destroy()
        } catch (t: Throwable) {
            Log.w(TAG, "DA3 module destroy failed", t)
        }
    }

    companion object {
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        /** Default DA3-Small working resolution (matches the Qualcomm NPU export). */
        const val DEFAULT_RES = 518

        /** Parse the fixed input resolution from a `..._r<N>.pte` filename, else [DEFAULT_RES]. */
        private fun resFromName(modelPath: String): Int =
            Regex("_r(\\d+)\\.pte$").find(modelPath)?.groupValues?.get(1)?.toIntOrNull() ?: DEFAULT_RES

        /**
         * Load a DA3 `.pte` from an absolute filesystem path. The input resolution is read from a
         * `_r<N>` suffix in the filename when present (the ExecuTorch input rank/shape is fixed at
         * export time and must match), otherwise [DEFAULT_RES]. Returns null if the file is missing
         * or the runtime can't load it (caller falls back to ARCore-only depth).
         */
        fun loadOrNull(modelPath: String, res: Int = resFromName(modelPath)): ExecuTorchDepthModel? {
            return try {
                val module = Module.load(modelPath)
                Log.i(TAG, "Loaded DA3 ExecuTorch model: $modelPath (res=$res)")
                ExecuTorchDepthModel(module, res)
            } catch (t: Throwable) {
                Log.w(TAG, "DA3 .pte not loaded ($modelPath); ARCore-only depth. ${t.message}")
                null
            }
        }
    }
}

/** depth -> disparity (`1/depth`), guarding non-finite/zero. Shared by both depth backends. */
internal fun depthToDisparity(depth: FloatArray, res: Int): FloatArray {
    val disp = FloatArray(res * res)
    val eps = 1e-6f
    for (i in disp.indices) {
        val d = depth[i]
        disp[i] = if (d.isFinite() && d > eps) 1f / d else 0f
    }
    return disp
}
