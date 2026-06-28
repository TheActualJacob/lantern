package com.lantern.recorder.recon

import android.util.Log
import com.google.ar.core.Frame
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

private const val TAG = "LANTERN"

/** A relative (affine-invariant) disparity map produced by a monocular depth net. */
class DisparityMap(val width: Int, val height: Int, val values: FloatArray) {
    init {
        require(values.size == width * height) { "values size ${values.size} != ${width}x$height" }
    }
}

/**
 * ARCore raw depth as a **metric** [DepthMap]. Always available on a depth-capable device
 * (RAW_DEPTH_ONLY), so the live mesh can light up with no `.pte` present. Sparse/noisy at
 * object scale — DA3 ([Da3DepthModel]) supplies the dense surface, this supplies the meter.
 *
 * Must be called on the GL thread while the Frame is current (it acquires images).
 */
class ArCoreRawDepthSource {
    fun acquire(frame: Frame): DepthMap? = try {
        val depth = frame.acquireRawDepthImage16Bits().use { ImageUtils.depth16ToMeters(it) }
        val confidence = try {
            frame.acquireRawDepthConfidenceImage().use { ImageUtils.confidence8ToFloat(it) }
        } catch (e: Exception) {
            null
        }
        if (confidence != null && confidence.size == depth.depthMeters.size) {
            DepthMap(depth.width, depth.height, depth.depthMeters, confidence)
        } else {
            depth
        }
    } catch (e: Exception) {
        Log.w(TAG, "ARCore raw depth unavailable this frame", e)
        null
    }
}

/**
 * Depth-Anything-3 Small monocular depth via the ExecuTorch runtime.
 *
 * Loads the `.pte` exported by `export_da3_executorch.py` (XNNPACK CPU or QNN/SM8750 NPU)
 * and returns a **relative disparity** map (`1 / da3_depth`) — the exact quantity the host
 * `pipeline_float.py` affine solver consumes. [AffineScaleSolver] turns it metric against
 * ARCore depth before fusion. Returns null (and logs) if the model is missing or fails, so
 * the pipeline degrades to ARCore-only depth instead of crashing.
 *
 * Input contract (must match the export): 1x3x[res]x[res] float32, ImageNet-normalized RGB.
 * Output: a single depth tensor [res]x[res] (or 1x[res]x[res]).
 */
class Da3DepthModel private constructor(
    private val module: Module,
    private val res: Int,
) {
    /**
     * Run DA3 on a packed-ARGB color image; returns relative disparity at [res]x[res].
     * CPU/NPU heavy — call off the GL thread.
     */
    fun inferDisparity(argb: ImageUtils.Argb): DisparityMap? {
        return try {
            val input = preprocess(argb)
            val tensor = Tensor.fromBlob(input, longArrayOf(1, 3, res.toLong(), res.toLong()))
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
            DisparityMap(res, res, depthToDisparity(depth))
        } catch (t: Throwable) {
            Log.e(TAG, "DA3 inference failed", t)
            null
        }
    }

    /** ARGB -> normalized NCHW float32 RGB at [res]x[res] (bilinear resize, ImageNet norm). */
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

    private fun depthToDisparity(depth: FloatArray): FloatArray {
        val disp = FloatArray(res * res)
        for (i in disp.indices) {
            val d = depth[i]
            disp[i] = if (d.isFinite() && d > DISP_EPS) 1f / d else 0f
        }
        return disp
    }

    fun close() {
        try {
            module.destroy()
        } catch (t: Throwable) {
            Log.w(TAG, "DA3 module destroy failed", t)
        }
    }

    companion object {
        private const val DISP_EPS = 1e-6f
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        /** Default DA3-Small working resolution (matches the Qualcomm NPU export). */
        const val DEFAULT_RES = 518

        /**
         * Load a DA3 `.pte` from an absolute filesystem path. Returns null if the file is
         * missing or the runtime can't load it (caller falls back to ARCore-only depth).
         */
        fun loadOrNull(modelPath: String, res: Int = DEFAULT_RES): Da3DepthModel? {
            return try {
                val module = Module.load(modelPath)
                Log.i(TAG, "Loaded DA3 model: $modelPath (res=$res)")
                Da3DepthModel(module, res)
            } catch (t: Throwable) {
                Log.w(TAG, "DA3 model not loaded ($modelPath); ARCore-only depth. ${t.message}")
                null
            }
        }
    }
}
