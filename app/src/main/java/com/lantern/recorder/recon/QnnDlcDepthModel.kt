package com.lantern.recorder.recon

import android.util.Log
import java.io.File

private const val TAG = "LANTERN"

/**
 * Depth-Anything-3 monocular depth run **natively on the Qualcomm QNN/QAIRT runtime**
 * ([DepthBackend]) via JNI, using the AI Hub `depth_anything_v3.dlc` (or a pre-compiled HTP
 * context binary). Targets the S25 / SM8750 Hexagon **v79** NPU. Falls back to null whenever the
 * native lib, QNN `.so`s, or model are missing, so the device degrades to ExecuTorch/ARCore
 * instead of crashing — see [QnnNative.available].
 *
 * Input contract (from the DLC `metadata.json`): **NHWC `[1,518,518,3]` float32, range `[0,1]`**
 * (divide by 255, **no** ImageNet normalization — unlike [ExecuTorchDepthModel]).
 * Output `depth_estimates`: `[1,518,518,1]` float32 monocular depth → converted to disparity.
 *
 * The native side ([app/src/main/cpp/qnn_depth.cpp]) is only compiled when the build is run with
 * `QNN_SDK_ROOT` set (see `app/build.gradle.kts`); on a stock build `libqnn_depth.so` is absent,
 * [QnnNative.available] is false, and [loadOrNull] returns null.
 */
class QnnDlcDepthModel private constructor(
    private val handle: Long,
    override val res: Int,
) : DepthBackend {
    override val kind = DepthBackendKind.QNN

    @Volatile private var closed = false

    override fun inferDisparity(argb: ImageUtils.Argb): DisparityMap? {
        if (closed) return null
        return try {
            val input = preprocess(argb) // NHWC [0,1], length res*res*3
            val out = FloatArray(res * res)
            val ok = QnnNative.nativeInfer(handle, input, out)
            if (!ok) {
                Log.w(TAG, "QNN inference returned failure")
                return null
            }
            DisparityMap(res, res, depthToDisparity(out, res))
        } catch (t: Throwable) {
            Log.e(TAG, "QNN DLC inference failed", t)
            null
        }
    }

    /**
     * ARGB -> **NHWC** float32 RGB at [res]x[res], scaled to `[0,1]` (no ImageNet norm), in the
     * channel order the DLC expects (R,G,B interleaved per pixel). Nearest-neighbor resize to
     * match [ExecuTorchDepthModel]'s sampling.
     */
    private fun preprocess(argb: ImageUtils.Argb): FloatArray {
        val out = FloatArray(res * res * 3)
        val sx = argb.width.toFloat() / res
        val sy = argb.height.toFloat() / res
        var o = 0
        for (oy in 0 until res) {
            val srcY = (oy * sy).toInt().coerceIn(0, argb.height - 1)
            val rowBase = srcY * argb.width
            for (ox in 0 until res) {
                val srcX = (ox * sx).toInt().coerceIn(0, argb.width - 1)
                val p = argb.pixels[rowBase + srcX]
                out[o] = ((p shr 16) and 0xFF) / 255f
                out[o + 1] = ((p shr 8) and 0xFF) / 255f
                out[o + 2] = (p and 0xFF) / 255f
                o += 3
            }
        }
        return out
    }

    override fun close() {
        if (closed) return
        closed = true
        try {
            QnnNative.nativeClose(handle)
        } catch (t: Throwable) {
            Log.w(TAG, "QNN context destroy failed", t)
        }
    }

    companion object {
        /** DA3-Small / DLC working resolution (NHWC 518x518x3, per metadata.json). */
        const val DEFAULT_RES = 518

        /**
         * Load the QNN runtime against a `.dlc` or pre-compiled HTP context-binary `.bin` at an
         * absolute path. Returns null (caller falls back) when:
         *  - the native `libqnn_depth.so` / QNN `.so`s aren't bundled ([QnnNative.available]),
         *  - the model file is missing, or
         *  - native init fails (unsupported SoC, wrong Hexagon skel, bad binary).
         */
        fun loadOrNull(modelPath: String, nativeLibDir: String? = null, res: Int = DEFAULT_RES): QnnDlcDepthModel? {
            if (!QnnNative.available) {
                Log.i(TAG, "QNN native lib unavailable; skipping QNN depth backend")
                return null
            }
            if (!File(modelPath).exists()) {
                Log.i(TAG, "QNN model not present ($modelPath); skipping QNN depth backend")
                return null
            }
            // The HTP backend loads the Hexagon skel via fastRPC (ADSP_LIBRARY_PATH); set it to the
            // app's extracted lib dir before init or device bring-up fails (0x36b1).
            QnnNative.setAdspLibraryPath(nativeLibDir)
            return try {
                val handle = QnnNative.nativeInit(modelPath, res)
                if (handle == 0L) {
                    Log.w(TAG, "QNN nativeInit failed for $modelPath")
                    null
                } else {
                    Log.i(TAG, "Loaded DA3 QNN model: $modelPath (res=$res)")
                    QnnDlcDepthModel(handle, res)
                }
            } catch (t: Throwable) {
                Log.w(TAG, "QNN model not loaded ($modelPath); ${t.message}")
                null
            }
        }
    }
}

/**
 * Thin JNI loader for the native QNN bridge. [available] is false (and the `native*` methods
 * must not be called) when `libqnn_depth.so` can't be loaded — i.e. on builds without the QNN
 * SDK, or on non-Qualcomm devices — which is the common case and is expected, not an error.
 */
internal object QnnNative {
    val available: Boolean = try {
        System.loadLibrary("qnn_depth")
        true
    } catch (t: Throwable) {
        Log.i(TAG, "libqnn_depth not loaded (QNN backend disabled): ${t.message}")
        false
    }

    /**
     * Point fastRPC at the app's extracted native-lib dir (where libQnnHtpV79Skel.so lives) so the
     * HTP backend can load the Hexagon skel onto the DSP; without it device bring-up fails (0x36b1).
     * Process-global and idempotent — call before any nativeInit. No-op if [nativeLibDir] is null.
     */
    fun setAdspLibraryPath(nativeLibDir: String?) {
        if (nativeLibDir == null) return
        try {
            val search = "$nativeLibDir;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;" +
                "/vendor/lib64/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
            android.system.Os.setenv("ADSP_LIBRARY_PATH", search, true)
        } catch (t: Throwable) {
            Log.w(TAG, "could not set ADSP_LIBRARY_PATH: ${t.message}")
        }
    }

    /** Init QNN HTP backend + context from a `.dlc`/`.bin`; returns an opaque handle or 0. */
    external fun nativeInit(modelPath: String, res: Int): Long

    /** Run one frame: NHWC `[0,1]` `input` (res*res*3) -> `output` depth (res*res). */
    external fun nativeInfer(handle: Long, input: FloatArray, output: FloatArray): Boolean

    /** Multi-IO inference (graph order). For models with several outputs (e.g. FastSAM). */
    external fun nativeInferN(handle: Long, inputs: Array<FloatArray>, outputs: Array<FloatArray>): Boolean

    /** Output tensor names in graph order, to map multi-output models by name. */
    external fun nativeOutputNames(handle: Long): Array<String>

    /** Element counts (graph order) for inputs (which=0) or outputs (which=1). */
    external fun nativeIoElems(handle: Long, which: Int): IntArray

    /** Free the QNN context/backend for [handle]. */
    external fun nativeClose(handle: Long)
}
