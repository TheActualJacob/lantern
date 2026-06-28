package com.lantern.recorder.recon

import android.util.Log
import java.io.File
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.roundToInt

private const val TAG = "LANTERN"

/**
 * FastSAM-s instance segmentation on the **Qualcomm QNN/QAIRT** Hexagon NPU ([QnnNative]), used to
 * isolate the object the camera is pointed at so the live mesh fuses only it (not the floor).
 *
 * Runs the AI Hub `fastsam_s` context binary (QAIRT 2.45, SM8750/v79). Contract (metadata.json):
 *  - input `image`: NHWC `[1,640,640,3]` float32 `[0,1]` (÷255, no ImageNet norm), letterboxed.
 *  - outputs: `mask_protos [1,160,160,32]`, `mask_coeffs [1,8400,32]`, `boxes [1,8400,4]`,
 *    `scores [1,8400]` (YOLOv8-seg head).
 *
 * We pick the **object under the screen center** by finding the anchor whose mask is strongest at
 * the center pixel (robust to box-format assumptions), then assemble that anchor's mask
 * (`sigmoid(coeffs·protos)`) and return it sampled into the depth image's frame for fusion gating.
 *
 * Falls back (returns null) when the model/native lib is absent — the pipeline then uses the
 * geometric culls alone.
 */
class QnnSegmentationModel private constructor(
    private val handle: Long,
    private val outElems: IntArray,        // graph-order output element counts
    private val idxProtos: Int,
    private val idxCoeffs: Int,
    private val idxScores: Int,
) {
    @Volatile private var closed = false

    /** Debug: highest object score over all anchors last frame (is FastSAM detecting anything?). */
    @Volatile var lastMaxScore = 0f
        private set

    /** Debug: fraction of the 160x160 mask set for the chosen object last frame. */
    @Volatile var lastCoverage = 0f
        private set

    /**
     * Object mask aligned to a depth image of [outW]x[outH] (1f = object, 0f = background), or null
     * if nothing confident is under the center. [argbW]/[argbH] are the source color-image dims
     * used to letterbox into 640x640.
     */
    fun inferObjectMask(argb: ImageUtils.Argb, outW: Int, outH: Int): FloatArray? {
        if (closed) return null
        return try {
            val lb = Letterbox(argb.width, argb.height)
            val input = preprocess(argb, lb)
            val outputs = Array(outElems.size) { FloatArray(outElems[it]) }
            if (!QnnNative.nativeInferN(handle, arrayOf(input), outputs)) {
                Log.w(TAG, "FastSAM inference failed"); return null
            }
            val protos = outputs[idxProtos]   // [160,160,32] HWC
            val coeffs = outputs[idxCoeffs]   // [8400,32]
            val scores = outputs[idxScores]   // [8400]
            val best = pickCenterAnchor(protos, coeffs, scores)
            if (best < 0) { lastCoverage = 0f; return null }
            val mask160 = anchorMask(protos, coeffs, best)
            keepLargestComponent(mask160, PROTO, PROTO) // drop disconnected edge-noise islands
            var set = 0
            for (m in mask160) if (m > 0f) set++
            lastCoverage = set.toFloat() / mask160.size
            sampleToDepth(mask160, lb, outW, outH)
        } catch (t: Throwable) {
            Log.e(TAG, "FastSAM segmentation failed", t)
            null
        }
    }

    /** Aspect-preserving fit of [srcW]x[srcH] into [SEG_RES] with centered padding. */
    private class Letterbox(srcW: Int, srcH: Int) {
        val scale = SEG_RES.toFloat() / max(srcW, srcH)
        val newW = (srcW * scale).roundToInt()
        val newH = (srcH * scale).roundToInt()
        val padX = (SEG_RES - newW) / 2
        val padY = (SEG_RES - newH) / 2
    }

    /** ARGB -> NHWC [0,1] 640x640 with letterbox padding (gray 0.5 like YOLO). */
    private fun preprocess(argb: ImageUtils.Argb, lb: Letterbox): FloatArray {
        val out = FloatArray(SEG_RES * SEG_RES * 3) { 0.5f }
        val sx = argb.width.toFloat() / lb.newW
        val sy = argb.height.toFloat() / lb.newH
        for (oy in 0 until lb.newH) {
            val srcY = (oy * sy).toInt().coerceIn(0, argb.height - 1)
            val rowBase = srcY * argb.width
            var o = ((lb.padY + oy) * SEG_RES + lb.padX) * 3
            for (ox in 0 until lb.newW) {
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

    /** Anchor whose mask is strongest at the center proto pixel (the object you're pointing at). */
    private fun pickCenterAnchor(protos: FloatArray, coeffs: FloatArray, scores: FloatArray): Int {
        val cx = PROTO / 2
        val cy = PROTO / 2
        val base = (cy * PROTO + cx) * PROTO_C
        var bestIdx = -1
        var bestKey = 0f
        var maxScore = 0f
        for (a in 0 until NUM_ANCHORS) {
            val s = scores[a]
            if (s > maxScore) maxScore = s
            if (s < SCORE_THRESH) continue
            var dot = 0f
            val cb = a * PROTO_C
            for (c in 0 until PROTO_C) dot += coeffs[cb + c] * protos[base + c]
            val m = sigmoid(dot)
            if (m < MASK_THRESH) continue
            val key = s * m
            if (key > bestKey) { bestKey = key; bestIdx = a }
        }
        lastMaxScore = maxScore
        return bestIdx
    }

    /** Assemble the binary 160x160 mask for one anchor: sigmoid(coeffs·protos) > MASK_THRESH. */
    private fun anchorMask(protos: FloatArray, coeffs: FloatArray, anchor: Int): FloatArray {
        val cb = anchor * PROTO_C
        val mask = FloatArray(PROTO * PROTO)
        for (py in 0 until PROTO) {
            for (px in 0 until PROTO) {
                val base = (py * PROTO + px) * PROTO_C
                var dot = 0f
                for (c in 0 until PROTO_C) dot += coeffs[cb + c] * protos[base + c]
                if (sigmoid(dot) > MASK_THRESH) mask[py * PROTO + px] = 1f
            }
        }
        return mask
    }

    /** Keep only the largest 4-connected blob of [mask] (in place); zero the rest. */
    private fun keepLargestComponent(mask: FloatArray, w: Int, h: Int) {
        val labels = IntArray(w * h) { -1 }
        val stack = IntArray(w * h)
        var bestLabel = -1
        var bestSize = 0
        var label = 0
        for (start in mask.indices) {
            if (mask[start] <= 0f || labels[start] != -1) continue
            var sp = 0
            stack[sp++] = start
            labels[start] = label
            var size = 0
            while (sp > 0) {
                val p = stack[--sp]
                size++
                val x = p % w
                val y = p / w
                if (x > 0 && mask[p - 1] > 0f && labels[p - 1] == -1) { labels[p - 1] = label; stack[sp++] = p - 1 }
                if (x < w - 1 && mask[p + 1] > 0f && labels[p + 1] == -1) { labels[p + 1] = label; stack[sp++] = p + 1 }
                if (y > 0 && mask[p - w] > 0f && labels[p - w] == -1) { labels[p - w] = label; stack[sp++] = p - w }
                if (y < h - 1 && mask[p + w] > 0f && labels[p + w] == -1) { labels[p + w] = label; stack[sp++] = p + w }
            }
            if (size > bestSize) { bestSize = size; bestLabel = label }
            label++
        }
        if (bestLabel < 0) return
        for (i in mask.indices) if (labels[i] != bestLabel) mask[i] = 0f
    }

    /** Resample the proto-space mask into a depth image of [outW]x[outH] via the letterbox map. */
    private fun sampleToDepth(mask160: FloatArray, lb: Letterbox, outW: Int, outH: Int): FloatArray {
        val out = FloatArray(outW * outH)
        for (dy in 0 until outH) {
            val ny = (dy + 0.5f) / outH
            val segY = lb.padY + ny * lb.newH        // 640-space
            val py = (segY / 4f).toInt()             // 160-space
            if (py < 0 || py >= PROTO) continue
            for (dx in 0 until outW) {
                val nx = (dx + 0.5f) / outW
                val segX = lb.padX + nx * lb.newW
                val px = (segX / 4f).toInt()
                if (px < 0 || px >= PROTO) continue
                out[dy * outW + dx] = mask160[py * PROTO + px]
            }
        }
        return out
    }

    fun close() {
        if (closed) return
        closed = true
        try {
            QnnNative.nativeClose(handle)
        } catch (t: Throwable) {
            Log.w(TAG, "FastSAM close failed", t)
        }
    }

    companion object {
        private const val SEG_RES = 640
        private const val PROTO = 160
        private const val PROTO_C = 32
        private const val NUM_ANCHORS = 8400
        private const val SCORE_THRESH = 0.4f
        private const val MASK_THRESH = 0.5f

        private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

        /**
         * Load FastSAM from a `.bin`/`.dlc` path. Null when the native lib / model is missing, or
         * the graph outputs don't match the expected FastSAM head.
         */
        fun loadOrNull(modelPath: String, nativeLibDir: String? = null): QnnSegmentationModel? {
            if (!QnnNative.available) return null
            if (!File(modelPath).exists()) {
                Log.i(TAG, "FastSAM model not present ($modelPath); segmentation off")
                return null
            }
            QnnNative.setAdspLibraryPath(nativeLibDir)
            return try {
                val handle = QnnNative.nativeInit(modelPath, SEG_RES)
                if (handle == 0L) { Log.w(TAG, "FastSAM nativeInit failed"); return null }
                val names = QnnNative.nativeOutputNames(handle)
                val outElems = QnnNative.nativeIoElems(handle, 1)
                val protos = names.indexOfFirst { it.contains("proto") }
                val coeffs = names.indexOfFirst { it.contains("coeff") }
                val scores = names.indexOfFirst { it.contains("score") }
                if (protos < 0 || coeffs < 0 || scores < 0) {
                    Log.w(TAG, "FastSAM outputs unexpected: ${names.joinToString()}")
                    QnnNative.nativeClose(handle); return null
                }
                Log.i(TAG, "Loaded FastSAM QNN model: $modelPath")
                QnnSegmentationModel(handle, outElems, protos, coeffs, scores)
            } catch (t: Throwable) {
                Log.w(TAG, "FastSAM not loaded ($modelPath); ${t.message}")
                null
            }
        }
    }
}
