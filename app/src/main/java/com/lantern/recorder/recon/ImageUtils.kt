package com.lantern.recorder.recon

import android.media.Image
import java.nio.ByteOrder

/**
 * Pixel-format helpers for turning ARCore [Image]s into the buffers the reconstruction
 * pipeline needs. All readers copy out of the native buffers so callers can close the
 * [Image] immediately (these must run while the Image is still acquired).
 */
object ImageUtils {

    /** Holds an unpacked color image as packed ARGB_8888. */
    class Argb(val pixels: IntArray, val width: Int, val height: Int)

    /**
     * ARCore raw depth (16-bit, little-endian millimeters) -> meters FloatArray
     * (row-major, 0 = invalid). Matches `FrameRecorder.extractGray16BigEndian` /
     * `MainActivity.estimateCenterDepthMeters` (both read little-endian mm).
     */
    fun depth16ToMeters(image: Image): DepthMap {
        val width = image.width
        val height = image.height
        val plane = image.planes[0]
        val buffer = plane.buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        val rowStride = plane.rowStride
        val pixelStride = if (plane.pixelStride > 0) plane.pixelStride else 2

        val out = FloatArray(width * height)
        for (y in 0 until height) {
            var rowBase = y * rowStride
            for (x in 0 until width) {
                val idx = rowBase + x * pixelStride
                val mm = (buffer.get(idx).toInt() and 0xFF) or
                    ((buffer.get(idx + 1).toInt() and 0xFF) shl 8)
                out[y * width + x] = mm / 1000f
            }
        }
        return DepthMap(width, height, out)
    }

    /** ARCore raw-depth confidence (8-bit) -> 0..1 FloatArray, row-major. */
    fun confidence8ToFloat(image: Image): FloatArray {
        val width = image.width
        val height = image.height
        val plane = image.planes[0]
        val buffer = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixelStride = if (plane.pixelStride > 0) plane.pixelStride else 1

        val out = FloatArray(width * height)
        for (y in 0 until height) {
            val rowBase = y * rowStride
            for (x in 0 until width) {
                out[y * width + x] = (buffer.get(rowBase + x * pixelStride).toInt() and 0xFF) / 255f
            }
        }
        return out
    }

    /** YUV_420_888 -> packed ARGB_8888 (BT.601), matching `FrameRecorder.extractYuv`. */
    fun yuvToArgb(image: Image): Argb {
        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer.duplicate()
        val uBuf = uPlane.buffer.duplicate()
        val vBuf = vPlane.buffer.duplicate()
        val yRowStride = yPlane.rowStride
        val yPixStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixStride = vPlane.pixelStride

        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            val uvRow = (y shr 1)
            for (x in 0 until width) {
                val uvCol = (x shr 1)
                val yVal = yBuf.get(y * yRowStride + x * yPixStride).toInt() and 0xFF
                val uVal = (uBuf.get(uvRow * uRowStride + uvCol * uPixStride).toInt() and 0xFF) - 128
                val vVal = (vBuf.get(uvRow * vRowStride + uvCol * vPixStride).toInt() and 0xFF) - 128

                var r = (yVal + 1.402f * vVal).toInt()
                var g = (yVal - 0.344136f * uVal - 0.714136f * vVal).toInt()
                var b = (yVal + 1.772f * uVal).toInt()
                r = r.coerceIn(0, 255)
                g = g.coerceIn(0, 255)
                b = b.coerceIn(0, 255)
                pixels[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Argb(pixels, width, height)
    }
}
