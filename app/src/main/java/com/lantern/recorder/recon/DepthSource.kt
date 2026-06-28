package com.lantern.recorder.recon

import android.util.Log
import com.google.ar.core.Frame

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
 * object scale — DA3 (a [DepthBackend]) supplies the dense surface, this supplies the meter.
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
