package com.lantern.recorder.recon

import android.util.Log
import com.google.ar.core.Frame
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs

private const val TAG = "LANTERN"

/**
 * Drives the on-device live mesh: ARCore depth (+ optional DA3 dense depth) -> affine scale
 * -> TSDF fusion -> marching cubes, off the GL thread.
 *
 * Lifecycle:
 *  - [onFrame] is called every GL frame with the current ARCore [Frame]. It cheaply checks
 *    the keyframe gate (camera translated enough), and only then acquires depth/RGB on the
 *    GL thread and hands a job to the worker thread.
 *  - The worker integrates and periodically re-meshes; [latestMesh] / [meshVersion] let the
 *    renderer pick up the newest mesh without blocking.
 *
 * Works with no model present (ARCore-only metric depth). If a DA3 `.pte` is supplied and
 * loads, its dense depth is scaled against ARCore and fused for a smoother surface.
 */
class LiveReconstructor(
    da3ModelPath: String? = null,
    qnnModelPath: String? = null,
    nativeLibDir: String? = null,
    segModelPath: String? = null,
) {
    private val volume = TsdfVolume()
    private val arcoreDepth = ArCoreRawDepthSource()

    /**
     * Dense-depth backend, picked once at construction. Preference order: native QNN DLC
     * (Hexagon NPU) -> ExecuTorch `.pte` (CPU/NPU) -> none (ARCore-only). Each loader returns
     * null when its model/runtime is absent, so this silently degrades on any device.
     */
    private val depth: DepthBackend? =
        qnnModelPath?.let { QnnDlcDepthModel.loadOrNull(it, nativeLibDir) }
            ?: da3ModelPath?.let { ExecuTorchDepthModel.loadOrNull(it) }

    /** FastSAM object segmenter (NPU). When loaded, fusion is gated to the object mask. */
    private val seg: QnnSegmentationModel? = segModelPath?.let { QnnSegmentationModel.loadOrNull(it, nativeLibDir) }

    val segActive: Boolean get() = seg != null

    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "live-recon").apply { isDaemon = true } }
    private val busy = AtomicBoolean(false)
    private val meshRef = AtomicReference(MeshData.EMPTY)
    private val versionRef = AtomicReference(0)

    @Volatile private var lastKeyframePos: FloatArray? = null
    @Volatile private var integratedFrames = 0
    @Volatile private var running = false

    /** Latest ARCore support-plane world-Y (object's surface); voxels below it are culled. */
    @Volatile private var lastGroundPlaneY: Float? = null

    /** Consecutive keyframes the look-target has been far from the current anchor (re-lock debounce). */
    private var reanchorStreak = 0

    @Volatile var debugEnabled = false
    @Volatile private var latestDebugFrame: DebugFrame? = null
    @Volatile private var loggedDims = false

    /** A debug snapshot for the on-screen overlay: a small thumbnail (camera + mask tint) + stats. */
    class DebugFrame(val argb: IntArray, val width: Int, val height: Int, val text: String)

    fun latestDebug(): DebugFrame? = latestDebugFrame

    /** Which dense-depth runtime is live (for the UI readout); ARCORE when no model loaded. */
    val depthBackend: DepthBackendKind get() = depth?.kind ?: DepthBackendKind.ARCORE

    fun start() {
        running = true
    }

    fun stop() {
        running = false
    }

    fun reset() {
        worker.execute {
            volume.reset()
            lastKeyframePos = null
            integratedFrames = 0
            reanchorStreak = 0
            meshRef.set(MeshData.EMPTY)
            versionRef.set(versionRef.get() + 1)
        }
    }

    /** Newest extracted mesh (world meters). Cheap; safe to call from the GL thread. */
    fun latestMesh(): MeshData = meshRef.get()

    /** Bumps whenever a new mesh is published; lets the renderer skip re-uploads. */
    fun meshVersion(): Int = versionRef.get()

    val frameCount: Int get() = integratedFrames

    /**
     * Per-GL-frame hook. Must run on the GL thread while [frame] is current.
     * @param poseColumnMajor ARCore camera-to-world, column-major (from `pose.toMatrix`).
     */
    fun onFrame(
        frame: Frame,
        intrinsics: CameraIntrinsics,
        poseColumnMajor: FloatArray,
        groundPlaneY: Float? = null,
    ) {
        if (!running) return
        val cameraToWorld = Mat4.fromColumnMajor(poseColumnMajor)
        val camX = cameraToWorld[3]
        val camY = cameraToWorld[7]
        val camZ = cameraToWorld[11]
        if (!isKeyframe(camX, camY, camZ)) return
        if (busy.get()) return // worker still chewing the previous keyframe; skip this one

        // Acquire metric depth now (Frame is only valid on this thread, this call).
        val depthMap = arcoreDepth.acquire(frame) ?: return
        val argb = if (depth != null || seg != null) {
            try {
                frame.acquireCameraImage().use { ImageUtils.yuvToArgb(it) }
            } catch (e: Exception) {
                null
            }
        } else {
            null
        }

        // Latch the latest support-plane height for the worker (off-thread integrate uses it).
        if (groundPlaneY != null) lastGroundPlaneY = groundPlaneY

        lastKeyframePos = floatArrayOf(camX, camY, camZ)
        busy.set(true)
        worker.execute {
            try {
                process(depthMap, argb, intrinsics, cameraToWorld)
            } catch (t: Throwable) {
                Log.e(TAG, "live recon worker failed", t)
            } finally {
                busy.set(false)
            }
        }
    }

    private fun process(
        arcoreMetric: DepthMap,
        argb: ImageUtils.Argb?,
        intrinsics: CameraIntrinsics,
        cameraToWorld: FloatArray,
    ) {
        val depthIntrinsics = intrinsics.scaledTo(arcoreMetric.width, arcoreMetric.height)
        if (argb != null && !loggedDims) {
            loggedDims = true
            Log.i(TAG, "DIMS depth=${arcoreMetric.width}x${arcoreMetric.height} " +
                "argb=${argb.width}x${argb.height} intr=${intrinsics.width}x${intrinsics.height}")
        }

        // Distance to whatever's centered in view (the object); used to focus the depth-scale
        // fit on the object instead of the background.
        val focusDepth = centerMedianDepth(arcoreMetric)

        // Object mask (FastSAM) aligned to the depth image; gates the scale fit and fusion so only
        // the segmented object is reconstructed. Null when segmentation is off/nothing at center.
        val mask = if (argb != null && seg != null) {
            seg.inferObjectMask(argb, arcoreMetric.width, arcoreMetric.height)
        } else {
            null
        }

        // Refine the SAM mask against ARCore depth: drop masked pixels whose (valid) depth is far
        // from the object's distance. Removes floor caught by a slight mask/registration offset or
        // the base rim, while keeping object pixels (in-band, or with no ARCore depth) intact.
        if (mask != null && focusDepth != null) {
            refineMaskByDepth(mask, arcoreMetric, focusDepth)
        }

        if (debugEnabled && argb != null) {
            captureDebug(argb, arcoreMetric, mask, focusDepth)
        }

        // Segmentation-gated capture: if the segmenter is active but found no object at the screen
        // center this frame, fuse nothing — otherwise the whole scene (floor) would be integrated
        // ungated and stick in the volume. Only frames with an object mask contribute.
        if (seg != null && mask == null) return

        // Reject bad-mask frames: when FastSAM momentarily grabs the floor/background instead of
        // the centered object, the masked region sits at a different distance than the object
        // (focusDepth). Skip those so transient garbage never bakes into the (permanent) TSDF.
        if (mask != null && focusDepth != null) {
            val maskDepth = maskedMedianDepth(arcoreMetric, mask)
            if (maskDepth != null && kotlin.math.abs(maskDepth - focusDepth) / focusDepth > MASK_DEPTH_TOL) {
                return
            }
        }

        // Prefer DA3 dense metric depth (scaled vs ARCore) when available.
        var fusionDepth = arcoreMetric
        if (argb != null && depth != null) {
            val disp = depth.inferDisparity(argb)
            if (disp != null) {
                val dense = AffineScaleSolver.buildMetricDepth(
                    disp, arcoreMetric, focusDepthM = focusDepth, mask = mask,
                )
                if (dense != null) fusionDepth = dense
            }
        }

        // What the camera is currently pointed at (object center in the world).
        val centroid = estimateCentroidWorld(fusionDepth, depthIntrinsics, cameraToWorld)
        if (centroid != null) {
            if (!volume.isCentered) {
                volume.centerOn(centroid[0], centroid[1], centroid[2])
                reanchorStreak = 0
            } else if (isNewTarget(centroid)) {
                // Pointed at a different object: re-lock there after a short debounce so a brief
                // glance away doesn't wipe the scan, then clear the old mesh and rebuild fresh.
                if (++reanchorStreak >= REANCHOR_FRAMES) {
                    volume.reset()
                    volume.centerOn(centroid[0], centroid[1], centroid[2])
                    meshRef.set(MeshData.EMPTY)
                    versionRef.set(versionRef.get() + 1)
                    integratedFrames = 0
                    reanchorStreak = 0
                }
            } else {
                reanchorStreak = 0
            }
        }
        if (!volume.isCentered) return

        volume.integrate(fusionDepth, depthIntrinsics, cameraToWorld, lastGroundPlaneY, mask)
        integratedFrames++

        if (integratedFrames % EXTRACT_EVERY == 0) {
            val mesh = volume.extractMesh()
            meshRef.set(mesh)
            versionRef.set(versionRef.get() + 1)
        }
    }

    /** True if [centroid] is far enough from the current volume center to be a different object. */
    private fun isNewTarget(centroid: FloatArray): Boolean {
        val half = volume.halfExtent
        val dx = centroid[0] - (volume.origin[0] + half)
        val dy = centroid[1] - (volume.origin[1] + half)
        val dz = centroid[2] - (volume.origin[2] + half)
        return dx * dx + dy * dy + dz * dz > REANCHOR_DIST_M * REANCHOR_DIST_M
    }

    /** Builds the debug thumbnail (camera at depth-res, object mask tinted green) + a stats string. */
    private fun captureDebug(
        argb: ImageUtils.Argb,
        depthMap: DepthMap,
        mask: FloatArray?,
        focusDepth: Float?,
    ) {
        val w = depthMap.width
        val h = depthMap.height
        val out = IntArray(w * h)
        for (dy in 0 until h) {
            val sy = ((dy + 0.5f) / h * argb.height).toInt().coerceIn(0, argb.height - 1)
            for (dx in 0 until w) {
                val sx = ((dx + 0.5f) / w * argb.width).toInt().coerceIn(0, argb.width - 1)
                var p = argb.pixels[sy * argb.width + sx]
                val i = dy * w + dx
                if (mask != null && mask[i] > 0.5f) {
                    // Blend toward green where the object mask is set.
                    val r = ((p shr 16) and 0xFF) / 2
                    val g = 0x80 + (((p shr 8) and 0xFF) / 2)
                    val b = (p and 0xFF) / 2
                    p = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                } else {
                    p = p or (0xFF shl 24)
                }
                out[i] = p
            }
        }
        val coverage = mask?.let { var s = 0; for (v in it) if (v > 0.5f) s++; s.toFloat() / it.size } ?: 0f
        val text = buildString {
            append("backend=${depth?.kind ?: DepthBackendKind.ARCORE}  seg=${if (seg != null) "on" else "off"}\n")
            append("seg.maxScore=${"%.2f".format(seg?.lastMaxScore ?: 0f)}  ")
            append("objFound=${mask != null}  maskCov=${"%.1f".format(coverage * 100)}%\n")
            append("focusDepth=${focusDepth?.let { "%.2fm".format(it) } ?: "—"}  ")
            append("ground=${lastGroundPlaneY?.let { "%.2f".format(it) } ?: "—"}\n")
            append("centered=${volume.isCentered}  frames=$integratedFrames\n")
            append("depth=${w}x$h  argb=${argb.width}x${argb.height}")
        }
        latestDebugFrame = DebugFrame(out, w, h, text)
    }

    /**
     * Refines [mask] in place against ARCore depth: clears any masked pixel whose *valid* depth is
     * farther than [MASK_REFINE_TOL] (fractional) from the object's distance [focusDepth]. Pixels
     * with no ARCore depth (0 / out of range) are kept — a dark/textureless object often has sparse
     * depth, so we must not erase it. This trims floor/background caught by a slight mask or
     * color↔depth registration offset and the base rim, while leaving the object intact.
     */
    private fun refineMaskByDepth(mask: FloatArray, depth: DepthMap, focusDepth: Float) {
        val d = depth.depthMeters
        val n = minOf(d.size, mask.size)
        val tol = focusDepth * MASK_REFINE_TOL
        for (i in 0 until n) {
            if (mask[i] < 0.5f) continue
            val v = d[i]
            if (v <= 0f || v >= 5f) continue // no reliable depth here — keep (could be the object)
            if (abs(v - focusDepth) > tol) mask[i] = 0f
        }
    }

    /** Median depth (m) over the masked (object) pixels — for the bad-mask depth-consistency gate. */
    private fun maskedMedianDepth(depth: DepthMap, mask: FloatArray): Float? {
        val d = depth.depthMeters
        val samples = ArrayList<Float>()
        val n = minOf(d.size, mask.size)
        for (i in 0 until n) {
            if (mask[i] < 0.5f) continue
            val v = d[i]
            if (v > 0f && v < 5f) samples.add(v)
        }
        if (samples.size < 8) return null
        samples.sort()
        return samples[samples.size / 2]
    }

    /** Median depth (m) over the center window of the frame — the distance to the centered object. */
    private fun centerMedianDepth(depth: DepthMap): Float? {
        val cw = depth.width / 4
        val ch = depth.height / 4
        val cx = depth.width / 2
        val cy = depth.height / 2
        val samples = ArrayList<Float>()
        for (y in (cy - ch).coerceAtLeast(0) until (cy + ch).coerceAtMost(depth.height)) {
            for (x in (cx - cw).coerceAtLeast(0) until (cx + cw).coerceAtMost(depth.width)) {
                val d = depth.depthMeters[y * depth.width + x]
                if (d > 0f && d < 5f) samples.add(d)
            }
        }
        if (samples.size < 8) return null
        samples.sort()
        return samples[samples.size / 2]
    }

    /** Center-window object distance back-projected along the camera's forward ray (world point). */
    private fun estimateCentroidWorld(
        depth: DepthMap,
        intrinsics: CameraIntrinsics,
        cameraToWorld: FloatArray,
    ): FloatArray? {
        val median = centerMedianDepth(depth) ?: return null

        // Camera world position + ARCore forward (-Z axis of the rotation) * median depth.
        val px = cameraToWorld[3]
        val py = cameraToWorld[7]
        val pz = cameraToWorld[11]
        val fx = -cameraToWorld[2]
        val fy = -cameraToWorld[6]
        val fz = -cameraToWorld[10]
        return floatArrayOf(px + fx * median, py + fy * median, pz + fz * median)
    }

    private fun isKeyframe(x: Float, y: Float, z: Float): Boolean {
        val last = lastKeyframePos ?: return true
        val dx = x - last[0]
        val dy = y - last[1]
        val dz = z - last[2]
        return abs(dx) + abs(dy) + abs(dz) >= KEYFRAME_TRANSLATION_M ||
            (dx * dx + dy * dy + dz * dz) >= KEYFRAME_TRANSLATION_M * KEYFRAME_TRANSLATION_M
    }

    fun close() {
        running = false
        worker.execute {
            depth?.close()
            seg?.close()
        }
        worker.shutdown()
    }

    companion object {
        private const val KEYFRAME_TRANSLATION_M = 0.02f // 2 cm between fused keyframes
        private const val EXTRACT_EVERY = 5 // re-mesh every N integrations (marching cubes is heavy)

        // Re-lock onto a new object only when the look-target leaves the current volume entirely
        // (well beyond the object) for REANCHOR_FRAMES *consecutive* keyframes. Deliberately large
        // + long so orbiting a single object — where the centroid wanders with handheld motion —
        // never wipes the accumulating scan; only genuinely moving to a new object/area resets.
        private const val REANCHOR_DIST_M = 0.45f
        private const val REANCHOR_FRAMES = 8

        // Skip a frame when the masked region's median depth differs from the centered object's by
        // more than this fraction — i.e. the mask grabbed the floor/background, not the object.
        private const val MASK_DEPTH_TOL = 0.25f

        // Per-pixel mask refinement: drop a masked pixel whose valid ARCore depth differs from the
        // object's distance by more than this fraction. Tighter than MASK_DEPTH_TOL since it's
        // trimming individual floor pixels (registration offset / base rim), not whole frames.
        private const val MASK_REFINE_TOL = 0.12f
    }
}
