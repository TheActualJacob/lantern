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

    /**
     * Object-lock lifecycle (segmentation mode only). The TSDF is permanent, so we never fuse on
     * faith: SEARCHING warms up until FastSAM has held a single object steadily, then LOCKED fuses
     * only frames consistent with that object. A transient floor grab is simply skipped; switching
     * to a genuinely new object requires holding it steadily for [SWITCH_LOCK_MS].
     */
    private enum class LockState { SEARCHING, LOCKED }
    @Volatile private var lockState = LockState.SEARCHING

    // Warmup candidate (SEARCHING): the object distance we're waiting to confirm, and since when.
    private var warmupDepth: Float? = null
    private var warmupSinceNs = 0L

    // The locked object's established distance (LOCKED), tracked as an EMA over consistent frames.
    private var lockedDepth: Float? = null

    // Timestamp of the last frame that was actually on the locked object. If we stay off it longer
    // than [UNLOCK_MS] the lock was wrong (e.g. warmup grabbed the floor) or the object is gone, so
    // we tear the mesh down and re-search instead of leaving bad geometry baked in forever.
    private var lastOnLockNs = 0L

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
            lockState = LockState.SEARCHING
            warmupDepth = null
            warmupSinceNs = 0L
            lockedDepth = null
            lastOnLockNs = 0L
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

        if (debugEnabled && argb != null) {
            captureDebug(argb, arcoreMetric, mask, focusDepth)
        }

        // Segmentation-gated capture: if the segmenter is active but found no object at the screen
        // center this frame, fuse nothing — otherwise the whole scene (floor) would be integrated
        // ungated and stick in the volume. Only frames with an object mask contribute.
        if (seg != null && mask == null) return

        // The object's metric distance (median depth over the mask) — the signal the lock tracks.
        val maskDepth = if (mask != null) maskedMedianDepth(arcoreMetric, mask) else null

        // In-frame sanity: the masked region must agree with the centered object this same frame.
        if (mask != null && maskDepth != null && focusDepth != null &&
            kotlin.math.abs(maskDepth - focusDepth) / focusDepth > MASK_DEPTH_TOL) {
            return
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

        // What the camera is currently pointed at (object surface point in the world).
        val centroid = estimateCentroidWorld(fusionDepth, depthIntrinsics, cameraToWorld)

        // Without segmentation there's nothing to lock onto: keep the original behavior — center
        // once on the look point, re-anchor only after a sustained move, and fuse every frame
        // (the geometric ground/cylinder culls keep the floor out in this mode).
        if (seg == null) {
            if (centroid != null) {
                if (!volume.isCentered) {
                    volume.centerOn(centroid[0], centroid[1], centroid[2]); reanchorStreak = 0
                } else if (isNewTarget(centroid)) {
                    if (++reanchorStreak >= REANCHOR_FRAMES) {
                        volume.reset(); volume.centerOn(centroid[0], centroid[1], centroid[2])
                        meshRef.set(MeshData.EMPTY); versionRef.set(versionRef.get() + 1)
                        integratedFrames = 0; reanchorStreak = 0
                    }
                } else reanchorStreak = 0
            }
            if (!volume.isCentered) return
            fuse(fusionDepth, depthIntrinsics, cameraToWorld, mask)
            return
        }

        // Segmentation mode: drive fusion through the object-lock state machine.
        runLock(maskDepth, centroid, fusionDepth, depthIntrinsics, cameraToWorld, mask)
    }

    /**
     * Object-lock state machine (segmentation mode). Decides whether this frame belongs to the
     * locked object and fuses only if so — so a transient FastSAM floor grab never reaches the
     * permanent TSDF, and a new object is only adopted after it's held steadily.
     */
    private fun runLock(
        maskDepth: Float?,
        centroid: FloatArray?,
        fusionDepth: DepthMap,
        depthIntrinsics: CameraIntrinsics,
        cameraToWorld: FloatArray,
        mask: FloatArray?,
    ) {
        val now = System.nanoTime()
        when (lockState) {
            LockState.SEARCHING -> {
                // Warm up: require FastSAM to hold one object at a steady distance for
                // WARMUP_LOCK_MS before committing. Nothing fuses here, so the mesh starts clean.
                if (maskDepth == null || centroid == null) return
                val cand = warmupDepth
                if (cand == null || kotlin.math.abs(maskDepth - cand) / cand > LOCK_DEPTH_TOL) {
                    warmupDepth = maskDepth
                    warmupSinceNs = now
                } else {
                    warmupDepth = cand + LOCK_DEPTH_EMA_ALPHA * (maskDepth - cand)
                    if (now - warmupSinceNs >= WARMUP_LOCK_MS * 1_000_000L) {
                        volume.centerOn(centroid[0], centroid[1], centroid[2])
                        lockedDepth = warmupDepth
                        integratedFrames = 0
                        lastOnLockNs = now
                        lockState = LockState.LOCKED
                    }
                }
            }
            LockState.LOCKED -> {
                if (maskDepth == null || centroid == null) return // unverifiable; keep lock, don't fuse
                val ld = lockedDepth
                val depthOk = ld != null && kotlin.math.abs(maskDepth - ld) / ld <= LOCK_DEPTH_TOL
                val posOk = !isNewTarget(centroid)
                if (depthOk && posOk) {
                    // On the locked object: track its drifting distance, refresh the timer, fuse.
                    lockedDepth = ld!! + LOCK_DEPTH_EMA_ALPHA * (maskDepth - ld)
                    lastOnLockNs = now
                    fuse(fusionDepth, depthIntrinsics, cameraToWorld, mask)
                } else if (now - lastOnLockNs >= UNLOCK_MS * 1_000_000L) {
                    // Off the locked object too long: the lock was wrong (warmup grabbed the floor)
                    // or the object left. Drop the mesh and re-search so bad geometry can't persist.
                    // A brief excursion (< UNLOCK_MS) is tolerated — the mesh is preserved and this
                    // frame simply doesn't fuse — so a quick glance away never wipes a good scan.
                    teardown()
                }
                // else: brief excursion — skip this frame, keep the mesh, wait it out.
            }
        }
    }

    /** Discard the current scan and return to warmup — used when the lock turns out to be wrong. */
    private fun teardown() {
        volume.reset()
        meshRef.set(MeshData.EMPTY)
        versionRef.set(versionRef.get() + 1)
        integratedFrames = 0
        lockState = LockState.SEARCHING
        warmupDepth = null
        warmupSinceNs = 0L
        lockedDepth = null
        lastOnLockNs = 0L
    }

    /** Integrate one frame into the volume and re-mesh on the extract cadence. */
    private fun fuse(
        fusionDepth: DepthMap,
        depthIntrinsics: CameraIntrinsics,
        cameraToWorld: FloatArray,
        mask: FloatArray?,
    ) {
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
            append("lock=$lockState  lockDepth=${lockedDepth?.let { "%.2fm".format(it) } ?: "—"}\n")
            append("centered=${volume.isCentered}  frames=$integratedFrames\n")
            append("depth=${w}x$h  argb=${argb.width}x${argb.height}")
        }
        latestDebugFrame = DebugFrame(out, w, h, text)
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

        // Object-lock state machine (segmentation mode):
        // Warmup: FastSAM must hold one object at a steady distance this long before anything fuses.
        private const val WARMUP_LOCK_MS = 5_000L
        // Unlock: once we've been off the locked object this long, the lock was wrong (or the object
        // left) — tear the mesh down and re-search, so a bad (e.g. floor) lock can't live forever.
        private const val UNLOCK_MS = 3_000L
        // A frame is "on the locked object" when its mask depth is within this fraction of the
        // locked distance. Keyframes are gated to 2 cm of motion, so a real object drifts only a
        // few percent per frame, while a mask jump to the floor spikes far past this.
        private const val LOCK_DEPTH_TOL = 0.18f
        // EMA smoothing for the locked/warmup distance (higher tracks deliberate motion faster).
        private const val LOCK_DEPTH_EMA_ALPHA = 0.2f
    }
}
