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

    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "live-recon").apply { isDaemon = true } }
    private val busy = AtomicBoolean(false)
    private val meshRef = AtomicReference(MeshData.EMPTY)
    private val versionRef = AtomicReference(0)

    @Volatile private var lastKeyframePos: FloatArray? = null
    @Volatile private var integratedFrames = 0
    @Volatile private var running = false

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
    fun onFrame(frame: Frame, intrinsics: CameraIntrinsics, poseColumnMajor: FloatArray) {
        if (!running) return
        val cameraToWorld = Mat4.fromColumnMajor(poseColumnMajor)
        val camX = cameraToWorld[3]
        val camY = cameraToWorld[7]
        val camZ = cameraToWorld[11]
        if (!isKeyframe(camX, camY, camZ)) return
        if (busy.get()) return // worker still chewing the previous keyframe; skip this one

        // Acquire metric depth now (Frame is only valid on this thread, this call).
        val depthMap = arcoreDepth.acquire(frame) ?: return
        val argb = if (depth != null) {
            try {
                frame.acquireCameraImage().use { ImageUtils.yuvToArgb(it) }
            } catch (e: Exception) {
                null
            }
        } else {
            null
        }

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

        // Prefer DA3 dense metric depth (scaled vs ARCore) when available.
        var fusionDepth = arcoreMetric
        if (argb != null && depth != null) {
            val disp = depth.inferDisparity(argb)
            if (disp != null) {
                val dense = AffineScaleSolver.buildMetricDepth(disp, arcoreMetric)
                if (dense != null) fusionDepth = dense
            }
        }

        if (!volume.isCentered) {
            val centroid = estimateCentroidWorld(fusionDepth, depthIntrinsics, cameraToWorld)
            if (centroid != null) volume.centerOn(centroid[0], centroid[1], centroid[2])
        }
        if (!volume.isCentered) return

        volume.integrate(fusionDepth, depthIntrinsics, cameraToWorld)
        integratedFrames++

        if (integratedFrames % EXTRACT_EVERY == 0) {
            val mesh = volume.extractMesh()
            meshRef.set(mesh)
            versionRef.set(versionRef.get() + 1)
        }
    }

    /** Median center-window depth back-projected along the camera's forward ray. */
    private fun estimateCentroidWorld(
        depth: DepthMap,
        intrinsics: CameraIntrinsics,
        cameraToWorld: FloatArray,
    ): FloatArray? {
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
        val median = samples[samples.size / 2]

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
        worker.execute { depth?.close() }
        worker.shutdown()
    }

    companion object {
        private const val KEYFRAME_TRANSLATION_M = 0.02f // 2 cm between fused keyframes
        private const val EXTRACT_EVERY = 5 // re-mesh every N integrations (marching cubes is heavy)
    }
}
