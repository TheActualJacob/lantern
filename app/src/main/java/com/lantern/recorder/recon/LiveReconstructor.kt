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
    private val da3ModelPath: String? = null,
    private val qnnModelPath: String? = null,
    private val nativeLibDir: String? = null,
    segModelPath: String? = null,
    private val mvModelPath: String? = null,
) {
    private val volume = TsdfVolume()
    private val arcoreDepth = ArCoreRawDepthSource()

    /**
     * Multi-view DA3 backend for the directed *build* (not live). Loaded lazily on first build so
     * its ~100 MB `.pte` doesn't add to startup; the live path never touches it. Null if no
     * multi-view `.pte` is present — directed build then uses the mono per-frame path.
     */
    @Volatile private var multiView: Da3MultiViewModel? = null
    @Volatile private var multiViewLoaded = false

    /**
     * Dense-depth backend. Preference order: native QNN DLC (Hexagon NPU) -> ExecuTorch `.pte`
     * (CPU/XNNPACK) -> none (ARCore-only). Loaded **asynchronously on the worker thread** (see the
     * `init` block) so the ~100 MB DA3 `.pte` load never blocks the GL/main thread or delays the
     * FastSAM segmenter — that's what previously made it look like SAM + DA3 "can't run together".
     * Null until the load finishes (or if no model/runtime is present); the pipeline uses ARCore
     * depth meanwhile and seamlessly upgrades to dense DA3 depth once this is set.
     */
    @Volatile private var depth: DepthBackend? = null

    /** FastSAM object segmenter (NPU). Loaded eagerly (fast, essential for the mask gate). */
    private val seg: QnnSegmentationModel? = segModelPath?.let { QnnSegmentationModel.loadOrNull(it, nativeLibDir) }

    val segActive: Boolean get() = seg != null

    /** Temporal hysteresis over the raw SAM mask: slow to accept new region (suppresses background
     *  noise), quick to drop a region SAM stops reporting (background removed / different object).
     *  Touched only on the preview worker. */
    private val maskStabilizer = MaskStabilizer()

    /** Single, stable DA3 disparity->metric scale for the whole scan. Replaces per-frame fitting
     *  against noisy ARCore depth (which jittered the object's size/pose and smeared the mesh).
     *  Touched only on the fusion worker. */
    private val scaleTracker = DepthScaleTracker()

    // Two workers, two cadences. [worker] runs the *fast* preview (FastSAM mask + debug overlay,
    // ~tens of ms) every gated frame so the live mask tracks the object in real time. [fusionWorker]
    // runs the *heavy* path (DA3 dense depth ~seconds on CPU + TSDF fusion + marching cubes) and
    // drops frames that arrive while it's busy — so dense-depth latency never freezes the mask.
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "live-recon").apply { isDaemon = true } }
    private val busy = AtomicBoolean(false)
    private val fusionWorker = Executors.newSingleThreadExecutor { r -> Thread(r, "live-fusion").apply { isDaemon = true } }
    private val fusionBusy = AtomicBoolean(false)
    private val meshRef = AtomicReference(MeshData.EMPTY)
    private val versionRef = AtomicReference(0)

    init {
        // DA3 runs on the ExecuTorch runtime (CPU/XNNPACK), a different stack from FastSAM's QNN/NPU
        // context, so the two coexist. Load it off-thread to keep startup snappy and SAM unblocked.
        worker.execute { loadMonoDepthIfNeeded() }
    }

    /** Load the live mono DA3 backend if not already present. Idempotent; safe to call on [start].
     *  Used both at init and to re-acquire it after a directed build freed it for memory headroom. */
    private fun loadMonoDepthIfNeeded() {
        if (depth != null) return
        val loaded = qnnModelPath?.let { QnnDlcDepthModel.loadOrNull(it, nativeLibDir) }
            ?: da3ModelPath?.let { ExecuTorchDepthModel.loadOrNull(it) }
        if (loaded != null) {
            depth = loaded
            Log.i(TAG, "Dense depth backend ready: ${loaded.kind}")
        }
    }

    @Volatile private var lastKeyframePos: FloatArray? = null
    @Volatile private var integratedFrames = 0
    @Volatile private var running = false

    /** Wall-clock time of the last processed frame; drives the warmup poll cadence when stationary. */
    @Volatile private var lastProcessNs = 0L

    /** Latest ARCore support-plane world-Y (object's surface); voxels below it are culled. */
    @Volatile private var lastGroundPlaneY: Float? = null

    /** Consecutive keyframes the look-target has been far from the current anchor (re-lock debounce). */
    private var reanchorStreak = 0

    /**
     * In-hand object-frame tracker (segmentation mode only, LIVE_MESH_PLAN §3). Registers each
     * frame's masked object cloud to the model and returns `T_OC` (OpenCV-camera -> object). Fusion
     * happens in the object's own frame, so turning the object in hand is the *expected* case — no
     * warmup/lock gating, because we no longer assume the object is static in the world.
     */
    private val tracker = ObjectTracker()

    /** Object-frame pose (`T_OC`, row-major) of the last tracked frame; null until first lock. */
    @Volatile private var latestObjectPose: FloatArray? = null

    /**
     * Object-frame -> ARCore world transform (row-major) of the last tracked frame, computed in the
     * worker from *that frame's* camera pose + `T_OC` (so the two are consistent). The renderer uses
     * this directly as the mesh model matrix; for a static object it's constant, so the overlay
     * stays glued smoothly even as the camera moves every frame. Null until first lock.
     */
    @Volatile private var latestObjectToWorld: FloatArray? = null

    /** `T_OC` at the last *fused* frame, so we only fuse genuinely new viewpoints of the object. */
    private var lastFusedPose: FloatArray? = null

    /**
     * State from the last *accepted* (trusted) track, used to seed the next ICP with ARCore
     * ego-motion: `T_OC` and the OpenCV camera->world at that frame. The relative camera motion
     * since then is a far stronger ICP init than constant velocity (verified offline: it removes
     * the single-bad-frame divergence that otherwise sends the whole track 40°+ off and never
     * recovers). Both advance together only on an accepted frame, so dropped frames don't poison it.
     */
    private var lastGoodObjectPose: FloatArray? = null
    private var lastGoodCamToWorldCv: FloatArray? = null

    @Volatile var debugEnabled = false
    @Volatile private var latestDebugFrame: DebugFrame? = null
    @Volatile private var loggedDims = false

    /** A debug snapshot for the on-screen overlay: a small thumbnail (camera + mask tint) + stats. */
    class DebugFrame(val argb: IntArray, val width: Int, val height: Int, val text: String)

    fun latestDebug(): DebugFrame? = latestDebugFrame

    /** Which dense-depth runtime is live (for the UI readout); ARCORE when no model loaded. */
    val depthBackend: DepthBackendKind get() = depth?.kind ?: DepthBackendKind.ARCORE

    /**
     * A [BatchReconstructor] sharing this reconstructor's already-loaded DA3 + FastSAM models, for
     * directed-capture's end-of-capture build pass. Reusing the loaded models avoids a second
     * ~100 MB DA3 load and a second QNN/NPU context (which would contend with the live segmenter).
     * Call only when live fusion is stopped (directed mode), so the models aren't used concurrently.
     */
    fun newBatchReconstructor(): BatchReconstructor {
        // Lazy-load the multi-view model on first build (directed mode only), so its weight load
        // never delays startup or the live segmenter. Cached for subsequent builds.
        if (!multiViewLoaded) {
            multiViewLoaded = true
            multiView = mvModelPath?.let { Da3MultiViewModel.loadOrNull(it) }
        }
        // Free the mono DA3 before the heavy multi-view build. It's a ~100 MB ExecuTorch/XNNPACK
        // model used only by the *live* path (LiveMesh) — never by the directed build — so keeping it
        // resident while the multi-view model's own XNNPACK weights cache loads makes two big models
        // coexist, which OOM-kills the process (crash in XNNWeightsCache::look_up_or_insert). Directed
        // capture stopped live fusion, so the live backend is idle; release it for build headroom.
        // ([start] re-acquires it via loadMonoDepthIfNeeded, so LiveMesh after a build isn't stranded.)
        if (multiView != null) {
            depth?.close()
            depth = null
        }
        return BatchReconstructor(depth, seg, multiView)
    }

    /**
     * Run FastSAM once on [argb], returning the object mask at [outW]x[outH] (1=object, 0=background)
     * or null if no segmenter / nothing found. For the directed-capture live preview overlay; call
     * off the GL thread (NPU). Independent of live fusion, so it's safe while [running] is false.
     */
    fun inferPreviewMask(argb: ImageUtils.Argb, outW: Int, outH: Int): FloatArray? =
        seg?.inferObjectMask(argb, outW, outH)

    fun start() {
        running = true
        // Re-acquire the mono DA3 backend if a prior directed build freed it (LiveMesh needs it).
        worker.execute { loadMonoDepthIfNeeded() }
    }

    fun stop() {
        running = false
    }

    fun reset() {
        // Mask hysteresis is owned by the preview worker — clear it there.
        worker.execute { maskStabilizer.reset() }
        // Volume/tracker/mesh are owned by the fusion worker — reset there to avoid racing a fuse.
        fusionWorker.execute {
            volume.reset()
            tracker.reset()
            scaleTracker.reset()
            lastKeyframePos = null
            integratedFrames = 0
            reanchorStreak = 0
            latestObjectPose = null
            latestObjectToWorld = null
            lastFusedPose = null
            lastGoodObjectPose = null
            lastGoodCamToWorldCv = null
            meshRef.set(MeshData.EMPTY)
            versionRef.set(versionRef.get() + 1)
        }
    }

    /** Newest extracted mesh (world meters). Cheap; safe to call from the GL thread. */
    fun latestMesh(): MeshData = meshRef.get()

    /** Bumps whenever a new mesh is published; lets the renderer skip re-uploads. */
    fun meshVersion(): Int = versionRef.get()

    /**
     * Object-frame -> world model matrix (row-major) for the renderer to overlay the object-frame
     * mesh on the real object. Null when not tracking (segmentation off / not yet locked) => draw
     * the mesh directly in world space.
     */
    fun latestObjectToWorld(): FloatArray? = latestObjectToWorld

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
        // Gating differs by mode. Without segmentation the object is assumed static in the world, so
        // a fresh viewpoint == camera motion (the keyframe gate). With segmentation we track the
        // object itself, which may move while the camera is still — camera motion is no longer a
        // proxy for "new viewpoint" — so we process on a time cadence and let the tracker's pose
        // delta decide what actually fuses.
        val nowNs = System.nanoTime()
        val gateOpen = if (seg != null) {
            (nowNs - lastProcessNs) >= SEG_MIN_INTERVAL_MS * 1_000_000L
        } else {
            isKeyframe(camX, camY, camZ)
        }
        if (!gateOpen) return
        if (busy.get()) return // worker still chewing the previous frame; skip this one

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
        lastProcessNs = nowNs
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
        // the segmented object is reconstructed. The raw per-frame mask flickers and sometimes grabs
        // a background anchor, so we run it through temporal hysteresis (slow to add, quick to drop)
        // and fuse only the *stabilized* mask. Null when segmentation is off / nothing locked.
        val tSam0 = System.nanoTime()
        val mask = if (seg != null) {
            val raw = if (argb != null) {
                seg.inferObjectMask(argb, arcoreMetric.width, arcoreMetric.height)
            } else {
                null
            }
            maskStabilizer.update(raw, arcoreMetric.width, arcoreMetric.height)
        } else {
            null
        }
        if (debugEnabled) {
            Log.i(TAG, "PERF sam=${(System.nanoTime() - tSam0) / 1_000_000L}ms on=${maskStabilizer.lastOnCount}")
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

        // Hand the freshest masked frame to the (slow) DA3 + fusion worker. SAM + the debug overlay
        // above already ran this frame for a real-time mask; DA3 (~seconds on CPU) + TSDF fusion run
        // on a separate worker that drops intermediate frames while busy, so dense-depth latency
        // never freezes the live mask/preview.
        if (fusionBusy.compareAndSet(false, true)) {
            val job = FusionJob(arcoreMetric, argb, mask, focusDepth, depthIntrinsics, cameraToWorld)
            fusionWorker.execute {
                try {
                    runFusion(job)
                } catch (t: Throwable) {
                    Log.e(TAG, "live fusion worker failed", t)
                } finally {
                    fusionBusy.set(false)
                }
            }
        }
    }

    /** Snapshot handed from the fast preview path to the heavy DA3 + TSDF fusion worker. */
    private class FusionJob(
        val arcoreMetric: DepthMap,
        val argb: ImageUtils.Argb?,
        val mask: FloatArray?,
        val focusDepth: Float?,
        val depthIntrinsics: CameraIntrinsics,
        val cameraToWorld: FloatArray,
    )

    /**
     * Heavy per-frame work, decoupled from the live mask preview: DA3 dense metric depth (scaled vs
     * ARCore) then TSDF fusion (world-frame for the orbit case, ICP for in-hand). Runs on
     * [fusionWorker]; frames arriving while it's busy are dropped — the preview keeps tracking.
     */
    private fun runFusion(job: FusionJob) {
        val arcoreMetric = job.arcoreMetric
        // Geometry comes from DA3 dense depth. ARCore depth is *only* used to fit a one-time scale
        // (DA3 is relative), which we lock via [scaleTracker] so its per-frame noise stops warping
        // the mesh. Fall back to raw ARCore depth solely when no DA3 model is loaded at all.
        var fusionDepth: DepthMap = arcoreMetric
        val denseBackend = depth // capture the volatile var for a stable smart-cast
        val tDa30 = System.nanoTime()
        if (job.argb != null && denseBackend != null) {
            val disp = denseBackend.inferDisparity(job.argb)
                ?: return // DA3 failed this frame; skip rather than fuse noisy ARCore depth.
            val candidate = AffineScaleSolver.fitAffine(
                disp, arcoreMetric, focusDepthM = job.focusDepth, mask = job.mask,
            )
            val scale = scaleTracker.update(candidate)
                ?: return // global scale not established yet (warmup); don't fuse until it locks in.
            fusionDepth = AffineScaleSolver.applyAffine(
                disp, arcoreMetric.width, arcoreMetric.height, scale,
            )
        }
        if (debugEnabled) {
            Log.i(TAG, "PERF da3=${(System.nanoTime() - tDa30) / 1_000_000L}ms " +
                "scale=${"%.3f".format(scaleTracker.s)},${"%.3f".format(scaleTracker.t)} " +
                "locked=${scaleTracker.locked}")
        }

        // Without segmentation there's nothing to track: keep the original world-frame behavior —
        // center once on the look point, re-anchor only after a sustained move, and fuse every
        // frame (the geometric ground/cylinder culls keep the floor out in this mode).
        if (seg == null) {
            val centroid = estimateCentroidWorld(fusionDepth, job.depthIntrinsics, job.cameraToWorld)
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
            fuse(fusionDepth, job.depthIntrinsics, job.cameraToWorld, job.mask)
            return
        }

        // Segmentation mode. Two regimes:
        //  * Static object, camera orbits (the common case): ARCore already tracks the camera, so
        //    the relative pose between frames is *known* — no registration needed. We fuse the
        //    SAM-masked depth straight into a world-frame TSDF using the ARCore pose. Robust and
        //    geometry-independent (works on flat/smooth objects ICP can't grip), because we never
        //    try to recover the object's motion — the object doesn't move.
        //  * Object turned in hand: ARCore camera pose no longer encodes the object's motion, so we
        //    must register cloud-to-model with ICP (fragile on low-relief/smooth objects).
        if (SEG_USE_ARCORE_POSE) {
            fuseStaticObject(job.cameraToWorld, fusionDepth, job.depthIntrinsics, job.mask!!)
        } else {
            trackAndFuse(job.cameraToWorld, fusionDepth, job.depthIntrinsics, job.mask!!)
        }
    }

    /**
     * "SAM mask + depth + known camera pose" fusion for a **static** object the camera orbits.
     * No ICP: ARCore's world tracking supplies the per-frame pose, the mask removes the floor, and
     * the TSDF's sphere cull bounds it to the object. This sidesteps the registration/drift problem
     * entirely whenever the object isn't moving — which is the orbit workflow.
     */
    private fun fuseStaticObject(
        cameraToWorld: FloatArray,
        fusionDepth: DepthMap,
        depthIntrinsics: CameraIntrinsics,
        mask: FloatArray,
    ) {
        if (!volume.isCentered) {
            val c = maskedCentroidWorld(fusionDepth, depthIntrinsics, mask, cameraToWorld) ?: return
            volume.centerOn(c[0], c[1], c[2])
            integratedFrames = 0
        }
        // Mesh is anchored in the world (object is static), so the renderer draws it directly —
        // no object->world matrix. Clear any stale value from a previous in-hand session.
        latestObjectPose = null
        latestObjectToWorld = null
        fuse(fusionDepth, depthIntrinsics, cameraToWorld, mask)
    }

    /** World-space centroid of the masked object cloud (for centering the TSDF grid on it). */
    private fun maskedCentroidWorld(
        depth: DepthMap,
        intr: CameraIntrinsics,
        mask: FloatArray,
        cameraToWorld: FloatArray,
    ): FloatArray? {
        val cloud = backprojectMasked(depth, intr, mask) // OpenCV camera frame
        val n = cloud.size / 3
        if (n < MIN_TRACK_POINTS) return null
        val c2wCv = Mat4.multiply(cameraToWorld, FLIP) // OpenCV-camera -> world
        var sx = 0f; var sy = 0f; var sz = 0f
        val w = FloatArray(3)
        for (i in 0 until n) {
            Mat4.transformPoint(c2wCv, cloud[i * 3], cloud[i * 3 + 1], cloud[i * 3 + 2], w)
            sx += w[0]; sy += w[1]; sz += w[2]
        }
        val inv = 1f / n
        return floatArrayOf(sx * inv, sy * inv, sz * inv)
    }

    /**
     * In-hand object-frame fusion. Back-projects the masked object into the camera frame, recovers
     * its pose `T_OC` against the model, and fuses the depth into the TSDF *in the object frame* —
     * so the reconstruction grows as the object is rotated, with no dependence on world pose.
     */
    private fun trackAndFuse(
        cameraToWorld: FloatArray,
        fusionDepth: DepthMap,
        depthIntrinsics: CameraIntrinsics,
        mask: FloatArray,
    ) {
        val cloud = backprojectMasked(fusionDepth, depthIntrinsics, mask)
        val n = cloud.size / 3
        if (n < MIN_TRACK_POINTS) return // too little object visible to register reliably

        // ARCore-ego-motion seed: predict this frame's T_OC by carrying the last trusted pose along
        // the camera's relative motion since that frame. camCv = OpenGL camera->world flipped into
        // OpenCV (the frame the cloud lives in). rel maps the current camera into the last-good one.
        val camCv = Mat4.multiply(cameraToWorld, FLIP)
        val seed = run {
            val lastTOC = lastGoodObjectPose
            val lastCam = lastGoodCamToWorldCv
            if (lastTOC != null && lastCam != null) {
                val rel = Mat4.multiply(Mat4.invertRigid(lastCam), camCv) // cur cam -> last-good cam
                Mat4.multiply(lastTOC, rel)
            } else {
                null
            }
        }

        val tOC = tracker.track(cloud, n, seed)
        // Drop untrusted registrations: keep the previous mesh + overlay, fuse nothing. This is what
        // stops bad frames from smearing the object into a drifting blob (bounded by the cull into a
        // "sphere") — only confidently-registered frames contribute.
        if (!tracker.lastAccepted) return
        latestObjectPose = tOC
        lastGoodObjectPose = tOC
        lastGoodCamToWorldCv = camCv

        // The first tracked frame defines object frame O (== that camera's OpenCV frame). Center the
        // TSDF grid on the object's centroid in O so the fixed grid surrounds it.
        if (!volume.isCentered) {
            val c = centroidInObject(cloud, n, tOC)
            volume.centerOn(c[0], c[1], c[2])
            integratedFrames = 0
            lastFusedPose = null
        }

        // Object->world from *this* frame's camera pose + T_OC (consistent pair), for a smooth,
        // non-swimming overlay. poseForFusion = T_OC·flip maps object-frame voxels into the OpenCV
        // camera for fusion; its inverse composed with cameraToWorld places the mesh in the world.
        val poseForFusion = Mat4.multiply(tOC, FLIP)
        latestObjectToWorld = Mat4.multiply(cameraToWorld, Mat4.invertRigid(poseForFusion))

        // Fuse only genuinely new viewpoints (the object/camera moved enough since the last fuse),
        // so a still object doesn't pile redundant weight into the same voxels.
        if (shouldFuse(tOC)) {
            fuse(fusionDepth, depthIntrinsics, poseForFusion, mask)
            lastFusedPose = tOC.copyOf()
        }
    }

    /** Back-project masked, valid-depth pixels into the OpenCV camera frame (flat xyz, metres). */
    private fun backprojectMasked(
        depth: DepthMap,
        intr: CameraIntrinsics,
        mask: FloatArray,
    ): FloatArray {
        val d = depth.depthMeters
        val w = depth.width
        val out = ArrayList<Float>(4096)
        val n = minOf(d.size, mask.size)
        var i = 0
        while (i < n) {
            if (mask[i] >= 0.5f) {
                val z = d[i]
                if (z > 0f && z < 5f) {
                    val u = i % w
                    val v = i / w
                    out.add((u - intr.cx) / intr.fx * z)
                    out.add((v - intr.cy) / intr.fy * z)
                    out.add(z)
                }
            }
            i++
        }
        return out.toFloatArray()
    }

    /** Centroid of a camera-frame cloud expressed in object frame O (transform by `T_OC`). */
    private fun centroidInObject(cloud: FloatArray, count: Int, tOC: FloatArray): FloatArray {
        var sx = 0f; var sy = 0f; var sz = 0f
        for (i in 0 until count) { sx += cloud[i * 3]; sy += cloud[i * 3 + 1]; sz += cloud[i * 3 + 2] }
        val inv = 1f / count
        val o = FloatArray(3)
        Mat4.transformPoint(tOC, sx * inv, sy * inv, sz * inv, o)
        return o
    }

    /** True if the object moved enough (rotation or translation) since the last fused frame. */
    private fun shouldFuse(tOC: FloatArray): Boolean {
        val last = lastFusedPose ?: return true
        val rel = Mat4.multiply(tOC, Mat4.invertRigid(last))
        val cos = ((rel[0] + rel[5] + rel[10]) - 1f) / 2f
        val angDeg = Math.toDegrees(kotlin.math.acos(cos.coerceIn(-1f, 1f)).toDouble())
        val t = kotlin.math.sqrt(rel[3] * rel[3] + rel[7] * rel[7] + rel[11] * rel[11])
        return angDeg >= FUSE_ROT_DEG || t >= FUSE_TRANS_M
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
        // Re-mesh early (first few fuses) for instant feedback, then on the steadier cadence so
        // marching cubes doesn't thrash the worker once the scan is established.
        val due = if (integratedFrames <= EXTRACT_EVERY) true else integratedFrames % EXTRACT_EVERY == 0
        if (due) {
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
            if (SEG_USE_ARCORE_POSE) {
                append("mode=ARCore-pose (static/orbit)\n")
            } else {
                append("mode=ICP  track: frames=${tracker.trackedFrames} ")
                append("fit=${"%.2f".format(tracker.lastFitness)} modelVox=${tracker.modelVoxelCount}\n")
            }
            append("scale=${"%.3f".format(scaleTracker.s)},${"%.3f".format(scaleTracker.t)} ")
            append("${if (scaleTracker.locked) "LOCKED" else if (scaleTracker.ready) "warming" else "—"}\n")
            append("centered=${volume.isCentered}  fused=$integratedFrames\n")
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
        worker.execute { seg?.close() }
        fusionWorker.execute { depth?.close(); multiView?.close() }
        worker.shutdown()
        fusionWorker.shutdown()
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

        // Segmentation mode: prefer the robust "SAM mask + ARCore camera pose" world-frame fusion
        // (static object, camera orbits). Set false to use the in-hand ICP object-frame tracker
        // instead (needed only when the object itself is turned/moved while scanning).
        private const val SEG_USE_ARCORE_POSE = true

        // Object-frame tracking (segmentation mode):
        // Min interval between processed frames; the object may move while the camera is still, so
        // we can't gate on camera motion. The worker + busy flag throttle the real rate further.
        private const val SEG_MIN_INTERVAL_MS = 80L
        // Minimum masked object points to attempt registration (too few => skip, don't corrupt).
        private const val MIN_TRACK_POINTS = 200
        // Fuse a frame only once the object has moved this much (new viewpoint) since the last fuse.
        private const val FUSE_ROT_DEG = 2.0
        private const val FUSE_TRANS_M = 0.01f

        /** OpenGL(ARCore)<->OpenCV camera flip, diag(1,-1,-1,1); its own inverse. */
        private val FLIP = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, -1f, 0f, 0f,
            0f, 0f, -1f, 0f,
            0f, 0f, 0f, 1f,
        )
    }
}
