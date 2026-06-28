package com.lantern.recorder

import android.content.Intent
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import com.lantern.recorder.recording.BoardSpec
import com.lantern.recorder.recording.FrameRecorder
import com.lantern.recorder.recording.OpenCvBoardDetector
import com.lantern.recorder.recording.OrbitPoseProvider
import com.lantern.recorder.recording.PoseProvider
import com.lantern.recorder.recording.TurntablePoseProvider
import com.lantern.recorder.recon.CameraIntrinsics
import com.lantern.recorder.recon.LiveReconstructor
import com.lantern.recorder.rendering.BackgroundRenderer
import com.lantern.recorder.rendering.DisplayRotationHelper
import com.lantern.recorder.rendering.MeshRenderer
import com.lantern.recorder.scanning.FlipDetector
import com.lantern.recorder.scanning.ObjectAngleCoverage
import com.lantern.recorder.sessions.SessionStore
import com.lantern.recorder.sessions.SessionsActivity
import com.lantern.recorder.ui.CaptureMode
import com.lantern.recorder.ui.CaptureOverlay
import com.lantern.recorder.ui.CaptureUiState
import com.lantern.recorder.ui.CoachOverlay
import com.lantern.recorder.ui.ScanPhase
import com.lantern.recorder.ui.theme.LanternTheme
import org.opencv.android.OpenCVLoader
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.acos

/**
 * Brings up an ARCore session, renders the live camera feed into a [GLSurfaceView],
 * and drives recording via [FrameRecorder]. The capture controls (status chip,
 * shutter, sessions FAB) are a Jetpack Compose overlay ([CaptureOverlay]) layered on
 * top of the GL feed through [AndroidView] — the GL rendering + ARCore session
 * lifecycle remain exactly as before.
 */
class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var surfaceView: GLSurfaceView
    private lateinit var displayRotationHelper: DisplayRotationHelper

    private val backgroundRenderer = BackgroundRenderer()
    private lateinit var frameRecorder: FrameRecorder

    // Board-free live mesh (recon package): ARCore depth + optional on-device DA3 depth
    // -> TSDF -> marching cubes, drawn over the camera feed in LiveMesh capture mode.
    private val meshRenderer = MeshRenderer()
    private var liveReconstructor: LiveReconstructor? = null
    private val liveViewMatrix = FloatArray(16)
    private val liveProjMatrix = FloatArray(16)
    private val liveMvpMatrix = FloatArray(16)
    private var lastLiveMeshVersion = -1
    private var lastLiveStatsMs = 0L

    /** Compose-observable mirror of capture state, updated on the UI thread. */
    private val uiState = CaptureUiState()

    // Read on the GL thread, written on the UI thread.
    @Volatile
    private var session: Session? = null

    private var installRequested = false
    private var messageShown: String? = null
    private var rawDepthSupported = false

    // Coverage gizmo: tracks when the recorder commits a new keyframe so we can fold
    // that keyframe's camera pose into the pose-based coverage ring (read on GL thread).
    private var lastCoverageCount = 0

    // Object centroid estimate (ARCore world meters), anchored from the raw depth at
    // frame center and refined as a running mean across keyframes. Because the object is
    // stationary, every (cameraPos + forward * centerDepth) sample points at roughly the
    // same world location, so averaging converges on the real object center. This is a
    // read-only consumer of depth for the coverage UI; the recorder is untouched.
    private var centroidSumX = 0f
    private var centroidSumY = 0f
    private var centroidSumZ = 0f
    private var centroidSamples = 0

    // Two-pass "flip the object" flow. The detector classifies disturb/settle from the
    // depth-derived center point; the rest is grouping metadata so the host can merge the
    // two passes. All read-only w.r.t. the recorder.
    private val flipDetector = FlipDetector()
    private var lastFlipTickMs = 0L
    private var scanGroupId: String? = null

    // Turntable mode (Object-Centric Capture, T2/T3). The board frame is the object
    // frame; the detector recovers T_CO per keyframe and the coverage tracks object
    // yaw. OpenCV is loaded once at startup; if it fails we fall back to orbit only.
    private val boardSpec = BoardSpec()
    private var openCvReady = false
    private var boardDetector: OpenCvBoardDetector? = null
    private val objectCoverage = ObjectAngleCoverage()
    // Seed for the second pass's centroid, captured when the object settles after a flip.
    private var passTwoSeedX = 0f
    private var passTwoSeedY = 0f
    private var passTwoSeedZ = 0f
    private var hasPassTwoSeed = false

    // Lightweight motion classifier for the "pure rotation / no translation" guardrail.
    // Evaluated over a sliding window from ARCore pose deltas; never touches the recorder.
    private var motionAnchorX = 0f
    private var motionAnchorY = 0f
    private var motionAnchorZ = 0f
    private var motionAnchorMs = 0L
    private var motionHasAnchor = false
    private var motionAccumRotationRad = 0f
    private var motionLastQuat: FloatArray? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Load OpenCV's bundled native lib up front so Turntable mode's ChArUco
        // detector can be constructed on demand. Orbit mode never touches OpenCV.
        openCvReady = OpenCVLoader.initLocal()
        Log.i(TAG, "OpenCV initLocal() = $openCvReady")

        displayRotationHelper = DisplayRotationHelper(this)

        // The camera feed MUST stay on a GLSurfaceView. We build it programmatically and
        // embed it in Compose via AndroidView; ARCore renders into it as before.
        surfaceView = GLSurfaceView(this).apply {
            keepScreenOn = true
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(this@MainActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
            setWillNotDraw(false)
        }

        // Live mesh reconstructor. Looks for an on-device DA3 .pte (exported by
        // export_da3_executorch.py) for dense NPU depth; if it's absent it still builds a
        // mesh from ARCore raw depth alone. We check the external files dir first since it's
        // directly `adb push`-able (/sdcard/Android/data/<pkg>/files/), then internal.
        val da3ModelPath = resolveDa3ModelPath()
        liveReconstructor = LiveReconstructor(da3ModelPath)

        frameRecorder = FrameRecorder(this)
        frameRecorder.onFrameSaved = { index, sessionName ->
            runOnUiThread { uiState.onFrameSaved(sessionName, index) }
        }
        frameRecorder.onStatus = { message ->
            // Only surface diagnostics while recording; ignore once frames are flowing.
            if (frameRecorder.isRecording) runOnUiThread { uiState.onMessage(message) }
        }
        // Turntable mode: each board detection advances the object-angle coverage ring
        // (and the live "board locked" hint), independent of keyframe commits.
        frameRecorder.onObjectPose = { pose ->
            runOnUiThread {
                if (pose.valid) {
                    objectCoverage.onObjectPose(pose.tCO)
                    uiState.onObjectCoverage(
                        sectorMask = objectCoverage.sectorMask,
                        currentSector = objectCoverage.currentSector,
                        coveragePercent = objectCoverage.coveragePercent,
                        tracking = true,
                    )
                } else {
                    uiState.onObjectCoverage(0, -1, 0, tracking = false)
                }
            }
        }
        // When a two-pass pass auto-completes, cut the recording cleanly before the user
        // handles the object. Fired on the UI thread from the coverage bookkeeping.
        uiState.onRequestStopRecording = { autoStopForFlip() }

        setContent {
            LanternTheme {
                // First-run coach shows automatically until dismissed once; the help
                // affordance can re-open it any time.
                var showCoach by remember {
                    mutableStateOf(!coachPrefs().getBoolean(KEY_COACH_SEEN, false))
                }
                AndroidView(
                    factory = { surfaceView },
                    modifier = Modifier.fillMaxSize(),
                )
                CaptureOverlay(
                    state = uiState,
                    onToggleRecord = ::toggleRecording,
                    onToggleTwoPass = { enabled -> uiState.toggleTwoPass(enabled) },
                    onSetCaptureMode = ::onSelectCaptureMode,
                    turntableAvailable = openCvReady,
                    onOpenSessions = {
                        startActivity(Intent(this, SessionsActivity::class.java))
                    },
                    onOpenHelp = { showCoach = true },
                )
                CoachOverlay(
                    visible = showCoach,
                    onDismiss = {
                        showCoach = false
                        coachPrefs().edit().putBoolean(KEY_COACH_SEEN, true).apply()
                    },
                )
            }
        }
    }

    /** Backing prefs for the one-time coach overlay. */
    private fun coachPrefs() = getSharedPreferences("lantern_ux", MODE_PRIVATE)

    private fun toggleRecording() {
        if (!rawDepthSupported) {
            showMessage(getString(R.string.depth_unsupported_no_record))
            return
        }
        // Block the shutter mid-flip: the object is being handled, not scanned.
        if (uiState.scanPhase == ScanPhase.NeedsFlip || uiState.scanPhase == ScanPhase.Flipping) {
            return
        }
        if (frameRecorder.isRecording) {
            val name = frameRecorder.sessionName ?: "—"
            val count = frameRecorder.savedFrameCount
            val path = frameRecorder.sessionPath
            frameRecorder.stop()
            // In a two-pass scan, a manual stop means "this side's done" — advance the
            // flip flow (and tag/arm) exactly like the auto-complete path. Set the phase
            // before onRecordingStopped so it doesn't reset the flow.
            val advanced = uiState.finishPassManually()
            uiState.onRecordingStopped(name, count)
            if (advanced) finalizePassAfterStop(path)
            resetMotionTracking()
            Log.i(TAG, "Saved $count frames to ${frameRecorder.sessionPath}")
        } else {
            // Starting a fresh two-pass scan after a completed one: reset the flow first.
            if (uiState.scanPhase == ScanPhase.Complete) {
                uiState.resetScanFlow()
                scanGroupId = null
            }
            val startingPassTwo = uiState.twoPassEnabled &&
                uiState.scanPhase == ScanPhase.ReadyForPassTwo
            // Stamp a group id when the first pass of a two-pass scan begins.
            if (uiState.twoPassEnabled && !startingPassTwo) {
                scanGroupId = "group_" + STAMP.format(Date())
            }
            // Pick orbit vs turntable capture before starting; turntable plugs in the
            // OpenCV board detector and resets the object-angle coverage ring.
            frameRecorder.poseProvider = poseProviderForMode()
            objectCoverage.reset()
            val name = frameRecorder.start()
            uiState.onRecordingStarted(name)
            lastCoverageCount = 0
            resetMotionTracking()
            flipDetector.clear()
            // Seed pass two's centroid with the settled object location for a fast lock.
            if (startingPassTwo && hasPassTwoSeed) {
                centroidSumX = passTwoSeedX
                centroidSumY = passTwoSeedY
                centroidSumZ = passTwoSeedZ
                centroidSamples = 1
            }
            hasPassTwoSeed = false
        }
    }

    /**
     * Resolves the DA3 `.pte` path, preferring the external files dir (adb-pushable to
     * `/sdcard/Android/data/com.lantern.recorder/files/da3_small_sm8750.pte`) and falling
     * back to internal `filesDir`. Returns the external path even if absent so the log
     * points the user at the right place; the loader degrades to ARCore depth when missing.
     */
    private fun resolveDa3ModelPath(): String {
        val name = "da3_small_sm8750.pte"
        val external = getExternalFilesDir(null)?.let { File(it, name) }
        if (external != null && external.exists()) return external.absolutePath
        val internal = File(filesDir, name)
        if (internal.exists()) return internal.absolutePath
        return (external ?: internal).absolutePath
    }

    /** Switches capture mode from the overlay toggle (idle only; turntable needs OpenCV). */
    private fun onSelectCaptureMode(mode: CaptureMode) {
        if (frameRecorder.isRecording) return
        if (mode == CaptureMode.Turntable && !openCvReady) {
            showMessage(getString(R.string.turntable_unavailable))
            return
        }
        uiState.selectCaptureMode(mode)
        // Live mesh runs whenever its mode is selected (independent of the recorder), so
        // start a fresh reconstruction on entry and pause it when leaving.
        if (mode == CaptureMode.LiveMesh) {
            lastLiveMeshVersion = -1
            liveReconstructor?.reset()
            liveReconstructor?.start()
        } else {
            liveReconstructor?.stop()
        }
    }

    /** Builds the [PoseProvider] for the selected capture mode (constructs OpenCV lazily). */
    private fun poseProviderForMode(): PoseProvider = when (uiState.captureMode) {
        CaptureMode.Turntable -> {
            val detector = boardDetector
                ?: OpenCvBoardDetector(boardSpec).also { boardDetector = it }
            TurntablePoseProvider(boardSpec, detector)
        }
        // Live mesh doesn't use the recorder's pose provider for its reconstruction, but
        // recording in this mode (if started) should behave like a free orbit.
        CaptureMode.Orbit, CaptureMode.LiveMesh -> OrbitPoseProvider()
    }

    /**
     * Cuts the current pass's recording when its coverage auto-completes, then writes the
     * pass's grouping metadata and arms flip detection. Runs on the UI thread; only uses
     * the recorder's public API.
     */
    private fun autoStopForFlip() {
        if (!frameRecorder.isRecording) return
        val name = frameRecorder.sessionName ?: "—"
        val count = frameRecorder.savedFrameCount
        val path = frameRecorder.sessionPath
        frameRecorder.stop()
        uiState.onRecordingStopped(name, count)
        Log.i(TAG, "Pass auto-stopped: saved $count frames to $path")
        finalizePassAfterStop(path)
    }

    /**
     * After a two-pass pass ends (auto or manual), tags the session as part of the scan
     * group and — for pass one — arms the [FlipDetector] against the object's current
     * location so we can tell when it's lifted and set back down.
     */
    private fun finalizePassAfterStop(path: String?) {
        val groupId = scanGroupId
        when (uiState.scanPhase) {
            ScanPhase.NeedsFlip -> {
                if (groupId != null && path != null) {
                    SessionStore.writeScanGroup(File(path), groupId, pass = 1, totalPasses = 2)
                }
                if (centroidSamples > 0) {
                    flipDetector.setReference(
                        centroidSumX / centroidSamples,
                        centroidSumY / centroidSamples,
                        centroidSumZ / centroidSamples,
                    )
                }
                lastFlipTickMs = 0L
            }
            ScanPhase.Complete -> {
                if (groupId != null && path != null) {
                    SessionStore.writeScanGroup(File(path), groupId, pass = 2, totalPasses = 2)
                }
                flipDetector.clear()
            }
            else -> { /* no-op */ }
        }
    }

    /** Clears the sliding-window motion classifier (between recordings / on pause). */
    private fun resetMotionTracking() {
        motionHasAnchor = false
        motionAccumRotationRad = 0f
        motionLastQuat = null
        // Also drop the object-centroid estimate so each session re-locks from scratch.
        centroidSumX = 0f
        centroidSumY = 0f
        centroidSumZ = 0f
        centroidSamples = 0
    }

    override fun onResume() {
        super.onResume()

        if (session == null && !ensureSessionCreated()) {
            // ensureSessionCreated() either prompted for install/permission and
            // will re-enter onResume, or showed a terminal error.
            return
        }

        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            showMessage("Camera not available. Try restarting the app.")
            session = null
            return
        }

        surfaceView.onResume()
        displayRotationHelper.onResume()
    }

    override fun onPause() {
        super.onPause()
        if (session != null) {
            // Stop recording before tearing down the session so no acquire races a pause.
            if (frameRecorder.isRecording) {
                val name = frameRecorder.sessionName ?: "—"
                val count = frameRecorder.savedFrameCount
                frameRecorder.stop()
                uiState.onRecordingStopped(name, count)
                resetMotionTracking()
            }
            // Abandon any in-progress flip flow rather than resuming mid-flip later.
            uiState.resetScanFlow()
            flipDetector.clear()
            // Pause live-mesh fusion so the worker doesn't touch a paused session.
            liveReconstructor?.stop()
            // Order matters: pause rendering before pausing the session.
            displayRotationHelper.onPause()
            surfaceView.onPause()
            session?.pause()
        }
    }

    override fun onDestroy() {
        frameRecorder.shutdown()
        liveReconstructor?.close()
        liveReconstructor = null
        session?.close()
        session = null
        super.onDestroy()
    }

    /**
     * Creates the ARCore [Session], driving the install/permission flow as needed.
     * Returns true only when a configured session is ready to resume.
     */
    private fun ensureSessionCreated(): Boolean {
        // 1. Make sure ARCore (Google Play Services for AR) is installed/up to date.
        try {
            when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    installRequested = true
                    return false // Play install UI shown; we'll be resumed afterward.
                }
                ArCoreApk.InstallStatus.INSTALLED -> { /* proceed */ }
            }
        } catch (e: UnavailableUserDeclinedInstallationException) {
            showMessage("ARCore installation declined; AR is required for capture.")
            return false
        } catch (e: UnavailableDeviceNotCompatibleException) {
            showMessage("This device does not support ARCore.")
            return false
        }

        // 2. Camera permission.
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
            CameraPermissionHelper.requestCameraPermission(this)
            return false
        }

        // 3. Construct and configure the session.
        return try {
            val newSession = Session(this)
            selectHighestResolutionCameraConfig(newSession)
            configureSession(newSession)
            session = newSession
            true
        } catch (e: UnavailableArcoreNotInstalledException) {
            showMessage("Please install ARCore (Google Play Services for AR).")
            false
        } catch (e: UnavailableApkTooOldException) {
            showMessage("Please update ARCore (Google Play Services for AR).")
            false
        } catch (e: UnavailableSdkTooOldException) {
            showMessage("Please update this app.")
            false
        } catch (e: UnavailableDeviceNotCompatibleException) {
            showMessage("This device does not support ARCore.")
            false
        } catch (e: Exception) {
            showMessage("Failed to create AR session: ${e.message}")
            Log.e(TAG, "Session creation failed", e)
            false
        }
    }

    /** Enables RAW_DEPTH_ONLY if the device supports it, and logs the result. */
    private fun configureSession(session: Session) {
        val config = session.config
        val supported = session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)
        rawDepthSupported = supported

        Log.i(TAG, "RAW_DEPTH_ONLY supported = $supported")

        if (supported) {
            config.depthMode = Config.DepthMode.RAW_DEPTH_ONLY
        } else {
            // Leave depth disabled; surface the gap loudly rather than silently
            // recording without depth.
            Log.w(TAG, "Raw depth NOT supported on this device — depth disabled.")
            config.depthMode = Config.DepthMode.DISABLED
        }

        // ARCore defaults to FIXED focus (tuned for tracking stability), which looks
        // soft on nearby objects. AUTO gives a sharp, camera-app-like image. We record
        // intrinsics per frame, so the focus-dependent focal-length change is captured.
        config.focusMode = Config.FocusMode.AUTO
        Log.i(TAG, "Focus mode = ${config.focusMode}")

        session.configure(config)

        runOnUiThread { uiState.onDepthResolved(supported) }
    }

    /**
     * Picks the ARCore [CameraConfig] with the largest CPU image size so recorded RGB
     * frames are as sharp as the device allows (default is often ~640×480). Must run
     * on a freshly-created, paused session before [configureSession]. Falls back
     * silently to the default config if enumeration fails.
     */
    private fun selectHighestResolutionCameraConfig(session: Session) {
        try {
            val filter = CameraConfigFilter(session)
            val configs: List<CameraConfig> = session.getSupportedCameraConfigs(filter)
            val best = configs.maxByOrNull {
                it.imageSize.width.toLong() * it.imageSize.height.toLong()
            } ?: return
            session.cameraConfig = best
            Log.i(TAG, "Camera config: CPU image ${best.imageSize.width}x${best.imageSize.height}")
        } catch (e: Exception) {
            // Non-fatal: keep ARCore's default config rather than failing session setup.
            Log.w(TAG, "Couldn't select a high-res camera config; using default.", e)
        }
    }

    private fun showMessage(message: String) {
        if (messageShown == message) return
        messageShown = message
        runOnUiThread {
            uiState.onMessage(message, isError = true)
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
        Log.w(TAG, message)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
            showMessage(getString(R.string.camera_permission_needed))
            if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
                // Permanently denied — send the user to app settings.
                CameraPermissionHelper.launchPermissionSettings(this)
            }
            finish()
        }
        // If granted, onResume() runs next and creates the session.
    }

    // ----- GLSurfaceView.Renderer (GL thread) -----

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        backgroundRenderer.createOnGlThread()
        meshRenderer.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        displayRotationHelper.onSurfaceChanged(width, height)
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        val session = this.session ?: return
        displayRotationHelper.updateSessionIfNeeded(session)

        try {
            // ARCore renders the camera image into this texture each update().
            session.setCameraTextureName(backgroundRenderer.textureId)
            val frame = session.update()
            backgroundRenderer.draw(frame)
            // Recording gates on camera translation; acquires/copies happen here on
            // the GL thread, encoding + disk I/O are offloaded inside the recorder.
            frameRecorder.onFrame(frame)
            // Pose-based scanning guidance (UI only): fold newly-committed keyframes
            // into the coverage ring and classify motion for the rotation guardrail.
            updateScanGuidance(frame)
            // Between passes of a two-pass scan, watch (depth-based) for the object being
            // flipped and set back down. Cheap, throttled, and only while not recording.
            updateFlipWatch(frame)
            // Board-free live mesh: fuse depth into the TSDF and draw the growing mesh
            // over the camera feed (only in LiveMesh capture mode).
            updateLiveMesh(frame)
        } catch (t: Throwable) {
            // Never let an exception escape the GL thread.
            Log.e(TAG, "Exception on the GL thread", t)
        }
    }

    /**
     * Live mesh hook (GL thread). In [CaptureMode.LiveMesh], hands the current ARCore
     * frame to the [LiveReconstructor] (which acquires depth/RGB here and fuses on a worker
     * thread), then draws the newest extracted mesh with the ARCore view/projection so it
     * sits correctly in the world over the camera feed. No-op in other modes.
     */
    private fun updateLiveMesh(frame: com.google.ar.core.Frame) {
        val recon = liveReconstructor ?: return
        if (uiState.captureMode != CaptureMode.LiveMesh) return
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) return

        val intr = camera.imageIntrinsics
        val intrinsics = CameraIntrinsics(
            fx = intr.focalLength[0],
            fy = intr.focalLength[1],
            cx = intr.principalPoint[0],
            cy = intr.principalPoint[1],
            width = intr.imageDimensions[0],
            height = intr.imageDimensions[1],
        )
        val pose = FloatArray(16)
        camera.pose.toMatrix(pose, 0)
        recon.onFrame(frame, intrinsics, pose)

        val version = recon.meshVersion()
        if (version != lastLiveMeshVersion) {
            meshRenderer.updateMesh(recon.latestMesh(), version)
            lastLiveMeshVersion = version
        }
        camera.getViewMatrix(liveViewMatrix, 0)
        camera.getProjectionMatrix(liveProjMatrix, 0, 0.05f, 10f)
        Matrix.multiplyMM(liveMvpMatrix, 0, liveProjMatrix, 0, liveViewMatrix, 0)
        meshRenderer.draw(liveMvpMatrix)

        val now = SystemClock.elapsedRealtime()
        if (now - lastLiveStatsMs > 250L) {
            lastLiveStatsMs = now
            val vertices = recon.latestMesh().vertexCount
            val frames = recon.frameCount
            val da3 = recon.da3Active
            runOnUiThread { uiState.onLiveMeshStats(vertices, frames, da3) }
        }
    }

    /**
     * While paused between two-pass passes ([ScanPhase.NeedsFlip] / [ScanPhase.Flipping] /
     * [ScanPhase.ReadyForPassTwo]), samples the center depth on a throttle and feeds the
     * [FlipDetector] so the UI can tell when the object is being flipped and when it has
     * settled into a new pose. Recorder is idle here, so the extra depth acquire is free.
     */
    private fun updateFlipWatch(frame: com.google.ar.core.Frame) {
        if (!uiState.twoPassEnabled) return
        val phase = uiState.scanPhase
        if (phase != ScanPhase.NeedsFlip && phase != ScanPhase.Flipping &&
            phase != ScanPhase.ReadyForPassTwo
        ) {
            return
        }
        val nowMs = SystemClock.elapsedRealtime()
        if (nowMs - lastFlipTickMs < FLIP_TICK_MS) return
        lastFlipTickMs = nowMs

        val camera = frame.camera
        if (camera.trackingState != com.google.ar.core.TrackingState.TRACKING) return
        val pose = camera.pose

        val depth = estimateCenterDepthMeters(frame)
        val hasPoint = depth != null
        var x = 0f; var y = 0f; var z = 0f
        if (depth != null) {
            val zAxis = pose.zAxis
            x = pose.tx() + (-zAxis[0]) * depth
            y = pose.ty() + (-zAxis[1]) * depth
            z = pose.tz() + (-zAxis[2]) * depth
        }
        val reading = flipDetector.update(hasPoint, x, y, z, nowMs)
        if (reading.settled) {
            passTwoSeedX = reading.settleX
            passTwoSeedY = reading.settleY
            passTwoSeedZ = reading.settleZ
            hasPassTwoSeed = true
        }
        runOnUiThread { uiState.onFlipReading(reading.disturbed, reading.settled) }
    }

    /**
     * Derives the scanning-coverage and motion-warning UI from ARCore poses, entirely
     * from data already available on the GL thread. The [FrameRecorder] is read-only
     * here (we only observe its committed keyframe count) and is never modified.
     */
    private fun updateScanGuidance(frame: com.google.ar.core.Frame) {
        if (!frameRecorder.isRecording) return
        // Turntable mode drives coverage from the object pose (onObjectPose), not the
        // camera-azimuth dome — the phone is near-fixed, so this path would mislabel it.
        if (uiState.captureMode == CaptureMode.Turntable) return
        val camera = frame.camera
        if (camera.trackingState != com.google.ar.core.TrackingState.TRACKING) return

        val pose = camera.pose
        val px = pose.tx(); val py = pose.ty(); val pz = pose.tz()

        // 1. Coverage: if the recorder committed a keyframe this frame, light its sector.
        val count = frameRecorder.savedFrameCount
        if (count > lastCoverageCount) {
            lastCoverageCount = count
            // Camera forward is -Z of the pose in world space.
            val zAxis = pose.zAxis
            val fx = -zAxis[0]; val fy = -zAxis[1]; val fz = -zAxis[2]

            // Anchor the object centroid from the real depth at frame center. The coach
            // asks the user to keep the object centered, so the center depth ~= object
            // distance. Refine a running world-space mean; fall back to a fixed distance
            // until depth is available so the dome still works on depth-less frames.
            val centerDepthM = estimateCenterDepthMeters(frame)
            val dist = centerDepthM ?: FALLBACK_CENTROID_M
            centroidSumX += px + fx * dist
            centroidSumY += py + fy * dist
            centroidSumZ += pz + fz * dist
            centroidSamples++

            val cx = centroidSumX / centroidSamples
            val cy = centroidSumY / centroidSamples
            val cz = centroidSumZ / centroidSamples
            // Surface is "observed" from this direction when this keyframe had a valid,
            // in-range center depth — i.e. the object's surface was actually seen, which
            // is what upgrades the dome from "orbit angle" to "surface coverage".
            val surfaceObserved = centerDepthM != null

            // Object-moved guard: the point the center ray hits should stay near the
            // running centroid while the object is still. A large, repeated deviation
            // means the object was bumped/moved mid-pass, which corrupts fusion.
            val objectMoved = if (centerDepthM != null && centroidSamples > MIN_SAMPLES_FOR_MOVE_CHECK) {
                val obsX = px + fx * dist
                val obsY = py + fy * dist
                val obsZ = pz + fz * dist
                val ddx = obsX - cx; val ddy = obsY - cy; val ddz = obsZ - cz
                kotlin.math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) > OBJECT_MOVED_M
            } else {
                false
            }

            runOnUiThread {
                uiState.onKeyframeCoverage(px, py, pz, cx, cy, cz, surfaceObserved)
                uiState.onObjectMoved(objectMoved)
            }
        }

        // 2. Motion classifier over a sliding window: lots of rotation + little net
        //    translation => pure spinning, which yields no parallax for raw depth.
        val q = floatArrayOf(pose.qx(), pose.qy(), pose.qz(), pose.qw())
        motionLastQuat?.let { last ->
            motionAccumRotationRad += quaternionAngle(last, q)
        }
        motionLastQuat = q

        val now = SystemClock.elapsedRealtime()
        if (!motionHasAnchor) {
            motionAnchorX = px; motionAnchorY = py; motionAnchorZ = pz
            motionAnchorMs = now
            motionAccumRotationRad = 0f
            motionHasAnchor = true
        } else if (now - motionAnchorMs >= MOTION_WINDOW_MS) {
            val dx = px - motionAnchorX
            val dy = py - motionAnchorY
            val dz = pz - motionAnchorZ
            val netDisp = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
            val pureRotation = motionAccumRotationRad >= MOTION_ROTATION_RAD &&
                netDisp < MOTION_TRANSLATION_M
            runOnUiThread { uiState.onMotionHint(pureRotation) }
            // Slide the window forward.
            motionAnchorX = px; motionAnchorY = py; motionAnchorZ = pz
            motionAnchorMs = now
            motionAccumRotationRad = 0f
        }
    }

    /** Smallest rotation angle (radians) between two unit quaternions [x,y,z,w]. */
    private fun quaternionAngle(a: FloatArray, b: FloatArray): Float {
        val dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
            .coerceIn(0f, 1f)
        return 2f * acos(dot)
    }

    /**
     * Robust object distance (meters) sampled from the raw depth at frame center, or null
     * if no plausible depth is available. Reads a small window around the depth image
     * center and takes the median of valid samples within [DEPTH_MIN_M]..[DEPTH_MAX_M]
     * (rejecting holes, hands very close, and far background/floor). The recorder is not
     * touched: we acquire our own short-lived copy and release it via `use`.
     *
     * Raw-depth samples are 16-bit millimeters, little-endian (matching the recorder's
     * extraction); RAW_DEPTH_ONLY confidence lives in a separate image we don't need here.
     */
    private fun estimateCenterDepthMeters(frame: com.google.ar.core.Frame): Float? {
        return try {
            frame.acquireRawDepthImage16Bits().use { image ->
                val width = image.width
                val height = image.height
                val plane = image.planes[0]
                val buffer = plane.buffer
                val rowStride = plane.rowStride
                val pixelStride = plane.pixelStride
                val cx = width / 2
                val cy = height / 2
                val half = (minOf(width, height) / 12).coerceIn(2, 8)
                val samples = ArrayList<Int>((2 * half + 1) * (2 * half + 1))
                var y = (cy - half).coerceAtLeast(0)
                val yMax = (cy + half).coerceAtMost(height - 1)
                val xMax = (cx + half).coerceAtMost(width - 1)
                while (y <= yMax) {
                    var x = (cx - half).coerceAtLeast(0)
                    while (x <= xMax) {
                        val p = y * rowStride + x * pixelStride
                        val lo = buffer.get(p).toInt() and 0xff
                        val hi = buffer.get(p + 1).toInt() and 0xff
                        val mm = (hi shl 8) or lo
                        if (mm > 0) samples.add(mm)
                        x++
                    }
                    y++
                }
                if (samples.isEmpty()) return null
                samples.sort()
                val meters = samples[samples.size / 2] / 1000f
                if (meters in DEPTH_MIN_M..DEPTH_MAX_M) meters else null
            }
        } catch (e: Exception) {
            // NotYetAvailable / transient acquire failures: skip this keyframe's depth.
            null
        }
    }

    companion object {
        private const val TAG = "LANTERN"

        // One-time coach overlay flag.
        private const val KEY_COACH_SEEN = "coach_seen"

        // Pure-rotation guardrail thresholds (evaluated per sliding window).
        private const val MOTION_WINDOW_MS = 1200L
        private const val MOTION_ROTATION_RAD = 0.45f   // ~26° of accumulated rotation
        private const val MOTION_TRANSLATION_M = 0.02f  // < 2 cm net displacement

        // Object-centroid depth estimation (meters). Plausible scan range from the coach
        // ("20–60 cm"), widened a little; the fallback is used until depth is available.
        private const val DEPTH_MIN_M = 0.12f
        private const val DEPTH_MAX_M = 1.2f
        private const val FALLBACK_CENTROID_M = 0.35f

        // Two-pass flip flow.
        private const val FLIP_TICK_MS = 120L
        private const val OBJECT_MOVED_M = 0.10f
        private const val MIN_SAMPLES_FOR_MOVE_CHECK = 4
        private val STAMP = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
    }
}
