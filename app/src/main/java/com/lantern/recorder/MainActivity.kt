package com.lantern.recorder

import android.content.Intent
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import com.lantern.recorder.recording.FrameRecorder
import com.lantern.recorder.rendering.BackgroundRenderer
import com.lantern.recorder.rendering.DisplayRotationHelper
import com.lantern.recorder.sessions.SessionsActivity
import com.lantern.recorder.ui.CaptureOverlay
import com.lantern.recorder.ui.CaptureUiState
import com.lantern.recorder.ui.theme.LanternTheme
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

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

    /** Compose-observable mirror of capture state, updated on the UI thread. */
    private val uiState = CaptureUiState()

    // Read on the GL thread, written on the UI thread.
    @Volatile
    private var session: Session? = null

    private var installRequested = false
    private var messageShown: String? = null
    private var rawDepthSupported = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

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

        frameRecorder = FrameRecorder(this)
        frameRecorder.onFrameSaved = { index, sessionName ->
            runOnUiThread { uiState.onFrameSaved(sessionName, index) }
        }
        frameRecorder.onStatus = { message ->
            // Only surface diagnostics while recording; ignore once frames are flowing.
            if (frameRecorder.isRecording) runOnUiThread { uiState.onMessage(message) }
        }

        setContent {
            LanternTheme {
                AndroidView(
                    factory = { surfaceView },
                    modifier = Modifier.fillMaxSize(),
                )
                CaptureOverlay(
                    state = uiState,
                    onToggleRecord = ::toggleRecording,
                    onOpenSessions = {
                        startActivity(Intent(this, SessionsActivity::class.java))
                    },
                )
            }
        }
    }

    private fun toggleRecording() {
        if (!rawDepthSupported) {
            showMessage(getString(R.string.depth_unsupported_no_record))
            return
        }
        if (frameRecorder.isRecording) {
            val name = frameRecorder.sessionName ?: "—"
            val count = frameRecorder.savedFrameCount
            frameRecorder.stop()
            uiState.onRecordingStopped(name, count)
            Log.i(TAG, "Saved $count frames to ${frameRecorder.sessionPath}")
        } else {
            val name = frameRecorder.start()
            uiState.onRecordingStarted(name)
        }
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
            }
            // Order matters: pause rendering before pausing the session.
            displayRotationHelper.onPause()
            surfaceView.onPause()
            session?.pause()
        }
    }

    override fun onDestroy() {
        frameRecorder.shutdown()
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
        } catch (t: Throwable) {
            // Never let an exception escape the GL thread.
            Log.e(TAG, "Exception on the GL thread", t)
        }
    }

    companion object {
        private const val TAG = "LANTERN"
    }
}
