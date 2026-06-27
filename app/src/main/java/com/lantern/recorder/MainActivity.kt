package com.lantern.recorder

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import com.lantern.recorder.databinding.ActivityMainBinding
import com.lantern.recorder.recording.FrameRecorder
import com.lantern.recorder.rendering.BackgroundRenderer
import com.lantern.recorder.rendering.DisplayRotationHelper
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Task P0b (first slice): bring up an ARCore session, render the live camera
 * feed, and confirm RAW_DEPTH_ONLY is supported on the device — logged under the
 * "LANTERN" tag and shown in the on-screen status overlay.
 *
 * Intentionally NOT here yet: recording, depth/pose readout, capture UI.
 */
class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var binding: ActivityMainBinding
    private lateinit var surfaceView: GLSurfaceView
    private lateinit var displayRotationHelper: DisplayRotationHelper

    private val backgroundRenderer = BackgroundRenderer()
    private lateinit var frameRecorder: FrameRecorder

    // Read on the GL thread, written on the UI thread.
    @Volatile
    private var session: Session? = null

    private var installRequested = false
    private var messageShown: String? = null
    private var rawDepthSupported = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        surfaceView = binding.surfaceView
        displayRotationHelper = DisplayRotationHelper(this)

        surfaceView.preserveEGLContextOnPause = true
        surfaceView.setEGLContextClientVersion(2)
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        surfaceView.setWillNotDraw(false)

        frameRecorder = FrameRecorder(this)
        frameRecorder.onFrameSaved = { index, sessionName ->
            runOnUiThread {
                binding.statusText.text = getString(R.string.recording_status, sessionName, index)
            }
        }
        frameRecorder.onStatus = { message ->
            // Only show diagnostics while recording; ignore once frames are flowing.
            if (frameRecorder.isRecording) runOnUiThread { binding.statusText.text = message }
        }
        binding.recordButton.setOnClickListener { toggleRecording() }
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
            binding.recordButton.setText(R.string.record_start)
            val summary = getString(R.string.recording_saved, count, name)
            binding.statusText.text = summary
            Toast.makeText(this, summary, Toast.LENGTH_LONG).show()
            Log.i(TAG, "Saved to ${frameRecorder.sessionPath}")
        } else {
            val name = frameRecorder.start()
            binding.recordButton.setText(R.string.record_stop)
            binding.statusText.text = getString(R.string.recording_status, name, 0)
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
                frameRecorder.stop()
                binding.recordButton.setText(R.string.record_start)
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
        session.configure(config)

        runOnUiThread {
            binding.statusText.text = getString(R.string.depth_supported, supported.toString())
            // Recording requires raw depth — only enable the control when supported.
            binding.recordButton.isEnabled = supported
        }
    }

    private fun showMessage(message: String) {
        if (messageShown == message) return
        messageShown = message
        runOnUiThread {
            binding.statusText.text = message
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
