package com.lantern.recorder.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlin.math.atan2

/**
 * What the status chip is currently communicating. Modeled as a sealed type so the
 * chip can pick an icon, color, and accessibility label per state, and so transitions
 * between states can crossfade cleanly.
 */
sealed interface CaptureStatus {
    /** AR session is still coming up. */
    data object Initializing : CaptureStatus

    /** Depth-support has been resolved; recording is only possible when [supported]. */
    data class DepthReady(val supported: Boolean) : CaptureStatus

    /** Actively recording a session. */
    data class Recording(val session: String, val frames: Int) : CaptureStatus

    /** A diagnostic or error message (camera unavailable, waiting for tracking, …). */
    data class Message(val text: String, val isError: Boolean = false) : CaptureStatus
}

/** A just-finished session, surfaced as a transient confirmation card (not the chip). */
data class SavedConfirmation(val session: String, val frames: Int)

/**
 * Compose-observable UI state for the capture screen. Owned by `MainActivity`, which
 * updates it (on the UI thread) from the ARCore session + [com.lantern.recorder.recording.FrameRecorder]
 * callbacks. The recorder and GL rendering remain the source of truth; this is a thin
 * presentation mirror.
 */
class CaptureUiState {
    var status by mutableStateOf<CaptureStatus>(CaptureStatus.Initializing)
        private set

    /** Whether RAW_DEPTH_ONLY is available; gates the shutter. */
    var depthSupported by mutableStateOf(false)
        private set

    var isRecording by mutableStateOf(false)
        private set

    var frameCount by mutableIntStateOf(0)
        private set

    /** Set when a recording finishes; drives the transient save card, then cleared. */
    var savedConfirmation by mutableStateOf<SavedConfirmation?>(null)
        private set

    // ----- Pose-based coverage ring (Deliverable 2) -----
    //
    // A v1 *ring* (2D azimuth coverage): the horizon around the object is split into
    // [COVERAGE_SECTORS] equal wedges. For each saved keyframe we compute the direction
    // from the object's estimated centroid to the camera and light up that wedge. The
    // mask is a bitfield so it is cheaply Compose-observable.

    /** Bitmask of covered azimuth sectors (bit i set => sector i captured). */
    var coveredMask by mutableIntStateOf(0)
        private set

    /** Percentage of the orbit captured so far (0..100). */
    var coveragePercent by mutableIntStateOf(0)
        private set

    /** The sector the camera most recently sat in, or -1 before the first keyframe. */
    var currentSector by mutableIntStateOf(-1)
        private set

    /** True once motion looks like pure rotation / no translation (a correctness risk). */
    var motionWarning by mutableStateOf(false)
        private set

    /**
     * True once the object centroid has been anchored from real depth (vs. the initial
     * fixed-distance guess). Lets the UI signal "locked onto the object".
     */
    var objectLocked by mutableStateOf(false)
        private set

    fun onDepthResolved(supported: Boolean) {
        depthSupported = supported
        // Don't clobber an active recording / error message with the idle depth state.
        if (!isRecording && status !is CaptureStatus.Message) {
            status = CaptureStatus.DepthReady(supported)
        }
    }

    fun onRecordingStarted(session: String) {
        isRecording = true
        frameCount = 0
        status = CaptureStatus.Recording(session, 0)
        resetCoverage()
    }

    /** Clears coverage so each recording starts from an empty ring. */
    private fun resetCoverage() {
        coveredMask = 0
        coveragePercent = 0
        currentSector = -1
        objectLocked = false
        motionWarning = false
    }

    /**
     * Folds one saved keyframe into the coverage ring, using the camera position and the
     * current best estimate of the object centroid (both ARCore world meters). The
     * centroid is estimated in `MainActivity` from the raw depth at frame center and
     * refined across keyframes; this only does the azimuth-sector bookkeeping. Called on
     * the UI thread when [com.lantern.recorder.recording.FrameRecorder] commits a frame;
     * the recorder itself is untouched.
     *
     * @param cameraX,cameraZ camera position on the gravity-horizontal plane
     * @param centroidX,centroidZ estimated object centroid on the same plane
     * @param depthAnchored whether [centroidX]/[centroidZ] came from real depth (vs guess)
     */
    fun onKeyframeCoverage(
        cameraX: Float, cameraZ: Float,
        centroidX: Float, centroidZ: Float,
        depthAnchored: Boolean,
    ) {
        if (depthAnchored) objectLocked = true
        // Direction from object to camera, projected onto the gravity-horizontal plane
        // (ARCore world Y is up). Azimuth 0 is +Z; sectors advance counter-clockwise.
        val dx = cameraX - centroidX
        val dz = cameraZ - centroidZ
        if (dx * dx + dz * dz < 1e-6f) return
        var az = atan2(dx, dz) // [-pi, pi]
        if (az < 0f) az += (2.0 * Math.PI).toFloat()
        val sector = ((az / (2.0 * Math.PI).toFloat()) * COVERAGE_SECTORS).toInt()
            .coerceIn(0, COVERAGE_SECTORS - 1)
        currentSector = sector
        coveredMask = coveredMask or (1 shl sector)
        coveragePercent = Integer.bitCount(coveredMask) * 100 / COVERAGE_SECTORS
        // Successful keyframes mean the user is translating; clear any rotation nag.
        motionWarning = false
    }

    /**
     * Surfaces / clears the "pure rotation, no translation" guidance. Driven by
     * `MainActivity` from ARCore pose deltas already available on the GL thread.
     */
    fun onMotionHint(pureRotation: Boolean) {
        if (isRecording) motionWarning = pureRotation
    }

    fun onFrameSaved(session: String, frames: Int) {
        frameCount = frames
        if (isRecording) status = CaptureStatus.Recording(session, frames)
    }

    fun onRecordingStopped(session: String, frames: Int) {
        isRecording = false
        frameCount = frames
        motionWarning = false
        // Revert the chip to its idle depth state and surface a transient save card
        // instead of letting a "Saved…" message linger in the chip.
        status = CaptureStatus.DepthReady(depthSupported)
        if (frames > 0) savedConfirmation = SavedConfirmation(session, frames)
    }

    /** Dismisses the transient save confirmation (after its timeout or on tap). */
    fun clearSavedConfirmation() {
        savedConfirmation = null
    }

    /** Surfaces a diagnostic/error message (only while it's meaningful to show one). */
    fun onMessage(text: String, isError: Boolean = false) {
        status = CaptureStatus.Message(text, isError)
    }

    companion object {
        /** Number of azimuth wedges in the coverage ring (15° each at 24). */
        const val COVERAGE_SECTORS = 24
    }
}
