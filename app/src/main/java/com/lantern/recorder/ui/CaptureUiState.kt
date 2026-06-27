package com.lantern.recorder.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlin.math.atan2
import kotlin.math.sqrt

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

    // ----- Pose-based coverage dome (Deliverable 2) -----
    //
    // A v2 *dome* (azimuth + elevation coverage). The viewing hemisphere over the object
    // is split into patches: two ringed elevation bands (sides + upper) each cut into
    // [AZIMUTH_SECTORS] wedges, plus a single top cap. For each saved keyframe we take the
    // direction from the object's depth-anchored centroid to the camera, bin it by azimuth
    // and elevation, and light that patch — but only when the keyframe actually observed
    // object surface from that direction (depth-gated). Masks are bitfields so they are
    // cheaply Compose-observable.

    /** Bitmask of covered azimuth sectors in the lower (sides) band. */
    var bandMaskSides by mutableIntStateOf(0)
        private set

    /** Bitmask of covered azimuth sectors in the upper band. */
    var bandMaskUpper by mutableIntStateOf(0)
        private set

    /** Whether the top cap (looking down on the object) has been covered. */
    var topCovered by mutableStateOf(false)
        private set

    /** Percentage of dome patches captured so far (0..100). */
    var coveragePercent by mutableIntStateOf(0)
        private set

    /** The band the camera most recently sat in (0=sides, 1=upper, 2=top), or -1. */
    var currentBand by mutableIntStateOf(-1)
        private set

    /** The azimuth sector the camera most recently sat in, or -1 (top cap / none). */
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

    /** Clears coverage so each recording starts from an empty dome. */
    private fun resetCoverage() {
        bandMaskSides = 0
        bandMaskUpper = 0
        topCovered = false
        coveragePercent = 0
        currentBand = -1
        currentSector = -1
        objectLocked = false
        motionWarning = false
    }

    /**
     * Folds one saved keyframe into the coverage dome, using the camera position and the
     * current best estimate of the object centroid (both ARCore world meters, 3D). The
     * centroid is estimated in `MainActivity` from the raw depth at frame center and
     * refined across keyframes; this only does the azimuth/elevation patch bookkeeping.
     * Called on the UI thread when [com.lantern.recorder.recording.FrameRecorder] commits
     * a frame; the recorder itself is untouched.
     *
     * @param cameraX,cameraY,cameraZ camera position (ARCore world meters; Y is up)
     * @param centroidX,centroidY,centroidZ estimated object centroid (same frame)
     * @param surfaceObserved whether this keyframe saw real object surface (depth gate)
     */
    fun onKeyframeCoverage(
        cameraX: Float, cameraY: Float, cameraZ: Float,
        centroidX: Float, centroidY: Float, centroidZ: Float,
        surfaceObserved: Boolean,
    ) {
        if (surfaceObserved) objectLocked = true
        // Only credit coverage when the keyframe actually observed object surface from
        // this direction — that's what makes it *surface* coverage, not just orbit angle.
        if (!surfaceObserved) {
            motionWarning = false
            return
        }

        // Direction from object to camera (ARCore world Y is up).
        val dx = cameraX - centroidX
        val dy = cameraY - centroidY
        val dz = cameraZ - centroidZ
        val horiz = sqrt(dx * dx + dz * dz)
        if (horiz < 1e-4f && kotlin.math.abs(dy) < 1e-4f) return

        // Elevation above the object's horizontal plane, in degrees.
        val elevationDeg = Math.toDegrees(atan2(dy, horiz).toDouble()).toFloat()

        if (elevationDeg >= TOP_CAP_ELEVATION_DEG) {
            // Looking down on the object — the azimuth-independent top cap.
            topCovered = true
            currentBand = 2
            currentSector = -1
        } else {
            // Azimuth 0 is +Z; sectors advance counter-clockwise.
            var az = atan2(dx, dz) // [-pi, pi]
            if (az < 0f) az += (2.0 * Math.PI).toFloat()
            val sector = ((az / (2.0 * Math.PI).toFloat()) * AZIMUTH_SECTORS).toInt()
                .coerceIn(0, AZIMUTH_SECTORS - 1)
            currentSector = sector
            if (elevationDeg >= UPPER_BAND_ELEVATION_DEG) {
                currentBand = 1
                bandMaskUpper = bandMaskUpper or (1 shl sector)
            } else {
                currentBand = 0
                bandMaskSides = bandMaskSides or (1 shl sector)
            }
        }

        val covered = Integer.bitCount(bandMaskSides) +
            Integer.bitCount(bandMaskUpper) +
            (if (topCovered) 1 else 0)
        coveragePercent = covered * 100 / TOTAL_PATCHES
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
        /** Azimuth wedges per ringed elevation band (30° each at 12). */
        const val AZIMUTH_SECTORS = 12

        /** Elevation (deg) at/above which a view counts as the azimuth-free top cap. */
        const val TOP_CAP_ELEVATION_DEG = 60f

        /** Elevation (deg) splitting the lower "sides" band from the "upper" band. */
        const val UPPER_BAND_ELEVATION_DEG = 30f

        /** Total dome patches: two ringed bands of [AZIMUTH_SECTORS] + one top cap. */
        const val TOTAL_PATCHES = AZIMUTH_SECTORS * 2 + 1
    }
}
