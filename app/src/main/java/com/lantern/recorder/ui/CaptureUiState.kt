package com.lantern.recorder.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.lantern.recorder.recon.DepthBackendKind
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
 * How the object is being captured (Object-Centric Capture plan, §3):
 *  - [Orbit] — today's default: the object is fixed, the phone orbits it; coverage
 *    tracks **camera azimuth** and keyframes are gated by camera translation.
 *  - [Turntable] — the phone is near-fixed and the object is spun on a fiducial
 *    board; coverage tracks **object yaw** (from `T_CO`) and keyframes are gated by
 *    object rotation. Requires the OpenCV board detector (phase T2).
 */
enum class CaptureMode {
    Orbit,
    Turntable,

    /**
     * Board-free **live mesh**: the phone moves around the object and a 3D mesh grows on
     * screen in real time (ARCore depth + optional on-device DA3 depth -> TSDF -> marching
     * cubes). See `com.lantern.recorder.recon.LiveReconstructor`.
     */
    LiveMesh,
}

/**
 * The stage of an optional **two-pass "flip the object" scan**, which captures the face
 * the object was resting on by having the user flip it and scan again. Single-pass scans
 * stay in [SinglePass] the whole time.
 */
enum class ScanPhase {
    /** Normal one-sided scan (or idle). The flip flow is not engaged. */
    SinglePass,

    /** Recording the first side. */
    PassOne,

    /** First side is sufficiently covered; waiting for the user to flip the object. */
    NeedsFlip,

    /** The object has been picked up / is being moved. */
    Flipping,

    /** The object has settled into a new resting pose; ready to scan the second side. */
    ReadyForPassTwo,

    /** Recording the second side. */
    PassTwo,

    /** Both sides captured. */
    Complete,
}

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

    /** Selected capture mode (orbit vs turntable). Switchable only while idle. */
    var captureMode by mutableStateOf(CaptureMode.Orbit)
        private set

    /**
     * Turntable mode only: whether the board is currently being tracked (a valid
     * `T_CO` this detection). Drives the live "Board locked / show the board" hint.
     */
    var boardTracking by mutableStateOf(false)
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

    // ----- Two-pass "flip the object" flow -----

    /**
     * Whether to capture both sides by flipping the object between two passes. Defaults
     * to on, since capturing the resting/bottom face is the whole reason the flow exists;
     * users who only want a quick one-sided scan can turn it off before recording.
     */
    var twoPassEnabled by mutableStateOf(true)
        private set

    /** Current stage of the (optional) flip flow. */
    var scanPhase by mutableStateOf(ScanPhase.SinglePass)
        private set

    /** Transient warning that the object appears to have moved mid-pass (breaks fusion). */
    var objectMovedWarning by mutableStateOf(false)
        private set

    /**
     * Fired (on the UI thread) when a pass auto-completes and recording should stop, so
     * `MainActivity` can cut the recording cleanly before the user flips the object. Set
     * by the activity; the recorder's own logic is never modified.
     */
    var onRequestStopRecording: (() -> Unit)? = null

    /** True once the current pass has covered enough of the reachable hemisphere. */
    private var passCompleteFired = false

    // ----- Live mesh (board-free real-time reconstruction) -----

    /** Vertices in the currently displayed live mesh. */
    var liveMeshVertices by mutableIntStateOf(0)
        private set

    /** Keyframes fused into the live TSDF so far. */
    var liveMeshFrames by mutableIntStateOf(0)
        private set

    /** Which dense-depth runtime is feeding the mesh (QNN NPU / ExecuTorch / ARCore-only). */
    var liveMeshDepthBackend by mutableStateOf(DepthBackendKind.ARCORE)
        private set

    /** Updates the live-mesh readout (called on the UI thread from the GL loop). */
    fun onLiveMeshStats(vertices: Int, frames: Int, depthBackend: DepthBackendKind) {
        liveMeshVertices = vertices
        liveMeshFrames = frames
        liveMeshDepthBackend = depthBackend
    }

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
        passCompleteFired = false
        objectMovedWarning = false
        // In two-pass mode, recording is always a labelled pass; otherwise single.
        scanPhase = when {
            !twoPassEnabled -> ScanPhase.SinglePass
            scanPhase == ScanPhase.ReadyForPassTwo -> ScanPhase.PassTwo
            else -> ScanPhase.PassOne
        }
    }

    /** Opts the next scan into (or out of) the two-pass flip flow. Idle/pre-record only. */
    fun toggleTwoPass(enabled: Boolean) {
        if (isRecording) return
        twoPassEnabled = enabled
        scanPhase = ScanPhase.SinglePass
        objectMovedWarning = false
    }

    /** Switches the capture mode (orbit ⇄ turntable). Idle/pre-record only. */
    fun selectCaptureMode(mode: CaptureMode) {
        if (isRecording) return
        captureMode = mode
        boardTracking = false
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
        boardTracking = false
    }

    /**
     * Turntable coverage update (plan §3.4): folds the object-angle ring derived from
     * `T_CO` (via [com.lantern.recorder.scanning.ObjectAngleCoverage]) into the same
     * dome UI the orbit path uses. The 12 object-yaw sectors map directly onto the
     * orbit ring's [AZIMUTH_SECTORS] wedges, so the existing gizmo carries over and a
     * full object spin can auto-complete a two-pass scan exactly like a full orbit.
     *
     * Called on the UI thread from `MainActivity` at board-detection rate; [tracking]
     * is false when the board isn't currently visible.
     *
     * @param sectorMask bitmask of covered object-yaw sectors
     * @param currentSector the sector the object is in now, or -1
     * @param coveragePercent covered fraction of the full turntable (0..100)
     * @param tracking whether the board is locked this detection
     */
    fun onObjectCoverage(
        sectorMask: Int,
        currentSector: Int,
        coveragePercent: Int,
        tracking: Boolean,
    ) {
        boardTracking = tracking
        if (tracking) objectLocked = true
        if (!tracking) return
        bandMaskSides = sectorMask
        bandMaskUpper = 0
        this.currentSector = currentSector
        currentBand = 0
        this.coveragePercent = coveragePercent
        motionWarning = false
        // A complete object spin ends the pass just like a complete orbit.
        maybeCompletePass()
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
        // Headline metric = orbit completeness (how many azimuth directions seen at ANY
        // elevation). A normal orbit keeps the phone near one height, so it fills one
        // elevation band per direction; requiring *both* bands full would never trigger.
        // "Did you get all the way around" is the meaningful flip-readiness signal.
        val azimuthCovered = Integer.bitCount(bandMaskSides or bandMaskUpper)
        coveragePercent = azimuthCovered * 100 / AZIMUTH_SECTORS
        // Successful keyframes mean the user is translating; clear any rotation nag.
        motionWarning = false

        // In two-pass mode, once the orbit is essentially complete, end this pass: ask
        // the activity to stop recording and advance the flip flow.
        maybeCompletePass()
    }

    /** Whether the current side has been orbited enough to end the pass / prompt a flip. */
    private fun passCoverageComplete(): Boolean {
        // Orbit completeness: nearly all azimuth directions seen (at any elevation), with
        // a minimum number of keyframes so a quick spin can't trip it early.
        val azimuthCovered = Integer.bitCount(bandMaskSides or bandMaskUpper)
        return azimuthCovered >= MIN_AZIMUTH_FOR_PASS && frameCount >= MIN_FRAMES_FOR_PASS
    }

    /** Fires the one-shot pass-complete transition during a two-pass scan. */
    private fun maybeCompletePass() {
        if (passCompleteFired) return
        if (scanPhase != ScanPhase.PassOne && scanPhase != ScanPhase.PassTwo) return
        if (!passCoverageComplete()) return
        advancePassPhase()
        // Cut the recording cleanly before the object is handled.
        onRequestStopRecording?.invoke()
    }

    /**
     * Advances the flip flow after a pass ends: pass one → needs flip, pass two → done.
     * One-shot (guarded by [passCompleteFired]); shared by the auto-complete path and a
     * manual shutter stop so the user can always force "this side's done, prompt me."
     */
    private fun advancePassPhase() {
        if (passCompleteFired) return
        passCompleteFired = true
        scanPhase = if (scanPhase == ScanPhase.PassOne) ScanPhase.NeedsFlip else ScanPhase.Complete
    }

    /**
     * Manually finishes the current pass (user tapped stop during a two-pass scan).
     * Returns true if it advanced the flip flow, so the caller can write the pass's
     * grouping metadata and arm flip detection just like the auto path.
     */
    fun finishPassManually(): Boolean {
        if (!twoPassEnabled) return false
        if (scanPhase != ScanPhase.PassOne && scanPhase != ScanPhase.PassTwo) return false
        advancePassPhase()
        return true
    }

    /**
     * Feeds a [com.lantern.recorder.scanning.FlipDetector] reading into the flip flow.
     * Driven by `MainActivity` from depth-derived disturb/settle signals on the GL thread.
     *
     * @param disturbed object appears picked up / moving
     * @param settled object has come to rest at a new pose
     */
    fun onFlipReading(disturbed: Boolean, settled: Boolean) {
        when (scanPhase) {
            ScanPhase.NeedsFlip -> if (disturbed) scanPhase = ScanPhase.Flipping
            ScanPhase.Flipping -> {
                if (settled) scanPhase = ScanPhase.ReadyForPassTwo
            }
            ScanPhase.ReadyForPassTwo -> {
                // If it gets picked up again before pass two starts, fall back.
                if (disturbed && !settled) scanPhase = ScanPhase.Flipping
            }
            else -> { /* not in the flip-watch window */ }
        }
    }

    /** Surfaces / clears the "object moved during the scan" warning (recording only). */
    fun onObjectMoved(moved: Boolean) {
        objectMovedWarning = isRecording && moved
    }

    /**
     * Manual override for the flip flow: if auto detection misses the flip (e.g. depth
     * never settles), tapping the prompt forces "ready for side two." Only valid while
     * waiting for / detecting a flip.
     */
    fun forceReadyForPassTwo() {
        if (scanPhase == ScanPhase.NeedsFlip || scanPhase == ScanPhase.Flipping) {
            scanPhase = ScanPhase.ReadyForPassTwo
        }
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
        objectMovedWarning = false
        // Revert the chip to its idle depth state and surface a transient save card
        // instead of letting a "Saved…" message linger in the chip.
        status = CaptureStatus.DepthReady(depthSupported)
        if (frames > 0) savedConfirmation = SavedConfirmation(session, frames)
        // If a single-pass scan (or a two-pass scan that wasn't auto-completing) just
        // stopped, return the flow to its idle single-pass state. Auto-completed passes
        // are advanced separately in maybeCompletePass(), so don't clobber those.
        if (!passCompleteFired && scanPhase != ScanPhase.ReadyForPassTwo) {
            scanPhase = ScanPhase.SinglePass
        }
    }

    /** Resets the flip flow to idle (e.g. after the two-pass scan is fully done). */
    fun resetScanFlow() {
        scanPhase = ScanPhase.SinglePass
        passCompleteFired = false
        objectMovedWarning = false
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

        /**
         * Azimuth directions (of [AZIMUTH_SECTORS]) that must be seen — at any elevation —
         * for a pass to count as a complete orbit. Slightly less than full so one stubborn
         * direction (against a wall, glare) can't stall the flip prompt forever.
         */
        const val MIN_AZIMUTH_FOR_PASS = 10

        /** Minimum saved keyframes before a pass can auto-complete (guards quick spins). */
        const val MIN_FRAMES_FOR_PASS = 10
    }
}
