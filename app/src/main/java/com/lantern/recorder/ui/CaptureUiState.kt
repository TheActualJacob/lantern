package com.lantern.recorder.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

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

    /** A finished session summary, shown briefly after stopping. */
    data class Saved(val frames: Int, val session: String) : CaptureStatus

    /** A diagnostic or error message (camera unavailable, waiting for tracking, …). */
    data class Message(val text: String, val isError: Boolean = false) : CaptureStatus
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
    }

    fun onFrameSaved(session: String, frames: Int) {
        frameCount = frames
        if (isRecording) status = CaptureStatus.Recording(session, frames)
    }

    fun onRecordingStopped(session: String, frames: Int) {
        isRecording = false
        frameCount = frames
        status = CaptureStatus.Saved(frames, session)
    }

    /** Surfaces a diagnostic/error message (only while it's meaningful to show one). */
    fun onMessage(text: String, isError: Boolean = false) {
        status = CaptureStatus.Message(text, isError)
    }
}
