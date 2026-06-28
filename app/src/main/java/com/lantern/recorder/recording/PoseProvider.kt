package com.lantern.recorder.recording

import com.google.ar.core.Frame
import org.json.JSONObject

/**
 * The capture-pose seam (Object-Centric Capture plan, phase **T1**).
 *
 * Today every capture fuses in ARCore's **world** frame: the object is fixed and
 * the phone orbits it. To rotate the *object* in front of a near-fixed phone
 * (turntable mode), the host must instead fuse in the **object's own frame** `O`,
 * which means each keyframe needs a per-frame transform `T_CO` (camera <- object).
 *
 * A [PoseProvider] is the single abstraction that supplies (or declines to supply)
 * that object pose:
 *
 *  - [OrbitPoseProvider] — today's behaviour. No object pose; the host falls back
 *    to world-frame fusion. Functionally identical to the pre-T1 recorder.
 *  - [TurntablePoseProvider] — detects a fiducial board per keyframe and returns
 *    `T_CO` (phase **T2**, behind the OpenCV dependency that the plan gates).
 *
 * [FrameRecorder] holds one provider, writes a `capture.json` sidecar describing
 * the mode, and stamps each keyframe's JSON with the object pose when valid. The
 * recorder is otherwise unchanged — this is a pure seam so orbit captures keep
 * their exact byte layout.
 */
interface PoseProvider {

    /** `"orbit"` or `"turntable"`; written into the session `capture.json`. */
    val captureMode: String

    /** The fiducial board definition for turntable mode, or null for orbit. */
    val boardSpec: BoardSpec?

    /**
     * Estimates the object pose for [frame], called on the GL thread only for
     * committed keyframes (never every GL frame). Returns [ObjectPose.INVALID]
     * when there is no object pose for this mode/frame (e.g. orbit, or the board
     * was not visible).
     */
    fun objectPose(frame: Frame): ObjectPose
}

/**
 * A per-keyframe object pose `T_CO` (camera <- object) as a **row-major 4x4**
 * matrix, in the OpenCV `solvePnP` camera convention (+X right, +Y down, +Z
 * forward) that the host's object-frame fusion expects.
 */
class ObjectPose(
    /** Row-major 4x4 `T_CO`; only meaningful when [valid]. */
    val tCO: FloatArray,
    /** False when the board was not seen / mode has no object pose. */
    val valid: Boolean,
    /** Board corners matched (diagnostic; 0 when not applicable). */
    val cornerCount: Int = 0,
    /** RMS reprojection error in pixels (NaN when not applicable). */
    val reprojPx: Float = Float.NaN,
) {
    init {
        require(tCO.size == 16) { "tCO must be a 16-element row-major 4x4, got ${tCO.size}" }
    }

    /** The pose as a JSON 4x4 array of rows, for the keyframe metadata. */
    fun toJsonRows(): org.json.JSONArray {
        val rows = org.json.JSONArray()
        for (r in 0 until 4) {
            val row = org.json.JSONArray()
            for (c in 0 until 4) row.put(tCO[r * 4 + c].toDouble())
            rows.put(row)
        }
        return rows
    }

    companion object {
        /** Shared "no object pose" result (identity matrix, [valid] = false). */
        val INVALID = ObjectPose(identity4x4(), valid = false)

        private fun identity4x4(): FloatArray {
            val m = FloatArray(16)
            m[0] = 1f; m[5] = 1f; m[10] = 1f; m[15] = 1f
            return m
        }
    }
}

/**
 * Definition of a printed ChArUco/ArUco board whose coordinate frame *is* the
 * object frame `O`. Mirrors the `board` block of the host `capture.json`
 * (plan §4.1) so the spec round-trips device -> disk -> host.
 */
data class BoardSpec(
    val dict: String = "DICT_5X5_100",
    val squaresX: Int = 5,
    val squaresY: Int = 7,
    val squareLenM: Float = 0.03f,
    val markerLenM: Float = 0.022f,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("dict", dict)
        put("squares_x", squaresX)
        put("squares_y", squaresY)
        put("square_len_m", squareLenM.toDouble())
        put("marker_len_m", markerLenM.toDouble())
    }
}

/**
 * Today's behaviour: the object is fixed in the world and the phone orbits it, so
 * there is no per-frame object pose. The host treats such sessions as world-frame
 * (orbit) captures. Selecting this provider is functionally identical to the
 * pre-T1 recorder.
 */
class OrbitPoseProvider : PoseProvider {
    override val captureMode: String = "orbit"
    override val boardSpec: BoardSpec? = null
    override fun objectPose(frame: Frame): ObjectPose = ObjectPose.INVALID
}

/**
 * Turntable mode (plan §3): the object sits on a fiducial board and is spun in
 * front of a near-fixed phone. Per keyframe it detects the board and returns
 * `T_CO`, delegating the actual detection to a [BoardDetector].
 *
 * The detector is the **OpenCV ArUco/ChArUco** dependency the plan gates behind a
 * mode switch (§5, §6 dependency note). Until that dependency lands (phase T2),
 * construct this with a null detector: object poses report invalid and the host
 * cleanly falls back to world-frame fusion, so the build stays green and orbit
 * captures are untouched.
 */
class TurntablePoseProvider(
    override val boardSpec: BoardSpec,
    private val detector: BoardDetector? = null,
) : PoseProvider {
    override val captureMode: String = "turntable"

    override fun objectPose(frame: Frame): ObjectPose {
        // T2 seam: an OpenCV-backed BoardDetector plugs in here. Without it we
        // record metadata-only turntable sessions (no object pose) rather than
        // pulling OpenCV into the default orbit path.
        return detector?.detect(frame) ?: ObjectPose.INVALID
    }
}

/**
 * Estimates `T_CO` from a single ARCore [Frame] using classical fiducial
 * geometry (marker detection + `solvePnP`). Implemented in phase **T2** with
 * OpenCV `cv::aruco`; kept as an interface so the recorder/provider compile and
 * test without the heavy dependency, and so the detection can be swapped or
 * mocked.
 *
 * Implementations must acquire the camera image on the GL thread (inside the
 * call) and may offload `solvePnP` itself; they should reject poses with too few
 * matched corners or high reprojection error (plan §9) by returning
 * [ObjectPose.INVALID].
 */
interface BoardDetector {
    fun detect(frame: Frame): ObjectPose
}
