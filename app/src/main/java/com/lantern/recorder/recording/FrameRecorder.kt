package com.lantern.recorder.recording

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.util.Log
import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.lantern.recorder.scanning.ObjectAngleCoverage
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Records synced capture frames to a per-session folder.
 *
 * Gating: a frame is saved whenever the camera has translated >= [TRANSLATION_THRESHOLD_M]
 * (~3 cm) since the last saved frame (and always for the first frame). Each saved
 * frame produces four files in the session folder:
 *
 *   frame_NNNN.png   RGB camera image (8-bit, from acquireCameraImage / YUV_420_888)
 *   frame_NNNN.json  camera pose (4x4), intrinsics, timestamps, image dimensions
 *   depth_NNNN.png   raw depth (16-bit grayscale, millimeters, acquireRawDepthImage16Bits)
 *   conf_NNNN.png    raw depth confidence (8-bit grayscale)
 *
 * [onFrame] runs on the GL thread: it acquires each ARCore image inside a
 * try-with-resources (`use`) block so native buffers are released promptly, copies
 * the pixel data out, then hands encoding + disk I/O to a background executor.
 */
class FrameRecorder(private val context: Context) {

    /** Invoked (on a background thread) after each frame's files are written. */
    @Volatile
    var onFrameSaved: ((index: Int, sessionName: String) -> Unit)? = null

    /** Invoked while recording with a short, human-readable status (why we are / aren't saving). */
    @Volatile
    var onStatus: ((message: String) -> Unit)? = null

    /**
     * Turntable mode only: invoked (on the GL thread) at detection rate with the
     * latest object pose — valid or not — so the UI can show live board-tracking
     * status and advance the object-angle coverage ring as the object spins, not
     * just on committed keyframes.
     */
    @Volatile
    var onObjectPose: ((ObjectPose) -> Unit)? = null

    /**
     * Supplies the per-keyframe object pose `T_CO` (turntable mode) or declines
     * to (orbit mode). Defaults to [OrbitPoseProvider], i.e. today's behaviour:
     * no object pose, host fuses in the world frame. Set before [start] to switch
     * capture mode; the chosen mode + board spec are written to `capture.json`.
     */
    @Volatile
    var poseProvider: PoseProvider = OrbitPoseProvider()

    private val ioExecutor = Executors.newSingleThreadExecutor()
    private val savedCount = AtomicInteger(0)

    // Logs the real camera intrinsics exactly once per session (first saved frame).
    private val loggedIntrinsics = java.util.concurrent.atomic.AtomicBoolean(false)

    // Throttle status spam from the per-frame GL loop.
    private var lastStatus = ""
    private var lastStatusMs = 0L

    @Volatile
    private var recording = false

    @Volatile
    private var sessionDir: File? = null

    // Translation (world meters) of the last saved camera pose.
    private var lastSavedX = 0f
    private var lastSavedY = 0f
    private var lastSavedZ = 0f
    private var hasLastSaved = false

    // Turntable gating: object rotation (3x3, row-major 9) of the last saved
    // keyframe, plus a detect-rate throttle so we don't run solvePnP every GL frame.
    private var lastSavedObjRot: FloatArray? = null
    private var hasLastSavedObj = false
    private var lastDetectMs = 0L

    val isRecording: Boolean get() = recording
    val sessionName: String? get() = sessionDir?.name
    val savedFrameCount: Int get() = savedCount.get()
    val sessionPath: String? get() = sessionDir?.absolutePath

    fun start(): String {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(sessionsRoot(), "session_$stamp")
        dir.mkdirs()
        sessionDir = dir
        savedCount.set(0)
        hasLastSaved = false
        hasLastSavedObj = false
        lastSavedObjRot = null
        lastDetectMs = 0L
        loggedIntrinsics.set(false)
        recording = true
        writeCaptureSidecar(dir, poseProvider)
        Log.i(TAG, "Recording started → ${dir.absolutePath} (mode=${poseProvider.captureMode})")
        return dir.name
    }

    /**
     * Writes the per-session `capture.json` describing the capture mode (orbit vs
     * turntable) and, for turntable, the board spec. The host reads this to pick
     * the fusion frame; a missing/orbit sidecar means today's world-frame path
     * (plan §4.1, backward compatible).
     */
    private fun writeCaptureSidecar(dir: File, provider: PoseProvider) {
        try {
            val json = JSONObject().apply {
                put("capture_mode", provider.captureMode)
                provider.boardSpec?.let {
                    put("object_frame", "charuco")
                    put("board", it.toJson())
                    put("axis_hint", "board_z")
                }
            }
            File(dir, "capture.json").writeText(json.toString(2))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write capture.json", e)
        }
    }

    fun stop() {
        if (!recording) return
        recording = false
        Log.i(TAG, "Recording stopped. ${savedCount.get()} frames in ${sessionDir?.name}")
    }

    fun shutdown() {
        recording = false
        ioExecutor.shutdown()
    }

    private fun sessionsRoot(): File =
        File(context.getExternalFilesDir(null), "sessions").apply { mkdirs() }

    /** Logs + surfaces a status message, throttled so the GL loop can't spam it. */
    private fun reportStatus(message: String) {
        val now = System.currentTimeMillis()
        if (message == lastStatus && now - lastStatusMs < 1000L) return
        lastStatus = message
        lastStatusMs = now
        Log.d(TAG, message)
        onStatus?.invoke(message)
    }

    /**
     * Called every render frame on the GL thread. Dispatches to the orbit
     * (camera-translation gated) or turntable (object-rotation gated) path based on
     * the active [poseProvider].
     */
    fun onFrame(frame: Frame) {
        if (!recording) return
        if (poseProvider.captureMode == "turntable") {
            onFrameTurntable(frame)
        } else {
            onFrameOrbit(frame)
        }
    }

    /**
     * Orbit mode: the object is fixed and the phone orbits it. Saves a keyframe
     * whenever the camera has translated past [TRANSLATION_THRESHOLD_M].
     */
    private fun onFrameOrbit(frame: Frame) {
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) {
            reportStatus("Move the phone slowly — waiting for tracking (${camera.trackingState})")
            return
        }

        val pose = camera.pose
        val x = pose.tx()
        val y = pose.ty()
        val z = pose.tz()
        if (hasLastSaved) {
            val dx = x - lastSavedX
            val dy = y - lastSavedY
            val dz = z - lastSavedZ
            if (dx * dx + dy * dy + dz * dz < TRANSLATION_THRESHOLD_M * TRANSLATION_THRESHOLD_M) {
                return
            }
        }

        val dir = sessionDir ?: return
        val index = savedCount.get() + 1

        val captured = try {
            capture(frame, index, precomputedObjectPose = null)
        } catch (e: NotYetAvailableException) {
            // Camera image or depth not ready yet this frame — retry next frame
            // without advancing the counter or the last-saved position. Raw depth
            // needs translational motion, so this is common until the phone moves.
            reportStatus("Keep moving — waiting for raw depth…")
            return
        } catch (e: Exception) {
            Log.e(TAG, "Capture failed for frame $index", e)
            reportStatus("Capture error: ${e.message}")
            return
        }

        // Commit: advance counter and last-saved position only once we have the data.
        savedCount.incrementAndGet()
        lastSavedX = x
        lastSavedY = y
        lastSavedZ = z
        hasLastSaved = true

        ioExecutor.execute { writeFrame(dir, captured) }
    }

    /**
     * Turntable mode (plan §3): the phone is near-fixed and the object is spun on a
     * fiducial board, so camera translation never advances — we gate on **object
     * rotation** instead. At a throttled detect rate we estimate `T_CO`; we surface
     * it to the UI (live board status + coverage ring) and commit a keyframe each
     * time the object has rotated past [TURNTABLE_YAW_THRESHOLD_DEG] about the board
     * axis.
     */
    private fun onFrameTurntable(frame: Frame) {
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) {
            reportStatus("Hold the phone steady — waiting for tracking (${camera.trackingState})")
            return
        }

        val now = System.currentTimeMillis()
        if (now - lastDetectMs < DETECT_INTERVAL_MS) return
        lastDetectMs = now

        val objectPose = try {
            poseProvider.objectPose(frame)
        } catch (e: Exception) {
            Log.w(TAG, "Object pose estimation failed", e)
            ObjectPose.INVALID
        }
        // Drive live UI (board lock + coverage) on every detection, valid or not.
        onObjectPose?.invoke(objectPose)

        if (!objectPose.valid) {
            reportStatus("Point the camera at the board — object not tracked")
            return
        }

        // Object-rotation gate: only commit once the turntable has advanced enough
        // about the board Z axis since the last saved keyframe.
        val curRot = ObjectAngleCoverage.rotationOf(objectPose.tCO)
        val lastRot = lastSavedObjRot
        if (hasLastSavedObj && lastRot != null) {
            val relative = ObjectAngleCoverage.multiply3x3(
                ObjectAngleCoverage.transpose3x3(lastRot), curRot,
            )
            val deltaYaw = signedYawDeg(ObjectAngleCoverage.yawAboutZDeg(relative))
            if (abs(deltaYaw) < TURNTABLE_YAW_THRESHOLD_DEG) {
                reportStatus("Keep rotating the object…")
                return
            }
        }

        val dir = sessionDir ?: return
        val index = savedCount.get() + 1

        val captured = try {
            capture(frame, index, precomputedObjectPose = objectPose)
        } catch (e: NotYetAvailableException) {
            reportStatus("Hold steady — waiting for raw depth…")
            return
        } catch (e: Exception) {
            Log.e(TAG, "Capture failed for frame $index", e)
            reportStatus("Capture error: ${e.message}")
            return
        }

        savedCount.incrementAndGet()
        lastSavedObjRot = curRot
        hasLastSavedObj = true

        ioExecutor.execute { writeFrame(dir, captured) }
    }

    /** Wraps a yaw (deg) into (-180, 180] so the gate sees a small signed delta. */
    private fun signedYawDeg(deg: Float): Float {
        var d = deg % 360f
        if (d > 180f) d -= 360f
        if (d <= -180f) d += 360f
        return d
    }

    /**
     * Acquires and copies all per-frame data on the GL thread. Buffers are released
     * via `use`. [precomputedObjectPose] lets the turntable path reuse the pose it
     * already estimated for gating; pass null to estimate here (orbit mode → INVALID).
     */
    private fun capture(frame: Frame, index: Int, precomputedObjectPose: ObjectPose?): CapturedFrame {
        val rgb = frame.acquireCameraImage().use { extractYuv(it) }
        val depth = frame.acquireRawDepthImage16Bits().use { extractGray16BigEndian(it) }
        val confidence = frame.acquireRawDepthConfidenceImage().use { extractGray8(it) }

        val camera = frame.camera
        val poseMatrix = FloatArray(16)
        camera.pose.toMatrix(poseMatrix, 0)
        val intrinsics = camera.imageIntrinsics

        // Turntable object pose (camera <- object). Orbit mode returns INVALID and
        // adds no cost. Runs only for committed keyframes, on the GL thread, as the
        // plan budgets (§5: keyed to keyframes, not every GL frame).
        val objectPose = precomputedObjectPose ?: try {
            poseProvider.objectPose(frame)
        } catch (e: Exception) {
            Log.w(TAG, "Object pose estimation failed for frame $index", e)
            ObjectPose.INVALID
        }

        return CapturedFrame(
            index = index,
            timestampNs = frame.timestamp,
            rgb = rgb,
            depth = depth,
            confidence = confidence,
            poseColumnMajor = poseMatrix,
            poseTranslation = floatArrayOf(camera.pose.tx(), camera.pose.ty(), camera.pose.tz()),
            poseQuaternion = camera.pose.let { floatArrayOf(it.qx(), it.qy(), it.qz(), it.qw()) },
            focalLength = intrinsics.focalLength,
            principalPoint = intrinsics.principalPoint,
            intrinsicsDimensions = intrinsics.imageDimensions,
            objectPose = objectPose,
        )
    }

    /** Background thread: encode and write the four files for one frame. */
    private fun writeFrame(dir: File, f: CapturedFrame) {
        val id = "%04d".format(f.index)
        try {
            // One-time sanity log: the real on-device camera intrinsics + image sizes.
            // Lets you eyeball the actual S25 focal length in logcat before pulling the
            // session, instead of guessing fx in the host pipeline. ARCore reports these
            // for the full CPU image (intrinsics width/height) — the host converter
            // rescales them to the saved depth resolution.
            if (loggedIntrinsics.compareAndSet(false, true)) {
                Log.i(
                    TAG,
                    "Intrinsics (full CPU image): fx=${f.focalLength[0]} fy=${f.focalLength[1]} " +
                        "cx=${f.principalPoint[0]} cy=${f.principalPoint[1]} " +
                        "size=${f.intrinsicsDimensions[0]}x${f.intrinsicsDimensions[1]} | " +
                        "saved rgb=${f.rgb.width}x${f.rgb.height} depth=${f.depth.width}x${f.depth.height}",
                )
            }
            // RGB → 8-bit PNG via Bitmap.
            val bitmap = Bitmap.createBitmap(f.rgb.argb, f.rgb.width, f.rgb.height, Bitmap.Config.ARGB_8888)
            FileOutputStream(File(dir, "frame_$id.png")).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            bitmap.recycle()

            // Depth → 16-bit grayscale PNG (millimeters).
            PngWriter.writeGrayscale(
                File(dir, "depth_$id.png"), f.depth.bytes, f.depth.width, f.depth.height, 16,
            )

            // Confidence → 8-bit grayscale PNG.
            PngWriter.writeGrayscale(
                File(dir, "conf_$id.png"), f.confidence.bytes, f.confidence.width, f.confidence.height, 8,
            )

            // Metadata JSON.
            writeJson(File(dir, "frame_$id.json"), f)

            Log.i(TAG, "Saved frame $id (${f.rgb.width}x${f.rgb.height} rgb, ${f.depth.width}x${f.depth.height} depth)")
            onFrameSaved?.invoke(f.index, dir.name)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write frame ${f.index}", e)
        }
    }

    private fun writeJson(file: File, f: CapturedFrame) {
        val json = JSONObject().apply {
            put("frame_index", f.index)
            put("timestamp_ns", f.timestampNs)
            put("rgb_width", f.rgb.width)
            put("rgb_height", f.rgb.height)
            put("depth_width", f.depth.width)
            put("depth_height", f.depth.height)
            put("confidence_width", f.confidence.width)
            put("confidence_height", f.confidence.height)
            put("depth_units", "millimeters")
            // Camera-to-world pose. Matrix is column-major (ARCore Pose.toMatrix / OpenGL convention).
            put("pose_matrix_column_major", JSONArray().apply { f.poseColumnMajor.forEach { put(it.toDouble()) } })
            put("translation_xyz", JSONArray().apply { f.poseTranslation.forEach { put(it.toDouble()) } })
            put("quaternion_xyzw", JSONArray().apply { f.poseQuaternion.forEach { put(it.toDouble()) } })
            put("intrinsics", JSONObject().apply {
                put("fx", f.focalLength[0].toDouble())
                put("fy", f.focalLength[1].toDouble())
                put("cx", f.principalPoint[0].toDouble())
                put("cy", f.principalPoint[1].toDouble())
                put("width", f.intrinsicsDimensions[0])
                put("height", f.intrinsicsDimensions[1])
            })
            // Turntable object pose T_CO (camera <- object), row-major 4x4, OpenCV
            // solvePnP convention. Absent/false ⇒ host uses world-frame fusion.
            put("object_pose_valid", f.objectPose.valid)
            if (f.objectPose.valid) {
                put("object_pose_T_CO", f.objectPose.toJsonRows())
                put("object_pose_corners", f.objectPose.cornerCount)
                if (!f.objectPose.reprojPx.isNaN()) {
                    put("object_pose_reproj_px", f.objectPose.reprojPx.toDouble())
                }
            }
        }
        file.writeText(json.toString(2))
    }

    // ----- Image extraction (GL thread) -----

    private data class RgbImage(val argb: IntArray, val width: Int, val height: Int)
    private data class GrayImage(val bytes: ByteArray, val width: Int, val height: Int)

    /** Converts a YUV_420_888 [Image] to packed ARGB_8888 (BT.601). */
    private fun extractYuv(image: Image): RgbImage {
        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBytes = toByteArray(yPlane.buffer)
        val uBytes = toByteArray(uPlane.buffer)
        val vBytes = toByteArray(vPlane.buffer)
        val yRowStride = yPlane.rowStride
        val yPixStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixStride = vPlane.pixelStride

        val argb = IntArray(width * height)
        for (j in 0 until height) {
            val uvRow = (j shr 1)
            for (i in 0 until width) {
                val yVal = yBytes[j * yRowStride + i * yPixStride].toInt() and 0xff
                val uvCol = (i shr 1)
                val uVal = uBytes[uvRow * uRowStride + uvCol * uPixStride].toInt() and 0xff
                val vVal = vBytes[uvRow * vRowStride + uvCol * vPixStride].toInt() and 0xff

                val c = yVal
                val d = uVal - 128
                val e = vVal - 128
                val r = clamp((c + 1.402f * e).toInt())
                val g = clamp((c - 0.344136f * d - 0.714136f * e).toInt())
                val b = clamp((c + 1.772f * d).toInt())
                argb[j * width + i] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return RgbImage(argb, width, height)
    }

    /** Copies a 16-bit single-channel [Image] to big-endian sample bytes for PNG. */
    private fun extractGray16BigEndian(image: Image): GrayImage {
        val width = image.width
        val height = image.height
        val plane = image.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val out = ByteArray(width * height * 2)
        var dst = 0
        for (j in 0 until height) {
            val rowStart = j * rowStride
            for (i in 0 until width) {
                val p = rowStart + i * pixelStride
                val lo = buffer.get(p).toInt() and 0xff
                val hi = buffer.get(p + 1).toInt() and 0xff
                // Source samples are little-endian (DEPTH16); write big-endian for PNG.
                out[dst++] = hi.toByte()
                out[dst++] = lo.toByte()
            }
        }
        return GrayImage(out, width, height)
    }

    /** Copies an 8-bit single-channel [Image] to tightly-packed sample bytes for PNG. */
    private fun extractGray8(image: Image): GrayImage {
        val width = image.width
        val height = image.height
        val plane = image.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val out = ByteArray(width * height)
        var dst = 0
        for (j in 0 until height) {
            val rowStart = j * rowStride
            for (i in 0 until width) {
                out[dst++] = buffer.get(rowStart + i * pixelStride)
            }
        }
        return GrayImage(out, width, height)
    }

    private fun toByteArray(buffer: ByteBuffer): ByteArray {
        val dup = buffer.duplicate()
        dup.rewind()
        val bytes = ByteArray(dup.remaining())
        dup.get(bytes)
        return bytes
    }

    private fun clamp(v: Int): Int = if (v < 0) 0 else if (v > 255) 255 else v

    private class CapturedFrame(
        val index: Int,
        val timestampNs: Long,
        val rgb: RgbImage,
        val depth: GrayImage,
        val confidence: GrayImage,
        val poseColumnMajor: FloatArray,
        val poseTranslation: FloatArray,
        val poseQuaternion: FloatArray,
        val focalLength: FloatArray,
        val principalPoint: FloatArray,
        val intrinsicsDimensions: IntArray,
        val objectPose: ObjectPose,
    )

    companion object {
        private const val TAG = "LANTERN"
        private const val TRANSLATION_THRESHOLD_M = 0.03f

        /** Turntable: commit a keyframe each time the object rotates this many degrees. */
        private const val TURNTABLE_YAW_THRESHOLD_DEG = 5f

        /** Turntable: minimum gap between board detections (ms) to bound solvePnP cost. */
        private const val DETECT_INTERVAL_MS = 66L
    }
}
