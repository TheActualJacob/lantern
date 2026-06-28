package com.lantern.recorder.recording

import android.graphics.ImageFormat
import android.util.Log
import com.google.ar.core.Frame
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import org.opencv.core.Size
import org.opencv.objdetect.CharucoBoard
import org.opencv.objdetect.CharucoDetector
import org.opencv.objdetect.Objdetect

/**
 * Phase **T2**: the OpenCV-backed [BoardDetector]. Detects the ChArUco board in the
 * ARCore camera frame and recovers the object pose `T_CO` (camera <- object) via
 * `solvePnP`. This is classical geometry — **not** ExecuTorch (plan §6).
 *
 * Pipeline per call (keyframe-rate, GL thread):
 *   1. Grab the camera image's luma (Y) plane → grayscale [Mat].
 *   2. `CharucoDetector.detectBoard` → interpolated chessboard corners + ids.
 *   3. Map ids → board object points; `solvePnP` (with ARCore intrinsics) → rvec/tvec.
 *   4. Reject too-few corners / high reprojection error (plan §9); else pack `T_CO`.
 *
 * The board frame *is* the object frame `O`, so spinning the board spins `O`, and the
 * host fuses everything into `O` (a rotating object becomes static).
 */
class OpenCvBoardDetector(
    private val spec: BoardSpec,
    private val minCorners: Int = DEFAULT_MIN_CORNERS,
    private val maxReprojPx: Float = DEFAULT_MAX_REPROJ_PX,
) : BoardDetector {

    private val board: CharucoBoard
    private val detector: CharucoDetector
    /** All chessboard corner object points in the board (object) frame, by id. */
    private val objCorners: Array<Point3>

    init {
        val dictionary = Objdetect.getPredefinedDictionary(dictionaryId(spec.dict))
        board = CharucoBoard(
            Size(spec.squaresX.toDouble(), spec.squaresY.toDouble()),
            spec.squareLenM,
            spec.markerLenM,
            dictionary,
        )
        detector = CharucoDetector(board)
        objCorners = MatOfPoint3f(board.chessboardCorners).toArray()
        Log.i(TAG, "OpenCV ChArUco detector ready: ${spec.dict} ${spec.squaresX}x${spec.squaresY}, " +
            "${objCorners.size} chessboard corners")
    }

    override fun detect(frame: Frame): ObjectPose {
        val gray = acquireGray(frame) ?: return ObjectPose.INVALID
        val charucoCorners = Mat()
        val charucoIds = Mat()
        try {
            detector.detectBoard(gray, charucoCorners, charucoIds)
            if (charucoIds.empty() || charucoIds.rows() < minCorners) {
                return ObjectPose.INVALID
            }

            val ids = MatOfInt(charucoIds).toArray()
            val corners = MatOfPoint2f(charucoCorners).toArray()
            if (ids.size != corners.size) return ObjectPose.INVALID

            val objList = ArrayList<Point3>(ids.size)
            val imgList = ArrayList<Point>(ids.size)
            for (k in ids.indices) {
                val id = ids[k]
                if (id in objCorners.indices) {
                    objList.add(objCorners[id])
                    imgList.add(corners[k])
                }
            }
            if (objList.size < minCorners) return ObjectPose.INVALID

            val objPts = MatOfPoint3f(*objList.toTypedArray())
            val imgPts = MatOfPoint2f(*imgList.toTypedArray())
            val cameraMatrix = cameraMatrix(frame)
            val distCoeffs = MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0)
            val rvec = Mat()
            val tvec = Mat()

            val ok = Calib3d.solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec)
            if (!ok) return ObjectPose.INVALID

            val reproj = reprojectionError(objPts, imgPts, rvec, tvec, cameraMatrix, distCoeffs)
            if (reproj.isNaN() || reproj > maxReprojPx) return ObjectPose.INVALID

            val tCO = poseMatrix(rvec, tvec)
            return ObjectPose(
                tCO = tCO,
                valid = true,
                cornerCount = objList.size,
                reprojPx = reproj.toFloat(),
            )
        } catch (e: Exception) {
            Log.w(TAG, "ChArUco detection failed", e)
            return ObjectPose.INVALID
        } finally {
            gray.release()
            charucoCorners.release()
            charucoIds.release()
        }
    }

    /** Builds K (3x3, CV_64F) from ARCore's per-frame image intrinsics. */
    private fun cameraMatrix(frame: Frame): Mat {
        val intr = frame.camera.imageIntrinsics
        val fx = intr.focalLength[0].toDouble()
        val fy = intr.focalLength[1].toDouble()
        val cx = intr.principalPoint[0].toDouble()
        val cy = intr.principalPoint[1].toDouble()
        val k = Mat.zeros(3, 3, CvType.CV_64F)
        k.put(0, 0, fx); k.put(0, 2, cx)
        k.put(1, 1, fy); k.put(1, 2, cy)
        k.put(2, 2, 1.0)
        return k
    }

    /**
     * Copies the camera image's luma plane into a single-channel grayscale [Mat].
     * ChArUco detection only needs luma. Returns null if the image isn't available
     * or isn't YUV_420_888. The Y plane resolution equals the intrinsics image size,
     * so K (above) lines up with this Mat without rescaling.
     */
    private fun acquireGray(frame: Frame): Mat? {
        return try {
            frame.acquireCameraImage().use { image ->
                if (image.format != ImageFormat.YUV_420_888) return null
                val width = image.width
                val height = image.height
                val plane = image.planes[0]
                val buffer = plane.buffer
                val rowStride = plane.rowStride
                val pixelStride = plane.pixelStride
                val data = ByteArray(width * height)
                val rowBuf = ByteArray(rowStride)
                var dst = 0
                for (row in 0 until height) {
                    buffer.position(row * rowStride)
                    val len = minOf(rowStride, buffer.remaining())
                    buffer.get(rowBuf, 0, len)
                    if (pixelStride == 1) {
                        System.arraycopy(rowBuf, 0, data, dst, width)
                    } else {
                        for (col in 0 until width) data[dst + col] = rowBuf[col * pixelStride]
                    }
                    dst += width
                }
                Mat(height, width, CvType.CV_8UC1).apply { put(0, 0, data) }
            }
        } catch (e: Exception) {
            // NotYetAvailable / transient acquire failures: skip this frame's pose.
            null
        }
    }

    private fun reprojectionError(
        objPts: MatOfPoint3f,
        imgPts: MatOfPoint2f,
        rvec: Mat,
        tvec: Mat,
        cameraMatrix: Mat,
        distCoeffs: MatOfDouble,
    ): Double {
        val projected = MatOfPoint2f()
        Calib3d.projectPoints(objPts, rvec, tvec, cameraMatrix, distCoeffs, projected)
        val proj = projected.toArray()
        val obs = imgPts.toArray()
        if (proj.size != obs.size || proj.isEmpty()) return Double.NaN
        var sumSq = 0.0
        for (i in proj.indices) {
            val dx = proj[i].x - obs[i].x
            val dy = proj[i].y - obs[i].y
            sumSq += dx * dx + dy * dy
        }
        return Math.sqrt(sumSq / proj.size)
    }

    /**
     * Packs a `solvePnP` (rvec, tvec) into a row-major 4x4 `T_CO` (camera <- object).
     * `X_cam = R·X_obj + t`, so `[R|t]` is exactly `T_CO`.
     */
    private fun poseMatrix(rvec: Mat, tvec: Mat): FloatArray {
        val r = Mat()
        Calib3d.Rodrigues(rvec, r)
        val m = FloatArray(16)
        for (row in 0 until 3) {
            for (col in 0 until 3) {
                m[row * 4 + col] = r.get(row, col)[0].toFloat()
            }
            m[row * 4 + 3] = tvec.get(row, 0)[0].toFloat()
        }
        m[15] = 1f
        r.release()
        return m
    }

    private fun dictionaryId(name: String): Int = try {
        Objdetect::class.java.getField(name).getInt(null)
    } catch (e: Exception) {
        Log.w(TAG, "Unknown ArUco dictionary $name; falling back to DICT_5X5_100")
        Objdetect.DICT_5X5_100
    }

    companion object {
        private const val TAG = "LANTERN"

        /** Minimum matched ChArUco corners for a trustworthy solvePnP (plan §9). */
        const val DEFAULT_MIN_CORNERS = 6

        /** Reject frames whose reprojection error exceeds this (px) — too jittery to fuse. */
        const val DEFAULT_MAX_REPROJ_PX = 2.0f
    }
}
