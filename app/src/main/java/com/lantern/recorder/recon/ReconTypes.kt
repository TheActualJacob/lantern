package com.lantern.recorder.recon

/**
 * Shared value types for the on-device live reconstruction pipeline
 * (depth -> scale -> TSDF -> marching cubes -> live mesh).
 *
 * Conventions (match the host pipeline in `tsdf_fuse.py` / `pipeline_float.py`):
 *  - Poses are **camera-to-world**, ARCore/OpenGL camera frame (+X right, +Y up, -Z fwd).
 *  - Depth is in **meters**, row-major, with 0 meaning "invalid / no measurement".
 *  - All 4x4 matrices here are **row-major** FloatArray(16) for the small math helpers
 *    in [Mat4]. ARCore hands out column-major matrices; convert at the boundary.
 */

/** Pinhole intrinsics for a specific image resolution. */
data class CameraIntrinsics(
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float,
    val width: Int,
    val height: Int,
) {
    /** Rescale intrinsics to a different resolution (e.g. RGB intrinsics -> depth res). */
    fun scaledTo(targetWidth: Int, targetHeight: Int): CameraIntrinsics {
        if (targetWidth == width && targetHeight == height) return this
        val sx = targetWidth.toFloat() / width
        val sy = targetHeight.toFloat() / height
        return CameraIntrinsics(
            fx = fx * sx,
            fy = fy * sy,
            cx = cx * sx,
            cy = cy * sy,
            width = targetWidth,
            height = targetHeight,
        )
    }
}

/**
 * A metric depth map. [depthMeters] is row-major length width*height; 0 = invalid.
 * [confidence] is optional, 0..1, same layout (null = treat all valid pixels equally).
 */
class DepthMap(
    val width: Int,
    val height: Int,
    val depthMeters: FloatArray,
    val confidence: FloatArray? = null,
) {
    init {
        require(depthMeters.size == width * height) {
            "depthMeters size ${depthMeters.size} != ${width}x$height"
        }
        require(confidence == null || confidence.size == width * height) {
            "confidence size ${confidence?.size} != ${width}x$height"
        }
    }
}

/**
 * Extracted triangle mesh, ready for GL upload. Vertices/normals are flat xyz triples in
 * **world** meters; [triangleCount] = indices.size / 3.
 */
class MeshData(
    val vertices: FloatArray,
    val normals: FloatArray,
    val indices: IntArray,
) {
    val vertexCount: Int get() = vertices.size / 3
    val triangleCount: Int get() = indices.size / 3

    /** No geometry at all (used to signal a failed/blank reconstruction). */
    val isEmpty: Boolean get() = vertices.isEmpty()

    /** Vertices but no triangles: render as a point cloud (GL_POINTS) instead of a surface. */
    val isPointCloud: Boolean get() = indices.isEmpty() && vertices.isNotEmpty()

    companion object {
        val EMPTY = MeshData(FloatArray(0), FloatArray(0), IntArray(0))
    }
}

/** Minimal row-major 4x4 matrix helpers (avoids a matrix dependency). */
object Mat4 {
    /** ARCore/OpenGL column-major 16-array -> row-major 16-array. */
    fun fromColumnMajor(cm: FloatArray): FloatArray {
        require(cm.size == 16) { "expected 16 elements, got ${cm.size}" }
        val rm = FloatArray(16)
        for (r in 0 until 4) for (c in 0 until 4) rm[r * 4 + c] = cm[c * 4 + r]
        return rm
    }

    /** Row-major 4x4 multiply: returns a*b. */
    fun multiply(a: FloatArray, b: FloatArray): FloatArray {
        val out = FloatArray(16)
        for (r in 0 until 4) {
            for (c in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) sum += a[r * 4 + k] * b[k * 4 + c]
                out[r * 4 + c] = sum
            }
        }
        return out
    }

    /** Invert a rigid-body (rotation+translation) row-major 4x4. */
    fun invertRigid(m: FloatArray): FloatArray {
        // R^T and -R^T t.
        val out = FloatArray(16)
        for (r in 0 until 3) for (c in 0 until 3) out[r * 4 + c] = m[c * 4 + r]
        val tx = m[0 * 4 + 3]
        val ty = m[1 * 4 + 3]
        val tz = m[2 * 4 + 3]
        out[0 * 4 + 3] = -(out[0] * tx + out[1] * ty + out[2] * tz)
        out[1 * 4 + 3] = -(out[4] * tx + out[5] * ty + out[6] * tz)
        out[2 * 4 + 3] = -(out[8] * tx + out[9] * ty + out[10] * tz)
        out[15] = 1f
        return out
    }

    /** Transform a point (x,y,z,1) by a row-major 4x4; writes into [out] (size>=3). */
    fun transformPoint(m: FloatArray, x: Float, y: Float, z: Float, out: FloatArray) {
        out[0] = m[0] * x + m[1] * y + m[2] * z + m[3]
        out[1] = m[4] * x + m[5] * y + m[6] * z + m[7]
        out[2] = m[8] * x + m[9] * y + m[10] * z + m[11]
    }

    fun identity(): FloatArray = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f,
    )
}
