package com.lantern.recorder.recon

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Markerless **object-frame** tracker (LIVE_MESH_PLAN.md §3): recovers the camera-from-object pose
 * `T_OC` for each frame by registering the current masked object point cloud against the model
 * accumulated so far — the in-hand replacement for the board's `solvePnP`.
 *
 * The model lives in a canonical object frame `O` defined by the first tracked frame (pose =
 * identity). Each new frame's masked cloud (camera frame, OpenCV convention: +X right, +Y down,
 * +Z forward, metres) is aligned to the model with point-to-point ICP, then merged in. Because we
 * track the object itself, it doesn't matter whether the camera moves or the object is turned in
 * hand — both produce the same frame-to-model registration problem.
 *
 * Pure Kotlin (no Android deps) so the registration math is unit-testable on the JVM.
 *
 * Returned poses are row-major 4x4 (the [Mat4] convention) mapping a **camera-frame** point to the
 * **object frame**: `X_O = T_OC · X_C`.
 */
class ObjectTracker(
    private val voxelSize: Float = 0.005f,
    private val maxIterations: Int = 18,
    private val maxCorrDist: Float = 0.03f,
    private val maxSourcePoints: Int = 1500,
    private val maxModelVoxels: Int = 300_000,
    /** Below this correspondence fitness the registration is untrusted: the frame is dropped (not
     *  fused, model not grown, seed not advanced) so bad alignments can't drift/pollute the model.
     *  Offline drift study (inhand_tracker.py): 0.5 turns a catastrophic 28°+ divergence into <7°. */
    private val minFitness: Float = 0.50f,
) {
    /** Voxel-hash model in object frame O: packed voxel key -> [sumX, sumY, sumZ, count]. */
    private val model = HashMap<Long, FloatArray>()

    /** Last two recovered T_OC (camera->object): seed + constant-velocity prediction. */
    private var lastPose: FloatArray? = null
    private var prevPose: FloatArray? = null

    /** Frames successfully tracked since the last [reset]. */
    var trackedFrames = 0
        private set

    /** Fitness (fraction of source points that found a model correspondence) of the last track. */
    var lastFitness = 0f
        private set

    /** Whether the last [track] was trusted (fitness >= minFitness) and thus fused into the model. */
    var lastAccepted = false
        private set

    val isEmpty: Boolean get() = model.isEmpty()
    val modelVoxelCount: Int get() = model.size

    fun reset() {
        model.clear()
        lastPose = null
        prevPose = null
        trackedFrames = 0
        lastFitness = 0f
        lastAccepted = false
    }

    /**
     * Track one masked object cloud and return its pose `T_OC` (camera->object, row-major 4x4).
     *
     * @param points flat xyz triples in the camera frame (OpenCV), metres.
     * @param count number of points (points.size may be larger than count*3).
     * @param seed optional external init for ICP (camera->object, row-major). When the caller has
     *  ARCore ego-motion it's a far stronger seed than constant velocity — the offline study showed
     *  it removes the single bad-frame divergence that otherwise wrecks the whole track. Null falls
     *  back to internal constant-velocity prediction.
     */
    fun track(points: FloatArray, count: Int, seed: FloatArray? = null): FloatArray {
        val src = subsample(points, count)
        if (model.isEmpty()) {
            val identity = Mat4.identity()
            integrate(src, identity)
            lastPose = identity
            trackedFrames = 1
            lastFitness = 1f
            lastAccepted = true
            return identity
        }

        // Prefer the caller's ARCore-derived seed; otherwise assume the same inter-frame motion as
        // last step (constant velocity) to give ICP an in-basin start. Either falls back to last pose.
        var pose = seed?.copyOf() ?: predictPose()
        val p = FloatArray(3)
        val srcMatched = ArrayList<Float>(src.size)
        val dstMatched = ArrayList<Float>(src.size)

        for (iter in 0 until maxIterations) {
            srcMatched.clear()
            dstMatched.clear()
            // Coarse-to-fine: shrink the correspondence radius as we converge.
            val corr = maxCorrDist * (1f - 0.5f * iter / maxIterations)
            val corr2 = corr * corr
            var i = 0
            while (i + 2 < src.size) {
                Mat4.transformPoint(pose, src[i], src[i + 1], src[i + 2], p)
                val nn = nearestModelPoint(p[0], p[1], p[2], corr)
                if (nn != null && dist2(p[0], p[1], p[2], nn[0], nn[1], nn[2]) <= corr2) {
                    srcMatched.add(p[0]); srcMatched.add(p[1]); srcMatched.add(p[2])
                    dstMatched.add(nn[0]); dstMatched.add(nn[1]); dstMatched.add(nn[2])
                }
                i += 3
            }
            val pairs = srcMatched.size / 3
            if (pairs < 6) break
            val delta = rigidFromCorrespondences(srcMatched, dstMatched) ?: break
            pose = Mat4.multiply(delta, pose)
            if (isSmallMotion(delta)) break
        }

        lastFitness = (srcMatched.size / 3).toFloat() / max(1, src.size / 3)
        // Reject untrusted registrations: don't fuse them, don't grow the model, and don't advance
        // the seed — otherwise a bad frame both pollutes the model and corrupts the next prediction,
        // which is what makes the reconstruction "drift everywhere" and balloon into a blob.
        if (lastFitness < minFitness) {
            lastAccepted = false
            return lastPose?.copyOf() ?: pose
        }
        integrate(src, pose)
        prevPose = lastPose
        lastPose = pose
        trackedFrames++
        lastAccepted = true
        return pose
    }

    /** Constant-velocity pose prediction: motion = last ∘ prev⁻¹, applied once more to last. */
    private fun predictPose(): FloatArray {
        val last = lastPose ?: return Mat4.identity()
        val prev = prevPose ?: return last.copyOf()
        val motion = Mat4.multiply(last, Mat4.invertRigid(prev))
        return Mat4.multiply(motion, last)
    }

    // ----------------------------------------------------------------- model ops

    /** Voxel-downsample the incoming cloud and cap point count for a bounded ICP cost. */
    private fun subsample(points: FloatArray, count: Int): FloatArray {
        val n = minOf(count, points.size / 3)
        if (n <= 0) return FloatArray(0)
        val seen = HashSet<Long>()
        val out = ArrayList<Float>(minOf(n, maxSourcePoints) * 3)
        // Stride so we sample across the whole cloud even when capping.
        val stride = max(1, n / maxSourcePoints)
        var i = 0
        while (i < n) {
            val x = points[i * 3]; val y = points[i * 3 + 1]; val z = points[i * 3 + 2]
            if (seen.add(voxelKey(x, y, z))) { out.add(x); out.add(y); out.add(z) }
            if (out.size / 3 >= maxSourcePoints) break
            i += stride
        }
        return out.toFloatArray()
    }

    /** Merge a camera-frame cloud into the model at [pose] (camera->object). */
    private fun integrate(src: FloatArray, pose: FloatArray) {
        val p = FloatArray(3)
        var i = 0
        while (i + 2 < src.size) {
            Mat4.transformPoint(pose, src[i], src[i + 1], src[i + 2], p)
            if (model.size < maxModelVoxels || model.containsKey(voxelKey(p[0], p[1], p[2]))) {
                val cell = model.getOrPut(voxelKey(p[0], p[1], p[2])) { FloatArray(4) }
                cell[0] += p[0]; cell[1] += p[1]; cell[2] += p[2]; cell[3] += 1f
            }
            i += 3
        }
    }

    /** Nearest model centroid to (x,y,z) within [radius], searching neighbouring voxels. */
    private fun nearestModelPoint(x: Float, y: Float, z: Float, radius: Float): FloatArray? {
        val r = max(1, kotlin.math.ceil(radius / voxelSize).toInt())
        val vx = quant(x); val vy = quant(y); val vz = quant(z)
        var best: FloatArray? = null
        var bestD = Float.MAX_VALUE
        var dx = -r
        while (dx <= r) {
            var dy = -r
            while (dy <= r) {
                var dz = -r
                while (dz <= r) {
                    val cell = model[packKey(vx + dx, vy + dy, vz + dz)]
                    if (cell != null && cell[3] > 0f) {
                        val cnt = cell[3]
                        val cxp = cell[0] / cnt; val cyp = cell[1] / cnt; val czp = cell[2] / cnt
                        val d = dist2(x, y, z, cxp, cyp, czp)
                        if (d < bestD) { bestD = d; best = floatArrayOf(cxp, cyp, czp) }
                    }
                    dz++
                }
                dy++
            }
            dx++
        }
        return best
    }

    // ----------------------------------------------------------------- voxel keys

    private fun quant(v: Float): Int = kotlin.math.floor(v / voxelSize).toInt()
    private fun voxelKey(x: Float, y: Float, z: Float): Long = packKey(quant(x), quant(y), quant(z))

    private fun packKey(ix: Int, iy: Int, iz: Int): Long {
        // 21 bits/axis, offset to keep positive (object span << 2^20 voxels).
        val ox = (ix + OFFSET).toLong() and MASK
        val oy = (iy + OFFSET).toLong() and MASK
        val oz = (iz + OFFSET).toLong() and MASK
        return (ox shl 42) or (oy shl 21) or oz
    }

    // -------------------------------------------------------------- rigid solve

    /**
     * Optimal rigid transform aligning point set P (src) to Q (dst), via Horn's unit-quaternion
     * method (largest-eigenvalue eigenvector of the 4x4 profile matrix N). Returns a row-major 4x4
     * mapping a src point to its dst, or null if degenerate.
     */
    internal fun rigidFromCorrespondences(srcFlat: List<Float>, dstFlat: List<Float>): FloatArray? {
        val n = srcFlat.size / 3
        if (n < 3) return null
        var pcx = 0f; var pcy = 0f; var pcz = 0f
        var qcx = 0f; var qcy = 0f; var qcz = 0f
        for (i in 0 until n) {
            pcx += srcFlat[i * 3]; pcy += srcFlat[i * 3 + 1]; pcz += srcFlat[i * 3 + 2]
            qcx += dstFlat[i * 3]; qcy += dstFlat[i * 3 + 1]; qcz += dstFlat[i * 3 + 2]
        }
        val inv = 1f / n
        pcx *= inv; pcy *= inv; pcz *= inv; qcx *= inv; qcy *= inv; qcz *= inv

        // Cross-covariance S = sum (p-pBar)(q-qBar)^T.
        var sxx = 0.0; var sxy = 0.0; var sxz = 0.0
        var syx = 0.0; var syy = 0.0; var syz = 0.0
        var szx = 0.0; var szy = 0.0; var szz = 0.0
        for (i in 0 until n) {
            val px = (srcFlat[i * 3] - pcx).toDouble()
            val py = (srcFlat[i * 3 + 1] - pcy).toDouble()
            val pz = (srcFlat[i * 3 + 2] - pcz).toDouble()
            val qx = (dstFlat[i * 3] - qcx).toDouble()
            val qy = (dstFlat[i * 3 + 1] - qcy).toDouble()
            val qz = (dstFlat[i * 3 + 2] - qcz).toDouble()
            sxx += px * qx; sxy += px * qy; sxz += px * qz
            syx += py * qx; syy += py * qy; syz += py * qz
            szx += pz * qx; szy += pz * qy; szz += pz * qz
        }

        // Horn's symmetric 4x4 N matrix (quaternion order w,x,y,z).
        val nMat = arrayOf(
            doubleArrayOf(sxx + syy + szz, syz - szy, szx - sxz, sxy - syx),
            doubleArrayOf(syz - szy, sxx - syy - szz, sxy + syx, szx + sxz),
            doubleArrayOf(szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy),
            doubleArrayOf(sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz),
        )
        val q = largestEigenvector4(nMat) ?: return null
        val w = q[0]; val x = q[1]; val y = q[2]; val z = q[3]

        // Quaternion -> row-major rotation.
        val r = FloatArray(16)
        r[0] = (1 - 2 * (y * y + z * z)).toFloat(); r[1] = (2 * (x * y - w * z)).toFloat(); r[2] = (2 * (x * z + w * y)).toFloat()
        r[4] = (2 * (x * y + w * z)).toFloat(); r[5] = (1 - 2 * (x * x + z * z)).toFloat(); r[6] = (2 * (y * z - w * x)).toFloat()
        r[8] = (2 * (x * z - w * y)).toFloat(); r[9] = (2 * (y * z + w * x)).toFloat(); r[10] = (1 - 2 * (x * x + y * y)).toFloat()
        r[15] = 1f
        // t = qBar - R pBar.
        r[3] = qcx - (r[0] * pcx + r[1] * pcy + r[2] * pcz)
        r[7] = qcy - (r[4] * pcx + r[5] * pcy + r[6] * pcz)
        r[11] = qcz - (r[8] * pcx + r[9] * pcy + r[10] * pcz)
        return r
    }

    /** Largest-eigenvalue eigenvector of a symmetric 4x4 via cyclic Jacobi rotations. */
    private fun largestEigenvector4(a: Array<DoubleArray>): DoubleArray? {
        val n = 4
        val v = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }
        val m = Array(n) { a[it].copyOf() }
        repeat(50) {
            // Largest off-diagonal magnitude.
            var p = 0; var q = 1; var maxOff = 0.0
            for (i in 0 until n) for (j in i + 1 until n) {
                if (abs(m[i][j]) > maxOff) { maxOff = abs(m[i][j]); p = i; q = j }
            }
            if (maxOff < 1e-12) return@repeat
            val app = m[p][p]; val aqq = m[q][q]; val apq = m[p][q]
            val phi = 0.5 * kotlin.math.atan2(2 * apq, aqq - app)
            val c = kotlin.math.cos(phi); val s = kotlin.math.sin(phi)
            for (i in 0 until n) {
                val mip = m[i][p]; val miq = m[i][q]
                m[i][p] = c * mip - s * miq
                m[i][q] = s * mip + c * miq
            }
            for (i in 0 until n) {
                val mpi = m[p][i]; val mqi = m[q][i]
                m[p][i] = c * mpi - s * mqi
                m[q][i] = s * mpi + c * mqi
            }
            for (i in 0 until n) {
                val vip = v[i][p]; val viq = v[i][q]
                v[i][p] = c * vip - s * viq
                v[i][q] = s * vip + c * viq
            }
        }
        var best = 0; var bestVal = m[0][0]
        for (i in 1 until n) if (m[i][i] > bestVal) { bestVal = m[i][i]; best = i }
        val q = doubleArrayOf(v[0][best], v[1][best], v[2][best], v[3][best])
        val norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if (norm < 1e-9) return null
        return doubleArrayOf(q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)
    }

    private fun isSmallMotion(delta: FloatArray): Boolean {
        // Rotation angle from trace, translation from the last column.
        val cos = ((delta[0] + delta[5] + delta[10]) - 1f) / 2f
        val ang = kotlin.math.acos(cos.coerceIn(-1f, 1f))
        val t = sqrt(delta[3] * delta[3] + delta[7] * delta[7] + delta[11] * delta[11])
        return ang < 1e-3f && t < 1e-4f
    }

    private fun dist2(ax: Float, ay: Float, az: Float, bx: Float, by: Float, bz: Float): Float {
        val dx = ax - bx; val dy = ay - by; val dz = az - bz
        return dx * dx + dy * dy + dz * dz
    }

    companion object {
        private const val OFFSET = 1 shl 20      // recenters signed voxel indices to positive
        private const val MASK = (1L shl 21) - 1 // 21 bits per axis
    }
}
