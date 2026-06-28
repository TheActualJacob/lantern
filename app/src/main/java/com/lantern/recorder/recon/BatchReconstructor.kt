package com.lantern.recorder.recon

import android.util.Log

private const val TAG = "LANTERN"

/**
 * One-shot **batch** reconstruction for directed-capture mode: a handful of deliberately-framed
 * keyframes (full-res color + ARCore pose + ARCore depth + intrinsics) are captured first, then this
 * replays them all at once into a single **point cloud**: DA3 dense depth → FastSAM object mask →
 * **one global** disparity→metric scale (robust median across frames) → back-project every masked
 * pixel with its ARCore pose → merge across views (voxel dedup).
 *
 * Why a point cloud, not TSDF marching cubes: with only a handful of deliberately-framed shots, every
 * SAM-masked pixel becomes a 3D point, so the full silhouette the segmenter sees is captured directly
 * — TSDF needs many overlapping views to close a watertight surface and tends to drop most of the
 * object from sparse captures. ARCore depth is used *only* to anchor the one global scale (overall
 * size); all geometry is pure DA3 + ARCore pose.
 *
 * Heavy (DA3 is ~1 s/frame on CPU); run off the main thread.
 */
class BatchReconstructor(
    private val depth: DepthBackend?,
    private val seg: QnnSegmentationModel?,
) {
    /** A captured viewpoint, held in memory (no 16-bit-PNG round-trip). */
    class Keyframe(
        val argb: ImageUtils.Argb,
        val depth: DepthMap,
        /** Intrinsics at the *color* resolution; rescaled to depth res internally. */
        val intrinsics: CameraIntrinsics,
        /** ARCore camera-to-world, column-major (from `pose.toMatrix`). */
        val poseColumnMajor: FloatArray,
        val groundY: Float? = null,
    )

    private class Prepared(
        val disp: DisparityMap,
        val mask: FloatArray?,
        val depthW: Int,
        val depthH: Int,
        val intr: CameraIntrinsics,
        val camToWorld: FloatArray,
        val groundY: Float?,
    )

    /** Progress callback: (framesDone, framesTotal, humanLabel). */
    fun interface Progress {
        fun update(done: Int, total: Int, label: String)
    }

    /**
     * Build a mesh from [frames]. Returns [MeshData.EMPTY] if there's no depth model, too few usable
     * frames, or no object could be located. Safe to call once; create a fresh instance per build.
     */
    fun reconstruct(frames: List<Keyframe>, progress: Progress? = null): MeshData {
        val da3 = depth
        if (da3 == null) {
            Log.w(TAG, "BatchReconstructor: no depth backend; cannot build")
            return MeshData.EMPTY
        }
        val total = frames.size
        if (total == 0) return MeshData.EMPTY

        // Pass 1: DA3 + SAM + a per-frame scale candidate.
        val prepared = ArrayList<Prepared>(total)
        val candS = ArrayList<Float>(total)
        val candT = ArrayList<Float>(total)
        for ((i, f) in frames.withIndex()) {
            progress?.update(i, total, "Analyzing ${i + 1}/$total")
            val dw = f.depth.width
            val dh = f.depth.height
            val intr = f.intrinsics.scaledTo(dw, dh)
            val mask = seg?.inferObjectMask(f.argb, dw, dh)
            val disp = da3.inferDisparity(f.argb)
            if (disp == null) {
                Log.w(TAG, "BatchReconstructor: DA3 failed on frame $i; skipping")
                continue
            }
            val focus = maskedMedianDepth(f.depth, mask)
            val cand = AffineScaleSolver.fitAffine(disp, f.depth, focusDepthM = focus, mask = mask)
            if (cand != null && cand.s > 0f && cand.s.isFinite() && cand.t.isFinite()) {
                candS.add(cand.s)
                candT.add(cand.t)
            }
            prepared.add(
                Prepared(disp, mask, dw, dh, intr, Mat4.fromColumnMajor(f.poseColumnMajor), f.groundY),
            )
        }
        if (prepared.isEmpty() || candS.isEmpty()) {
            Log.w(TAG, "BatchReconstructor: no usable frames/scale (prepared=${prepared.size}, cand=${candS.size})")
            return MeshData.EMPTY
        }

        // One global scale for the whole capture: robust median over per-frame fits. Order-independent
        // and immune to a few degenerate frames (flat ARCore depth → near-zero/negative s, already
        // filtered above), so every view registers at the same metric size. ARCore depth only sets
        // the *scale* here; all geometry below is pure DA3.
        val scale = AffineScaleSolver.Affine(median(candS), median(candT))
        Log.i(TAG, "BatchReconstructor: global scale s=${scale.s} t=${scale.t} from ${candS.size}/${total} frames")

        // Locate the object so we can reject stray background points far from it.
        var ctr: FloatArray? = null
        for (p in prepared) {
            val md = AffineScaleSolver.applyAffine(p.disp, p.depthW, p.depthH, scale)
            ctr = maskedCentroidWorld(md, p.intr, p.mask, p.camToWorld) ?: continue
            break
        }
        val c = ctr ?: run {
            Log.w(TAG, "BatchReconstructor: could not locate object centroid")
            return MeshData.EMPTY
        }

        // Pass 2: back-project every SAM-masked pixel's DA3 depth into world space and merge across
        // views. This is the "take all the depth inside the SAM area" approach — every masked pixel
        // becomes a point, so the full silhouette is captured even from a handful of frames (vs. TSDF
        // marching cubes, which needs many views to close a surface). A voxel grid dedups overlap.
        val seen = HashSet<Long>(1 shl 16)
        val verts = ArrayList<Float>(1 shl 16)
        val r2 = OBJECT_RADIUS * OBJECT_RADIUS
        val wp = FloatArray(3)
        outer@ for ((i, p) in prepared.withIndex()) {
            progress?.update(i, total, "Fusing ${i + 1}/$total")
            val md = AffineScaleSolver.applyAffine(p.disp, p.depthW, p.depthH, scale)
            val d = md.depthMeters
            val w = p.depthW
            val h = p.depthH
            val mask = p.mask
            val c2w = Mat4.multiply(p.camToWorld, FLIP) // OpenGL cam-to-world ∘ flip = OpenCV cam→world
            val fx = p.intr.fx; val fy = p.intr.fy; val cx = p.intr.cx; val cy = p.intr.cy
            val groundCut = p.groundY?.let { it + GROUND_MARGIN }
            var idx = 0
            while (idx < w * h) {
                if (mask != null && (idx >= mask.size || mask[idx] < 0.5f)) { idx++; continue }
                val z = d[idx]
                if (z <= 0f || z >= DEPTH_MAX) { idx++; continue }
                val u = idx % w
                val v = idx / w
                val camX = (u - cx) / fx * z
                val camY = (v - cy) / fy * z
                Mat4.transformPoint(c2w, camX, camY, z, wp)
                val px = wp[0]; val py = wp[1]; val pz = wp[2]
                idx++
                if (groundCut != null && py < groundCut) continue
                val dx = px - c[0]; val dy = py - c[1]; val dz = pz - c[2]
                if (dx * dx + dy * dy + dz * dz > r2) continue
                if (seen.add(voxelKey(px, py, pz))) {
                    verts.add(px); verts.add(py); verts.add(pz)
                    if (verts.size / 3 >= MAX_POINTS) break@outer
                }
            }
        }

        if (verts.size / 3 < MIN_POINTS) {
            Log.w(TAG, "BatchReconstructor: too few points (${verts.size / 3})")
            return MeshData.EMPTY
        }
        progress?.update(total, total, "Finishing\u2026")
        val v = verts.toFloatArray()
        // Point cloud: no triangles. Up-normals keep the renderer's shader happy; the viewer colors
        // by height anyway.
        val normals = FloatArray(v.size) { if (it % 3 == 1) 1f else 0f }
        Log.i(TAG, "BatchReconstructor: point cloud ${v.size / 3} points")
        return MeshData(v, normals, IntArray(0))
    }

    /** Pack a world point into a voxel-grid key (~[VOXEL_M] m cells) for dedup across views. */
    private fun voxelKey(x: Float, y: Float, z: Float): Long {
        val inv = 1f / VOXEL_M
        val ix = Math.round(x * inv) + 524288L // +2^19 bias keeps coords non-negative within ±5 km
        val iy = Math.round(y * inv) + 524288L
        val iz = Math.round(z * inv) + 524288L
        return (ix and 0xFFFFF) or ((iy and 0xFFFFF) shl 20) or ((iz and 0xFFFFF) shl 40)
    }

    /** Median depth (m) over masked pixels (or all valid pixels if no mask); null if too sparse. */
    private fun maskedMedianDepth(depth: DepthMap, mask: FloatArray?): Float? {
        val d = depth.depthMeters
        val samples = ArrayList<Float>(d.size / 4)
        val n = if (mask != null) minOf(d.size, mask.size) else d.size
        for (i in 0 until n) {
            if (mask != null && mask[i] < 0.5f) continue
            val v = d[i]
            if (v > 0f && v < 5f) samples.add(v)
        }
        if (samples.size < 8) return null
        samples.sort()
        return samples[samples.size / 2]
    }

    /** World centroid of the masked object cloud (for centering the TSDF grid). */
    private fun maskedCentroidWorld(
        depth: DepthMap,
        intr: CameraIntrinsics,
        mask: FloatArray?,
        cameraToWorld: FloatArray,
    ): FloatArray? {
        val d = depth.depthMeters
        val w = depth.width
        val c2wCv = Mat4.multiply(cameraToWorld, FLIP) // OpenCV-camera -> world
        var sx = 0f
        var sy = 0f
        var sz = 0f
        var count = 0
        val wp = FloatArray(3)
        val n = if (mask != null) minOf(d.size, mask.size) else d.size
        for (i in 0 until n) {
            if (mask != null && mask[i] < 0.5f) continue
            val z = d[i]
            if (z <= 0f || z >= 5f) continue
            val u = i % w
            val v = i / w
            val cx = (u - intr.cx) / intr.fx * z
            val cy = (v - intr.cy) / intr.fy * z
            Mat4.transformPoint(c2wCv, cx, cy, z, wp)
            sx += wp[0]; sy += wp[1]; sz += wp[2]
            count++
        }
        if (count < MIN_POINTS) return null
        val inv = 1f / count
        return floatArrayOf(sx * inv, sy * inv, sz * inv)
    }

    private fun median(values: List<Float>): Float {
        val sorted = values.sorted()
        return sorted[sorted.size / 2]
    }

    companion object {
        private const val MIN_POINTS = 200

        /** Reject points farther than this from the object centroid (drops stray background). */
        private const val OBJECT_RADIUS = 0.35f

        /** Lift the ground cut slightly above the detected support plane to shed the contact rim. */
        private const val GROUND_MARGIN = 0.005f

        /** Ignore depths beyond this (m); DA3 metric depth past here is unreliable at object scale. */
        private const val DEPTH_MAX = 5f

        /** Dedup voxel size (m): merges overlapping points across views into one. */
        private const val VOXEL_M = 0.003f

        /** Hard cap on emitted points so a noisy scan can't OOM the renderer. */
        private const val MAX_POINTS = 400_000

        /** OpenGL(ARCore)<->OpenCV camera flip, diag(1,-1,-1,1); its own inverse. */
        private val FLIP = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, -1f, 0f, 0f,
            0f, 0f, -1f, 0f,
            0f, 0f, 0f, 1f,
        )
    }
}
