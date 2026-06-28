package com.lantern.recorder.recon

/**
 * Dense truncated signed-distance (TSDF) volume in **world** meters. Fuses many metric
 * depth frames into one averaged surface, then [extractMesh] runs marching cubes.
 *
 * Mirrors the host `tsdf_fuse.py` contract: ARCore camera-to-world poses (OpenGL frame,
 * +Y up / -Z forward) are flipped into the OpenCV/Open3D camera frame (+Y down / +Z fwd)
 * before projection, voxel length and `sdf_trunc = voxelSize * sdfTruncMult` match.
 *
 * A fixed dense grid (default 160 m-cube of voxels) is fine for a hand-held object roughly
 * centered in view. Call [centerOn] once with the object centroid before integrating.
 *
 * Not thread-safe: integrate + extract from a single worker thread.
 */
class TsdfVolume(
    val resolution: Int = DEFAULT_RESOLUTION,
    val voxelSize: Float = DEFAULT_VOXEL_SIZE,
    private val sdfTruncMult: Float = 5f,
    private val depthTrunc: Float = 5f,
    private val maxWeight: Float = 64f,
    /** Horizontal radius (m) around the center axis to fuse; voxels outside are ignored so the
     *  floor/clutter spreading out around the object don't get reconstructed. <=0 disables. */
    private val objectRadius: Float = DEFAULT_OBJECT_RADIUS,
) {
    val tsdf = FloatArray(resolution * resolution * resolution) { 1f }
    val weight = FloatArray(resolution * resolution * resolution)

    /** World coordinate of voxel-center (0,0,0). Set by [centerOn]. */
    val origin = FloatArray(3)
    private var centered = false
    private val sdfTrunc = voxelSize * sdfTruncMult

    /** Half the grid's world extent; [origin] + this on each axis is the volume center. */
    val halfExtent: Float get() = resolution * voxelSize * 0.5f

    /** Camera-to-OpenCV-camera flip (diag(1,-1,-1,1)) applied to world->camera. */
    private val flip = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, -1f, 0f, 0f,
        0f, 0f, -1f, 0f,
        0f, 0f, 0f, 1f,
    )

    val isCentered: Boolean get() = centered

    /** Center the grid so [worldX,worldY,worldZ] is at the volume's middle. Idempotent-ish. */
    fun centerOn(worldX: Float, worldY: Float, worldZ: Float) {
        val half = resolution * voxelSize * 0.5f
        origin[0] = worldX - half
        origin[1] = worldY - half
        origin[2] = worldZ - half
        centered = true
    }

    /**
     * Integrate one metric depth frame.
     *
     * @param depth metric depth (meters, 0 = invalid) at [intrinsics] resolution.
     * @param intrinsics pinhole intrinsics matching [depth] resolution.
     * @param cameraToWorld row-major 4x4 ARCore camera-to-world pose.
     * @param groundY optional world-Y of the support surface; voxels at/below it (plus a small
     *   margin) are skipped so the object reconstructs without the table/floor it rests on.
     * @param mask optional foreground mask (object saliency 0..1) at [depth] resolution; a voxel
     *   projecting onto a background pixel is skipped, so only the segmented object is fused.
     */
    fun integrate(
        depth: DepthMap,
        intrinsics: CameraIntrinsics,
        cameraToWorld: FloatArray,
        groundY: Float? = null,
        mask: FloatArray? = null,
    ) {
        if (!centered) return
        // When a segmentation mask is supplied it IS the object filter, so the coarse geometric
        // culls (which can clip the object or, if ARCore locks the object's own top face as the
        // "ground", delete it entirely) are disabled — the mask alone decides what's the object.
        val maskActive = mask != null
        val cullBelowY = if (maskActive) null else groundY?.let { it + GROUND_MARGIN }
        val centerX = origin[0] + halfExtent
        val centerY = origin[1] + halfExtent
        val centerZ = origin[2] + halfExtent
        // Spatial cull bounds what fuses so drift / mask noise can't smear across the whole grid
        // (which also blows up the mesh -> OOM). Mask/object-frame mode uses a sphere around the
        // object centre (orientation-free, since the object frame's axes are arbitrary); world-frame
        // mode keeps the vertical cylinder + ground plane.
        val radius2 = if (objectRadius > 0f) objectRadius * objectRadius else Float.MAX_VALUE
        val worldToCamCv = Mat4.multiply(flip, Mat4.invertRigid(cameraToWorld))
        val fx = intrinsics.fx
        val fy = intrinsics.fy
        val cx = intrinsics.cx
        val cy = intrinsics.cy
        val dw = depth.width
        val dh = depth.height
        val d = depth.depthMeters
        val conf = depth.confidence
        val p = FloatArray(3)
        val n = resolution

        for (k in 0 until n) {
            val wz0 = origin[2] + (k + 0.5f) * voxelSize
            val dz = wz0 - centerZ
            val dz2 = dz * dz
            if (dz2 > radius2) continue // whole z-slice is outside the cull region
            val kBase = k * n * n
            for (j in 0 until n) {
                val wy0 = origin[1] + (j + 0.5f) * voxelSize
                // Drop the support surface (and anything below it) so only the object remains.
                if (cullBelowY != null && wy0 < cullBelowY) continue
                // Sphere cull (object-frame mode) also bounds the vertical extent; cylinder doesn't.
                val dy = wy0 - centerY
                val dy2 = if (maskActive) dy * dy else 0f
                if (dy2 > radius2) continue
                val jBase = kBase + j * n
                for (i in 0 until n) {
                    val wx0 = origin[0] + (i + 0.5f) * voxelSize
                    val dx = wx0 - centerX
                    if (dx * dx + dy2 + dz2 > radius2) continue // outside the object sphere/cylinder
                    Mat4.transformPoint(worldToCamCv, wx0, wy0, wz0, p)
                    val camZ = p[2]
                    if (camZ <= 0f || camZ > depthTrunc) continue
                    val u = (fx * p[0] / camZ + cx)
                    val v = (fy * p[1] / camZ + cy)
                    val ui = u.toInt()
                    val vi = v.toInt()
                    if (ui < 0 || ui >= dw || vi < 0 || vi >= dh) continue
                    val pix = vi * dw + ui
                    val dz = d[pix]
                    // The segmentation mask IS the object filter. A voxel projecting onto a
                    // background (non-object) pixel is not just skipped — it's *carved*: its weight
                    // is bled off and, once gone, its TSDF reset to empty, so background the mask has
                    // stopped reporting fades out of the mesh instead of persisting forever (the
                    // additive TSDF would otherwise keep any background it ever saw).
                    //
                    // Carve ONLY clear free space — a voxel sitting well in front of the observed
                    // surface (camZ < dz - sdfTrunc), i.e. somewhere we can plainly see *through*.
                    // Voxels at/near a surface (camZ ~= dz) are left alone even when masked out:
                    // they may be the real object that SAM under-segmented this frame, so carving
                    // them would chew holes in the reconstruction. Occluded voxels (behind the
                    // surface) and depth-less pixels are also left untouched.
                    if (mask != null && pix < mask.size && mask[pix] < 0.5f) {
                        val ci = jBase + i
                        if (weight[ci] > 0f && dz > 0f && camZ < dz - sdfTrunc) {
                            val wNew = weight[ci] - CARVE_RATE
                            if (wNew <= 0f) {
                                weight[ci] = 0f
                                tsdf[ci] = 1f
                            } else {
                                weight[ci] = wNew
                            }
                        }
                        continue
                    }
                    if (dz <= 0f) continue
                    val sdf = dz - camZ
                    if (sdf < -sdfTrunc) continue
                    var tsdfVal = sdf / sdfTrunc
                    if (tsdfVal > 1f) tsdfVal = 1f
                    if (tsdfVal < -1f) tsdfVal = -1f

                    val sampleW = if (conf != null) conf[pix].coerceIn(0f, 1f) else 1f
                    if (sampleW <= 0f) continue

                    val idx = jBase + i
                    val wOld = weight[idx]
                    val wNew = wOld + sampleW
                    if (wNew <= 0f) continue
                    tsdf[idx] = (tsdf[idx] * wOld + tsdfVal * sampleW) / wNew
                    weight[idx] = if (wNew > maxWeight) maxWeight else wNew
                }
            }
        }
    }

    /** Run marching cubes over the integrated region. */
    fun extractMesh(): MeshData = MarchingCubes.extract(this)

    fun reset() {
        java.util.Arrays.fill(tsdf, 1f)
        java.util.Arrays.fill(weight, 0f)
        centered = false
    }

    companion object {
        const val DEFAULT_RESOLUTION = 160
        const val DEFAULT_VOXEL_SIZE = 0.004f // 4 mm, in LIVE_MESH_PLAN's 2-4 mm range

        /** Default object cylinder radius: ~the volume half-extent, i.e. a ~32 cm-wide object. */
        const val DEFAULT_OBJECT_RADIUS = 0.16f

        /** Cull plane is lifted this far above the detected surface to clear plane noise/thickness
         *  while keeping the object's base (ARCore plane Y is only cm-accurate). */
        private const val GROUND_MARGIN = 0.01f // 1 cm

        /** Weight bled off a voxel each frame it's visibly observed as background (mask=0). Lower
         *  than the per-frame object weight gain so flickering object edges recover, but enough that
         *  a region the mask truly abandons clears within a handful of fused frames. */
        private const val CARVE_RATE = 4f
    }
}
