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
     */
    fun integrate(
        depth: DepthMap,
        intrinsics: CameraIntrinsics,
        cameraToWorld: FloatArray,
        groundY: Float? = null,
    ) {
        if (!centered) return
        val cullBelowY = groundY?.let { it + GROUND_MARGIN }
        // Vertical-cylinder cull around the object's center axis (removes surrounding floor).
        val centerX = origin[0] + halfExtent
        val centerZ = origin[2] + halfExtent
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
            if (dz2 > radius2) continue // whole z-slice is outside the object cylinder
            val kBase = k * n * n
            for (j in 0 until n) {
                val wy0 = origin[1] + (j + 0.5f) * voxelSize
                // Drop the support surface (and anything below it) so only the object remains.
                if (cullBelowY != null && wy0 < cullBelowY) continue
                val jBase = kBase + j * n
                for (i in 0 until n) {
                    val wx0 = origin[0] + (i + 0.5f) * voxelSize
                    val dx = wx0 - centerX
                    if (dx * dx + dz2 > radius2) continue // outside the object cylinder
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
    }
}
