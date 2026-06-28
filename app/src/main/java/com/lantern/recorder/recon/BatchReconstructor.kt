package com.lantern.recorder.recon

import android.util.Log

private const val TAG = "LANTERN"

/**
 * One-shot **batch** reconstruction for directed-capture mode: a handful of deliberately-framed
 * keyframes (full-res color + ARCore pose + ARCore depth + intrinsics) are captured first, then this
 * replays them all at once into a single **point cloud**: DA3 dense depth → FastSAM object mask →
 * **per-frame** disparity→metric scale (each frame anchored to its own ARCore depth) → back-project
 * every masked pixel with its ARCore pose → merge across views (voxel dedup).
 *
 * Why per-frame scale, not one global scale: DA3 predicts only *relative*, per-image-normalized
 * disparity, so the disparity→metric mapping genuinely differs frame-to-frame (the same object at the
 * same distance can have very different raw disparity depending on what else is in view). Forcing one
 * global scale therefore mis-sizes most views, so even with a perfect SAM mask the per-view clouds
 * land at inconsistent depths and the merged shape is warped (or culled away by the ground/sphere
 * gates). Because each keyframe carries its own metric ARCore depth, we instead anchor every frame to
 * *itself*: one good frame already reconstructs its visible surface correctly, and extra frames add
 * coverage that actually aligns. (The live-streaming path locks ONE scale on purpose — that's to stop
 * a single TSDF grid from temporally smearing as the fit jitters; it doesn't apply here, where each
 * frame contributes independent world-space points.) A frame whose own object fit fails falls back to
 * a coherent global scale (the median *frame's* (s,t) pair, chosen jointly — not independent medians).
 *
 * Why a point cloud, not TSDF marching cubes: with only a handful of deliberately-framed shots, every
 * SAM-masked pixel becomes a 3D point, so the full silhouette the segmenter sees is captured directly
 * — TSDF needs many overlapping views to close a watertight surface and tends to drop most of the
 * object from sparse captures. ARCore depth is used *only* to anchor the per-frame scale (overall
 * size); all geometry is pure DA3 + ARCore pose.
 *
 * Heavy (DA3 is ~1 s/frame on CPU); run off the main thread.
 */
class BatchReconstructor(
    private val depth: DepthBackend?,
    private val seg: QnnSegmentationModel?,
    private val multiView: Da3MultiViewModel? = null,
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
        /** This frame's own object-anchored DA3 scale; null if its fit failed (use the global one). */
        val affine: AffineScaleSolver.Affine?,
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
        // Preferred path: native multi-view DA3 (one joint solve -> consistent geometry). Falls
        // through to the mono per-frame path below if the model is absent or it can't build.
        if (multiView != null && seg != null && frames.isNotEmpty()) {
            val mv = try {
                reconstructMultiView(frames, progress)
            } catch (t: Throwable) {
                Log.e(TAG, "BatchReconstructor: multi-view path failed; falling back to mono", t)
                MeshData.EMPTY
            }
            if (!mv.isEmpty) return mv
            Log.w(TAG, "BatchReconstructor: multi-view produced no mesh; trying mono fallback")
        }

        val da3 = depth
        if (da3 == null) {
            Log.w(TAG, "BatchReconstructor: no depth backend; cannot build")
            return MeshData.EMPTY
        }
        val total = frames.size
        if (total == 0) return MeshData.EMPTY

        // Pass 1: DA3 + SAM + a per-frame, object-anchored scale.
        val prepared = ArrayList<Prepared>(total)
        val objAffines = ArrayList<AffineScaleSolver.Affine>(total)
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
            // Anchor THIS frame's DA3 depth to ITS OWN ARCore depth over the object region. DA3
            // disparity is per-image normalized, so a per-frame fit is metrically correct for the
            // view; a single shared scale would mis-size most views and warp the merged cloud.
            val aff = AffineScaleSolver.fitObjectAffine(disp, f.depth, focus, mask)
                ?.takeIf { it.s > 0f && it.s.isFinite() && it.t.isFinite() }
            if (aff != null) objAffines.add(aff)
            prepared.add(
                Prepared(disp, mask, dw, dh, intr, Mat4.fromColumnMajor(f.poseColumnMajor), f.groundY, aff),
            )
        }
        if (prepared.isEmpty()) {
            Log.w(TAG, "BatchReconstructor: no usable frames")
            return MeshData.EMPTY
        }

        // Coherent global scale, used only as the fallback for frames whose own object fit failed.
        // It's the median *frame's* (s,t) pair chosen jointly (sort by s, take the middle pair) — NOT
        // median(s) paired with median(t) independently. s and t are correlated regression
        // coefficients; mixing medians from different frames yields a line that fits no frame at all.
        val globalScale: AffineScaleSolver.Affine? =
            if (objAffines.isEmpty()) null else objAffines.sortedBy { it.s }[objAffines.size / 2]
        if (prepared.all { it.affine == null } && globalScale == null) {
            Log.w(TAG, "BatchReconstructor: no usable scale on any frame")
            return MeshData.EMPTY
        }
        Log.i(TAG, "BatchReconstructor: per-frame scale on ${objAffines.size}/${prepared.size} frames; " +
            "global fallback s=${globalScale?.s} t=${globalScale?.t}")

        // Per-frame scale: this frame's own object fit, else the coherent global fallback.
        fun scaleFor(p: Prepared): AffineScaleSolver.Affine? = p.affine ?: globalScale

        // Locate the object (to reject stray background points) from the *median* of each frame's
        // masked centroid under its own scale — robust to a single off frame, unlike taking the first.
        val cxs = ArrayList<Float>(prepared.size)
        val cys = ArrayList<Float>(prepared.size)
        val czs = ArrayList<Float>(prepared.size)
        for (p in prepared) {
            val sc = scaleFor(p) ?: continue
            val md = AffineScaleSolver.applyAffine(p.disp, p.depthW, p.depthH, sc)
            val pc = maskedCentroidWorld(md, p.intr, p.mask, p.camToWorld) ?: continue
            cxs.add(pc[0]); cys.add(pc[1]); czs.add(pc[2])
        }
        if (cxs.isEmpty()) {
            Log.w(TAG, "BatchReconstructor: could not locate object centroid")
            return MeshData.EMPTY
        }
        val c = floatArrayOf(median(cxs), median(cys), median(czs))

        // Pass 2: back-project every SAM-masked pixel's DA3 depth into world space and merge across
        // views. This is the "take all the depth inside the SAM area" approach — every masked pixel
        // becomes a point, so the full silhouette is captured even from a handful of frames (vs. TSDF
        // marching cubes, which needs many views to close a surface). A voxel grid dedups overlap.
        val seen = HashSet<Long>(1 shl 16)
        val verts = ArrayList<Float>(1 shl 16)
        val vertKeys = ArrayList<Long>(1 shl 16) // voxel key per emitted point, for the density filter
        val r2 = OBJECT_RADIUS * OBJECT_RADIUS
        val wp = FloatArray(3)
        outer@ for ((i, p) in prepared.withIndex()) {
            progress?.update(i, total, "Fusing ${i + 1}/$total")
            val sc = scaleFor(p) ?: continue
            val md = AffineScaleSolver.applyAffine(p.disp, p.depthW, p.depthH, sc)
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
                val key = voxelKey(px, py, pz)
                if (seen.add(key)) {
                    verts.add(px); verts.add(py); verts.add(pz)
                    vertKeys.add(key)
                    if (verts.size / 3 >= MAX_POINTS) break@outer
                }
            }
        }

        if (verts.size / 3 < MIN_POINTS) {
            Log.w(TAG, "BatchReconstructor: too few points (${verts.size / 3})")
            return MeshData.EMPTY
        }

        // Density filter: drop isolated points \u2014 DA3 depth floaters and mask-edge bleed that the
        // per-frame metric scale doesn't catch. A real surface is locally dense at the 3 mm voxel
        // grid, so any point whose 26-voxel neighborhood holds fewer than [MIN_NEIGHBOR_VOXELS]
        // occupied cells is a stray and is removed; the object's silhouette edges keep enough
        // in-plane neighbors to survive. This is the cleanup that makes the cloud read as the object
        // rather than a noisy shell.
        val v = densityFilter(verts, vertKeys, seen)
        if (v.size / 3 < MIN_POINTS) {
            Log.w(TAG, "BatchReconstructor: too few points after density filter (${v.size / 3})")
            return MeshData.EMPTY
        }
        progress?.update(total, total, "Finishing\u2026")
        // Point cloud: no triangles. Up-normals keep the renderer's shader happy; the viewer colors
        // by height anyway.
        val normals = FloatArray(v.size) { if (it % 3 == 1) 1f else 0f }
        Log.i(TAG, "BatchReconstructor: point cloud ${v.size / 3} points")
        return MeshData(v, normals, IntArray(0))
    }

    /**
     * Native multi-view reconstruction: one [Da3MultiViewModel] forward over the model's fixed
     * number of selected keyframes yields per-view depth + DA3-estimated poses/intrinsics in **one
     * shared frame**, so
     * back-projecting them merges into a coherent object (no affine fit, no ARCore-pose stitching —
     * DA3 supplies the geometry). The SAM mask per view isolates the object; confidence + the shared
     * density filter clean it. Returns [MeshData.EMPTY] if the model output is unusable.
     *
     * Scale note: DA3 depth is up-to-scale in its own frame; the result is metrically *arbitrary*
     * (the viewer normalizes by extent). A future pass can fit one global factor vs ARCore depth.
     */
    private fun reconstructMultiView(frames: List<Keyframe>, progress: Progress?): MeshData {
        val mv = multiView ?: return MeshData.EMPTY
        @Suppress("NAME_SHADOWING") val seg = seg ?: return MeshData.EMPTY
        val n = mv.numViews
        val resW = mv.resW
        val resH = mv.resH
        progress?.update(0, 1, "Selecting views…")
        val views = selectViews(frames, n)

        progress?.update(0, 1, "Multi-view depth…")
        val result = mv.infer(views.map { it.argb }) ?: return MeshData.EMPTY

        // SAM mask per view, sampled into the same resW x resH letterboxed grid as the depth.
        progress?.update(0, 1, "Masking…")
        val masks = ArrayList<FloatArray>(n)
        for (i in 0 until n) {
            val g = result.geom[i]
            val content = seg.inferObjectMask(views[i].argb, g.newW, g.newH) // 1=object over the image
            masks.add(placeInRect(content, g, resW, resH))
        }

        // Back-project every masked, confident pixel of each view with that view's K + extrinsics.
        val seen = HashSet<Long>(1 shl 16)
        val verts = ArrayList<Float>(1 shl 16)
        val vertKeys = ArrayList<Long>(1 shl 16)
        val plane = resW * resH
        val xc = FloatArray(3)
        val xw = FloatArray(3)
        outer@ for (i in 0 until n) {
            progress?.update(i, n, "Fusing ${i + 1}/$n")
            val dBase = i * plane
            val mask = masks[i]
            // Reject this view's mask if it looks like a background grab rather than the object:
            // background-sized (covers most of the frame), empty, or off-center (e.g. SAM briefly
            // latched the monitor/desk). Such a frame's depth would smear background into the cloud,
            // so we skip fusing it entirely — the other views still carry the object.
            if (!maskIsObjectLike(mask, result.geom[i], resW)) {
                Log.i(TAG, "BatchReconstructor(MV): skipping view $i (mask not object-like)")
                continue
            }
            // Per-view confidence gate: drop the lowest [CONF_DROP_PCT] of masked pixels (edges/holes).
            val confThr = maskedConfPercentile(result.conf, dBase, mask, CONF_DROP_PCT)
            val fx = result.intrinsics[i * 9 + 0]
            val fy = result.intrinsics[i * 9 + 4]
            val cx = result.intrinsics[i * 9 + 2]
            val cy = result.intrinsics[i * 9 + 5]
            val e = i * 12 // 3x4 row-major world->cam [R|t]
            for (idx in 0 until plane) {
                if (mask[idx] < 0.5f) continue
                val z = result.depth[dBase + idx]
                if (z <= 0f || !z.isFinite()) continue
                if (result.conf[dBase + idx] < confThr) continue
                val u = idx % resW
                val v = idx / resW
                xc[0] = (u - cx) / fx * z
                xc[1] = (v - cy) / fy * z
                xc[2] = z
                worldFromCam(result.extrinsics, e, xc, xw) // Xw = R^T (Xc - t)
                val key = voxelKey(xw[0], xw[1], xw[2])
                if (seen.add(key)) {
                    verts.add(xw[0]); verts.add(xw[1]); verts.add(xw[2])
                    vertKeys.add(key)
                    if (verts.size / 3 >= MAX_POINTS) break@outer
                }
            }
        }
        if (verts.size / 3 < MIN_POINTS) {
            Log.w(TAG, "BatchReconstructor(MV): too few points (${verts.size / 3})")
            return MeshData.EMPTY
        }
        val v = densityFilter(verts, vertKeys, seen)
        if (v.size / 3 < MIN_POINTS) {
            Log.w(TAG, "BatchReconstructor(MV): too few points after density filter (${v.size / 3})")
            return MeshData.EMPTY
        }

        if (HULL_FILL) {
            // "Find a hull and fill it": trim stray background, then wrap the cloud in a watertight
            // convex hull. Robust to the partial-shell / layered-views problem (overlapping shells
            // collapse into one solid) and yields real triangles for OBJ export. Convex-only — fills
            // concavities, fine for cans/boxes/cases. Falls back to the raw cloud if the hull fails.
            progress?.update(1, 1, "Filling hull…")
            val core = trimToCentralCluster(v, HULL_TRIM_PCT)
            val hull = ConvexHull3D.compute(core)
            if (!hull.isEmpty) {
                Log.i(TAG, "BatchReconstructor(MV): hull ${hull.vertexCount} verts / ${hull.triangleCount} tris from $n views")
                // The renderer draws triangle soup (glDrawArrays, sequential order like marching
                // cubes); expand the indexed hull so its faces render and the OBJ stays valid.
                return expandToSoup(hull)
            }
            Log.w(TAG, "BatchReconstructor(MV): hull failed; returning point cloud")
        }

        progress?.update(1, 1, "Finishing…")
        val normals = FloatArray(v.size) { if (it % 3 == 1) 1f else 0f }
        Log.i(TAG, "BatchReconstructor(MV): point cloud ${v.size / 3} points from $n views")
        return MeshData(v, normals, IntArray(0))
    }

    /**
     * Convert an indexed [mesh] to triangle soup (3 fresh vertices per face, sequential indices) —
     * the order [com.lantern.recorder.rendering.MeshRenderer] draws with `glDrawArrays`. Vertex
     * normals are carried through per corner so the surface still shades smoothly.
     */
    private fun expandToSoup(mesh: MeshData): MeshData {
        val idx = mesh.indices
        if (idx.isEmpty()) return mesh
        val sv = mesh.vertices
        val sn = mesh.normals
        val hasN = sn.size == sv.size
        val v = FloatArray(idx.size * 3)
        val nrm = FloatArray(idx.size * 3)
        for (k in idx.indices) {
            val s = idx[k] * 3
            val d = k * 3
            v[d] = sv[s]; v[d + 1] = sv[s + 1]; v[d + 2] = sv[s + 2]
            if (hasN) { nrm[d] = sn[s]; nrm[d + 1] = sn[s + 1]; nrm[d + 2] = sn[s + 2] } else nrm[d + 1] = 1f
        }
        return MeshData(v, nrm, IntArray(idx.size) { it })
    }

    /** Drop points beyond the [pct]-percentile distance from the cloud median — removes stray
     *  background chunks that would balloon the convex hull. Returns the surviving flat-xyz points. */
    private fun trimToCentralCluster(v: FloatArray, pct: Int): FloatArray {
        val n = v.size / 3
        if (n < 8) return v
        var mx = 0f; var my = 0f; var mz = 0f
        // Median via per-axis sort (robust center; mean would chase outliers).
        val xs = FloatArray(n); val ys = FloatArray(n); val zs = FloatArray(n)
        for (i in 0 until n) { xs[i] = v[i * 3]; ys[i] = v[i * 3 + 1]; zs[i] = v[i * 3 + 2] }
        xs.sort(); ys.sort(); zs.sort()
        mx = xs[n / 2]; my = ys[n / 2]; mz = zs[n / 2]
        val d = FloatArray(n)
        for (i in 0 until n) {
            val dx = v[i * 3] - mx; val dy = v[i * 3 + 1] - my; val dz = v[i * 3 + 2] - mz
            d[i] = dx * dx + dy * dy + dz * dz
        }
        val sorted = d.clone(); sorted.sort()
        val thr = sorted[((n * pct / 100).coerceIn(0, n - 1))]
        val out = ArrayList<Float>(n * 3)
        for (i in 0 until n) if (d[i] <= thr) { out.add(v[i * 3]); out.add(v[i * 3 + 1]); out.add(v[i * 3 + 2]) }
        return out.toFloatArray()
    }

    /** Pick exactly [n] keyframes: even spread if more were captured, cyclic pad if fewer. */
    private fun selectViews(frames: List<Keyframe>, n: Int): List<Keyframe> {
        if (frames.size == n) return frames
        val out = ArrayList<Keyframe>(n)
        if (frames.size > n) {
            for (i in 0 until n) {
                val idx = Math.round(i.toFloat() * (frames.size - 1) / (n - 1)).coerceIn(0, frames.size - 1)
                out.add(frames[idx])
            }
        } else {
            for (i in 0 until n) out.add(frames[i % frames.size]) // pad by repetition
        }
        return out
    }

    /** Place a [content]-sized (newW x newH) mask into the padded [resW]x[resH] grid at the offset. */
    private fun placeInRect(content: FloatArray?, g: Da3MultiViewModel.ViewGeom, resW: Int, resH: Int): FloatArray {
        val out = FloatArray(resW * resH)
        if (content == null) return out
        for (oy in 0 until g.newH) {
            val dstRow = (g.padY + oy) * resW + g.padX
            val srcRow = oy * g.newW
            for (ox in 0 until g.newW) out[dstRow + ox] = content[srcRow + ox]
        }
        return out
    }

    /** Xw = R^T (Xc - t) for a 3x4 row-major world->cam `[R|t]` at offset [e] in [ext]. */
    private fun worldFromCam(ext: FloatArray, e: Int, xc: FloatArray, out: FloatArray) {
        val dx = xc[0] - ext[e + 3]
        val dy = xc[1] - ext[e + 7]
        val dz = xc[2] - ext[e + 11]
        // R rows are (e0,e1,e2),(e4,e5,e6),(e8,e9,e10); R^T multiply = columns dotted with d.
        out[0] = ext[e + 0] * dx + ext[e + 4] * dy + ext[e + 8] * dz
        out[1] = ext[e + 1] * dx + ext[e + 5] * dy + ext[e + 9] * dz
        out[2] = ext[e + 2] * dx + ext[e + 6] * dy + ext[e + 10] * dz
    }

    /**
     * Whether [mask] (in the res x res letterboxed grid) reads as the centered object rather than a
     * background grab: enough but not too-many object pixels, and a centroid near the content center.
     * Guards against SAM briefly latching the monitor/desk in one frame (the leak that smears the
     * cloud). [geom] gives the content box so coverage is measured against the image, not the padding.
     */
    private fun maskIsObjectLike(mask: FloatArray, geom: Da3MultiViewModel.ViewGeom, resW: Int): Boolean {
        var on = 0
        var sx = 0.0
        var sy = 0.0
        for (idx in mask.indices) {
            if (mask[idx] < 0.5f) continue
            on++
            sx += idx % resW
            sy += idx / resW
        }
        val area = (geom.newW * geom.newH).coerceAtLeast(1)
        val cov = on.toFloat() / area
        if (on < MV_MIN_MASK_PIX || cov > MV_MAX_MASK_FRAC) return false
        // Centroid must sit inside the central band of the content box (object is framed centrally).
        val cx = (sx / on - geom.padX) / geom.newW
        val cy = (sy / on - geom.padY) / geom.newH
        return cx in MV_CENTER_LO..MV_CENTER_HI && cy in MV_CENTER_LO..MV_CENTER_HI
    }

    /** The [pct]-percentile confidence over masked pixels of view at [dBase] (the drop threshold). */
    private fun maskedConfPercentile(conf: FloatArray, dBase: Int, mask: FloatArray, pct: Int): Float {
        val vals = ArrayList<Float>(4096)
        for (idx in mask.indices) if (mask[idx] >= 0.5f) vals.add(conf[dBase + idx])
        if (vals.size < 16) return Float.NEGATIVE_INFINITY // too few to gate; keep all
        vals.sort()
        return vals[(vals.size * pct / 100).coerceIn(0, vals.size - 1)]
    }

    /** Pack a world point into a voxel-grid key (~[VOXEL_M] m cells) for dedup across views. */
    private fun voxelKey(x: Float, y: Float, z: Float): Long {
        val inv = 1f / VOXEL_M
        val ix = Math.round(x * inv) + 524288L // +2^19 bias keeps coords non-negative within ±5 km
        val iy = Math.round(y * inv) + 524288L
        val iz = Math.round(z * inv) + 524288L
        return packVoxel(ix, iy, iz)
    }

    /** Pack already-biased integer voxel coords into the 20-bits-per-axis key (matches [voxelKey]). */
    private fun packVoxel(ix: Long, iy: Long, iz: Long): Long =
        (ix and 0xFFFFF) or ((iy and 0xFFFFF) shl 20) or ((iz and 0xFFFFF) shl 40)

    /**
     * Remove isolated points (DA3 floaters / mask-edge bleed): keep a point only if its 26-voxel
     * neighborhood contains at least [MIN_NEIGHBOR_VOXELS] other occupied cells. [occupied] is the
     * dedup set built during back-projection, so this is a pure lookup — no spatial structure to
     * rebuild. Returns the surviving points as a flat xyz array.
     */
    private fun densityFilter(verts: ArrayList<Float>, keys: ArrayList<Long>, occupied: HashSet<Long>): FloatArray {
        val out = ArrayList<Float>(verts.size)
        for (k in keys.indices) {
            val key = keys[k]
            val ix = key and 0xFFFFF
            val iy = (key ushr 20) and 0xFFFFF
            val iz = (key ushr 40) and 0xFFFFF
            var neighbors = 0
            loop@ for (dz in -1L..1L) for (dy in -1L..1L) for (dx in -1L..1L) {
                if (dx == 0L && dy == 0L && dz == 0L) continue
                if (occupied.contains(packVoxel(ix + dx, iy + dy, iz + dz))) {
                    if (++neighbors >= MIN_NEIGHBOR_VOXELS) break@loop
                }
            }
            if (neighbors >= MIN_NEIGHBOR_VOXELS) {
                val b = k * 3
                out.add(verts[b]); out.add(verts[b + 1]); out.add(verts[b + 2])
            }
        }
        return out.toFloatArray()
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

        /** Multi-view: drop the lowest this-percent of masked pixels by DA3 confidence per view. */
        private const val CONF_DROP_PCT = 20

        // Convex-hull "fill" for the multi-view cloud: wraps the points in a watertight solid.
        // Robust to partial/layered shells; convex-only (fills concavities). Toggle off for raw cloud.
        private const val HULL_FILL = true
        // Keep points within this distance percentile of the cloud median before hulling (drops
        // background floaters that would otherwise balloon the hull).
        private const val HULL_TRIM_PCT = 92

        // Multi-view per-view mask sanity gate (reject background grabs / off-center latches):
        private const val MV_MIN_MASK_PIX = 400          // too few object pixels => empty/failed mask
        private const val MV_MAX_MASK_FRAC = 0.65f       // covers most of the frame => background, not object
        private const val MV_CENTER_LO = 0.12f           // centroid must lie within the central
        private const val MV_CENTER_HI = 0.88f           // [12%,88%] band of the content box

        /** Density filter: min occupied cells in a point's 26-voxel neighborhood to keep it (else a
         *  stray floater). 2 kills singletons and isolated pairs while sparing silhouette edges. */
        private const val MIN_NEIGHBOR_VOXELS = 2

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
