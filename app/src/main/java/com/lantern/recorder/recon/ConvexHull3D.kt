package com.lantern.recorder.recon

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Watertight 3D **convex hull** of a point cloud — the "find a hull and fill it" path for small
 * **convex** objects (cans, boxes, cases) captured from only one or two views.
 *
 * Why this matters here: the multi-view DA3 cloud is a thin, partial shell (you only see the front
 * of the object) and, when views don't register perfectly, it splits into overlapping layers. The
 * convex hull wraps *all* the points into a single solid, so it (a) gives a filled, watertight mesh
 * with real triangles instead of a sparse/holey point cloud, and (b) is inherently robust to the
 * layering — overlapping shells collapse into one outer surface. The trade-off: it fills concavities,
 * so it's only faithful for roughly-convex objects (a mug handle or an L-shape would be bridged over).
 *
 * Algorithm: incremental hull. Build a seed tetrahedron from extreme points, then add each remaining
 * point — find the faces it can "see" (is outside of), remove them, and stitch new faces from the
 * horizon edges to the point. Faces are oriented outward against a fixed interior reference point, so
 * no winding bookkeeping is needed. Dense clouds are strided down first (the hull only needs extreme
 * points, so this is exact for the shape and keeps it fast).
 */
object ConvexHull3D {

    /** Max input points fed to the hull; denser clouds are strided down (hull is unchanged). */
    private const val MAX_INPUT = 3000

    /** A hull face: indices into the working point array, plus its outward unit normal + plane d. */
    private class Face(val a: Int, val b: Int, val c: Int, val nx: Float, val ny: Float, val nz: Float, val d: Float)

    /**
     * Compute the convex hull of [points] (flat xyz) as a triangle [MeshData] with per-vertex
     * normals. Returns [MeshData.EMPTY] if there are fewer than 4 points or they're degenerate
     * (all collinear/coplanar) so no volume exists.
     */
    fun compute(points: FloatArray): MeshData {
        val p = stride(points, MAX_INPUT)
        val n = p.size / 3
        if (n < 4) return MeshData.EMPTY

        val seed = initialTetra(p, n) ?: return MeshData.EMPTY
        // Fixed interior reference: the seed tetra's centroid (stays interior as the hull only grows).
        val ix = (p[seed[0] * 3] + p[seed[1] * 3] + p[seed[2] * 3] + p[seed[3] * 3]) * 0.25f
        val iy = (p[seed[0] * 3 + 1] + p[seed[1] * 3 + 1] + p[seed[2] * 3 + 1] + p[seed[3] * 3 + 1]) * 0.25f
        val iz = (p[seed[0] * 3 + 2] + p[seed[1] * 3 + 2] + p[seed[2] * 3 + 2] + p[seed[3] * 3 + 2]) * 0.25f
        val interior = floatArrayOf(ix, iy, iz)

        var faces = arrayListOf(
            makeFace(p, seed[0], seed[1], seed[2], interior),
            makeFace(p, seed[0], seed[1], seed[3], interior),
            makeFace(p, seed[0], seed[2], seed[3], interior),
            makeFace(p, seed[1], seed[2], seed[3], interior),
        )

        // Scale-relative epsilon so visibility tests are stable across object sizes.
        val eps = hullEps(p, n)
        val inSeed = seed.toHashSet()
        for (i in 0 until n) {
            if (i in inSeed) continue
            val px = p[i * 3]; val py = p[i * 3 + 1]; val pz = p[i * 3 + 2]
            val visible = ArrayList<Face>()
            for (f in faces) {
                // Signed distance to the face plane; > eps means the point is outside this face.
                if (f.nx * px + f.ny * py + f.nz * pz - f.d > eps) visible.add(f)
            }
            if (visible.isEmpty()) continue // inside the current hull

            // Horizon = undirected edges that belong to exactly one visible face (the boundary
            // between the visible cap being removed and the faces that stay).
            val edgeCount = HashMap<Long, Int>(visible.size * 3)
            for (f in visible) {
                bump(edgeCount, f.a, f.b); bump(edgeCount, f.b, f.c); bump(edgeCount, f.c, f.a)
            }
            val keep = ArrayList<Face>(faces.size)
            val visibleSet = visible.toHashSet()
            for (f in faces) if (f !in visibleSet) keep.add(f)
            // Stitch new faces from each horizon edge to the new point.
            for ((key, count) in edgeCount) {
                if (count != 1) continue
                val u = (key ushr 32).toInt()
                val v = (key and 0xFFFFFFFFL).toInt()
                keep.add(makeFace(p, u, v, i, interior))
            }
            faces = keep
        }

        return toMesh(p, faces)
    }

    /** Stride [points] (flat xyz) down to at most [max] points; returns the original if small enough. */
    private fun stride(points: FloatArray, max: Int): FloatArray {
        val n = points.size / 3
        if (n <= max) return points
        val step = (n + max - 1) / max
        val outN = (n + step - 1) / step
        val out = FloatArray(outN * 3)
        var o = 0
        var i = 0
        while (i < n) {
            out[o * 3] = points[i * 3]
            out[o * 3 + 1] = points[i * 3 + 1]
            out[o * 3 + 2] = points[i * 3 + 2]
            o++
            i += step
        }
        return out.copyOf(o * 3)
    }

    /** Four affinely-independent extreme points for the seed tetra, or null if degenerate. */
    private fun initialTetra(p: FloatArray, n: Int): IntArray? {
        // p0: min-x extreme. p1: farthest from p0. p2: farthest from line p0p1. p3: farthest from plane.
        var p0 = 0
        for (i in 1 until n) if (p[i * 3] < p[p0 * 3]) p0 = i
        val p1 = farthestFromPoint(p, n, p0)
        if (p1 < 0) return null
        val p2 = farthestFromLine(p, n, p0, p1)
        if (p2 < 0) return null
        val p3 = farthestFromPlane(p, n, p0, p1, p2)
        if (p3 < 0) return null
        return intArrayOf(p0, p1, p2, p3)
    }

    private fun farthestFromPoint(p: FloatArray, n: Int, a: Int): Int {
        var best = -1; var bestD = 0f
        for (i in 0 until n) {
            if (i == a) continue
            val dx = p[i * 3] - p[a * 3]; val dy = p[i * 3 + 1] - p[a * 3 + 1]; val dz = p[i * 3 + 2] - p[a * 3 + 2]
            val d = dx * dx + dy * dy + dz * dz
            if (d > bestD) { bestD = d; best = i }
        }
        return if (bestD > 1e-12f) best else -1
    }

    private fun farthestFromLine(p: FloatArray, n: Int, a: Int, b: Int): Int {
        val ex = p[b * 3] - p[a * 3]; val ey = p[b * 3 + 1] - p[a * 3 + 1]; val ez = p[b * 3 + 2] - p[a * 3 + 2]
        val elen = sqrt(ex * ex + ey * ey + ez * ez)
        if (elen < 1e-9f) return -1
        val ux = ex / elen; val uy = ey / elen; val uz = ez / elen
        var best = -1; var bestD = 0f
        for (i in 0 until n) {
            val ax = p[i * 3] - p[a * 3]; val ay = p[i * 3 + 1] - p[a * 3 + 1]; val az = p[i * 3 + 2] - p[a * 3 + 2]
            val t = ax * ux + ay * uy + az * uz
            val cx = ax - t * ux; val cy = ay - t * uy; val cz = az - t * uz
            val d = cx * cx + cy * cy + cz * cz
            if (d > bestD) { bestD = d; best = i }
        }
        return if (bestD > 1e-12f) best else -1
    }

    private fun farthestFromPlane(p: FloatArray, n: Int, a: Int, b: Int, c: Int): Int {
        val nrm = triNormal(p, a, b, c)
        val len = sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2])
        if (len < 1e-12f) return -1
        var best = -1; var bestD = 0f
        for (i in 0 until n) {
            val dx = p[i * 3] - p[a * 3]; val dy = p[i * 3 + 1] - p[a * 3 + 1]; val dz = p[i * 3 + 2] - p[a * 3 + 2]
            val dist = abs((dx * nrm[0] + dy * nrm[1] + dz * nrm[2]) / len)
            if (dist > bestD) { bestD = dist; best = i }
        }
        return if (bestD > 1e-9f) best else -1
    }

    /** Build a face (a,b,c) with its normal oriented to point away from [interior]. */
    private fun makeFace(p: FloatArray, a: Int, b: Int, c: Int, interior: FloatArray): Face {
        val nrm = triNormal(p, a, b, c)
        val len = sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).coerceAtLeast(1e-20f)
        var nx = nrm[0] / len; var ny = nrm[1] / len; var nz = nrm[2] / len
        var d = nx * p[a * 3] + ny * p[a * 3 + 1] + nz * p[a * 3 + 2]
        // Flip so the interior point is behind the plane (n·interior - d < 0 => normal points outward).
        if (nx * interior[0] + ny * interior[1] + nz * interior[2] - d > 0f) {
            nx = -nx; ny = -ny; nz = -nz; d = -d
        }
        return Face(a, b, c, nx, ny, nz, d)
    }

    private fun triNormal(p: FloatArray, a: Int, b: Int, c: Int): FloatArray {
        val ux = p[b * 3] - p[a * 3]; val uy = p[b * 3 + 1] - p[a * 3 + 1]; val uz = p[b * 3 + 2] - p[a * 3 + 2]
        val vx = p[c * 3] - p[a * 3]; val vy = p[c * 3 + 1] - p[a * 3 + 1]; val vz = p[c * 3 + 2] - p[a * 3 + 2]
        return floatArrayOf(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    }

    /** Scale-relative visibility epsilon: a tiny fraction of the cloud's bounding-box diagonal. */
    private fun hullEps(p: FloatArray, n: Int): Float {
        var minX = p[0]; var minY = p[1]; var minZ = p[2]; var maxX = p[0]; var maxY = p[1]; var maxZ = p[2]
        for (i in 1 until n) {
            val x = p[i * 3]; val y = p[i * 3 + 1]; val z = p[i * 3 + 2]
            if (x < minX) minX = x; if (x > maxX) maxX = x
            if (y < minY) minY = y; if (y > maxY) maxY = y
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
        }
        val dx = maxX - minX; val dy = maxY - minY; val dz = maxZ - minZ
        return sqrt(dx * dx + dy * dy + dz * dz) * 1e-5f + 1e-9f
    }

    /** Undirected edge key (min<<32 | max). */
    private fun bump(m: HashMap<Long, Int>, a: Int, b: Int) {
        val lo = minOf(a, b); val hi = maxOf(a, b)
        val key = (lo.toLong() shl 32) or hi.toLong()
        m[key] = (m[key] ?: 0) + 1
    }

    /** Emit only the vertices actually used by [faces], with per-vertex (area-weighted) normals. */
    private fun toMesh(p: FloatArray, faces: List<Face>): MeshData {
        if (faces.isEmpty()) return MeshData.EMPTY
        val remap = HashMap<Int, Int>()
        val verts = ArrayList<Float>()
        fun vi(idx: Int): Int = remap.getOrPut(idx) {
            verts.add(p[idx * 3]); verts.add(p[idx * 3 + 1]); verts.add(p[idx * 3 + 2])
            (verts.size / 3) - 1
        }
        val tris = IntArray(faces.size * 3)
        var t = 0
        for (f in faces) {
            tris[t++] = vi(f.a); tris[t++] = vi(f.b); tris[t++] = vi(f.c)
        }
        val v = verts.toFloatArray()
        val normals = FloatArray(v.size)
        var i = 0
        while (i < tris.size) {
            val a = tris[i]; val b = tris[i + 1]; val c = tris[i + 2]
            val nrm = triNormal(v, a, b, c) // area-weighted (un-normalized cross)
            for (idx in intArrayOf(a, b, c)) {
                normals[idx * 3] += nrm[0]; normals[idx * 3 + 1] += nrm[1]; normals[idx * 3 + 2] += nrm[2]
            }
            i += 3
        }
        var j = 0
        while (j < normals.size) {
            val len = sqrt(normals[j] * normals[j] + normals[j + 1] * normals[j + 1] + normals[j + 2] * normals[j + 2])
            if (len > 1e-12f) { normals[j] /= len; normals[j + 1] /= len; normals[j + 2] /= len } else { normals[j + 1] = 1f }
            j += 3
        }
        return MeshData(v, normals, tris)
    }
}
