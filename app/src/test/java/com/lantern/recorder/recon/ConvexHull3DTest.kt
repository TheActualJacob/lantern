package com.lantern.recorder.recon

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Verifies [ConvexHull3D] produces a correct watertight hull: it wraps all input points (every
 * point is inside-or-on the result), the surface is closed (each edge shared by exactly two
 * triangles), and interior/noise points don't change the hull.
 */
class ConvexHull3DTest {

    private fun cubeCorners(s: Float = 1f): FloatArray = floatArrayOf(
        0f, 0f, 0f, s, 0f, 0f, 0f, s, 0f, s, s, 0f,
        0f, 0f, s, s, 0f, s, 0f, s, s, s, s, s,
    )

    @Test
    fun emptyBelowFourPoints() {
        assertTrue(ConvexHull3D.compute(floatArrayOf(0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f)).isEmpty)
    }

    @Test
    fun degeneratePlanarIsEmpty() {
        // All points coplanar (z=0) => no volume => empty hull.
        val planar = floatArrayOf(0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f, 1f, 1f, 0f, 0.5f, 0.5f, 0f)
        assertTrue(ConvexHull3D.compute(planar).isEmpty)
    }

    @Test
    fun cubeHullIsClosedAndWrapsAllPoints() {
        // Cube corners + a dense jitter of interior/surface points the hull must still contain.
        val pts = ArrayList<Float>()
        cubeCorners().forEach { pts.add(it) }
        val rnd = java.util.Random(7)
        repeat(300) {
            pts.add(rnd.nextFloat()); pts.add(rnd.nextFloat()); pts.add(rnd.nextFloat())
        }
        val mesh = ConvexHull3D.compute(pts.toFloatArray())
        assertTrue("hull should have triangles", mesh.triangleCount > 0)

        // Closed surface: every undirected edge is shared by exactly two triangles.
        val edges = HashMap<Long, Int>()
        val idx = mesh.indices
        var i = 0
        while (i < idx.size) {
            for (e in arrayOf(idx[i] to idx[i + 1], idx[i + 1] to idx[i + 2], idx[i + 2] to idx[i])) {
                val lo = minOf(e.first, e.second).toLong(); val hi = maxOf(e.first, e.second).toLong()
                val k = (lo shl 32) or hi
                edges[k] = (edges[k] ?: 0) + 1
            }
            i += 3
        }
        assertTrue("surface must be closed (all edges shared by 2 tris)", edges.values.all { it == 2 })

        // Every input point lies inside or on the hull (within tolerance): build outward planes and
        // check no point is far outside any face.
        val v = mesh.vertices
        val cx = (0..7).sumOf { v[idx[it] * 3].toDouble() } // rough interior ~ cube center 0.5
        val interior = floatArrayOf(0.5f, 0.5f, 0.5f)
        var maxOutside = 0f
        var t = 0
        while (t < idx.size) {
            val a = idx[t]; val b = idx[t + 1]; val c = idx[t + 2]
            val ux = v[b * 3] - v[a * 3]; val uy = v[b * 3 + 1] - v[a * 3 + 1]; val uz = v[b * 3 + 2] - v[a * 3 + 2]
            val vx = v[c * 3] - v[a * 3]; val vy = v[c * 3 + 1] - v[a * 3 + 1]; val vz = v[c * 3 + 2] - v[a * 3 + 2]
            var nx = uy * vz - uz * vy; var ny = uz * vx - ux * vz; var nz = ux * vy - uy * vx
            val len = sqrt(nx * nx + ny * ny + nz * nz)
            if (len < 1e-9f) { t += 3; continue }
            nx /= len; ny /= len; nz /= len
            var d = nx * v[a * 3] + ny * v[a * 3 + 1] + nz * v[a * 3 + 2]
            if (nx * interior[0] + ny * interior[1] + nz * interior[2] - d > 0f) { nx = -nx; ny = -ny; nz = -nz; d = -d }
            var k = 0
            while (k < pts.size) {
                val s = nx * pts[k] + ny * pts[k + 1] + nz * pts[k + 2] - d
                if (s > maxOutside) maxOutside = s
                k += 3
            }
            t += 3
        }
        assertTrue("all points within hull (maxOutside=$maxOutside)", maxOutside < 1e-3f)
    }

    @Test
    fun hullVerticesAreCubeCornersOnly() {
        // Interior noise must not add hull vertices: a cube's hull has exactly its 8 corners.
        val pts = ArrayList<Float>()
        cubeCorners(2f).forEach { pts.add(it) }
        val rnd = java.util.Random(3)
        repeat(500) { pts.add(0.2f + rnd.nextFloat() * 1.6f); pts.add(0.2f + rnd.nextFloat() * 1.6f); pts.add(0.2f + rnd.nextFloat() * 1.6f) }
        val mesh = ConvexHull3D.compute(pts.toFloatArray())
        assertEquals("cube hull has 8 vertices", 8, mesh.vertexCount)
        // 8 verts, closed triangulated convex polyhedron => 2V-4 = 12 triangles (Euler).
        assertEquals("cube hull has 12 triangles", 12, mesh.triangleCount)
    }
}
