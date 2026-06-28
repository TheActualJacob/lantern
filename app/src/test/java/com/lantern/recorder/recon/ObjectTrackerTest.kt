package com.lantern.recorder.recon

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random
import org.junit.Test

/**
 * Off-device correctness checks for the in-hand [ObjectTracker]'s frame-to-model ICP. We synthesise
 * a rigid object cloud, seed the model with it (defining object frame O = camera 0), then render it
 * from a *known* second camera pose and assert the tracker recovers that pose. This validates the
 * registration math (Horn quaternion solve + ICP) without a device or ground-truth capture.
 */
class ObjectTrackerTest {

    /**
     * Points on the *surface* of an asymmetric ellipsoid (semi-axes 10x7x5 cm), centred ~0.5 m
     * ahead. Unequal axes make orientation observable, and the everywhere-curved surface is
     * well-conditioned for point-to-point ICP in translation too (no flat faces to slide along —
     * unlike a box, where tangential translation is unconstrained).
     */
    private fun objectCloud(n: Int, seed: Int): FloatArray {
        val rng = Random(seed)
        val ax = 0.10f; val ay = 0.07f; val az = 0.05f
        val cz = 0.5f
        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            // Uniform-ish direction on the unit sphere, then scale by the semi-axes.
            var dx = rng.nextFloat() * 2f - 1f
            var dy = rng.nextFloat() * 2f - 1f
            var dz = rng.nextFloat() * 2f - 1f
            var len = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
            if (len < 1e-4f) { dx = 1f; dy = 0f; dz = 0f; len = 1f }
            dx /= len; dy /= len; dz /= len
            out[i * 3] = dx * ax
            out[i * 3 + 1] = dy * ay
            out[i * 3 + 2] = cz + dz * az
        }
        return out
    }

    /** Row-major rigid transform from a Y-axis rotation (rad) + translation. */
    private fun rigidY(angle: Float, tx: Float, ty: Float, tz: Float): FloatArray {
        val c = cos(angle); val s = sin(angle)
        return floatArrayOf(
            c, 0f, s, tx,
            0f, 1f, 0f, ty,
            -s, 0f, c, tz,
            0f, 0f, 0f, 1f,
        )
    }

    /** Apply a row-major 4x4 to every xyz triple. */
    private fun apply(m: FloatArray, pts: FloatArray): FloatArray {
        val out = FloatArray(pts.size)
        val p = FloatArray(3)
        var i = 0
        while (i + 2 < pts.size) {
            Mat4.transformPoint(m, pts[i], pts[i + 1], pts[i + 2], p)
            out[i] = p[0]; out[i + 1] = p[1]; out[i + 2] = p[2]
            i += 3
        }
        return out
    }

    private fun rotationAngleDeg(err: FloatArray): Double {
        val cosA = ((err[0] + err[5] + err[10]) - 1f) / 2f
        return Math.toDegrees(kotlin.math.acos(cosA.coerceIn(-1f, 1f)).toDouble())
    }

    private fun assertRecovers(T_OC_true: FloatArray, recovered: FloatArray, rotTolDeg: Double, transTol: Float) {
        // err = recovered * inv(true) should be identity.
        val err = Mat4.multiply(recovered, Mat4.invertRigid(T_OC_true))
        val rotDeg = rotationAngleDeg(err)
        val t = kotlin.math.sqrt(err[3] * err[3] + err[7] * err[7] + err[11] * err[11])
        assertTrue("rotation error too large ($rotDeg deg)", rotDeg < rotTolDeg)
        assertTrue("translation error too large ($t)", t < transTol)
    }

    @Test
    fun rigidSolverRecoversExactTransform() {
        // Isolates Horn's quaternion solve from ICP/nearest-neighbour: feed exact correspondences
        // Q = T*P and check it recovers T (rotation + translation) to float precision.
        val tracker = ObjectTracker()
        val p = objectCloud(500, seed = 3)
        val T = rigidY(angle = 0.5f, tx = 0.04f, ty = 0.02f, tz = -0.03f)
        val q = apply(T, p)
        val solved = tracker.rigidFromCorrespondences(p.toList(), q.toList())!!
        assertRecovers(T, solved, rotTolDeg = 1e-2, transTol = 1e-4f)
    }

    @Test
    fun recoversKnownRotationAndTranslation() {
        val tracker = ObjectTracker(voxelSize = 0.004f)
        val cloud0 = objectCloud(2000, seed = 1)

        // Frame 0 seeds the model; O == camera 0, so T_OC0 = identity.
        val pose0 = tracker.track(cloud0, 2000)
        assertRecovers(Mat4.identity(), pose0, rotTolDeg = 1e-3, transTol = 1e-4f)

        // A known camera->object pose for frame 1 (object frame == camera 0 frame). Kept within
        // ICP's convergence basin (small inter-frame motion, like a 2 cm keyframe gate on device).
        val T_OC1 = rigidY(angle = 0.08f, tx = 0.015f, ty = -0.008f, tz = 0.01f)
        // The cloud as seen by camera 1 is the object points pushed into that camera: inv(T_OC1)*X_O.
        val cloud1 = apply(Mat4.invertRigid(T_OC1), cloud0)

        val pose1 = tracker.track(cloud1, 2000)
        assertRecovers(T_OC1, pose1, rotTolDeg = 1.0, transTol = 0.005f)
    }

    @Test
    fun tracksIncrementalRotationSequence() {
        val tracker = ObjectTracker(voxelSize = 0.004f)
        val cloud0 = objectCloud(2500, seed = 7)
        tracker.track(cloud0, 2500)

        // March the object pose in small steps (like turning it in hand a few degrees per keyframe).
        var angle = 0f
        repeat(8) {
            angle += 0.10f // ~5.7 deg per step
            val T_OC = rigidY(angle, tx = 0.01f * it, ty = 0f, tz = 0f)
            val cloud = apply(Mat4.invertRigid(T_OC), cloud0)
            val recovered = tracker.track(cloud, 2500)
            assertRecovers(T_OC, recovered, rotTolDeg = 2.0, transTol = 0.01f)
        }
        assertEquals(9, tracker.trackedFrames)
    }
}
