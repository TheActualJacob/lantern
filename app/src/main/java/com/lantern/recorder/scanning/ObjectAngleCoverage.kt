package com.lantern.recorder.scanning

import kotlin.math.atan2
import kotlin.math.roundToInt

/**
 * Object-angle coverage for turntable capture (Object-Centric Capture plan, **T3**).
 *
 * In orbit mode, coverage tracks **camera azimuth** around a fixed object — so
 * rotating the object does nothing. In turntable mode the object spins about the
 * board's Z axis, so coverage must instead track **how much of the turntable
 * rotation has been observed**. This class derives the object's yaw about the
 * board axis from each keyframe's `T_CO` (camera <- object) and lights an angular
 * ring of [sectors] wedges by *object angle*.
 *
 * The maths mirror the host `charuco_pose.relative_object_yaw_deg`: with a
 * (mostly) fixed camera, `T_CO(t) = T_CO(0) · Rz(theta)`, so the relative rotation
 * `R_ref^T · R(t)` is a pure Z rotation whose angle is the turntable azimuth
 * advanced since the first keyframe. The existing dome/ring UI carries over
 * unchanged — only the angle source swaps (plan §3.4).
 *
 * Pure Kotlin (no Android / ARCore types) so it is unit-testable and shareable
 * between the recorder and any live-preview surface.
 *
 * Matrices are **row-major 4x4** float arrays in the OpenCV `solvePnP` convention,
 * matching [com.lantern.recorder.recording.ObjectPose].
 */
class ObjectAngleCoverage(val sectors: Int = DEFAULT_SECTORS) {

    /** Reference rotation `R_ref` (3x3, row-major 9) from the first valid keyframe. */
    private var refRotation: FloatArray? = null

    /** Bitmask of covered object-yaw sectors. */
    var sectorMask: Int = 0
        private set

    /** Most recent object yaw about the board Z axis, in degrees [0, 360). */
    var lastYawDeg: Float = 0f
        private set

    /** Number of distinct yaw sectors observed so far. */
    val coveredSectors: Int get() = Integer.bitCount(sectorMask)

    /** Coverage as a percentage of the full turntable rotation (0..100). */
    val coveragePercent: Int get() = coveredSectors * 100 / sectors

    /** Most recent sector index, or -1 before any valid pose. */
    var currentSector: Int = -1
        private set

    /** Clears all coverage (between sessions / passes). */
    fun reset() {
        refRotation = null
        sectorMask = 0
        lastYawDeg = 0f
        currentSector = -1
    }

    /**
     * Folds one keyframe's object pose into the coverage ring.
     *
     * @param tCO row-major 4x4 `T_CO` (camera <- object) for this keyframe.
     * @return the relative object yaw (degrees, [0,360)) for this keyframe.
     */
    fun onObjectPose(tCO: FloatArray): Float {
        require(tCO.size == 16) { "tCO must be a 16-element row-major 4x4, got ${tCO.size}" }
        val rot = rotationOf(tCO)
        val ref = refRotation
        if (ref == null) {
            refRotation = rot
            lastYawDeg = 0f
            return creditYaw(0f)
        }
        val relative = multiply3x3(transpose3x3(ref), rot)
        val yaw = normalizeDeg(yawAboutZDeg(relative))
        lastYawDeg = yaw
        return creditYaw(yaw)
    }

    /** Lights the sector for [yawDeg] and returns it. */
    private fun creditYaw(yawDeg: Float): Float {
        val sector = ((yawDeg / 360f) * sectors).toInt().coerceIn(0, sectors - 1)
        currentSector = sector
        sectorMask = sectorMask or (1 shl sector)
        return yawDeg
    }

    companion object {
        /** Azimuth wedges around the turntable (matches the orbit ring's 12 at 30° each). */
        const val DEFAULT_SECTORS = 12

        /** Extracts the 3x3 rotation (row-major 9) from a row-major 4x4. */
        fun rotationOf(m4: FloatArray): FloatArray = floatArrayOf(
            m4[0], m4[1], m4[2],
            m4[4], m4[5], m4[6],
            m4[8], m4[9], m4[10],
        )

        /** Transpose of a 3x3 (row-major 9). */
        fun transpose3x3(m: FloatArray): FloatArray = floatArrayOf(
            m[0], m[3], m[6],
            m[1], m[4], m[7],
            m[2], m[5], m[8],
        )

        /** Product of two 3x3 matrices (row-major 9). */
        fun multiply3x3(a: FloatArray, b: FloatArray): FloatArray {
            val out = FloatArray(9)
            for (r in 0 until 3) {
                for (c in 0 until 3) {
                    var sum = 0f
                    for (k in 0 until 3) sum += a[r * 3 + k] * b[k * 3 + c]
                    out[r * 3 + c] = sum
                }
            }
            return out
        }

        /**
         * Rotation angle about +Z (degrees) read from a 3x3 (row-major 9). Uses
         * the upper-left 2x2 block so a small handheld off-axis wobble doesn't
         * corrupt the turntable yaw (mirrors `charuco_pose.yaw_about_z_deg`).
         */
        fun yawAboutZDeg(r: FloatArray): Float {
            // r[3]=r10, r[1]=r01, r[0]=r00, r[4]=r11
            val angle = atan2((r[3] - r[1]).toDouble(), (r[0] + r[4]).toDouble())
            return Math.toDegrees(angle).toFloat()
        }

        /** Wraps an angle in degrees into [0, 360). */
        fun normalizeDeg(deg: Float): Float {
            var d = deg % 360f
            if (d < 0f) d += 360f
            return d
        }

        /** Convenience: round a percentage for display. */
        fun percent(covered: Int, total: Int): Int =
            ((covered.toFloat() / total) * 100f).roundToInt()
    }
}
