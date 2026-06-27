package com.lantern.recorder.recon

import com.lantern.recorder.sessions.SessionInfo
import java.util.Locale
import kotlin.math.cbrt
import kotlin.math.roundToInt

/**
 * Stand-in metrics for a reconstructed mesh. Until the on-device reconstruction
 * runtime (ExecuTorch / QNN) lands, these are derived deterministically from the
 * captured session so the viewer + detail UI show plausible, stable numbers.
 */
data class ModelStats(
    val vertices: Int,
    val faces: Int,
    val sizeBytes: Long,
    val widthMeters: Float,
    val heightMeters: Float,
    val depthMeters: Float,
    val watertight: Boolean,
) {
    fun formattedDimensions(): String = String.format(
        Locale.US, "%.2f × %.2f × %.2f m", widthMeters, heightMeters, depthMeters,
    )

    fun formattedSize(): String = SessionInfo.formatBytes(sizeBytes)

    companion object {
        /**
         * Deterministic mock so repeated visits to the same session show identical
         * stats. Roughly models TSDF output growing with the number of fused frames.
         */
        fun mockFor(session: SessionInfo): ModelStats {
            val frames = session.frameCount.coerceAtLeast(1)
            val vertices = (frames * 1800 + 4200).coerceAtMost(2_500_000)
            val faces = (vertices * 1.9f).roundToInt()
            // Vague object footprint that grows slowly with capture coverage.
            val span = 0.18f + cbrt(frames.toFloat()) * 0.06f
            return ModelStats(
                vertices = vertices,
                faces = faces,
                sizeBytes = vertices.toLong() * 38L,
                widthMeters = span,
                heightMeters = span * 1.25f,
                depthMeters = span * 0.9f,
                watertight = false,
            )
        }
    }
}
