package com.lantern.recorder.rendering

import android.opengl.GLES20
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws a single-channel object mask as a translucent green tint over the camera feed, aligned to
 * the live image. The mask is produced (off-thread) in **camera-image** coordinates that map
 * uniformly onto the CPU image, so we texture a full-screen NDC quad whose texture coordinates come
 * from ARCore's `OPENGL_NDC -> IMAGE_NORMALIZED` transform — the same mechanism [BackgroundRenderer]
 * uses for the camera, which is why this lines up exactly (no manual rotation/crop guessing).
 *
 * [setMask] may be called from any thread; the upload happens lazily on the GL thread in [draw].
 */
class MaskOverlayRenderer {

    private var program = 0
    private var positionAttrib = 0
    private var texCoordAttrib = 0
    private var textureUniform = 0
    private var textureId = 0

    private val quadCoords: FloatBuffer = newFloatBuffer(QUAD_COORDS)
    private val quadTexCoords: FloatBuffer = newFloatBuffer(FloatArray(8))

    // Pending mask handed in from the segmentation worker; uploaded on the next GL draw.
    @Volatile private var pendingMask: ByteBuffer? = null
    @Volatile private var pendingW = 0
    @Volatile private var pendingH = 0
    @Volatile private var hasMask = false

    // Whether [quadTexCoords] has been filled from ARCore's NDC->image transform yet. Until then the
    // buffer is all zeros, which would sample one texel across the whole quad (overlay invisible).
    private var texCoordsValid = false

    fun createOnGlThread() {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        textureId = textures[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        program = ShaderUtil.createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        positionAttrib = GLES20.glGetAttribLocation(program, "a_Position")
        texCoordAttrib = GLES20.glGetAttribLocation(program, "a_TexCoord")
        textureUniform = GLES20.glGetUniformLocation(program, "uMask")
        ShaderUtil.checkGlError("MaskOverlayRenderer.createOnGlThread")
    }

    /**
     * Hand in a new mask: [mask] is row-major [w]x[h], 1f = object. Copied into a luminance byte
     * buffer immediately (so the caller's array can be reused), uploaded on the next [draw].
     */
    fun setMask(mask: FloatArray, w: Int, h: Int) {
        if (w <= 0 || h <= 0 || mask.size < w * h) return
        val buf = ByteBuffer.allocateDirect(w * h).order(ByteOrder.nativeOrder())
        for (i in 0 until w * h) buf.put(if (mask[i] > 0.5f) 0xFF.toByte() else 0)
        buf.position(0)
        pendingMask = buf
        pendingW = w
        pendingH = h
        hasMask = true
    }

    /** Clear the overlay (e.g. when the directed session stops). */
    fun clear() {
        hasMask = false
        pendingMask = null
    }

    fun draw(frame: Frame) {
        if (frame.timestamp == 0L) return

        // Map the NDC quad into camera-image-normalized coords (the mask's space). Recompute on
        // geometry change AND on first use: the change event fires once early in the session — often
        // before any mask exists — so gating only on it (plus the !hasMask early-return below) would
        // leave the texcoords zeroed when the first mask finally arrives, hiding the overlay until a
        // later rotation happened to re-fire the event. Consuming it here every frame fixes that.
        if (frame.hasDisplayGeometryChanged() || !texCoordsValid) {
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                quadCoords,
                Coordinates2d.IMAGE_NORMALIZED,
                quadTexCoords,
            )
            texCoordsValid = true
        }

        if (!hasMask) return

        val upload = pendingMask
        if (upload != null) {
            pendingMask = null
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
            GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1)
            GLES20.glTexImage2D(
                GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, pendingW, pendingH, 0,
                GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, upload,
            )
        }

        quadCoords.position(0)
        quadTexCoords.position(0)

        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)

        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        GLES20.glUniform1i(textureUniform, 0)

        GLES20.glVertexAttribPointer(positionAttrib, 2, GLES20.GL_FLOAT, false, 0, quadCoords)
        GLES20.glVertexAttribPointer(texCoordAttrib, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
        GLES20.glEnableVertexAttribArray(positionAttrib)
        GLES20.glEnableVertexAttribArray(texCoordAttrib)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(positionAttrib)
        GLES20.glDisableVertexAttribArray(texCoordAttrib)

        GLES20.glDepthMask(true)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glDisable(GLES20.GL_BLEND)
        ShaderUtil.checkGlError("MaskOverlayRenderer.draw")
    }

    private fun newFloatBuffer(data: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(data.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(data)
                position(0)
            }

    companion object {
        // Same full-screen NDC quad / winding as BackgroundRenderer, so the transforms agree.
        private val QUAD_COORDS = floatArrayOf(
            -1.0f, -1.0f,
            +1.0f, -1.0f,
            -1.0f, +1.0f,
            +1.0f, +1.0f,
        )

        private const val VERTEX_SHADER = """
            attribute vec4 a_Position;
            attribute vec2 a_TexCoord;
            varying vec2 v_TexCoord;
            void main() {
                gl_Position = a_Position;
                v_TexCoord = a_TexCoord;
            }
        """

        private const val FRAGMENT_SHADER = """
            precision mediump float;
            varying vec2 v_TexCoord;
            uniform sampler2D uMask;
            void main() {
                float m = texture2D(uMask, v_TexCoord).r;
                if (m < 0.5) discard;
                gl_FragColor = vec4(0.13, 0.87, 0.43, 0.45);
            }
        """
    }
}
