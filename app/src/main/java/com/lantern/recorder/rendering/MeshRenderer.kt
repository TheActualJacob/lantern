package com.lantern.recorder.rendering

import android.opengl.GLES20
import com.lantern.recorder.recon.MeshData
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws the live reconstruction mesh as shaded triangles, in ARCore world space, after the
 * camera background. Marching cubes emits independent triangles (sequential vertices), so we
 * draw with `glDrawArrays` and skip an index buffer entirely.
 *
 * GL-thread only. [updateMesh] re-uploads buffers; [draw] renders with the supplied
 * model-view-projection (column-major, OpenGL convention) from ARCore.
 */
class MeshRenderer {
    private var program = 0
    private var aPos = 0
    private var aNormal = 0
    private var uMvp = 0

    private var vertexBuffer: FloatBuffer? = null
    private var normalBuffer: FloatBuffer? = null
    private var vertexCount = 0
    private var uploadedVersion = -1
    private var pointCloud = false

    fun createOnGlThread() {
        program = ShaderUtil.createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        aPos = GLES20.glGetAttribLocation(program, "aPos")
        aNormal = GLES20.glGetAttribLocation(program, "aNormal")
        uMvp = GLES20.glGetUniformLocation(program, "uMvp")
        ShaderUtil.checkGlError("MeshRenderer program")
    }

    /** Upload a new mesh if [version] differs from the last upload. Cheap no-op otherwise. */
    fun updateMesh(mesh: MeshData, version: Int) {
        if (version == uploadedVersion) return
        uploadedVersion = version
        vertexCount = mesh.vertexCount
        pointCloud = mesh.isPointCloud
        if (mesh.isEmpty) {
            vertexBuffer = null
            normalBuffer = null
            return
        }
        vertexBuffer = newFloatBuffer(mesh.vertices)
        // Point clouds may ship without normals; synthesize up-normals so the shader stays happy.
        val normals = if (mesh.normals.size == mesh.vertices.size) {
            mesh.normals
        } else {
            FloatArray(mesh.vertices.size) { if (it % 3 == 1) 1f else 0f }
        }
        normalBuffer = newFloatBuffer(normals)
    }

    fun draw(mvpColumnMajor: FloatArray) {
        val vb = vertexBuffer ?: return
        val nb = normalBuffer ?: return
        if (vertexCount == 0) return

        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(uMvp, 1, false, mvpColumnMajor, 0)

        GLES20.glEnableVertexAttribArray(aPos)
        vb.position(0)
        GLES20.glVertexAttribPointer(aPos, 3, GLES20.GL_FLOAT, false, 0, vb)

        GLES20.glEnableVertexAttribArray(aNormal)
        nb.position(0)
        GLES20.glVertexAttribPointer(aNormal, 3, GLES20.GL_FLOAT, false, 0, nb)

        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glDrawArrays(if (pointCloud) GLES20.GL_POINTS else GLES20.GL_TRIANGLES, 0, vertexCount)

        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aNormal)
        ShaderUtil.checkGlError("MeshRenderer draw")
    }

    private fun newFloatBuffer(data: FloatArray): FloatBuffer {
        val bb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
        val fb = bb.asFloatBuffer()
        fb.put(data)
        fb.position(0)
        return fb
    }

    companion object {
        private const val VERTEX_SHADER = """
            uniform mat4 uMvp;
            attribute vec4 aPos;
            attribute vec3 aNormal;
            varying vec3 vNormal;
            void main() {
                vNormal = aNormal;
                gl_Position = uMvp * aPos;
                gl_PointSize = 7.0;
            }
        """

        // Two-sided Lambert (abs(dot)) so flat/flipped marching-cubes normals still shade.
        private const val FRAGMENT_SHADER = """
            precision mediump float;
            varying vec3 vNormal;
            void main() {
                vec3 n = normalize(vNormal);
                vec3 l = normalize(vec3(0.3, 0.8, 0.6));
                float diff = abs(dot(n, l)) * 0.7 + 0.3;
                vec3 base = vec3(0.45, 0.78, 1.0);
                gl_FragColor = vec4(base * diff, 1.0);
            }
        """
    }
}
