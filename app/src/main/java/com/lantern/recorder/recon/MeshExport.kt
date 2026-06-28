package com.lantern.recorder.recon

import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

private const val TAG = "LANTERN"

/**
 * Writes a [MeshData] (world-meter triangle mesh from the live TSDF + marching cubes) to a
 * Wavefront **OBJ** file. OBJ is chosen over glTF/PLY because it's plain text with zero
 * dependencies and imports everywhere (Blender, MeshLab, Preview, the pipeline's
 * `import_and_clean.py`). Vertices/normals are emitted in meters; faces are triangles.
 */
object MeshExport {

    /**
     * Serialize [mesh] to [file] as OBJ. Returns the file on success, or null if the mesh is
     * empty or the write fails (callers degrade gracefully — a failed export must never crash a
     * scan). Safe to call off the GL thread.
     */
    fun writeObj(mesh: MeshData, file: File): File? {
        if (mesh.isEmpty || mesh.vertexCount == 0) {
            Log.w(TAG, "MeshExport: nothing to write (empty mesh)")
            return null
        }
        return try {
            file.parentFile?.mkdirs()
            BufferedWriter(FileWriter(file), 1 shl 16).use { w ->
                val v = mesh.vertices
                val nrm = mesh.normals
                val idx = mesh.indices
                val hasNormals = nrm.size == v.size
                w.write("# Lantern live-mesh export\n")
                w.write("# vertices=${mesh.vertexCount} triangles=${mesh.triangleCount}\n")
                w.write("o lantern_object\n")
                var i = 0
                while (i < v.size) {
                    w.write("v ${v[i]} ${v[i + 1]} ${v[i + 2]}\n")
                    i += 3
                }
                if (hasNormals) {
                    i = 0
                    while (i < nrm.size) {
                        w.write("vn ${nrm[i]} ${nrm[i + 1]} ${nrm[i + 2]}\n")
                        i += 3
                    }
                }
                // OBJ indices are 1-based.
                i = 0
                while (i < idx.size) {
                    val a = idx[i] + 1
                    val b = idx[i + 1] + 1
                    val c = idx[i + 2] + 1
                    if (hasNormals) {
                        w.write("f $a//$a $b//$b $c//$c\n")
                    } else {
                        w.write("f $a $b $c\n")
                    }
                    i += 3
                }
            }
            Log.i(TAG, "MeshExport: wrote ${mesh.vertexCount} verts / ${mesh.triangleCount} tris -> ${file.absolutePath}")
            file
        } catch (t: Throwable) {
            Log.e(TAG, "MeshExport: failed to write ${file.absolutePath}", t)
            null
        }
    }

    /** Directory where live-mesh OBJ exports are written (external files dir, adb-pullable). */
    fun modelsDir(filesRoot: File?): File = File(filesRoot, "models")

    /** The most recently exported `.obj` model, or null if none exist. */
    fun latestModel(filesRoot: File?): File? =
        modelsDir(filesRoot).listFiles { f -> f.isFile && f.name.endsWith(".obj") }
            ?.maxByOrNull { it.lastModified() }

    /**
     * Parse an OBJ written by [writeObj] back into a [MeshData] (for the in-app viewer). Only the
     * `v`/`vn`/`f` records we emit are handled; faces are assumed triangulated. Returns null on
     * any failure so the viewer can fall back to the placeholder. Run off the main thread.
     */
    fun readObj(file: File): MeshData? {
        return try {
            val verts = ArrayList<Float>(1 shl 16)
            val norms = ArrayList<Float>(1 shl 16)
            val idx = ArrayList<Int>(1 shl 16)
            file.bufferedReader().useLines { lines ->
                for (line in lines) {
                    when {
                        line.startsWith("v ") -> {
                            val t = line.split(' ')
                            verts.add(t[1].toFloat()); verts.add(t[2].toFloat()); verts.add(t[3].toFloat())
                        }
                        line.startsWith("vn ") -> {
                            val t = line.split(' ')
                            norms.add(t[1].toFloat()); norms.add(t[2].toFloat()); norms.add(t[3].toFloat())
                        }
                        line.startsWith("f ") -> {
                            val t = line.split(' ')
                            // "f a//na b//nb c//nc" or "f a b c"; OBJ is 1-based.
                            for (k in 1..3) idx.add(t[k].substringBefore('/').toInt() - 1)
                        }
                    }
                }
            }
            // A face-less OBJ is a valid point cloud (directed-capture export), not a failure.
            if (verts.isEmpty()) return null
            MeshData(verts.toFloatArray(), norms.toFloatArray(), idx.toIntArray())
        } catch (t: Throwable) {
            Log.e(TAG, "MeshExport: failed to read ${file.absolutePath}", t)
            null
        }
    }
}
