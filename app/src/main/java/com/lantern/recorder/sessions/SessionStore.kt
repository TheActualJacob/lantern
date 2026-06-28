package com.lantern.recorder.sessions

import android.content.Context
import android.util.Log
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Read/maintenance layer for recorded capture sessions living under
 * `getExternalFilesDir(null)/sessions/`. Pure file-system work — no UI, no ARCore —
 * so it can be unit-reasoned and reused by the [SessionsActivity] browser.
 *
 * A "session" is one `session_<stamp>` folder written by `FrameRecorder`, containing
 * `frame_NNNN.{png,json}`, `depth_NNNN.png`, `conf_NNNN.png`.
 */
object SessionStore {

    private const val TAG = "LANTERN"

    /** Root folder that holds every `session_*` directory (created on demand). */
    fun sessionsRoot(context: Context): File =
        File(context.getExternalFilesDir(null), "sessions").apply { mkdirs() }

    /**
     * Writes a `scan_group.json` sidecar into a recorded session folder, marking it as one
     * pass of a multi-pass "flip the object" capture so the host reconstruction pipeline
     * can register and merge the passes (e.g. ICP) into a single watertight model.
     *
     * This is a separate metadata file written *after* recording stops — it does not touch
     * the [com.lantern.recorder.recording.FrameRecorder] or any captured frame data.
     *
     * @param sessionDir the `session_*` folder to annotate
     * @param groupId shared id linking all passes of one object
     * @param pass 1-based pass index within the group
     * @param totalPasses expected number of passes in the group
     */
    fun writeScanGroup(sessionDir: File, groupId: String, pass: Int, totalPasses: Int) {
        try {
            val json = """
                {
                  "group_id": "$groupId",
                  "pass": $pass,
                  "total_passes": $totalPasses,
                  "merge": "flip",
                  "note": "Pass $pass of $totalPasses. Object was flipped between passes; the recorded poses are not in a shared frame across passes — register/merge (e.g. ICP) downstream."
                }
            """.trimIndent()
            File(sessionDir, "scan_group.json").writeText(json)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to write scan_group.json in ${sessionDir.absolutePath}", e)
        }
    }

    /** Lists sessions newest-first. Frame count is derived from `frame_*.json` files. */
    fun listSessions(context: Context): List<SessionInfo> {
        val root = sessionsRoot(context)
        val dirs = root.listFiles { f -> f.isDirectory && f.name.startsWith("session_") }
            ?: return emptyList()
        return dirs
            .map { dir -> toInfo(dir) }
            .sortedByDescending { it.lastModified }
    }

    private fun toInfo(dir: File): SessionInfo {
        val files = dir.listFiles() ?: emptyArray()
        var sizeBytes = 0L
        var frameCount = 0
        for (file in files) {
            sizeBytes += file.length()
            val name = file.name
            // One JSON per saved frame — the cheapest unambiguous frame tally.
            if (name.startsWith("frame_") && name.endsWith(".json")) frameCount++
        }
        return SessionInfo(
            dir = dir,
            name = dir.name,
            frameCount = frameCount,
            sizeBytes = sizeBytes,
            lastModified = dir.lastModified(),
        )
    }

    /** Recursively deletes a session folder. Returns true if it's gone afterward. */
    fun delete(session: SessionInfo): Boolean {
        val ok = session.dir.deleteRecursively()
        if (!ok) Log.w(TAG, "Failed to fully delete ${session.dir.absolutePath}")
        return !session.dir.exists()
    }

    /**
     * Zips a session into `cacheDir/shared/<name>.zip` for sharing via FileProvider.
     * Overwrites any stale zip with the same name. Returns the zip file.
     */
    fun zipForSharing(context: Context, session: SessionInfo): File {
        val shareDir = File(context.cacheDir, "shared").apply { mkdirs() }
        val zipFile = File(shareDir, "${session.name}.zip")
        if (zipFile.exists()) zipFile.delete()

        val files = session.dir.listFiles()?.filter { it.isFile }?.sortedBy { it.name } ?: emptyList()
        ZipOutputStream(zipFile.outputStream().buffered()).use { zip ->
            val buffer = ByteArray(64 * 1024)
            for (file in files) {
                zip.putNextEntry(ZipEntry("${session.name}/${file.name}"))
                file.inputStream().buffered().use { input ->
                    var read = input.read(buffer)
                    while (read >= 0) {
                        zip.write(buffer, 0, read)
                        read = input.read(buffer)
                    }
                }
                zip.closeEntry()
            }
        }
        return zipFile
    }
}
