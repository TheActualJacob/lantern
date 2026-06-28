package com.lantern.recorder.sessions

import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Immutable snapshot of one recorded capture session on disk. */
data class SessionInfo(
    val dir: File,
    val name: String,
    val frameCount: Int,
    val sizeBytes: Long,
    val lastModified: Long,
) {
    /** Human date parsed from the `session_yyyyMMdd_HHmmss` name, falling back to mtime. */
    fun displayDate(): String {
        parseStampDate()?.let { return PRETTY.format(it) }
        return PRETTY.format(Date(lastModified))
    }

    fun displaySize(): String = formatBytes(sizeBytes)

    private fun parseStampDate(): Date? {
        val stamp = name.removePrefix("session_")
        return try {
            STAMP.parse(stamp)
        } catch (_: Exception) {
            null
        }
    }

    companion object {
        private val STAMP = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        private val PRETTY = SimpleDateFormat("MMM d, yyyy · h:mm a", Locale.US)

        fun formatBytes(bytes: Long): String {
            if (bytes < 1024) return "$bytes B"
            val units = arrayOf("KB", "MB", "GB", "TB")
            var value = bytes.toDouble() / 1024.0
            var unit = 0
            while (value >= 1024.0 && unit < units.lastIndex) {
                value /= 1024.0
                unit++
            }
            return String.format(Locale.US, "%.1f %s", value, units[unit])
        }
    }
}
