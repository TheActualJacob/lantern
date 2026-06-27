package com.lantern.recorder.sessions

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.FileProvider
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.lantern.recorder.R
import com.lantern.recorder.ui.theme.LanternTheme

/**
 * Browser for recorded capture sessions (roadmap P6 UX layer). Lets the user see
 * what they've captured and act on it: **share** a session as a `.zip` (via
 * FileProvider) or **delete** it (button or swipe).
 *
 * The UI is Jetpack Compose ([SessionsScreen]); this activity owns the file-system
 * side effects and feeds an observable list from [SessionStore], refreshing on resume
 * so deletions and new recordings are always reflected.
 */
class SessionsActivity : AppCompatActivity() {

    private var sessions by mutableStateOf<List<SessionInfo>>(emptyList())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            LanternTheme {
                SessionsScreen(
                    sessions = sessions,
                    onBack = { finish() },
                    onShare = ::shareSession,
                    onDelete = ::confirmDelete,
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    private fun refresh() {
        sessions = SessionStore.listSessions(this)
    }

    private fun shareSession(session: SessionInfo) {
        Toast.makeText(this, getString(R.string.preparing_share), Toast.LENGTH_SHORT).show()
        // Zipping touches disk; keep it off the main thread.
        Thread {
            val result = runCatching { SessionStore.zipForSharing(this, session) }
            runOnUiThread {
                result.onSuccess { zip ->
                    val uri = FileProvider.getUriForFile(
                        this, "$packageName.fileprovider", zip,
                    )
                    val send = Intent(Intent.ACTION_SEND).apply {
                        type = "application/zip"
                        putExtra(Intent.EXTRA_STREAM, uri)
                        putExtra(Intent.EXTRA_SUBJECT, session.name)
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    startActivity(Intent.createChooser(send, getString(R.string.share_session_via)))
                }.onFailure { e ->
                    Log.e(TAG, "Failed to zip ${session.name}", e)
                    Toast.makeText(this, getString(R.string.share_failed), Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun confirmDelete(session: SessionInfo) {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.delete_session_title)
            .setMessage(getString(R.string.delete_session_message, session.displayDate()))
            .setNegativeButton(R.string.action_cancel, null)
            .setPositiveButton(R.string.action_delete) { _, _ ->
                if (SessionStore.delete(session)) {
                    refresh()
                } else {
                    Toast.makeText(this, getString(R.string.delete_failed), Toast.LENGTH_LONG).show()
                }
            }
            .show()
    }

    private companion object {
        const val TAG = "LANTERN"
    }
}
