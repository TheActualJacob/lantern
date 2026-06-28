package com.lantern.recorder.sessions

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.FileProvider
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.lantern.recorder.R
import com.lantern.recorder.recon.MeshData
import com.lantern.recorder.recon.MeshExport
import com.lantern.recorder.recon.ModelStats
import com.lantern.recorder.recon.ModelViewerScreen
import com.lantern.recorder.recon.SessionDetailScreen
import com.lantern.recorder.ui.theme.LanternTheme
import java.io.File

/**
 * Browser for recorded capture sessions. Hosts a small in-activity navigation stack
 * (list → session detail → 3D model viewer) using Compose state — no navigation
 * library — so it stays dependency-light. Owns the file-system side effects (share
 * zip via FileProvider, delete) and feeds an observable list from [SessionStore].
 */
class SessionsActivity : AppCompatActivity() {

    /** In-activity navigation target. */
    private sealed interface Screen {
        data object List : Screen
        data class Detail(val session: SessionInfo) : Screen
        data class Viewer(val session: SessionInfo, val stats: ModelStats) : Screen
    }

    private var sessions by mutableStateOf<List<SessionInfo>>(emptyList())
    private var screen by mutableStateOf<Screen>(Screen.List)

    /** The reconstructed mesh shown in the viewer (latest live-mesh OBJ export); null while loading. */
    private var viewerMesh by mutableStateOf<MeshData?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            LanternTheme {
                // System / predictive back pops the in-activity stack before finishing.
                BackHandler(enabled = screen !is Screen.List) {
                    screen = when (val s = screen) {
                        is Screen.Viewer -> Screen.Detail(s.session)
                        else -> Screen.List
                    }
                }
                AnimatedContent(
                    targetState = screen,
                    transitionSpec = {
                        val forward = targetState !is Screen.List
                        if (forward) {
                            (slideInHorizontally(tween(260)) { it / 3 } + fadeIn(tween(260))) togetherWith
                                fadeOut(tween(180))
                        } else {
                            fadeIn(tween(220)) togetherWith
                                (slideOutHorizontally(tween(220)) { it / 3 } + fadeOut(tween(180)))
                        }
                    },
                    label = "sessions-nav",
                ) { target ->
                    when (target) {
                        is Screen.List -> SessionsScreen(
                            sessions = sessions,
                            onBack = { finish() },
                            onOpen = { screen = Screen.Detail(it) },
                            onShare = ::shareSession,
                            onDelete = ::confirmDelete,
                        )
                        is Screen.Detail -> SessionDetailScreen(
                            session = target.session,
                            onBack = { screen = Screen.List },
                            onOpenViewer = { stats ->
                                screen = Screen.Viewer(target.session, stats)
                                loadLatestMesh()
                            },
                            onShare = { shareSession(target.session) },
                        )
                        is Screen.Viewer -> ModelViewerScreen(
                            session = target.session,
                            stats = target.stats,
                            mesh = viewerMesh,
                            onBack = { screen = Screen.Detail(target.session) },
                            onExport = { shareLatestModel() },
                        )
                    }
                }
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

    /** Loads the most recent live-mesh OBJ export into [viewerMesh] off the main thread. */
    private fun loadLatestMesh() {
        viewerMesh = null
        Thread {
            val file = MeshExport.latestModel(getExternalFilesDir(null))
            val mesh = file?.let { MeshExport.readObj(it) }
            runOnUiThread { viewerMesh = mesh }
        }.start()
    }

    /** Shares the most recent reconstructed OBJ model via the system share sheet. */
    private fun shareLatestModel() {
        val file: File? = MeshExport.latestModel(getExternalFilesDir(null))
        if (file == null || !file.exists()) {
            Toast.makeText(this, getString(R.string.export_unavailable), Toast.LENGTH_SHORT).show()
            return
        }
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, file.name)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(send, getString(R.string.share_session_via)))
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
