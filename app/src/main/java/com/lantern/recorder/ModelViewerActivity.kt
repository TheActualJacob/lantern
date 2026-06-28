package com.lantern.recorder

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.FileProvider
import com.lantern.recorder.recon.MeshData
import com.lantern.recorder.recon.MeshExport
import com.lantern.recorder.recon.ModelCloudViewer
import com.lantern.recorder.ui.theme.LanternTheme
import java.io.File

/**
 * Lightweight, standalone viewer for a single reconstructed model file (a directed-capture point
 * cloud OBJ). Loads the mesh off-thread and shows it in the orbitable [ModelCloudViewer], with a
 * share action. Decoupled from [com.lantern.recorder.sessions.SessionsActivity] so the capture
 * screen can jump straight here after a build.
 */
class ModelViewerActivity : AppCompatActivity() {

    private var mesh by mutableStateOf<MeshData?>(null)
    private var loading by mutableStateOf(true)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val path = intent.getStringExtra(EXTRA_MODEL_PATH)
        loadMesh(path)
        setContent {
            LanternTheme {
                ModelCloudViewer(
                    mesh = mesh,
                    title = getString(R.string.model_viewer_title),
                    onBack = { finish() },
                    onShare = path?.let { { shareModel(File(it)) } },
                )
            }
        }
    }

    private fun loadMesh(path: String?) {
        if (path == null) {
            loading = false
            return
        }
        Thread {
            val loaded = runCatching { MeshExport.readObj(File(path)) }.getOrNull()
            runOnUiThread {
                mesh = loaded
                loading = false
            }
        }.start()
    }

    private fun shareModel(file: File) {
        if (!file.exists()) {
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

    companion object {
        private const val EXTRA_MODEL_PATH = "model_path"

        fun intent(context: Context, modelPath: String): Intent =
            Intent(context, ModelViewerActivity::class.java)
                .putExtra(EXTRA_MODEL_PATH, modelPath)
    }
}
