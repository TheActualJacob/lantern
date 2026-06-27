package com.lantern.recorder.recon

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.lantern.recorder.R
import com.lantern.recorder.sessions.SessionInfo
import kotlin.math.cos
import kotlin.math.sin

/**
 * Full-screen viewer for a reconstructed model. The 3D surface is a lightweight
 * Compose-`Canvas` wireframe placeholder (auto-rotating, drag to orbit) — no GL / 3D
 * engine dependency — standing in until the real `.glb` from the reconstruction
 * pipeline can be loaded. Below it: model stats and export/share actions (stubbed).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelViewerScreen(
    session: SessionInfo,
    stats: ModelStats,
    onBack: () -> Unit,
    onExport: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.model_viewer_title)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            painter = painterResource(R.drawable.ic_arrow_back),
                            contentDescription = stringResource(R.string.action_back),
                        )
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            Wireframe3DSurface(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
            )

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                PreviewDisclaimer()

                OutlinedCard(shape = RoundedCornerShape(16.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        Text(
                            text = session.displayDate(),
                            style = MaterialTheme.typography.titleMedium,
                        )
                        Spacer(Modifier.height(12.dp))
                        StatRow(stringResource(R.string.model_stat_vertices), "%,d".format(stats.vertices))
                        StatRow(stringResource(R.string.model_stat_faces), "%,d".format(stats.faces))
                        StatRow(stringResource(R.string.model_stat_dimensions), stats.formattedDimensions())
                        StatRow(stringResource(R.string.model_stat_size), stats.formattedSize())
                        StatRow(
                            label = stringResource(R.string.model_stat_watertight),
                            value = stringResource(
                                if (stats.watertight) R.string.value_yes else R.string.value_no,
                            ),
                        )
                    }
                }

                FilledTonalButton(
                    onClick = onExport,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(
                        painter = painterResource(R.drawable.ic_share),
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.size(8.dp))
                    Text(stringResource(R.string.model_export_glb))
                }
            }
        }
    }
}

@Composable
private fun StatRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
        )
    }
}

@Composable
private fun PreviewDisclaimer() {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
        shape = RoundedCornerShape(12.dp),
    ) {
        Text(
            text = stringResource(R.string.model_preview_disclaimer),
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
        )
    }
}

/**
 * The placeholder 3D surface: a wireframe icosahedron + point cloud that slowly
 * auto-rotates and responds to drag. Pure [Canvas] math — projects rotated 3D points
 * to 2D — so it carries no rendering-engine dependency.
 */
@Composable
private fun Wireframe3DSurface(modifier: Modifier = Modifier) {
    val infinite = rememberInfiniteTransition(label = "turntable")
    val autoYaw by infinite.animateFloat(
        initialValue = 0f,
        targetValue = (2 * Math.PI).toFloat(),
        animationSpec = infiniteRepeatable(tween(14000, easing = LinearEasing), RepeatMode.Restart),
        label = "yaw",
    )

    // User drag offsets (added on top of the auto-rotation).
    var dragYaw by remember { mutableFloatStateOf(0f) }
    var dragPitch by remember { mutableFloatStateOf(0.5f) }

    Box(
        modifier = modifier
            .background(
                Brush.radialGradient(
                    colors = listOf(Color(0xFF15212E), Color(0xFF0A1119)),
                ),
            )
            .pointerInput(Unit) {
                detectDragGestures { _, drag ->
                    dragYaw += drag.x * 0.01f
                    dragPitch = (dragPitch + drag.y * 0.01f).coerceIn(-1.4f, 1.4f)
                }
            },
    ) {
        Canvas(Modifier.fillMaxSize()) {
            val yaw = autoYaw + dragYaw
            val pitch = dragPitch
            val center = Offset(size.width / 2f, size.height / 2f)
            val radius = size.minDimension * 0.30f
            val focal = 3.2f

            fun project(v: FloatArray): Pair<Offset, Float> {
                // Rotate around Y (yaw) then X (pitch).
                val cy = cos(yaw); val sy = sin(yaw)
                val x1 = v[0] * cy + v[2] * sy
                val z1 = -v[0] * sy + v[2] * cy
                val cp = cos(pitch); val sp = sin(pitch)
                val y2 = v[1] * cp - z1 * sp
                val z2 = v[1] * sp + z1 * cp
                val scale = focal / (focal + z2)
                return Offset(
                    center.x + x1 * scale * radius,
                    center.y - y2 * scale * radius,
                ) to z2
            }

            // Edges (depth-independent line set for a clean wireframe).
            for ((a, b) in ICOSA_EDGES) {
                val (pa, za) = project(ICOSA_VERTS[a])
                val (pb, zb) = project(ICOSA_VERTS[b])
                val depth = ((za + zb) / 2f + 1.6f) / 3.2f
                drawLine(
                    color = Color(0xFF5FA8E0).copy(alpha = (0.25f + depth * 0.6f).coerceIn(0.2f, 0.95f)),
                    start = pa,
                    end = pb,
                    strokeWidth = 2.2f,
                    cap = StrokeCap.Round,
                )
            }

            // Vertices as glowing points to suggest a scanned point cloud.
            for (v in ICOSA_VERTS) {
                val (p, z) = project(v)
                val depth = (z + 1.6f) / 3.2f
                drawCircle(
                    color = Color(0xFFE53935).copy(alpha = (0.35f + depth * 0.6f).coerceIn(0.3f, 1f)),
                    radius = 3.5f + depth * 3f,
                    center = p,
                )
            }
        }
    }
}

// --- Icosahedron geometry (12 vertices, 30 edges) used for the wireframe placeholder.

private val PHI = ((1.0 + Math.sqrt(5.0)) / 2.0).toFloat()

private val ICOSA_VERTS: Array<FloatArray> = arrayOf(
    floatArrayOf(-1f, PHI, 0f),
    floatArrayOf(1f, PHI, 0f),
    floatArrayOf(-1f, -PHI, 0f),
    floatArrayOf(1f, -PHI, 0f),
    floatArrayOf(0f, -1f, PHI),
    floatArrayOf(0f, 1f, PHI),
    floatArrayOf(0f, -1f, -PHI),
    floatArrayOf(0f, 1f, -PHI),
    floatArrayOf(PHI, 0f, -1f),
    floatArrayOf(PHI, 0f, 1f),
    floatArrayOf(-PHI, 0f, -1f),
    floatArrayOf(-PHI, 0f, 1f),
).also { verts ->
    // Normalize to a unit-ish radius so projection scaling is predictable.
    val norm = Math.sqrt((1 + PHI * PHI).toDouble()).toFloat()
    for (v in verts) {
        v[0] /= norm; v[1] /= norm; v[2] /= norm
    }
}

private val ICOSA_EDGES: Array<IntArray> = arrayOf(
    intArrayOf(0, 1), intArrayOf(0, 5), intArrayOf(0, 7), intArrayOf(0, 10), intArrayOf(0, 11),
    intArrayOf(1, 5), intArrayOf(1, 7), intArrayOf(1, 8), intArrayOf(1, 9),
    intArrayOf(2, 3), intArrayOf(2, 4), intArrayOf(2, 6), intArrayOf(2, 10), intArrayOf(2, 11),
    intArrayOf(3, 4), intArrayOf(3, 6), intArrayOf(3, 8), intArrayOf(3, 9),
    intArrayOf(4, 5), intArrayOf(4, 9), intArrayOf(4, 11),
    intArrayOf(5, 9), intArrayOf(5, 11),
    intArrayOf(6, 7), intArrayOf(6, 8), intArrayOf(6, 10),
    intArrayOf(7, 8), intArrayOf(7, 10),
    intArrayOf(8, 9),
    intArrayOf(10, 11),
)
