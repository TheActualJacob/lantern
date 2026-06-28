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
import androidx.compose.ui.graphics.Path
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
    mesh: MeshData? = null,
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
                mesh = mesh,
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
                        StatRow(
                            stringResource(R.string.model_stat_vertices),
                            "%,d".format(mesh?.vertexCount ?: stats.vertices),
                        )
                        StatRow(
                            stringResource(R.string.model_stat_faces),
                            "%,d".format(mesh?.triangleCount ?: stats.faces),
                        )
                        StatRow(
                            stringResource(R.string.model_stat_dimensions),
                            mesh?.let { formatMeshDimensions(it) } ?: stats.formattedDimensions(),
                        )
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

/**
 * Standalone viewer for a freshly reconstructed [mesh] (e.g. a directed-capture point cloud), not
 * tied to a recorded session. Same orbit/auto-rotate surface as [ModelViewerScreen], with stats
 * derived straight from the mesh and an optional share action.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelCloudViewer(
    mesh: MeshData?,
    title: String,
    onBack: () -> Unit,
    onShare: (() -> Unit)? = null,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(title) },
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
                mesh = mesh,
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
                OutlinedCard(shape = RoundedCornerShape(16.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        StatRow(
                            stringResource(R.string.model_stat_vertices),
                            "%,d".format(mesh?.vertexCount ?: 0),
                        )
                        StatRow(
                            stringResource(R.string.model_stat_dimensions),
                            mesh?.let { formatMeshDimensions(it) } ?: "—",
                        )
                    }
                }
                if (onShare != null) {
                    FilledTonalButton(onClick = onShare, modifier = Modifier.fillMaxWidth()) {
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
 * The 3D surface. When a reconstructed [mesh] is supplied it renders that mesh as a
 * height/depth-shaded point cloud (subsampled for smooth interaction), auto-centered and
 * normalized to fit. With no mesh it falls back to the placeholder wireframe icosahedron.
 * Pure [Canvas] math — projects rotated 3D points to 2D — so it carries no rendering-engine
 * dependency. Slowly auto-rotates and responds to drag.
 */
@Composable
private fun Wireframe3DSurface(modifier: Modifier = Modifier, mesh: MeshData? = null) {
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

    // Surface meshes (e.g. the convex-hull fill) carry triangles — render them filled. Point-cloud
    // results (no triangles) fall back to the depth-shaded dot rendering below.
    val surface: NormalizedMesh? = remember(mesh) { mesh?.takeIf { it.triangleCount > 0 }?.let { normalizedMesh(it, MAX_VIEW_TRIS) } }
    // Centered + unit-normalized + subsampled point cloud, computed once per mesh.
    val points: FloatArray? = remember(mesh) { if (mesh != null && mesh.triangleCount == 0) normalizedPoints(mesh, MAX_VIEW_POINTS) else null }

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
            val radius = size.minDimension * (if (points != null) 0.42f else 0.30f)
            val focal = 3.2f
            val cy = cos(yaw); val sy = sin(yaw)
            val cp = cos(pitch); val sp = sin(pitch)

            fun project(x: Float, y: Float, z: Float): Pair<Offset, Float> {
                // Rotate around Y (yaw) then X (pitch).
                val x1 = x * cy + z * sy
                val z1 = -x * sy + z * cy
                val y2 = y * cp - z1 * sp
                val z2 = y * sp + z1 * cp
                val scale = focal / (focal + z2)
                return Offset(center.x + x1 * scale * radius, center.y - y2 * scale * radius) to z2
            }

            if (surface != null) {
                // Surface mesh (hull fill / marching cubes): filled triangles via painter's algorithm
                // (sort far->near, no z-buffer needed) with two-sided Lambert shading + height hue.
                val v = surface.verts
                val idx = surface.indices
                val nv = v.size / 3
                // Project + cache rotated 3D (for normals) and screen pos per vertex.
                val sxArr = FloatArray(nv); val syArr = FloatArray(nv)
                val rxArr = FloatArray(nv); val ryArr = FloatArray(nv); val rzArr = FloatArray(nv)
                for (k in 0 until nv) {
                    val x = v[k * 3]; val y = v[k * 3 + 1]; val z = v[k * 3 + 2]
                    val x1 = x * cy + z * sy
                    val z1 = -x * sy + z * cy
                    val y2 = y * cp - z1 * sp
                    val z2 = y * sp + z1 * cp
                    rxArr[k] = x1; ryArr[k] = y2; rzArr[k] = z2
                    val scale = focal / (focal + z2)
                    sxArr[k] = center.x + x1 * scale * radius
                    syArr[k] = center.y - y2 * scale * radius
                }
                val nt = idx.size / 3
                val order = (0 until nt).sortedByDescending { f ->
                    rzArr[idx[f * 3]] + rzArr[idx[f * 3 + 1]] + rzArr[idx[f * 3 + 2]]
                }
                // Light from over the viewer's shoulder (view space).
                val lx = 0.35f; val ly = 0.45f; val lz = -0.82f
                for (f in order) {
                    val a = idx[f * 3]; val b = idx[f * 3 + 1]; val c = idx[f * 3 + 2]
                    // View-space face normal for shading.
                    val ux = rxArr[b] - rxArr[a]; val uy = ryArr[b] - ryArr[a]; val uz = rzArr[b] - rzArr[a]
                    val wx = rxArr[c] - rxArr[a]; val wy = ryArr[c] - ryArr[a]; val wz = rzArr[c] - rzArr[a]
                    var nx = uy * wz - uz * wy; var ny = uz * wx - ux * wz; var nz = ux * wy - uy * wx
                    val nl = kotlin.math.sqrt(nx * nx + ny * ny + nz * nz)
                    if (nl < 1e-9f) continue
                    nx /= nl; ny /= nl; nz /= nl
                    val lambert = kotlin.math.abs(nx * lx + ny * ly + nz * lz) // two-sided
                    val shade = (0.35f + 0.65f * lambert).coerceIn(0f, 1f)
                    val h = ((ryArr[a] + ryArr[b] + ryArr[c]) / 3f + 1f) / 2f
                    val col = Color(
                        red = ((0.30f + 0.55f * h.coerceIn(0f, 1f)) * shade),
                        green = ((0.55f + 0.30f * (1f - kotlin.math.abs(h - 0.5f) * 2f)) * shade).coerceIn(0f, 1f),
                        blue = ((0.65f + 0.30f * (1f - h.coerceIn(0f, 1f))) * shade),
                    )
                    val path = Path().apply {
                        moveTo(sxArr[a], syArr[a]); lineTo(sxArr[b], syArr[b]); lineTo(sxArr[c], syArr[c]); close()
                    }
                    drawPath(path, color = col)
                }
                return@Canvas
            }

            if (points != null) {
                // Real reconstructed object: depth-shaded point cloud, colored by height.
                var i = 0
                while (i < points.size) {
                    val px = points[i]; val py = points[i + 1]; val pz = points[i + 2]
                    val (p, z) = project(px, py, pz)
                    val depth = ((z + 1.5f) / 3f).coerceIn(0f, 1f)
                    val h = ((py + 1f) / 2f).coerceIn(0f, 1f)
                    val col = Color(
                        red = (0.20f + 0.70f * h),
                        green = (0.45f + 0.40f * (1f - kotlin.math.abs(h - 0.5f) * 2f)),
                        blue = (0.55f + 0.40f * (1f - h)),
                    )
                    drawCircle(
                        color = col.copy(alpha = (0.30f + depth * 0.65f).coerceIn(0.25f, 1f)),
                        radius = 1.6f + depth * 2.4f,
                        center = p,
                    )
                    i += 3
                }
                return@Canvas
            }

            // Placeholder: wireframe icosahedron + glowing point cloud.
            for ((a, b) in ICOSA_EDGES) {
                val (pa, za) = project(ICOSA_VERTS[a][0], ICOSA_VERTS[a][1], ICOSA_VERTS[a][2])
                val (pb, zb) = project(ICOSA_VERTS[b][0], ICOSA_VERTS[b][1], ICOSA_VERTS[b][2])
                val depth = ((za + zb) / 2f + 1.6f) / 3.2f
                drawLine(
                    color = Color(0xFF5FA8E0).copy(alpha = (0.25f + depth * 0.6f).coerceIn(0.2f, 0.95f)),
                    start = pa,
                    end = pb,
                    strokeWidth = 2.2f,
                    cap = StrokeCap.Round,
                )
            }
            for (v in ICOSA_VERTS) {
                val (p, z) = project(v[0], v[1], v[2])
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

/** Bounding-box dimensions of a mesh formatted in millimeters (e.g. "120 × 85 × 60 mm"). */
private fun formatMeshDimensions(mesh: MeshData): String {
    val v = mesh.vertices
    var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE; var minZ = Float.MAX_VALUE
    var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
    var i = 0
    while (i < v.size) {
        val x = v[i]; val y = v[i + 1]; val z = v[i + 2]
        if (x < minX) minX = x; if (x > maxX) maxX = x
        if (y < minY) minY = y; if (y > maxY) maxY = y
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
        i += 3
    }
    val mm = 1000f
    return "%.0f × %.0f × %.0f mm".format((maxX - minX) * mm, (maxY - minY) * mm, (maxZ - minZ) * mm)
}

/** Max points drawn in the viewer; the mesh is strided down to this for smooth interaction. */
private const val MAX_VIEW_POINTS = 6000

/** Max triangles filled by the software canvas; denser meshes are decimated for smooth interaction. */
private const val MAX_VIEW_TRIS = 12000

/** Normalized surface mesh for the viewer: unit-fit vertices (flat xyz) + triangle indices. */
private class NormalizedMesh(val verts: FloatArray, val indices: IntArray)

/**
 * Center + unit-normalize a triangle mesh's vertices (so it fits a unit cube) and decimate to at
 * most [triBudget] triangles (stride whole faces) so the Compose software canvas stays smooth.
 * Returns null if the mesh has no triangles.
 */
private fun normalizedMesh(mesh: MeshData, triBudget: Int): NormalizedMesh? {
    if (mesh.triangleCount == 0) return null
    val v = mesh.vertices
    var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE; var minZ = Float.MAX_VALUE
    var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
    var i = 0
    while (i < v.size) {
        val x = v[i]; val y = v[i + 1]; val z = v[i + 2]
        if (x < minX) minX = x; if (x > maxX) maxX = x
        if (y < minY) minY = y; if (y > maxY) maxY = y
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
        i += 3
    }
    val cx = (minX + maxX) / 2f; val cy = (minY + maxY) / 2f; val cz = (minZ + maxZ) / 2f
    val half = maxOf(maxX - minX, maxY - minY, maxZ - minZ) / 2f
    val inv = if (half > 1e-6f) 1f / half else 1f
    val nv = FloatArray(v.size)
    var j = 0
    while (j < v.size) {
        nv[j] = (v[j] - cx) * inv; nv[j + 1] = (v[j + 1] - cy) * inv; nv[j + 2] = (v[j + 2] - cz) * inv
        j += 3
    }
    val src = mesh.indices
    val nt = src.size / 3
    val indices = if (nt <= triBudget) {
        src
    } else {
        val step = (nt + triBudget - 1) / triBudget
        val out = ArrayList<Int>(triBudget * 3)
        var f = 0
        while (f < nt) {
            out.add(src[f * 3]); out.add(src[f * 3 + 1]); out.add(src[f * 3 + 2])
            f += step
        }
        out.toIntArray()
    }
    return NormalizedMesh(nv, indices)
}

/**
 * Center a mesh's vertices on their bounding-box midpoint and scale by the inverse of the
 * largest half-extent so the object fits a unit cube, striding down to ~[budget] points so the
 * Compose canvas stays smooth. Returns a flat xyz array.
 */
private fun normalizedPoints(mesh: MeshData, budget: Int): FloatArray? {
    val v = mesh.vertices
    val n = mesh.vertexCount
    if (n == 0) return null
    var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE; var minZ = Float.MAX_VALUE
    var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
    var i = 0
    while (i < v.size) {
        val x = v[i]; val y = v[i + 1]; val z = v[i + 2]
        if (x < minX) minX = x; if (x > maxX) maxX = x
        if (y < minY) minY = y; if (y > maxY) maxY = y
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
        i += 3
    }
    val cx = (minX + maxX) / 2f; val cy = (minY + maxY) / 2f; val cz = (minZ + maxZ) / 2f
    val half = maxOf(maxX - minX, maxY - minY, maxZ - minZ) / 2f
    val inv = if (half > 1e-6f) 1f / half else 1f
    val stride = maxOf(1, n / budget)
    val out = ArrayList<Float>(minOf(n, budget) * 3)
    var idx = 0
    while (idx < n) {
        val o = idx * 3
        out.add((v[o] - cx) * inv)
        out.add((v[o + 1] - cy) * inv)
        out.add((v[o + 2] - cz) * inv)
        idx += stride
    }
    return out.toFloatArray()
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
