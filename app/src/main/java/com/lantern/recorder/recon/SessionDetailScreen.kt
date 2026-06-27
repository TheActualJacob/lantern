package com.lantern.recorder.recon

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.lantern.recorder.R
import com.lantern.recorder.sessions.SessionInfo
import kotlinx.coroutines.delay

/** One stage of the host reconstruction pipeline, surfaced as a progress step. */
private data class ReconStage(val labelRes: Int, val durationMs: Long)

private val RECON_STAGES = listOf(
    ReconStage(R.string.stage_depth, 1400),
    ReconStage(R.string.stage_scale, 900),
    ReconStage(R.string.stage_tsdf, 1500),
    ReconStage(R.string.stage_mesh, 1100),
    ReconStage(R.string.stage_cleanup, 1000),
)

private enum class ReconPhase { Intro, Running, Ready }

/**
 * Detail screen for a captured session: shows capture stats and lets the user kick
 * off reconstruction. The reconstruction itself is **mocked** for now (the on-device
 * runtime isn't wired up) — it animates through the real pipeline stages
 * (depth → scale → TSDF → mesh → cleanup) and then unlocks the model viewer.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SessionDetailScreen(
    session: SessionInfo,
    onBack: () -> Unit,
    onOpenViewer: (ModelStats) -> Unit,
    onShare: () -> Unit,
) {
    var phase by remember(session.name) { mutableStateOf(ReconPhase.Intro) }
    var stageIndex by remember(session.name) { mutableIntStateOf(0) }
    var stageProgress by remember(session.name) { mutableFloatStateOf(0f) }
    var startToken by remember(session.name) { mutableIntStateOf(0) }

    // Drives the mocked pipeline once the user taps Build (startToken > 0).
    LaunchedEffect(startToken) {
        if (startToken == 0) return@LaunchedEffect
        phase = ReconPhase.Running
        for (i in RECON_STAGES.indices) {
            stageIndex = i
            val steps = 24
            repeat(steps) { s ->
                stageProgress = (s + 1) / steps.toFloat()
                delay(RECON_STAGES[i].durationMs / steps)
            }
        }
        phase = ReconPhase.Ready
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.detail_title)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            painter = painterResource(R.drawable.ic_arrow_back),
                            contentDescription = stringResource(R.string.action_back),
                        )
                    }
                },
                actions = {
                    IconButton(onClick = onShare) {
                        Icon(
                            painter = painterResource(R.drawable.ic_share),
                            contentDescription = stringResource(R.string.action_share),
                        )
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Spacer(Modifier.height(0.dp))
            SessionSummaryCard(session)

            AnimatedContent(
                targetState = phase,
                transitionSpec = { fadeIn(tween(220)) togetherWith fadeOut(tween(160)) },
                label = "recon-phase",
            ) { p ->
                when (p) {
                    ReconPhase.Intro -> IntroPanel(onBuild = { startToken++ })
                    ReconPhase.Running -> RunningPanel(stageIndex, stageProgress)
                    ReconPhase.Ready -> ReadyPanel(
                        onView = { onOpenViewer(ModelStats.mockFor(session)) },
                        onRebuild = {
                            stageIndex = 0
                            stageProgress = 0f
                            startToken++
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun SessionSummaryCard(session: SessionInfo) {
    OutlinedCard(shape = RoundedCornerShape(16.dp), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text(
                text = session.displayDate(),
                style = MaterialTheme.typography.titleMedium,
            )
            Text(
                text = pluralStringResource(
                    R.plurals.session_frames,
                    session.frameCount,
                    session.frameCount,
                    session.displaySize(),
                ),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 2.dp),
            )
        }
    }
}

@Composable
private fun IntroPanel(onBuild: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(
            text = stringResource(R.string.build_model_desc),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        DisclaimerBanner()
        Button(onClick = onBuild, modifier = Modifier.fillMaxWidth()) {
            Icon(
                painter = painterResource(R.drawable.ic_cube),
                contentDescription = null,
                modifier = Modifier.size(18.dp),
            )
            Spacer(Modifier.size(8.dp))
            Text(stringResource(R.string.build_model))
        }
    }
}

@Composable
private fun RunningPanel(stageIndex: Int, stageProgress: Float) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = stringResource(R.string.recon_running_title),
            style = MaterialTheme.typography.titleMedium,
        )
        RECON_STAGES.forEachIndexed { i, stage ->
            val status = when {
                i < stageIndex -> StageStatus.Done
                i == stageIndex -> StageStatus.Active
                else -> StageStatus.Pending
            }
            StageRow(
                label = stringResource(stage.labelRes),
                status = status,
                progress = if (status == StageStatus.Active) stageProgress else 0f,
            )
        }
    }
}

private enum class StageStatus { Pending, Active, Done }

@Composable
private fun StageRow(label: String, status: StageStatus, progress: Float) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp),
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.size(24.dp),
        ) {
            when (status) {
                StageStatus.Done -> Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .size(24.dp)
                        .clip(CircleShape),
                ) {
                    Icon(
                        painter = painterResource(R.drawable.ic_check),
                        contentDescription = null,
                        tint = Color(0xFF34C759),
                        modifier = Modifier.size(22.dp),
                    )
                }
                StageStatus.Active -> CircularProgressIndicator(
                    strokeWidth = 2.5.dp,
                    modifier = Modifier.size(20.dp),
                )
                StageStatus.Pending -> Box(
                    Modifier
                        .size(10.dp)
                        .background(MaterialTheme.colorScheme.outlineVariant, CircleShape),
                )
            }
        }
        Spacer(Modifier.size(14.dp))
        Column(Modifier.weight(1f)) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyLarge,
                color = if (status == StageStatus.Pending) {
                    MaterialTheme.colorScheme.onSurfaceVariant
                } else {
                    MaterialTheme.colorScheme.onSurface
                },
            )
            if (status == StageStatus.Active) {
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 6.dp),
                )
            }
        }
    }
}

@Composable
private fun ReadyPanel(onView: () -> Unit, onRebuild: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                painter = painterResource(R.drawable.ic_check),
                contentDescription = null,
                tint = Color(0xFF34C759),
                modifier = Modifier.size(24.dp),
            )
            Spacer(Modifier.size(8.dp))
            Text(
                text = stringResource(R.string.recon_done_title),
                style = MaterialTheme.typography.titleMedium,
            )
        }
        DisclaimerBanner()
        Button(onClick = onView, modifier = Modifier.fillMaxWidth()) {
            Icon(
                painter = painterResource(R.drawable.ic_cube),
                contentDescription = null,
                modifier = Modifier.size(18.dp),
            )
            Spacer(Modifier.size(8.dp))
            Text(stringResource(R.string.view_model))
        }
        TextButton(onClick = onRebuild, modifier = Modifier.fillMaxWidth()) {
            Text(stringResource(R.string.rebuild_model))
        }
    }
}

@Composable
private fun DisclaimerBanner() {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
        shape = RoundedCornerShape(12.dp),
    ) {
        Text(
            text = stringResource(R.string.recon_disclaimer),
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
        )
    }
}
