package com.lantern.recorder.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.lantern.recorder.R
import com.lantern.recorder.ui.theme.RecordRed

/**
 * The Compose overlay drawn on top of the live ARCore camera feed (a `GLSurfaceView`
 * hosted via `AndroidView`). It owns no camera state — it renders [state] and reports
 * intent through [onToggleRecord] / [onOpenSessions].
 *
 * Layout (edge-to-edge, portrait):
 *  - top center: status chip (crossfades between depth/recording/error states)
 *  - bottom center: shutter button (dot ⇄ square morph, pulse + frame badge while recording)
 *  - bottom start: sessions/gallery FAB
 */
@Composable
fun CaptureOverlay(
    state: CaptureUiState,
    onToggleRecord: () -> Unit,
    onOpenSessions: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(modifier = modifier.fillMaxSize()) {
        StatusChip(
            status = state.status,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(top = 16.dp, start = 24.dp, end = 24.dp),
        )

        // Sessions FAB, anchored to the bottom-start above the gesture bar.
        FloatingActionButton(
            onClick = onOpenSessions,
            containerColor = Color.Black.copy(alpha = 0.35f),
            contentColor = Color.White,
            modifier = Modifier
                .align(Alignment.BottomStart)
                .windowInsetsPadding(WindowInsets.navigationBars)
                .padding(start = 28.dp, bottom = 40.dp)
                .size(52.dp),
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_collections),
                contentDescription = stringResource(R.string.sessions_title),
            )
        }

        ShutterButton(
            recording = state.isRecording,
            enabled = state.depthSupported,
            frameCount = state.frameCount,
            onClick = onToggleRecord,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .windowInsetsPadding(WindowInsets.navigationBars)
                .padding(bottom = 32.dp),
        )
    }
}

/**
 * Pill chip describing the current capture state. The text + accent color crossfade
 * whenever [status] changes; a translucent scrim keeps it legible over any camera
 * scene. Announced to TalkBack as a polite live region.
 */
@Composable
private fun StatusChip(
    status: CaptureStatus,
    modifier: Modifier = Modifier,
) {
    val (label, accent, recording) = statusPresentation(status)
    Surface(
        color = Color.Black.copy(alpha = 0.55f),
        contentColor = Color.White,
        shape = RoundedCornerShape(50),
        modifier = modifier.semantics {
            liveRegion = LiveRegionMode.Polite
            contentDescription = label
        },
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
        ) {
            // Small status dot; pulses red while recording.
            StatusDot(accent = accent, pulsing = recording)
            AnimatedContent(
                targetState = label,
                transitionSpec = { fadeIn(tween(220)) togetherWith fadeOut(tween(160)) },
                label = "status-text",
            ) { text ->
                Text(
                    text = text,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Medium,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}

@Composable
private fun StatusDot(accent: Color, pulsing: Boolean) {
    val transition = rememberInfiniteTransition(label = "dot-pulse")
    val alpha by transition.animateFloat(
        initialValue = 1f,
        targetValue = if (pulsing) 0.25f else 1f,
        animationSpec = infiniteRepeatable(tween(700), RepeatMode.Reverse),
        label = "dot-alpha",
    )
    Box(
        Modifier
            .size(8.dp)
            .clip(CircleShape)
            .drawBehind { drawCircle(accent.copy(alpha = if (pulsing) alpha else 1f)) },
    )
}

/**
 * Camera-app-style shutter: a fixed white ring with a red core that morphs between a
 * filled circle (idle) and a rounded square (recording). While recording it gently
 * pulses and shows a live frame-count badge. Disabled (depth-unsupported) state dims
 * the whole control and blocks clicks.
 */
@Composable
private fun ShutterButton(
    recording: Boolean,
    enabled: Boolean,
    frameCount: Int,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val interactionSource = remember { MutableInteractionSource() }
    val pressed by interactionSource.collectIsPressedAsState()

    // Press feedback: a subtle scale-down on the whole button.
    val pressScale by animateFloatAsState(
        targetValue = if (pressed && enabled) 0.92f else 1f,
        animationSpec = tween(120),
        label = "shutter-press",
    )

    // Recording pulse on the ring.
    val pulse = rememberInfiniteTransition(label = "shutter-pulse")
    val ringScale by pulse.animateFloat(
        initialValue = 1f,
        targetValue = if (recording) 1.06f else 1f,
        animationSpec = infiniteRepeatable(tween(650), RepeatMode.Reverse),
        label = "ring-scale",
    )

    // Core morph: corner radius (round dot -> rounded square) + size.
    val coreCorner by animateDpAsState(
        targetValue = if (recording) 8.dp else 32.dp,
        animationSpec = tween(260),
        label = "core-corner",
    )
    val coreSize by animateDpAsState(
        targetValue = if (recording) 34.dp else 58.dp,
        animationSpec = tween(260),
        label = "core-size",
    )

    val contentDesc = when {
        !enabled -> stringResource(R.string.depth_unsupported_no_record)
        recording -> stringResource(R.string.a11y_stop_recording)
        else -> stringResource(R.string.a11y_start_recording)
    }

    Box(contentAlignment = Alignment.Center, modifier = modifier) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(80.dp)
                .scale(pressScale)
                .clip(CircleShape)
                .clickable(
                    interactionSource = interactionSource,
                    indication = ripple(bounded = true, color = Color.White),
                    enabled = enabled,
                    role = Role.Button,
                    onClickLabel = contentDesc,
                    onClick = onClick,
                )
                .semantics { contentDescription = contentDesc },
        ) {
            val coreColor = if (enabled) RecordRed else RecordRed.copy(alpha = 0.35f)
            val ringColor = if (enabled) Color.White else Color.White.copy(alpha = 0.4f)

            // White ring.
            Canvas(Modifier.size(80.dp).scale(if (recording) ringScale else 1f)) {
                drawCircle(
                    color = ringColor,
                    radius = size.minDimension / 2f - 4.dp.toPx(),
                    center = Offset(size.width / 2f, size.height / 2f),
                    style = Stroke(width = 5.dp.toPx()),
                )
            }
            // Morphing red core.
            Box(
                Modifier
                    .size(coreSize)
                    .clip(RoundedCornerShape(coreCorner))
                    .drawBehind { drawRect(coreColor) },
            )
        }

        // Frame-count badge while recording (top-right of the shutter).
        AnimatedVisibility(
            visible = recording,
            enter = scaleIn() + fadeIn(),
            exit = scaleOut() + fadeOut(),
            modifier = Modifier.align(Alignment.TopEnd),
        ) {
            Surface(
                color = RecordRed,
                contentColor = Color.White,
                shape = RoundedCornerShape(50),
            ) {
                Text(
                    text = frameCount.toString(),
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier
                        .padding(horizontal = 8.dp, vertical = 2.dp)
                        .clearAndSetSemantics {},
                )
            }
        }
    }
}

/** Maps a [CaptureStatus] to its (label, accent color, isRecording) presentation. */
@Composable
private fun statusPresentation(status: CaptureStatus): Triple<String, Color, Boolean> = when (status) {
    CaptureStatus.Initializing ->
        Triple(stringResource(R.string.status_initializing), Color.White, false)

    is CaptureStatus.DepthReady ->
        if (status.supported) {
            Triple(stringResource(R.string.status_depth_ready), Color(0xFF6BE675), false)
        } else {
            Triple(stringResource(R.string.status_depth_unsupported), Color(0xFFFFC074), false)
        }

    is CaptureStatus.Recording ->
        Triple(
            stringResource(R.string.recording_status, status.session, status.frames),
            RecordRed,
            true,
        )

    is CaptureStatus.Saved ->
        Triple(
            stringResource(R.string.recording_saved, status.frames, status.session),
            Color(0xFF6BE675),
            false,
        )

    is CaptureStatus.Message ->
        Triple(status.text, if (status.isError) RecordRed else Color.White, false)
}
