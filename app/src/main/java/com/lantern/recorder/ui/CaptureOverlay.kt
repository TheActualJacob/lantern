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
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
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
import com.lantern.recorder.ui.theme.LanternNavy
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
    onOpenHelp: () -> Unit,
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

        // Help affordance to re-open the scanning coach overlay (top-end corner).
        IconButton(
            onClick = onOpenHelp,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(top = 8.dp, end = 8.dp)
                .size(48.dp)
                .clip(CircleShape)
                .drawBehind { drawCircle(Color.Black.copy(alpha = 0.35f)) },
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_help),
                contentDescription = stringResource(R.string.coach_help_cd),
                tint = Color.White,
            )
        }

        // Persistent, low-key guidance: the "place on a surface" guardrail when idle,
        // a "move around" hint as recording starts, and a pure-rotation warning.
        GuidanceBanner(
            state = state,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(top = 72.dp, start = 24.dp, end = 24.dp),
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

        // Live pose-based coverage dome, bottom-end, while a session is recording.
        CoverageGizmo(
            visible = state.isRecording,
            bandMaskSides = state.bandMaskSides,
            bandMaskUpper = state.bandMaskUpper,
            topCovered = state.topCovered,
            currentBand = state.currentBand,
            currentSector = state.currentSector,
            coveragePercent = state.coveragePercent,
            objectLocked = state.objectLocked,
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .windowInsetsPadding(WindowInsets.navigationBars)
                .padding(end = 24.dp, bottom = 44.dp),
        )

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

        // Transient "saved" confirmation, floating just above the shutter. Far nicer
        // than a system toast: on-brand, auto-dismissing, with a direct "View" action.
        SaveConfirmation(
            saved = state.savedConfirmation,
            onView = {
                state.clearSavedConfirmation()
                onOpenSessions()
            },
            onDismiss = state::clearSavedConfirmation,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .windowInsetsPadding(WindowInsets.navigationBars)
                .padding(bottom = 140.dp, start = 24.dp, end = 24.dp),
        )
    }
}

/**
 * A floating confirmation card shown briefly after a recording stops. Slides up from
 * behind the shutter with a green check, the frame count, and the session name, plus
 * a "View" action. Auto-dismisses after a few seconds (or immediately when actioned).
 */
@Composable
private fun SaveConfirmation(
    saved: SavedConfirmation?,
    onView: () -> Unit,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    // Drive auto-dismiss off the latest non-null value so the timer restarts per save.
    LaunchedEffect(saved) {
        if (saved != null) {
            kotlinx.coroutines.delay(4500)
            onDismiss()
        }
    }

    AnimatedVisibility(
        visible = saved != null,
        enter = slideInVertically(tween(280)) { it / 2 } + fadeIn(tween(280)),
        exit = slideOutVertically(tween(220)) { it / 2 } + fadeOut(tween(180)),
        modifier = modifier,
    ) {
        // Keep the last value during the exit animation so it doesn't blank out.
        val shown = remember(saved) { saved } ?: return@AnimatedVisibility
        Surface(
            color = LanternNavy.copy(alpha = 0.94f),
            contentColor = Color.White,
            shape = RoundedCornerShape(20.dp),
            shadowElevation = 8.dp,
            modifier = Modifier.widthIn(max = 420.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(start = 14.dp, end = 6.dp, top = 12.dp, bottom = 12.dp),
            ) {
                // Green check badge.
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .size(36.dp)
                        .clip(CircleShape)
                        .drawBehind { drawCircle(Color(0xFF34C759)) },
                ) {
                    Icon(
                        painter = painterResource(R.drawable.ic_check),
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(22.dp),
                    )
                }
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 12.dp),
                ) {
                    Text(
                        text = stringResource(R.string.save_confirm_title, shown.frames),
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = shown.session,
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.White.copy(alpha = 0.7f),
                        maxLines = 1,
                    )
                }
                TextButton(onClick = onView) {
                    Text(
                        text = stringResource(R.string.save_confirm_view),
                        color = Color(0xFF8FC9FF),
                        fontWeight = FontWeight.SemiBold,
                    )
                }
            }
        }
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

    is CaptureStatus.Message ->
        Triple(status.text, if (status.isError) RecordRed else Color.White, false)
}

/**
 * A single line of contextual guidance shown just under the status chip. It is a
 * correctness aid, not decoration — in priority order it surfaces:
 *  1. a pure-rotation warning (recording but the recorder isn't getting parallax),
 *  2. a "move around the object" hint while a fresh recording warms up,
 *  3. the "place it on a surface, don't hold/rotate" guardrail while idle.
 * Once frames are flowing normally it yields to the status chip and hides itself.
 */
@Composable
private fun GuidanceBanner(
    state: CaptureUiState,
    modifier: Modifier = Modifier,
) {
    data class Guidance(val text: String, val iconRes: Int, val warning: Boolean)

    val guidance: Guidance? = when {
        state.motionWarning ->
            Guidance(stringResource(R.string.warn_pure_rotation), R.drawable.ic_no_spin, true)

        state.isRecording && state.frameCount == 0 ->
            Guidance(stringResource(R.string.hint_move_around), R.drawable.ic_orbit, false)

        !state.isRecording && state.depthSupported ->
            Guidance(stringResource(R.string.guardrail_surface), R.drawable.ic_surface, false)

        else -> null
    }

    AnimatedVisibility(
        visible = guidance != null,
        enter = fadeIn(tween(220)) + slideInVertically(tween(220)) { -it / 3 },
        exit = fadeOut(tween(160)) + slideOutVertically(tween(160)) { -it / 3 },
        modifier = modifier,
    ) {
        val shown = remember(guidance) { guidance } ?: return@AnimatedVisibility
        val accent = if (shown.warning) Color(0xFFFFC074) else Color.White
        Surface(
            color = if (shown.warning) Color(0xCC5A3A12) else Color.Black.copy(alpha = 0.5f),
            contentColor = Color.White,
            shape = RoundedCornerShape(50),
            modifier = Modifier.semantics {
                liveRegion = LiveRegionMode.Polite
                contentDescription = shown.text
            },
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 9.dp),
            ) {
                Icon(
                    painter = painterResource(shown.iconRes),
                    contentDescription = null,
                    tint = accent,
                    modifier = Modifier.size(18.dp),
                )
                Text(
                    text = shown.text,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Medium,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.clearAndSetSemantics {},
                )
            }
        }
    }
}

/**
 * The pose-based coverage guide (Deliverable 2, v2 = **dome / azimuth + elevation**).
 * A small top-down polar map of the viewing hemisphere over the object: the filled
 * center is the top cap (looking straight down), the middle ring is the upper band, and
 * the outer ring is the sides. Each ring is cut into azimuth wedges. Captured patches
 * light green, the current one is highlighted white, uncovered patches stay dim. Updates
 * as `onFrameSaved` folds each depth-gated keyframe into the dome masks. A live coverage
 * percentage and a "what's missing" hint sit beneath the dome.
 */
@Composable
private fun CoverageGizmo(
    visible: Boolean,
    bandMaskSides: Int,
    bandMaskUpper: Int,
    topCovered: Boolean,
    currentBand: Int,
    currentSector: Int,
    coveragePercent: Int,
    objectLocked: Boolean,
    modifier: Modifier = Modifier,
) {
    AnimatedVisibility(
        visible = visible,
        enter = scaleIn(tween(220)) + fadeIn(tween(220)),
        exit = scaleOut(tween(160)) + fadeOut(tween(160)),
        modifier = modifier,
    ) {
        val coverageDesc = stringResource(R.string.coverage_cd, coveragePercent)
        // Once the user has orbited the sides but hasn't looked down on it, nudge them
        // toward the top — the dome's whole point is exposing that missing dimension.
        val sidesDone = Integer.bitCount(bandMaskSides) >= CaptureUiState.AZIMUTH_SECTORS
        val sublabel = when {
            !objectLocked -> stringResource(R.string.coverage_locating)
            !topCovered && sidesDone -> stringResource(R.string.coverage_hint_top)
            else -> stringResource(R.string.coverage_title)
        }
        Surface(
            color = Color.Black.copy(alpha = 0.42f),
            shape = RoundedCornerShape(18.dp),
            modifier = Modifier.semantics {
                liveRegion = LiveRegionMode.Polite
                contentDescription = coverageDesc
            },
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            ) {
                val coveredColor = Color(0xFF6BE675)
                val currentColor = Color.White
                val dimColor = Color.White.copy(alpha = 0.16f)
                val sectors = CaptureUiState.AZIMUTH_SECTORS

                Canvas(Modifier.size(74.dp)) {
                    val center = Offset(size.width / 2f, size.height / 2f)
                    val rOuter = size.minDimension / 2f - 1.dp.toPx()
                    val stroke = rOuter * 0.26f
                    val gap = rOuter * 0.07f
                    val sidesRadius = rOuter - stroke / 2f
                    val upperRadius = sidesRadius - stroke - gap
                    val capRadius = (upperRadius - stroke / 2f - gap).coerceAtLeast(3.dp.toPx())

                    val sweep = 360f / sectors
                    val angGap = sweep * 0.2f

                    // Draw one ringed band of azimuth wedges at [radius].
                    fun drawBand(mask: Int, band: Int, radius: Float) {
                        val boxTopLeft = Offset(center.x - radius, center.y - radius)
                        val boxSize = Size(radius * 2f, radius * 2f)
                        for (i in 0 until sectors) {
                            val covered = (mask and (1 shl i)) != 0
                            val color = when {
                                band == currentBand && i == currentSector -> currentColor
                                covered -> coveredColor
                                else -> dimColor
                            }
                            // Sector 0 at the top (12 o'clock); advance clockwise.
                            val start = -90f + i * sweep + angGap / 2f
                            drawArc(
                                color = color,
                                startAngle = start,
                                sweepAngle = sweep - angGap,
                                useCenter = false,
                                topLeft = boxTopLeft,
                                size = boxSize,
                                style = Stroke(width = stroke, cap = StrokeCap.Round),
                            )
                        }
                    }

                    // Outer = sides, middle = upper band.
                    drawBand(bandMaskSides, band = 0, radius = sidesRadius)
                    drawBand(bandMaskUpper, band = 1, radius = upperRadius)

                    // Center disc = top cap.
                    val capColor = when {
                        currentBand == 2 -> currentColor
                        topCovered -> coveredColor
                        else -> dimColor
                    }
                    drawCircle(color = capColor, radius = capRadius, center = center)
                }

                Text(
                    text = stringResource(R.string.coverage_percent, coveragePercent),
                    style = MaterialTheme.typography.titleMedium,
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier
                        .padding(top = 6.dp)
                        .clearAndSetSemantics {},
                )
                Text(
                    text = sublabel,
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.White.copy(alpha = 0.7f),
                    textAlign = TextAlign.Center,
                    modifier = Modifier.clearAndSetSemantics {},
                )
            }
        }
    }
}
