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
import androidx.compose.foundation.Image
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
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import com.lantern.recorder.R
import com.lantern.recorder.recon.DepthBackendKind
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
    onToggleTwoPass: (Boolean) -> Unit,
    onSetCaptureMode: (CaptureMode) -> Unit,
    turntableAvailable: Boolean,
    onOpenSessions: () -> Unit,
    onOpenHelp: () -> Unit,
    onToggleDebug: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    Box(modifier = modifier.fillMaxSize()) {
        // One structured top panel: a toolbar row (mode toggle · help), an options row
        // (two-sided scan), the status chip, and a single mode-aware context line. Each
        // is its own full-width row in a column, so nothing floats or overlaps.
        TopPanel(
            state = state,
            turntableAvailable = turntableAvailable,
            onSetCaptureMode = onSetCaptureMode,
            onToggleTwoPass = onToggleTwoPass,
            onOpenHelp = onOpenHelp,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(top = 10.dp, start = 16.dp, end = 12.dp),
        )

        // Prominent, can't-miss flip-flow card (only during the two-pass flip sequence).
        FlipFlowBanner(
            phase = state.scanPhase,
            onManualReady = state::forceReadyForPassTwo,
            modifier = Modifier
                .align(Alignment.Center)
                .padding(horizontal = 32.dp),
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

        // Live-mesh readout (board-free real-time reconstruction), bottom-end.
        if (state.captureMode == CaptureMode.LiveMesh) {
            LiveMeshStatsChip(
                vertices = state.liveMeshVertices,
                frames = state.liveMeshFrames,
                depthBackend = state.liveMeshDepthBackend,
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .windowInsetsPadding(WindowInsets.navigationBars)
                    .padding(end = 24.dp, bottom = 44.dp)
                    .clickable { onToggleDebug() }, // tap to toggle the segmentation debug view
            )
            if (state.liveMeshDebug) {
                LiveMeshDebugPanel(
                    bitmap = state.liveMeshDebugBitmap,
                    text = state.liveMeshDebugText,
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .windowInsetsPadding(WindowInsets.statusBars)
                        .padding(start = 12.dp, top = 96.dp),
                )
            }
        }

        // The shutter is blocked mid-flip (object being handled); in ReadyForPassTwo it
        // invites the user to start the second side.
        val flipBlocked = state.scanPhase == ScanPhase.NeedsFlip ||
            state.scanPhase == ScanPhase.Flipping
        val startPassTwo = state.scanPhase == ScanPhase.ReadyForPassTwo
        ShutterButton(
            recording = state.isRecording,
            enabled = state.depthSupported && !flipBlocked,
            frameCount = state.frameCount,
            startPassTwo = startPassTwo,
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
    startPassTwo: Boolean,
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
        startPassTwo -> stringResource(R.string.a11y_start_pass_two)
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
        // Orbit ring = every azimuth direction seen at ANY elevation (matches the %).
        val orbitMask = bandMaskSides or bandMaskUpper
        val sublabel = when {
            !objectLocked -> stringResource(R.string.coverage_locating)
            !topCovered && coveragePercent >= 100 -> stringResource(R.string.coverage_hint_top)
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
                    val stroke = rOuter * 0.30f
                    val ringRadius = rOuter - stroke / 2f
                    val capRadius = (ringRadius - stroke / 2f - rOuter * 0.16f)
                        .coerceAtLeast(4.dp.toPx())

                    val sweep = 360f / sectors
                    val angGap = sweep * 0.2f

                    // Single orbit ring: one wedge per azimuth direction, lit if that
                    // direction was captured at any elevation (the headline metric).
                    val boxTopLeft = Offset(center.x - ringRadius, center.y - ringRadius)
                    val boxSize = Size(ringRadius * 2f, ringRadius * 2f)
                    for (i in 0 until sectors) {
                        val covered = (orbitMask and (1 shl i)) != 0
                        val color = when {
                            i == currentSector && currentBand != 2 -> currentColor
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

                    // Center disc = the top-down "looked down on it" cap.
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

/**
 * The single, structured top panel. Laid out strictly top-to-bottom as full-width
 * rows so elements never float or overlap:
 *
 *   1. **Toolbar** — capture-mode toggle (or live pass indicator) on the left, help
 *      on the right.
 *   2. **Options** — the two-sided-scan toggle (idle only).
 *   3. **Status** — the capture status chip.
 *   4. **Context** — exactly one mode-aware guidance line (never the orbit
 *      "don't rotate" guardrail in turntable mode).
 */
@Composable
private fun TopPanel(
    state: CaptureUiState,
    turntableAvailable: Boolean,
    onSetCaptureMode: (CaptureMode) -> Unit,
    onToggleTwoPass: (Boolean) -> Unit,
    onOpenHelp: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val inFlow = state.scanPhase != ScanPhase.SinglePass
    Column(
        modifier = modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        // 1. Toolbar row.
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Box(contentAlignment = Alignment.CenterStart) {
                // Live Mesh is the only capture mode now, so there's no mode selector. The pass
                // indicator still shows during a two-sided recording.
                if (state.isRecording && state.twoPassEnabled && inFlow) {
                    PassIndicator(state.scanPhase)
                }
            }
            HelpButton(onOpenHelp)
        }

        // 2. Options row: two-sided scan toggle (idle, pre-record).
        if (!state.isRecording && state.depthSupported) {
            Row(modifier = Modifier.fillMaxWidth()) {
                TwoPassToggle(on = state.twoPassEnabled, onToggle = onToggleTwoPass)
            }
        }

        // 3. Status chip.
        StatusChip(status = state.status, modifier = Modifier.widthIn(max = 360.dp))

        // 4. One mode-aware context line.
        ContextLine(state = state)
    }
}

/** Round, translucent help button (re-opens the scanning coach). */
@Composable
private fun HelpButton(onOpenHelp: () -> Unit) {
    IconButton(
        onClick = onOpenHelp,
        modifier = Modifier
            .size(44.dp)
            .clip(CircleShape)
            .drawBehind { drawCircle(Color.Black.copy(alpha = 0.4f)) },
    ) {
        Icon(
            painter = painterResource(R.drawable.ic_help),
            contentDescription = stringResource(R.string.coach_help_cd),
            tint = Color.White,
        )
    }
}

/**
 * A small segmented Orbit | Turntable selector shown while idle. Orbit keeps the
 * object still and moves the phone; Turntable keeps the phone still and spins the
 * object on a fiducial board (Object-Centric Capture, §3).
 */
@Composable
private fun CaptureModeToggle(
    mode: CaptureMode,
    turntableAvailable: Boolean,
    onSetCaptureMode: (CaptureMode) -> Unit,
) {
    Surface(
        color = Color.Black.copy(alpha = 0.42f),
        shape = RoundedCornerShape(50),
    ) {
        Row(
            modifier = Modifier.padding(3.dp),
            horizontalArrangement = Arrangement.spacedBy(3.dp),
        ) {
            ModeSegment(
                label = stringResource(R.string.mode_orbit),
                cd = stringResource(R.string.mode_orbit_cd),
                selected = mode == CaptureMode.Orbit,
                onClick = { onSetCaptureMode(CaptureMode.Orbit) },
            )
            if (turntableAvailable) {
                ModeSegment(
                    label = stringResource(R.string.mode_turntable),
                    cd = stringResource(R.string.mode_turntable_cd),
                    selected = mode == CaptureMode.Turntable,
                    onClick = { onSetCaptureMode(CaptureMode.Turntable) },
                )
            }
            ModeSegment(
                label = stringResource(R.string.mode_livemesh),
                cd = stringResource(R.string.mode_livemesh_cd),
                selected = mode == CaptureMode.LiveMesh,
                onClick = { onSetCaptureMode(CaptureMode.LiveMesh) },
            )
        }
    }
}

/** Live-mesh readout (vertex count, fused keyframes, depth backend) shown in LiveMesh mode. */
@Composable
private fun LiveMeshStatsChip(
    vertices: Int,
    frames: Int,
    depthBackend: DepthBackendKind,
    modifier: Modifier = Modifier,
) {
    val dense = depthBackend != DepthBackendKind.ARCORE
    val backend = stringResource(
        when (depthBackend) {
            DepthBackendKind.QNN -> R.string.livemesh_backend_qnn
            DepthBackendKind.EXECUTORCH -> R.string.livemesh_backend_da3
            DepthBackendKind.ARCORE -> R.string.livemesh_backend_arcore
        }
    )
    val label = stringResource(R.string.livemesh_stats, vertices, frames, backend)
    Surface(
        color = Color.Black.copy(alpha = 0.5f),
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
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 9.dp),
        ) {
            Box(
                Modifier
                    .size(9.dp)
                    .clip(CircleShape)
                    .drawBehind { drawCircle(if (dense) Color(0xFF6BE675) else Color(0xFF8FC9FF)) },
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Medium,
                modifier = Modifier.clearAndSetSemantics {},
            )
        }
    }
}

/** Debug overlay (tap the stats chip to toggle): camera-at-depth-res with the object mask tinted
 *  green, plus pipeline stats — to see what segmentation/depth are actually producing. */
@Composable
private fun LiveMeshDebugPanel(
    bitmap: android.graphics.Bitmap?,
    text: String,
    modifier: Modifier = Modifier,
) {
    Surface(
        color = Color.Black.copy(alpha = 0.6f),
        contentColor = Color.White,
        shape = RoundedCornerShape(10.dp),
        modifier = modifier,
    ) {
        Column(modifier = Modifier.padding(8.dp)) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Live mesh debug view",
                    filterQuality = FilterQuality.None,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .widthIn(max = 180.dp)
                        .clip(RoundedCornerShape(6.dp)),
                )
            }
            Text(
                text = text,
                style = MaterialTheme.typography.labelSmall,
                color = Color(0xFF9BE7A0),
                modifier = Modifier.padding(top = 6.dp),
            )
        }
    }
}

@Composable
private fun ModeSegment(
    label: String,
    cd: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        color = if (selected) Color(0xCC10324F) else Color.Transparent,
        contentColor = Color.White,
        shape = RoundedCornerShape(50),
        modifier = Modifier
            .clip(RoundedCornerShape(50))
            .clickable(role = Role.Button, onClickLabel = cd, onClick = onClick)
            .semantics {
                this.contentDescription = cd
                this.selected = selected
            },
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium,
            color = if (selected) Color.White else Color.White.copy(alpha = 0.7f),
            modifier = Modifier
                .padding(horizontal = 14.dp, vertical = 7.dp)
                .clearAndSetSemantics {},
        )
    }
}

/** Turntable board-lock status pill (green = locked, amber = searching). */
@Composable
private fun BoardStatusChip(tracking: Boolean) {
    val accent = if (tracking) Color(0xFF6BE675) else Color(0xFFFFC074)
    val label = stringResource(if (tracking) R.string.board_locked else R.string.board_searching)
    Surface(
        color = Color.Black.copy(alpha = 0.5f),
        contentColor = Color.White,
        shape = RoundedCornerShape(50),
        modifier = Modifier.semantics {
            liveRegion = LiveRegionMode.Polite
            contentDescription = label
        },
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
        ) {
            Box(
                Modifier
                    .size(9.dp)
                    .clip(CircleShape)
                    .drawBehind { drawCircle(accent) },
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Medium,
                modifier = Modifier.clearAndSetSemantics {},
            )
        }
    }
}

/** Live "Side N of 2" indicator shown on the toolbar during a two-pass flip scan. */
@Composable
private fun PassIndicator(phase: ScanPhase) {
    val sideNumber = when (phase) {
        ScanPhase.PassOne, ScanPhase.NeedsFlip -> 1
        ScanPhase.Flipping, ScanPhase.ReadyForPassTwo, ScanPhase.PassTwo, ScanPhase.Complete -> 2
        else -> 1
    }
    val label = stringResource(R.string.flip_pass_indicator, sideNumber)
    Surface(
        color = Color.Black.copy(alpha = 0.5f),
        contentColor = Color.White,
        shape = RoundedCornerShape(50),
        modifier = Modifier.semantics { contentDescription = label },
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            modifier = Modifier.padding(horizontal = 13.dp, vertical = 8.dp),
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_flip),
                contentDescription = null,
                tint = Color(0xFF8FC9FF),
                modifier = Modifier.size(16.dp),
            )
            Text(
                text = label,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.clearAndSetSemantics {},
            )
        }
    }
}

/** Pre-record "Two-sided scan" opt-in toggle. */
@Composable
private fun TwoPassToggle(on: Boolean, onToggle: (Boolean) -> Unit) {
    val cd = stringResource(if (on) R.string.twopass_disable_cd else R.string.twopass_enable_cd)
    Surface(
        color = if (on) Color(0xCC10324F) else Color.Black.copy(alpha = 0.42f),
        contentColor = Color.White,
        shape = RoundedCornerShape(50),
        modifier = Modifier
            .clip(RoundedCornerShape(50))
            .clickable(role = Role.Switch, onClickLabel = cd, onClick = { onToggle(!on) })
            .semantics { contentDescription = cd },
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(7.dp),
            modifier = Modifier.padding(horizontal = 13.dp, vertical = 8.dp),
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_flip),
                contentDescription = null,
                tint = if (on) Color(0xFF8FC9FF) else Color.White.copy(alpha = 0.85f),
                modifier = Modifier.size(16.dp),
            )
            Text(
                text = stringResource(R.string.twopass_label),
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Medium,
                modifier = Modifier.clearAndSetSemantics {},
            )
            Box(
                Modifier
                    .size(9.dp)
                    .clip(CircleShape)
                    .drawBehind {
                        drawCircle(if (on) Color(0xFF6BE675) else Color.White.copy(alpha = 0.3f))
                    },
            )
        }
    }
}

/**
 * The single contextual guidance line under the status chip — **mode aware**, so the
 * orbit "place it down, don't rotate" guardrail never appears in turntable mode (where
 * rotating the object is the whole point). At most one message shows at a time; the
 * prominent flip card speaks for itself during the flip flow, so we stay quiet then.
 */
@Composable
private fun ContextLine(state: CaptureUiState) {
    val inFlipFlow = state.scanPhase == ScanPhase.NeedsFlip ||
        state.scanPhase == ScanPhase.Flipping ||
        state.scanPhase == ScanPhase.ReadyForPassTwo ||
        state.scanPhase == ScanPhase.Complete

    when {
        inFlipFlow -> {}

        // Turntable: rotating IS the instruction. Show board lock while recording,
        // and a setup hint while idle — never the orbit guardrail.
        state.captureMode == CaptureMode.Turntable -> {
            if (state.isRecording) {
                BoardStatusChip(tracking = state.boardTracking)
            } else {
                InfoChip(stringResource(R.string.turntable_hint), R.drawable.ic_orbit)
            }
        }

        // Orbit guidance, in priority order.
        state.objectMovedWarning ->
            InfoChip(stringResource(R.string.object_moved_warning), R.drawable.ic_surface, warning = true)

        state.motionWarning ->
            InfoChip(stringResource(R.string.warn_pure_rotation), R.drawable.ic_no_spin, warning = true)

        state.isRecording && state.frameCount == 0 ->
            InfoChip(stringResource(R.string.hint_move_around), R.drawable.ic_orbit)

        !state.isRecording && state.depthSupported ->
            InfoChip(stringResource(R.string.guardrail_surface), R.drawable.ic_surface)

        else -> {}
    }
}

/** A compact pill carrying one line of guidance (info = dark, warning = amber). */
@Composable
private fun InfoChip(text: String, iconRes: Int, warning: Boolean = false) {
    val accent = if (warning) Color(0xFFFFC074) else Color.White
    Surface(
        color = if (warning) Color(0xCC5A3A12) else Color.Black.copy(alpha = 0.5f),
        contentColor = Color.White,
        shape = RoundedCornerShape(50),
        modifier = Modifier
            .widthIn(max = 360.dp)
            .semantics {
                liveRegion = LiveRegionMode.Polite
                contentDescription = text
            },
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 9.dp),
        ) {
            Icon(
                painter = painterResource(iconRes),
                contentDescription = null,
                tint = accent,
                modifier = Modifier.size(18.dp),
            )
            Text(
                text = text,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Medium,
                textAlign = TextAlign.Center,
                modifier = Modifier.clearAndSetSemantics {},
            )
        }
    }
}

/**
 * The prominent, can't-miss card that drives the two-pass flip sequence. It only shows
 * during the flip phases and tells the user exactly what to do: flip the object, wait
 * while it's detected moving, then scan side two — or that both sides are done. The
 * three "needs flip / flipping / ready" states are detected automatically from depth.
 */
@Composable
private fun FlipFlowBanner(
    phase: ScanPhase,
    onManualReady: () -> Unit,
    modifier: Modifier = Modifier,
) {
    data class FlipCopy(val title: String, val body: String, val iconRes: Int, val accent: Color, val pulse: Boolean)

    // The "needs flip" / "flipping" states accept a manual tap to advance, in case depth
    // detection misses the settle (object too small, glare, occluded).
    val manualAdvanceable = phase == ScanPhase.NeedsFlip || phase == ScanPhase.Flipping

    val copy: FlipCopy? = when (phase) {
        ScanPhase.NeedsFlip -> FlipCopy(
            stringResource(R.string.flip_needed_title),
            stringResource(R.string.flip_needed_body),
            R.drawable.ic_flip, Color(0xFFFFC074), pulse = true,
        )
        ScanPhase.Flipping -> FlipCopy(
            stringResource(R.string.flip_inprogress_title),
            stringResource(R.string.flip_inprogress_body),
            R.drawable.ic_flip, Color(0xFF8FC9FF), pulse = true,
        )
        ScanPhase.ReadyForPassTwo -> FlipCopy(
            stringResource(R.string.flip_ready_title),
            stringResource(R.string.flip_ready_body),
            R.drawable.ic_check, Color(0xFF6BE675), pulse = false,
        )
        ScanPhase.Complete -> FlipCopy(
            stringResource(R.string.flip_complete_title),
            stringResource(R.string.flip_complete_body),
            R.drawable.ic_check, Color(0xFF6BE675), pulse = false,
        )
        else -> null
    }

    AnimatedVisibility(
        visible = copy != null,
        enter = fadeIn(tween(220)) + scaleIn(tween(220), initialScale = 0.9f),
        exit = fadeOut(tween(160)) + scaleOut(tween(160), targetScale = 0.9f),
        modifier = modifier,
    ) {
        val shown = remember(copy) { copy } ?: return@AnimatedVisibility
        val manualHint = stringResource(R.string.flip_manual_hint)
        val manualCd = stringResource(R.string.flip_manual_cd)
        val pulseT = rememberInfiniteTransition(label = "flip-pulse")
        val iconScale by pulseT.animateFloat(
            initialValue = 1f,
            targetValue = if (shown.pulse) 1.14f else 1f,
            animationSpec = infiniteRepeatable(tween(620), RepeatMode.Reverse),
            label = "flip-icon-scale",
        )
        Surface(
            color = LanternNavy.copy(alpha = 0.95f),
            contentColor = Color.White,
            shape = RoundedCornerShape(24.dp),
            shadowElevation = 12.dp,
            modifier = Modifier
                .widthIn(max = 420.dp)
                .then(
                    if (manualAdvanceable) {
                        Modifier
                            .clip(RoundedCornerShape(24.dp))
                            .clickable(
                                role = Role.Button,
                                onClickLabel = manualCd,
                                onClick = onManualReady,
                            )
                    } else {
                        Modifier
                    },
                )
                .semantics {
                    liveRegion = LiveRegionMode.Assertive
                    contentDescription =
                        "${shown.title}. ${shown.body}" +
                        if (manualAdvanceable) ". $manualCd" else ""
                },
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.padding(20.dp),
            ) {
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .size(48.dp)
                        .scale(if (shown.pulse) iconScale else 1f)
                        .clip(CircleShape)
                        .drawBehind { drawCircle(shown.accent.copy(alpha = 0.22f)) },
                ) {
                    Icon(
                        painter = painterResource(shown.iconRes),
                        contentDescription = null,
                        tint = shown.accent,
                        modifier = Modifier.size(26.dp),
                    )
                }
                Column(modifier = Modifier.clearAndSetSemantics {}) {
                    Text(
                        text = shown.title,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = shown.body,
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color.White.copy(alpha = 0.82f),
                    )
                    if (manualAdvanceable) {
                        Text(
                            text = manualHint,
                            style = MaterialTheme.typography.labelMedium,
                            color = Color(0xFF8FC9FF),
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.padding(top = 6.dp),
                        )
                    }
                }
            }
        }
    }
}
