package com.lantern.recorder.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.lantern.recorder.R
import com.lantern.recorder.ui.theme.LanternNavy
import com.lantern.recorder.ui.theme.RecordRed

/** One coaching rule: a small icon, a short title, and a one-line explanation. */
private data class CoachRule(val iconRes: Int, val titleRes: Int, val descRes: Int, val critical: Boolean = false)

private val coachRules = listOf(
    // The reconstruction-correctness rule comes first and is visually emphasized.
    CoachRule(R.drawable.ic_surface, R.string.coach_rule_surface_title, R.string.coach_rule_surface_desc, critical = true),
    CoachRule(R.drawable.ic_orbit, R.string.coach_rule_orbit_title, R.string.coach_rule_orbit_desc),
    CoachRule(R.drawable.ic_no_spin, R.string.coach_rule_no_spin_title, R.string.coach_rule_no_spin_desc),
    CoachRule(R.drawable.ic_distance, R.string.coach_rule_distance_title, R.string.coach_rule_distance_desc),
    CoachRule(R.drawable.ic_cube, R.string.coach_rule_sides_title, R.string.coach_rule_sides_desc),
    CoachRule(R.drawable.ic_light, R.string.coach_rule_light_title, R.string.coach_rule_light_desc),
    CoachRule(R.drawable.ic_hand, R.string.coach_rule_hands_title, R.string.coach_rule_hands_desc),
)

/**
 * The scanning coach overlay (Deliverable 1). Shown once on first run and re-openable
 * any time from the help affordance. It lists the core capture rules — each a short
 * line with a small icon — and leads with the "place it on a surface, don't hold or
 * rotate it" correctness guardrail. Dismissed via [onDismiss].
 *
 * Rendered as a full-screen scrim + card over the live feed; copy lives in `strings.xml`.
 */
@Composable
fun CoachOverlay(
    visible: Boolean,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    AnimatedVisibility(
        visible = visible,
        enter = fadeIn(tween(200)),
        exit = fadeOut(tween(160)),
        modifier = modifier,
    ) {
        // Full-screen contrast scrim. Tapping it dismisses (without a ripple).
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.72f))
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null,
                    onClick = onDismiss,
                ),
        ) {
            AnimatedVisibility(
                visible = visible,
                enter = scaleIn(tween(220), initialScale = 0.92f) + fadeIn(tween(220)),
                exit = scaleOut(tween(160), targetScale = 0.92f) + fadeOut(tween(120)),
            ) {
                Surface(
                    color = LanternNavy,
                    contentColor = Color.White,
                    shape = RoundedCornerShape(28.dp),
                    shadowElevation = 12.dp,
                    modifier = Modifier
                        .windowInsetsPadding(WindowInsets.safeDrawing)
                        .padding(24.dp)
                        .widthIn(max = 460.dp)
                        // Consume the tap so it doesn't fall through to the dismiss scrim.
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null,
                            onClick = {},
                        ),
                ) {
                    Column(
                        modifier = Modifier
                            .verticalScroll(rememberScrollState())
                            .padding(24.dp),
                    ) {
                        Text(
                            text = stringResource(R.string.coach_title),
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = stringResource(R.string.coach_subtitle),
                            style = MaterialTheme.typography.bodyMedium,
                            color = Color.White.copy(alpha = 0.72f),
                            modifier = Modifier.padding(top = 4.dp, bottom = 20.dp),
                        )

                        coachRules.forEach { rule ->
                            CoachRuleRow(rule)
                        }

                        Button(
                            onClick = onDismiss,
                            shape = RoundedCornerShape(16.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 20.dp),
                        ) {
                            Text(
                                text = stringResource(R.string.coach_dismiss),
                                style = MaterialTheme.typography.titleMedium,
                                modifier = Modifier.padding(vertical = 4.dp),
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun CoachRuleRow(rule: CoachRule) {
    val title = stringResource(rule.titleRes)
    val desc = stringResource(rule.descRes)
    Row(
        verticalAlignment = Alignment.Top,
        horizontalArrangement = Arrangement.spacedBy(14.dp),
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
            .clearAndSetSemantics { contentDescription = "$title. $desc" },
    ) {
        // Critical rule gets the record-red badge; the rest a calm navy tint.
        val badge = if (rule.critical) RecordRed else Color.White.copy(alpha = 0.14f)
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(badge),
        ) {
            Icon(
                painter = painterResource(rule.iconRes),
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(22.dp),
            )
        }
        Column(modifier = Modifier.padding(top = 1.dp)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = desc,
                style = MaterialTheme.typography.bodySmall,
                color = Color.White.copy(alpha = 0.72f),
            )
        }
    }
}
