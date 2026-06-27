package com.lantern.recorder.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

/** Lantern brand navy used for the launcher background and as the M3 primary seed. */
val LanternNavy = Color(0xFF0B1F33)

/** Shutter / recording accent. Matches the legacy `record_red` resource. */
val RecordRed = Color(0xFFE53935)

/** A lighter navy used as a secondary brand tint on cards and chips. */
private val LanternNavyLight = Color(0xFF1C3A57)
private val LanternSky = Color(0xFF5FA8E0)

private val LanternLightColors = lightColorScheme(
    primary = LanternNavy,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD3E4F5),
    onPrimaryContainer = Color(0xFF071829),
    secondary = LanternNavyLight,
    onSecondary = Color.White,
    tertiary = LanternSky,
    error = RecordRed,
    onError = Color.White,
    background = Color(0xFFF7F9FC),
    onBackground = Color(0xFF111417),
    surface = Color(0xFFF7F9FC),
    onSurface = Color(0xFF111417),
    surfaceVariant = Color(0xFFE1E7EE),
    onSurfaceVariant = Color(0xFF424A52),
    outlineVariant = Color(0xFFC3CCD6),
)

private val LanternDarkColors = darkColorScheme(
    primary = LanternSky,
    onPrimary = Color(0xFF06243B),
    primaryContainer = LanternNavyLight,
    onPrimaryContainer = Color(0xFFD3E4F5),
    secondary = Color(0xFF9FC3E2),
    onSecondary = Color(0xFF0A2236),
    tertiary = LanternSky,
    error = Color(0xFFFF8A80),
    onError = Color(0xFF410002),
    background = Color(0xFF0B1119),
    onBackground = Color(0xFFE3E7EC),
    surface = Color(0xFF0E1620),
    onSurface = Color(0xFFE3E7EC),
    surfaceVariant = Color(0xFF2A323B),
    onSurfaceVariant = Color(0xFFBEC7D1),
    outlineVariant = Color(0xFF3A434D),
)

/**
 * Material 3 theme for the Lantern capture + sessions UI.
 *
 * Defaults to the fixed brand palette so the navy/red identity is consistent across
 * devices; pass [dynamicColor] = true to opt into Material You wallpaper-based color
 * on Android 12+ (the camera overlay still draws its own scrims for contrast).
 */
@Composable
fun LanternTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> LanternDarkColors
        else -> LanternLightColors
    }
    MaterialTheme(
        colorScheme = colorScheme,
        typography = MaterialTheme.typography,
        content = content,
    )
}
