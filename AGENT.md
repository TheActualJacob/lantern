# Lantern Recorder — Agent / Developer Notes

ARCore raw-depth **capture recorder** for Galaxy S25 (Snapdragon 8 Elite / SM8750).
This app records RGB + pose + intrinsics + raw depth + confidence to disk for an
**offline** reconstruction pipeline (see `roadmap.md`). It is a recorder, **not** a
live-inference app.

## Current status

**Done:**
1. ARCore session + live camera feed + `RAW_DEPTH_ONLY` support confirmation.
2. **Translation-gated recording** (see below).

Verified to **compile** via headless `./gradlew :app:assembleDebug` (Studio's JBR 21
+ SDK 35). Not yet verified on-device — needs the S25.

Not yet implemented: replay/loader for recorded sessions, sync/pose validator,
round-trip projection test (roadmap Phase P0b gate).

## Recording

Tap **Record** to start; tap **Stop** to end. A new frame is saved whenever the
camera has **translated ≥ 3 cm** (`TRANSLATION_THRESHOLD_M` in `FrameRecorder`)
since the last saved frame (first tracked frame always saves). Recording is only
enabled when `RAW_DEPTH_ONLY` is supported, and auto-stops on `onPause`.

- **Output:** `getExternalFilesDir(null)/sessions/session_<yyyyMMdd_HHmmss>/`
  (app-specific external storage, no runtime storage permission needed). Pull with
  `adb pull /sdcard/Android/data/com.lantern.recorder/files/sessions`.
- **Per saved frame (NNNN = 0001…):**
  | File | Contents |
  |------|----------|
  | `frame_NNNN.png` | RGB, 8-bit (YUV_420_888 → ARGB → PNG, BT.601) |
  | `depth_NNNN.png` | raw depth, **16-bit grayscale, millimeters** (`acquireRawDepthImage16Bits`) |
  | `conf_NNNN.png`  | raw depth confidence, 8-bit grayscale (`acquireRawDepthConfidenceImage`) |
  | `frame_NNNN.json`| pose 4×4 (column-major, camera→world) + translation + quaternion, intrinsics (fx, fy, cx, cy, w, h), timestamps, image dims |
- Depth/confidence use a hand-rolled `PngWriter` because `Bitmap.compress` can't emit
  16-bit grayscale; depth must stay lossless for the offline pipeline. Read in Python
  with PIL (`Image.open(...)` → mode `I;16` for depth) or `cv2.IMREAD_UNCHANGED`.
- All ARCore images are acquired inside `use {}` (try-with-resources) on the GL
  thread so native buffers release immediately; pixel data is copied out and
  encoding + disk I/O run on a single background executor.

> NOTE on intrinsics vs depth resolution: saved intrinsics are for the **RGB** image
> (`camera.imageIntrinsics`). The depth/confidence maps are lower-res; the offline
> pipeline must rescale intrinsics by the resolution ratio (roadmap Phase 1, Day 1).

## How to verify this slice

1. Open the project root in Android Studio and let it Gradle-sync.
2. Connect the S25 (USB debugging on) and press **Run**.
3. Grant the camera permission when prompted; accept any ARCore install/update
   prompt ("Google Play Services for AR").
4. Expect: a live camera feed fills the screen, and a status overlay reads
   `RAW_DEPTH_ONLY supported: true`.
5. Confirm in Logcat (filter tag **`LANTERN`**):
   `RAW_DEPTH_ONLY supported = true`

## Key decisions baked in

- **Package / applicationId:** `com.lantern.recorder`
- **App label:** "Lantern Recorder"
- **AR Required app** (`com.google.ar.core` = `required`, `android.hardware.camera.ar`
  required) — Play Store only installs it on ARCore-capable devices and
  auto-provisions ARCore. First launch runs the full `ArCoreApk.requestInstall`
  prompt flow.
- **Depth mode:** `Config.DepthMode.RAW_DEPTH_ONLY` (raw per-frame depth +
  confidence, needed for offline scale recovery — see roadmap Decision 3 / Card E).
  If unsupported, depth is set to `DISABLED` and the gap is logged loudly rather
  than silently recording without depth.
- **Camera feed:** GLSurfaceView + external-OES texture (`BackgroundRenderer`),
  ARCore-supplied texcoord transform via `DisplayRotationHelper`. This GL path is
  reused when recording is added.

## Toolchain / versions

| Component        | Version | Where |
|------------------|---------|-------|
| Gradle           | 8.11.1  | `gradle/wrapper/gradle-wrapper.properties` |
| Android Gradle Plugin | 8.7.3 | `gradle/libs.versions.toml` |
| Kotlin           | 2.1.0   | `gradle/libs.versions.toml` |
| ARCore (`com.google.ar:core`) | 1.54.0 | `gradle/libs.versions.toml` |
| compileSdk / targetSdk | 35 | `app/build.gradle.kts` |
| minSdk           | 24 (ARCore minimum) | `app/build.gradle.kts` |
| Java/Kotlin JVM target | 17 | `app/build.gradle.kts` |

Bump versions in `gradle/libs.versions.toml` (single source of truth).

## Getting data off the device + converting to pipeline input

The recorder writes to app-private external storage (no on-device viewer — it's a
capture harness). Pull a session and convert it to the float-pipeline layout:

```sh
adb pull /sdcard/Android/data/com.lantern.recorder/files/sessions ./sessions
python3 convert_session.py ./sessions/session_<stamp>     # → ./session_<stamp>_dataset/{frames,arcore}/
```

`convert_session.py` bridges the recorder's flat session (`frame_NNNN.png`,
`depth_NNNN.png`, `conf_NNNN.png`, `frame_NNNN.json`) to what `pipeline_float.py`
expects (`frames/<base>.color.png`, `arcore/<base>.depth.png`, `<base>.pose.txt`,
`intrinsics.txt`). It transposes the pose from ARCore's column-major (OpenGL)
matrix to the row-major form `np.loadtxt` reads, and copies PNGs byte-for-byte so
16-bit depth stays lossless. Emitted `intrinsics.txt` is the **color-image**
intrinsics; the disparity/calibration step (the AI part) owns resizing the
lower-res raw depth to the disparity resolution (`solve_affine` needs matching
shapes). The DA-V2 `disparities/` input is produced separately and is out of scope
for this recorder.

## Prerequisites to build/run

- **Android Studio** (recent; bundles a JDK 17 and SDK manager). The system JDK on
  the original dev host was 1.8 — too old for AGP 8.7; use Studio's bundled JDK
  (Settings → Build Tools → Gradle → Gradle JDK = 17) or install JDK 17.
- **Android SDK Platform 35** + Build-Tools (install via SDK Manager).
- `local.properties` with `sdk.dir=...` — Android Studio generates this on first
  open; it is git-ignored. For CLI builds, create it manually pointing at your SDK.
- A physical **ARCore-supported device** (the S25). AR cannot run on a stock
  emulator without the AR-capable emulator image.

> NOTE: This project was scaffolded on a host with no Android SDK/Studio and only
> JDK 1.8, so it has **not** been compiled or run here. First build + on-device
> verification happens in Android Studio on the S25.

## Source map

```
app/src/main/AndroidManifest.xml                      perms, AR-required meta-data, launcher activity
app/src/main/java/com/lantern/recorder/
  MainActivity.kt                                     session lifecycle, install/permission flow,
                                                      RAW_DEPTH_ONLY check + log, GL render loop,
                                                      record button wiring
  CameraPermissionHelper.kt                           runtime CAMERA permission
  recording/FrameRecorder.kt                          translation gating, image acquire/copy,
                                                      pose+intrinsics, YUV→ARGB, JSON, IO offload
  recording/PngWriter.kt                              lossless 8/16-bit grayscale PNG encoder
  rendering/BackgroundRenderer.kt                     camera feed via external-OES texture quad
  rendering/DisplayRotationHelper.kt                  pushes display rotation into the session
  rendering/ShaderUtil.kt                             GLSL compile/link + GL error logging
app/src/main/res/layout/activity_main.xml             GLSurfaceView + status overlay + record button
```
