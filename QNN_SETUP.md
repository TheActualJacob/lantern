# QNN on-device setup — wiring the QAIRT SDK so the app uses the Hexagon NPU

> ## STATUS (2026-06-27) — wired & building; blocked on SDK version
> Everything below is **done and verified on the connected S25** EXCEPT the SDK version:
> - ✅ `local.properties` → `qnn.sdkRoot` points at the SDK; Gradle reads it (`app/build.gradle.kts`).
> - ✅ NDK `27.0.12077973` + CMake `3.22.1` installed; **`libqnn_depth.so` compiles** against the
>   real 2.44 headers and is bundled with the 5 Qualcomm `.so`s in the debug APK.
> - ✅ APK installed, `depth_anything_v3.dlc` pushed to the app's files dir, app finds & loads it.
> - ✅ At runtime the native lib dlopens the QNN runtime and reads the model — gets all the way to
>   DLC parse.
> - ❌ **BLOCKER: version mismatch.** Installed SDK is **2.44.0**; the AI Hub DLC was exported with
>   **2.45.0** (`metadata.json` / DLC `converterVersion: 2.45.0.260326154327`). 2.44's DLC reader
>   rejects it: `Failed to create dlc handle with code 1002` → `ComposeGraphs Failed`. On-device
>   logcat (`LANTERN_QNN`) shows `systemContextGetBinaryInfo failed` → falls back to ARCore.
>
> **Fix = match the versions, then re-run:**
> 1. Download **QAIRT 2.45.0** (same minor as the DLC), unzip, update `qnn.sdkRoot` in
>    `local.properties`, and re-copy the 5 `.so`s into `app/src/main/jniLibs/arm64-v8a/` from the
>    2.45 SDK (see step 3). *(Alternative: re-export the DLC from AI Hub targeting QAIRT 2.44.)*
> 2. The raw `.dlc` still isn't a context binary, so convert it once (the on-device generator is
>    already staged at `/data/local/tmp/qnn`) — re-push the 2.45 libs there first:
>    ```
>    adb shell "cd /data/local/tmp/qnn && LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=dsp \
>      ./qnn-context-binary-generator --backend lib/libQnnHtp.so --model lib/libQnnModelDlc.so \
>      --dlc_path depth_anything_v3.dlc --binary_file da3_qnn_v79 --output_dir output"
>    adb push <pull output/da3_qnn_v79.bin> /sdcard/Android/data/com.lantern.recorder/files/da3_qnn_v79.bin
>    ```
>    The app prefers `da3_qnn_v79.bin` over the `.dlc`, and `qnn_depth.cpp`'s
>    `contextCreateFromBinary` path is built for exactly that binary.
> 3. Relaunch → live-mesh chip should read **`QNN NPU`** (green).
>
> If `contextCreateFromBinary` then fails on a struct-version assert, flip the `graphInfoV1` /
> `Qnn_Tensor_t.v1` branches in `qnn_depth.cpp` to the V2 variants (the other "known unknown").

---


You downloaded the QAIRT/QNN SDK zip. This is the checklist to turn the gated QNN depth backend
on so a plain **Run** in Android Studio uses the AI Hub `depth_anything_v3.dlc` on the S25's
Hexagon-v79 NPU. The code is already implemented and gated off (see `QNN_RUNTIME_PLAN.md`); this
just supplies the SDK pieces it needs.

**Context for a fresh session:** the architecture is done — `recon/DepthBackend.kt`,
`recon/QnnDlcDepthModel.kt`, `app/src/main/cpp/qnn_depth.cpp`, `app/src/main/cpp/CMakeLists.txt`,
and the Gradle gate in `app/build.gradle.kts` (property `qnn.sdkRoot` / env `QNN_SDK_ROOT`).
Everything below is wiring + assets, not new feature code. The native QNN code has **never been
run on-device**, so step 6 (verify/debug) is the real unknown.

---

## 0. Prereqs

- Android Studio with the **NDK** and **CMake 3.22.1** installed:
  *SDK Manager → SDK Tools → check "NDK (Side by side)" and "CMake"*. (CMake version is pinned in
  `app/build.gradle.kts`.)
- The S25 connected, `adb devices` shows it.
- The DLC model present locally at `depth_anything_v3-qnn_dlc-float/depth_anything_v3.dlc`
  (it's gitignored — 143 MB — but should still be on disk).

## 1. Unzip the SDK

Unzip somewhere stable, e.g.:

```
~/qairt/2.45.0.xxxxxxxx/
```

The folder that contains `include/`, `lib/`, `bin/` is your **`QNN_SDK_ROOT`**. Confirm:

```
ls $QNN_SDK_ROOT/include/QNN/QnnInterface.h          # headers exist
ls $QNN_SDK_ROOT/lib/aarch64-android/                  # android arm64 .so's
ls $QNN_SDK_ROOT/lib/hexagon-v79/unsigned/             # v79 skel
```

(If the arch dir isn't `hexagon-v79`, list `$QNN_SDK_ROOT/lib/` and find the `hexagon-v79*` one.
SM8750 = Hexagon **v79** — the skel arch MUST match or it silently runs nothing.)

## 2. Point Gradle at the SDK

Add one line to **`gradle.properties`** (repo root). After this, plain Run enables the native
build automatically:

```
qnn.sdkRoot=/Users/jacob/qairt/2.45.0.xxxxxxxx
```

(Use your real path. Alternatively `export QNN_SDK_ROOT=...` in the env Studio launches from.)

## 3. Copy the 5 runtime `.so`s into the app

These are NOT in git (Qualcomm licensing). Copy from the SDK into
`app/src/main/jniLibs/arm64-v8a/` (see that dir's `README.md` for the canonical list):

```
SDK=$QNN_SDK_ROOT
DST=app/src/main/jniLibs/arm64-v8a
cp $SDK/lib/aarch64-android/libQnnHtp.so          $DST/
cp $SDK/lib/aarch64-android/libQnnSystem.so       $DST/
cp $SDK/lib/aarch64-android/libQnnHtpPrepare.so   $DST/
cp $SDK/lib/aarch64-android/libQnnHtpV79Stub.so   $DST/
cp $SDK/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so $DST/
```

Verify 5 `.so`s are present:

```
ls -1 app/src/main/jniLibs/arm64-v8a/*.so
```

## 4. Push the model to the phone

The app looks in its external files dir (`MainActivity.resolveQnnModelPath()`), preferring a
pre-compiled context binary `da3_qnn_v79.bin`, then the raw `depth_anything_v3.dlc`. Easiest is
to push the DLC:

```
adb push depth_anything_v3-qnn_dlc-float/depth_anything_v3.dlc \
  /sdcard/Android/data/com.lantern.recorder/files/
```

*(Optional, faster init later: generate an HTP context binary offline with the SDK's
`qnn-context-binary-generator` and push it as `da3_qnn_v79.bin` instead. Not required for
first bring-up.)*

## 5. Build & Run

Press **Run** in Android Studio (or `./gradlew :app:assembleDebug` with `JAVA_HOME` = Studio's
JBR 21). With `qnn.sdkRoot` set, Gradle now compiles `libqnn_depth.so` via CMake and bundles the
`.so`s. Switch the app to **Live mesh** mode.

## 6. Verify it's actually on the NPU

- **On-screen:** the live-mesh readout chip should say **`QNN NPU`** with a green dot (not blue
  `ARCore depth`). Blue = QNN didn't load and it fell back.
- **Logcat** (filter tag `LANTERN` and `LANTERN_QNN`):
  - `Loaded DA3 QNN model: ...`  → success.
  - `libqnn_depth not loaded` → step 3/NDK problem (native lib didn't build/bundle).
  - `QNN model not present` → step 4 (model not pushed to the right path).
  - `QNN nativeInit failed` / `graphRetrieve failed` / `contextCreateFromBinary failed` →
    step 6 debugging (see below). App stays up and falls back; it won't crash.

## Troubleshooting / known unknowns

The native bridge (`app/src/main/cpp/qnn_depth.cpp`) was written against the QAIRT 2.45 C API but
never executed. Likely first-run friction points:

- **`.dlc` vs context binary:** `contextCreateFromBinary` expects an HTP **context binary**, not
  necessarily a raw `.dlc`. If init fails on the raw DLC, convert it offline first:
  `qnn-context-binary-generator` (in `$QNN_SDK_ROOT/bin/`) → push the resulting `.bin` as
  `da3_qnn_v79.bin`. This is the most probable fix.
- **Tensor struct versions:** the `graphInfoV1` / `Qnn_Tensor_t.v1` accessors in the cpp assume
  the V1 variants; if the SDK reports V2, adjust the `version` branches in `inspectBinary` and the
  `in.v1` / `out.v1` field access in `nativeInfer`.
- **Skel mismatch:** wrong Hexagon arch = silent no-op/garbage. Re-check step 1's `hexagon-v79`.
- **NDK version:** if CMake/NDK errors, set an explicit `ndkVersion = "26.x.x"` in
  `app/build.gradle.kts android { }` matching an installed NDK.

## Separate issue (not QNN): the "mountain" mesh

The noisy blue mountain blobs are the **ARCore-only** live mesh fusing raw room depth with no
object isolation — a `LiveReconstructor` / `TsdfVolume` problem, independent of the depth backend.
Getting QNN working gives denser depth but won't by itself remove background geometry. Track that
fix separately.

---

### One-line summary for next session
SDK is downloaded. Do: set `qnn.sdkRoot` in `gradle.properties`, copy 5 `.so`s into
`app/src/main/jniLibs/arm64-v8a/`, `adb push` the `.dlc` to the app's files dir, install NDK+CMake,
Run, then watch logcat `LANTERN_QNN` — if `contextCreateFromBinary` fails, convert the DLC to a
context binary with `qnn-context-binary-generator` and push that instead.
