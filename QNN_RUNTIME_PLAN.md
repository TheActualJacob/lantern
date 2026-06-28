# Plan: Switch on-device DA3 depth to the Qualcomm QNN runtime (AI Hub `.dlc`)

Goal: run the **AI Hub `depth_anything_v3.dlc`** directly on the S25 Hexagon NPU via the
Qualcomm **QNN / QAIRT** runtime, instead of ExecuTorch `.pte`. Reuse everything downstream
(`recon/*`, mesh, UI) by hiding QNN behind the existing depth interface.

## Why
- We already downloaded the NPU-optimized `.dlc` (`depth_anything_v3-qnn_dlc-float/`); QNN runs
  it natively — no `.pte` re-export, no torch tracing.
- ExecuTorch can't load a `.dlc`; the two are different runtimes.

## What we have (from `metadata.json`)
- Runtime `qnn_dlc`, **float**, QAIRT **2.45.0**.
- Input `image`: **NHWC `[1,518,518,3]`, float32, range `[0,1]`** (no ImageNet norm).
- Output `depth_estimates`: `[1,518,518,1]` float32, monocular depth.

## Design (keep the pipeline; swap only the depth source)
1. Introduce a `DepthBackend` interface that returns a relative `DisparityMap`
   (extract from today's `Da3DepthModel`). Two impls:
   - `ExecuTorchDepthModel` (existing `.pte` path).
   - `QnnDlcDepthModel` (new; JNI → QNN C API).
2. `LiveReconstructor` picks the backend; everything after (scale solve, TSDF, MC, render)
   is unchanged.

## Native integration (the real work)
- **SDK:** install **QAIRT / QNN SDK ≥ 2.45** (match the DLC's QAIRT version).
- **Bundle `.so`s** in `app/src/main/jniLibs/arm64-v8a/` (from the SDK):
  `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpPrepare.so`,
  `libQnnHtpV79Stub.so`, + the V79 **skel** `libQnnHtpV79Skel.so` (SM8750 = Hexagon **v79**).
- **Load path:** prefer converting the DLC to an HTP **context binary** offline with
  `qnn-context-binary-generator` (faster init, no on-device prepare), ship the `.bin`; else
  load the DLC and prepare on device.
- **JNI wrapper** (`app/src/main/cpp/qnn_depth.cpp` + `CMakeLists.txt`): init QNN HTP backend,
  create context from the binary, query graph IO, and expose
  `infer(floatInput, out)` to Kotlin. Set `CMAKE`/`externalNativeBuild` in `build.gradle.kts`.
- **Preprocess to the DLC contract:** NHWC, `[0,1]` (divide by 255, **no** ImageNet norm),
  518×518 — differs from the current ExecuTorch preprocessing; implement per-backend.
- **Postprocess:** depth `[1,518,518,1]` → `disp = 1/depth` (reuse existing convention).

## Milestones
1. **M1 – Bring-up:** JNI loads QNN, runs the DLC on a static test image, logs output stats.
   Gate: depth map is sane vs host `depth_anything_v3.py`.
2. **M2 – Wire-in:** `QnnDlcDepthModel` behind `DepthBackend`; LiveMesh uses it; ARCore
   fallback intact. Gate: live mesh builds with `QNN NPU` shown in the readout.
3. **M3 – Perf/robustness:** context-binary caching, threading off the GL thread, graceful
   fallback (QNN missing/unsupported SoC → ExecuTorch/ARCore). Gate: ~real-time, no thrash.

## File changes
- New: `recon/DepthBackend.kt`, `recon/QnnDlcDepthModel.kt`, `app/src/main/cpp/*`,
  `app/src/main/jniLibs/arm64-v8a/*.so`, model asset (`.dlc`/`.bin`).
- Edit: `LiveReconstructor.kt` (choose backend), `build.gradle.kts` (NDK/CMake +
  `jniLibs`), UI backend label.

## Risks / decisions
- **Redistribution:** confirm licensing before committing Qualcomm `.so`s; otherwise pull at
  build time or ship via a separate flavor.
- **Qualcomm-only:** QNN path is SM8750-class only — must keep ExecuTorch/ARCore fallback.
- **Skel/arch version** must match the device (v79 for SM8750); wrong skel = silent CPU/no-op.
- **Decision:** ship QNN in a dedicated product flavor (`qnn`) vs. always-bundled (APK size +
  ~tens of MB of `.so` + the DLC).

## TL;DR
Hide depth behind `DepthBackend`; add a JNI QNN runtime that loads the AI Hub DLC (NHWC,
`[0,1]`, v79 skel) and returns disparity; keep ExecuTorch/ARCore as fallback. Heaviest part is
the native QNN bring-up + bundling the right Hexagon-v79 `.so`s.
