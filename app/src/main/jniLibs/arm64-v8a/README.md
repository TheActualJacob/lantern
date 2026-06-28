# Qualcomm QNN runtime libraries (arm64-v8a)

These `.so`s are **not committed** (Qualcomm licensing — see QNN_RUNTIME_PLAN.md "Risks"). To
build the QNN Hexagon-NPU depth backend, copy them here from your QAIRT/QNN SDK
(`$QNN_SDK_ROOT`, version ≥ 2.45 to match the DLC) and build with `-Pqnn.sdkRoot=$QNN_SDK_ROOT`.

Required (from `$QNN_SDK_ROOT/lib/aarch64-android/`):

- `libQnnHtp.so`          — HTP backend (dlopen'd by qnn_depth.cpp)
- `libQnnSystem.so`       — System API used to read the context binary
- `libQnnHtpPrepare.so`   — needed if preparing a raw `.dlc` on device
- `libQnnHtpV79Stub.so`   — SM8750 = Hexagon **v79** stub

Plus the v79 **skel** from `$QNN_SDK_ROOT/lib/hexagon-v79/unsigned/`:

- `libQnnHtpV79Skel.so`   — runs on the DSP; wrong arch = silent CPU/no-op

The model asset (`depth_anything_v3.dlc`, or a `da3_qnn_v79.bin` HTP context binary produced
offline with `qnn-context-binary-generator`) is pushed to the app's external files dir at
runtime — see `MainActivity.resolveQnnModelPath()` — not bundled in the APK.
