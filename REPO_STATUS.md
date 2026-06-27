# Repository Status

This is the current post-merge state of the Lantern repo.

## Short Version

The repo now has a working host-side reconstruction path:

```text
RGB frames -> DA-V2 disparity -> affine metric calibration -> TSDF fusion -> output/mesh.glb
```

It also has:

- An Android ARCore recorder app.
- A Blender cleanup script for making meshes more CAD-friendly.
- Unit tests for the Python helper code.
- Manual verification docs.

The main missing piece is the bridge from Android recorder sessions into the
Python pipeline layout.

## What Works Now

### Host Float Pipeline

These files form the current working reconstruction pipeline:

```text
depth_model.py
scale_solver.py
pipeline_float.py
tsdf_fuse.py
```

Current capabilities:

- Run Depth Anything V2 locally on `frames/*.color.png`.
- Save pipeline-compatible disparity maps to `disparities/`.
- Recover metric scale/shift from sparse metric depth.
- Fuse calibrated depth frames into a colored `.glb` mesh.
- Run on the included/local 50-frame 7-Scenes chess-style sample.

Known local output:

```text
output/mesh.glb
output/mesh_preview.png
```

Observed mesh quality from the local run:

- Recognizable partial room/chess-scene reconstruction.
- Vertex colors are present.
- The mesh is one connected component.
- It is not watertight.
- Some surfaces are sparse, wispy, or incomplete.
- This is expected for the current short-sequence TSDF fusion output.

### Android Recorder

The Android app is present under `app/`.

Current capabilities:

- ARCore camera session.
- Live camera preview.
- `RAW_DEPTH_ONLY` support check.
- Record/stop UI.
- Translation-gated capture.
- Saves RGB, raw depth, confidence, pose, intrinsics, and timestamps.

Current limitation:

- The app output format is not directly consumed by `pipeline_float.py` yet.

### Blender Cleanup

`import_and_clean.py` exists for post-processing raw meshes.

Current capabilities:

- Imports raw mesh files.
- Joins mesh parts.
- Voxel-remeshes for watertightness.
- Fixes normals.
- Scales from meters to millimeters.
- Exports cleaned `.glb`.

Validation scripts:

```text
test_harness.sh
orientation_test.sh
```

## Important Files

### Reconstruction Scripts

| File | Purpose |
|---|---|
| `depth_model.py` | Runs Depth Anything V2 and writes `*.disp.png` plus `disp_maxes.json`. |
| `scale_solver.py` | Fits affine scale/shift from relative disparity to metric disparity. |
| `pipeline_float.py` | Runs calibration and TSDF fusion end-to-end. |
| `tsdf_fuse.py` | Uses Open3D to fuse RGB-D frames into a `.glb` mesh. |
| `import_and_clean.py` | Blender cleanup and CAD-scale export. |

### Android Files

| File | Purpose |
|---|---|
| `app/src/main/java/com/lantern/recorder/MainActivity.kt` | ARCore lifecycle, camera preview, depth support, record button. |
| `app/src/main/java/com/lantern/recorder/recording/FrameRecorder.kt` | Captures RGB, depth, confidence, pose, and intrinsics. |
| `app/src/main/java/com/lantern/recorder/recording/PngWriter.kt` | Writes 8-bit and 16-bit grayscale PNGs losslessly. |
| `app/src/main/java/com/lantern/recorder/rendering/` | OpenGL camera background rendering. |

### Docs

| File | Purpose |
|---|---|
| `README.md` | Product vision and Blender cleanup summary. |
| `roadmap.md` | Longer hackathon execution plan and risk register. |
| `AGENT.md` | Android recorder developer notes. |
| `FEATURES.md` | Feature explanation for the Python mesh pipeline. |
| `VERIFY.md` | Manual verification checklist. |
| `ARCHITECTURE.md` | System architecture and data contracts. |
| `REPO_STATUS.md` | Current post-merge status snapshot. |

### Tests

| File | Purpose |
|---|---|
| `tests/test_scale_solver.py` | Tests affine calibration and RANSAC behavior. |
| `tests/test_tsdf_fuse.py` | Tests pose loading and TSDF helper behavior. |
| `tests/test_pipeline_float.py` | Tests intrinsics parsing, smoothing, disparity loading, and depth conversion. |
| `tests/conftest.py` | Test import helpers and lightweight fakes for heavy modules. |

## Generated or Data Folders

These folders are data/output, not core source:

```text
frames/
arcore/
disparities/
output/
.venv/
app/build/
Depth-Anything-V2/checkpoints/
```

Notes:

- `frames/` contains RGB input frames.
- `arcore/` contains metric depth and poses for the sample data.
- `disparities/` contains Depth Anything V2 output.
- `output/` contains generated mesh artifacts.
- `.venv/` is the local Python environment.
- `Depth-Anything-V2/checkpoints/` should hold downloaded model weights and is
  ignored by git.

## How To Run The Current Pipeline

Generate disparities:

```sh
./.venv/bin/python depth_model.py --frames frames --output disparities
```

Run calibration and TSDF fusion:

```sh
./.venv/bin/python pipeline_float.py
```

Expected final output:

```text
output/mesh.glb
```

Optional Blender cleanup:

```sh
blender --background --python import_and_clean.py -- output/mesh.glb output/mesh_clean.glb
```

## How To Verify

Python syntax check:

```sh
python3 -m py_compile depth_model.py scale_solver.py pipeline_float.py tsdf_fuse.py
```

Scale solver self-test:

```sh
python3 scale_solver.py
```

Pytest suite:

```sh
python3 -m pytest
```

Blender validation:

```sh
./test_harness.sh
./orientation_test.sh
```

Android build check:

```sh
./gradlew :app:assembleDebug
```

## Known Gaps

### Android Session Converter Missing

The Android app writes:

```text
frame_0001.png
depth_0001.png
conf_0001.png
frame_0001.json
```

The Python pipeline expects:

```text
frame-000000.color.png
frame-000000.depth.png
frame-000000.pose.txt
intrinsics.txt
```

A converter script is needed.

### Depth and RGB Resolution Handling

`AGENT.md` notes that Android RGB intrinsics and depth image resolution can
differ. The pipeline needs explicit intrinsics rescaling before Android captures
are used directly.

### Confidence Maps Are Not Used Yet

The Android app records confidence PNGs, but the Python calibration currently
uses only:

```text
depth > 0 and depth < 65535
```

Confidence-aware filtering would likely improve fusion.

### On-Device Inference Is Not Implemented Yet

The roadmap targets ExecuTorch/QNN on Snapdragon, but the repo currently does
not contain:

- `.pte` export.
- QNN lowering scripts.
- PT2E quantization scripts.
- Android ExecuTorch runtime integration.

Depth inference is currently host-side through `depth_model.py`.

### Blender Cleanup Is Manual

`pipeline_float.py` writes the raw TSDF mesh. It does not automatically call
`import_and_clean.py`.

### Dependency Manifest Is Missing

The repo does not currently have a root `requirements.txt` or `pyproject.toml`
for the Python pipeline dependencies.

## Next Best Tasks

1. Write an Android-session-to-pipeline converter.
2. Add explicit intrinsics rescaling for RGB/depth resolution mismatch.
3. Use ARCore confidence maps in calibration and/or TSDF weighting.
4. Add a root Python dependency file.
5. Wire optional Blender cleanup into the pipeline or document it as a required
   final export step.
6. Start the ExecuTorch/QNN export path once the float pipeline is stable.

