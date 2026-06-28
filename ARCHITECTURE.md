# Lantern Architecture

Lantern is a phone-scan-to-3D-mesh project. The current repository has three
main parts:

1. A host-side Python reconstruction pipeline.
2. An Android ARCore recorder app.
3. A Blender cleanup/export step for CAD-friendly meshes.

The long-term roadmap is on-device reconstruction with Depth Anything V2 running
through ExecuTorch/QNN on Snapdragon. The code in this repo currently proves the
host-side float pipeline and capture foundation first.

## High-Level Flow

```text
RGB frames
  -> Depth Anything V2 disparity
  -> affine scale/shift using metric ARCore/7-Scenes depth
  -> calibrated metric depth
  -> Open3D TSDF fusion
  -> raw GLB mesh
  -> optional Blender cleanup for watertight/CAD output
```

## Main Folders

```text
lantern/
  app/                 Android ARCore recorder app
  Depth-Anything-V2/   Local Depth Anything V2 clone
  frames/              Input RGB frames, usually *.color.png
  arcore/              Metric depth PNGs and camera poses
  disparities/         DA-V2 disparity PNGs and disp_maxes.json
  output/              Generated mesh outputs
  tests/               Pytest-style unit tests for Python helpers
```

`frames/`, `disparities/`, and `output/` are generated or data folders, not core
source. They are ignored by git. The current local workspace also has `arcore/`
7-Scenes-style depth/pose fixtures.

## Python Reconstruction Pipeline

The Python pipeline is the part that already runs end-to-end on the local
7-Scenes chess sample.

### `depth_model.py`

Runs Depth Anything V2 on RGB frames.

Input:

```text
frames/*.color.png
Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
```

Output:

```text
disparities/*.disp.png
disparities/disp_maxes.json
```

The `.disp.png` files are uint16 normalized disparity images. The original float
range is recovered later using `disp_maxes.json`.

Typical command:

```sh
./.venv/bin/python depth_model.py --frames frames --output disparities
```

### `scale_solver.py`

Calibrates DA-V2 relative disparity into metric disparity.

It fits:

```text
metric_disp ~= s * pred_disp + t
```

where:

```text
metric_disp = 1 / depth_in_meters
```

Important details:

- Uses valid metric depth samples only.
- Rejects invalid depth `0` and saturated depth `65535`.
- Uses SciPy Huber loss to reduce edge/outlier influence.
- Includes a RANSAC fallback.

### `pipeline_float.py`

The end-to-end host runner.

It:

1. Loads RGB frames, DA-V2 disparities, metric depth, camera poses, and intrinsics.
2. Reconstructs float disparity from each uint16 `.disp.png`.
3. Solves per-frame `(s, t)` with `scale_solver.py`.
4. Uses median `s` across frames as the global scale.
5. Smooths per-frame `t` with a trailing 5-frame moving average.
6. Writes calibrated metric depth PNGs into a temporary folder.
7. Calls `tsdf_fuse.py`.
8. Writes `output/mesh.glb`.

Typical command:

```sh
./.venv/bin/python pipeline_float.py
```

### `tsdf_fuse.py`

Fuses calibrated RGB-D frames into a mesh with Open3D.

It:

- Loads sorted `*.pose.txt` files as 4x4 camera-to-world transforms.
- Inverts poses for Open3D world-to-camera extrinsics.
- Integrates RGB-D frames into `ScalableTSDFVolume`.
- Extracts a triangle mesh.
- Keeps the largest connected triangle component.
- Converts the Open3D mesh to `trimesh`.
- Exports `.glb`.

## Android Recorder App

The Android app lives in `app/` and records ARCore data for an offline pipeline.
It is not currently a live reconstruction app.

Important files:

```text
app/src/main/java/com/lantern/recorder/MainActivity.kt
app/src/main/java/com/lantern/recorder/recording/FrameRecorder.kt
app/src/main/java/com/lantern/recorder/recording/PngWriter.kt
app/src/main/java/com/lantern/recorder/rendering/
```

What it does now:

- Starts an ARCore session.
- Renders the live camera feed with OpenGL.
- Checks whether `RAW_DEPTH_ONLY` is supported.
- Records frames when the camera moves at least about 3 cm.
- Saves RGB, raw depth, confidence, pose, intrinsics, and timestamps.

Android session output format:

```text
sessions/session_<timestamp>/
  frame_0001.png
  depth_0001.png
  conf_0001.png
  frame_0001.json
```

This is not the same layout expected by `pipeline_float.py`. A converter from
Android recorder sessions to the host pipeline layout is still needed.

## Blender Cleanup

`import_and_clean.py` is a host-side Blender script for post-processing a raw
mesh.

It:

- Imports `.glb`, `.gltf`, `.obj`, `.ply`, or `.stl`.
- Joins mesh parts.
- Optionally voxel-remeshes for watertightness.
- Recomputes consistent normals.
- Scales meters to millimeters by default.
- Exports a cleaned `.glb`.

Typical command:

```sh
blender --background --python import_and_clean.py -- output/mesh.glb output/mesh_clean.glb
```

Related validation scripts:

```text
test_harness.sh
orientation_test.sh
```

## Data Contracts

### Host Pipeline Input Layout

```text
frames/
  frame-000000.color.png

arcore/
  frame-000000.depth.png
  frame-000000.pose.txt
  intrinsics.txt              optional

disparities/
  frame-000000.disp.png
  disp_maxes.json
```

Files are matched by sorted index across folders.

### Depth and Pose Conventions

- Depth PNGs are uint16 millimeters.
- `0` means invalid depth.
- `65535` is treated as saturated/invalid.
- Pose files are 4x4 camera-to-world matrices.
- Open3D integration receives the inverse world-to-camera matrix.
- Mesh units are meters before Blender cleanup.

### Disparity Encoding

`depth_model.py` writes:

```text
encoded_disp = disp / disp.max() * 65535
```

`pipeline_float.py` reads:

```text
pred_disp = encoded_disp / 65535 * disp_maxes[filename]
```

The `disp_maxes.json` key must be the output disparity filename, such as:

```text
frame-000000.disp.png
```

## Dependency Areas

Python reconstruction:

- `numpy`
- `scipy`
- `opencv-python`
- `open3d`
- `trimesh`
- `torch`
- `torchvision`

Android app:

- Gradle
- Kotlin
- Android SDK 35
- ARCore

Blender cleanup:

- Blender Python runtime

## Current Architecture Boundary

The biggest boundary is between the Android recorder and the Python pipeline.

The Android app records a real ARCore session, but the Python pipeline currently
expects a 7-Scenes-style folder layout. To connect the two, the repo needs a
session converter that:

- Reads each `frame_NNNN.json`.
- Converts the column-major ARCore pose matrix into a 4x4 `.pose.txt`.
- Writes RGB as `*.color.png`.
- Writes raw depth as `*.depth.png`.
- Writes or rescales intrinsics.
- Handles RGB/depth resolution differences.

