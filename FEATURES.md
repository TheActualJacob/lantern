# Lantern Depth-to-Mesh Features

This project now has an end-to-end pipeline for turning Depth Anything V2
relative disparity predictions plus sparse 7-Scenes/ARCore metric depth into a
fused 3D mesh.

## What Was Built

### `scale_solver.py`

This file calibrates relative DA-V2 disparity into metric disparity.

Main functions:

- `solve_affine(pred_disp, metric_depth, conf_mask) -> (s, t)`
- `solve_affine_ransac(pred_disp, metric_depth, conf_mask, n_iter=100) -> (s, t)`

What it does:

- Takes DA-V2 relative disparity as `pred_disp`.
- Takes 7-Scenes depth PNG values in millimeters as `metric_depth`.
- Uses `conf_mask` to keep only valid depth pixels.
- Converts metric depth to metric disparity with:

```text
metric_disp = 1 / depth_in_meters
```

- Fits:

```text
metric_disp ~= s * pred_disp + t
```

- Uses SciPy Huber loss to reject edge bleed and other outliers.
- Falls back to RANSAC if the Huber solve fails or returns invalid values.

Manual check:

```sh
python3 scale_solver.py
```

Expected result: it prints recovered `s` and `t` values close to the synthetic
ground truth.

## `tsdf_fuse.py`

This file fuses calibrated metric depth frames into a 3D mesh.

Main functions:

- `load_poses(arcore_dir) -> list[np.ndarray]`
- `fuse(depth_paths, poses, intrinsics, voxel_size=0.01, output_path="output/mesh.glb")`

What it does:

- Loads sorted `*.pose.txt` files as 4x4 camera-to-world poses.
- Loads 16-bit depth PNGs in millimeters.
- Finds the matching color image for each depth image using the same base name
  and `.color.png`.
- Builds an Open3D `ScalableTSDFVolume`.
- Integrates each RGB-D frame into the TSDF volume.
- Converts camera-to-world poses into world-to-camera extrinsics for Open3D.
- Extracts a triangle mesh.
- Keeps the largest connected triangle component.
- Exports the mesh as `.glb` using `trimesh`.

Manual check:

```sh
python3 -m py_compile tsdf_fuse.py
```

Expected result: the file compiles without syntax errors.

## `pipeline_float.py`

This is the full runner that connects calibration and fusion.

Default input/output layout:

```text
lantern/
  frames/        *.color.png
  arcore/        *.depth.png, *.pose.txt, intrinsics.txt optional
  disparities/   *.disp.png, disp_maxes.json
  output/        mesh.glb
```

What it does:

1. Parses CLI args:

```sh
--frames
--disparities
--arcore
--output
```

2. Loads camera intrinsics from `arcore/intrinsics.txt` when present.

3. If `intrinsics.txt` is missing, uses standard 7-Scenes Kinect intrinsics
   scaled to the frame size.

4. Loads `disparities/disp_maxes.json`.

5. Matches color frames, disparity images, metric depth images, and poses by
   sorted index.

6. For each frame:

- Reads the 16-bit `.disp.png`.
- Converts it back to float disparity:

```text
pred_disp = disp_png / 65535 * disp_max
```

- Reads the 16-bit `.depth.png` in millimeters.
- Builds a confidence mask from valid metric depth:

```text
depth > 0 and depth < 65535
```

- Calls `solve_affine(...)` to estimate per-frame `s` and `t`.
- Prints calibration diagnostics:

```text
frame_name: s=..., t=..., mean_abs_residual=...
```

7. Uses the median `s` across all frames as the global scale.

8. Smooths per-frame `t` values with a trailing 5-frame moving average.

9. Converts calibrated disparity back into metric depth PNGs.

10. Copies matching color images into a temporary folder.

11. Calls `tsdf_fuse.fuse(...)`.

12. Writes the final mesh to `output/mesh.glb`.

Manual check:

```sh
./.venv/bin/python pipeline_float.py
```

Expected result:

- Per-frame calibration diagnostics are printed.
- `global median s=...` is printed.
- The script ends with:

```text
Wrote mesh to .../output/mesh.glb
```

## Generated Output

The pipeline was run successfully on the local 50-frame chess scene.

Generated files:

- `output/mesh.glb`
- `output/mesh_preview.png`

Observed mesh stats:

- Vertices: `305,298`
- Faces: `584,679`
- Connected components: `1`
- Vertex colors: present
- Boundary edges: `33,449`
- Nonmanifold edges: `2`

Visual quality:

- The mesh is a recognizable partial room/chess-scene reconstruction.
- It has visible RGB coloring.
- The surface is sparse and wispy in places.
- Walls and floor areas are incomplete.
- The mesh is connected but not watertight.
- The artifacts are consistent with partial RGB-D fusion from a short sequence.

## Automated Tests

Tests were added under `tests/`.

They cover:

- Synthetic affine recovery in `scale_solver.py`.
- Invalid and saturated depth filtering.
- Shape mismatch errors.
- RANSAC recovery with outliers.
- Pose loading and helper behavior in `tsdf_fuse.py`.
- Intrinsics parsing, smoothing, disparity recovery, and depth conversion in
  `pipeline_float.py`.

Run tests with:

```sh
python3 -m pytest
```

If `pytest` is not installed, install it into the active environment first.

Syntax check:

```sh
python3 -m py_compile scale_solver.py tsdf_fuse.py pipeline_float.py
```

