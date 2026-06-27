# Manual Verification

Use these checks after finishing changes to the scale calibration, TSDF fusion, or
end-to-end floating pipeline scripts.

## Automated Tests

Run the lightweight regression suite:

```sh
python3 -m pytest
```

If `pytest` is not installed in the active environment, install or activate the
project environment that provides it, then rerun the command.

Run syntax checks for the standalone scripts:

```sh
python3 -m py_compile scale_solver.py tsdf_fuse.py pipeline_float.py
```

## `scale_solver.py`

Run the built-in synthetic self-test:

```sh
python3 scale_solver.py
```

Expected result: the script prints recovered `s` and `t` values close to the
hard-coded synthetic values and exits without an assertion failure.

## `tsdf_fuse.py`

Check import and syntax:

```sh
python3 -m py_compile tsdf_fuse.py
python3 - <<'PY'
from tsdf_fuse import fuse, load_poses
print("tsdf_fuse import ok")
PY
```

Manual fusion is intended to be called from Python with aligned color, depth,
pose, and intrinsics inputs. Example:

```sh
python3 - <<'PY'
from pathlib import Path
from tsdf_fuse import fuse, load_poses

root = Path(".")
depth_paths = sorted(str(p) for p in (root / "arcore").glob("*.depth.png"))
poses = load_poses(str(root / "arcore"))
intrinsics = {
    "fx": 500.0,
    "fy": 500.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480,
}
fuse(depth_paths, poses, intrinsics, output_path="output/mesh.glb")
PY
```

Expected result: the command completes and writes `output/mesh.glb`. Replace the
sample intrinsics with values from your capture, such as `arcore/intrinsics.txt`
when available.

## `pipeline_float.py`

Run the end-to-end pipeline against the default folders:

```sh
python3 pipeline_float.py
```

Or provide explicit folders:

```sh
python3 pipeline_float.py \
  --frames frames \
  --disparities disparities \
  --arcore arcore \
  --output output/mesh.glb
```

Expected result: the script prints per-frame `s`, `t`, and
`mean_abs_residual` diagnostics, then prints `global median s=...` and
`Wrote mesh to .../output/mesh.glb`. The output mesh should exist at
`output/mesh.glb`.
