"""Convert a Lantern recorder session into the float-pipeline input layout.

The on-device recorder (Android app) writes one flat folder per session:

    session_<stamp>/
        frame_NNNN.png    RGB (8-bit)
        depth_NNNN.png    raw depth (16-bit grayscale, millimeters)
        conf_NNNN.png     raw depth confidence (8-bit grayscale)
        frame_NNNN.json   pose (4x4 column-major) + intrinsics + timestamps

`pipeline_float.py` instead expects (see FEATURES.md):

    <out>/frames/  <base>.color.png
    <out>/arcore/  <base>.depth.png
                   <base>.pose.txt      4x4 camera-to-world, row-major, whitespace text
                   intrinsics.txt       "fx=.. fy=.. cx=.. cy=.. width=.. height=.."

This converter bridges the two: it renames/splits the streams, transposes the pose
from ARCore's column-major (OpenGL) layout to the row-major matrix np.loadtxt reads,
and emits intrinsics.txt. PNGs are copied byte-for-byte, so the 16-bit depth is
preserved losslessly. The confidence map is carried along as <base>.conf.png for
later use (the current pipeline does not consume it).

Resolution note: the emitted intrinsics describe the *color* image. ARCore's raw
depth map is lower-resolution than the color image, and `solve_affine` requires the
DA-V2 disparity and the metric depth to share a shape -- so the disparity/calibration
step (the AI part, out of scope here) is responsible for resampling depth to the
disparity resolution. We hand off each stream at its faithful native resolution.

Usage:
    python3 convert_session.py path/to/session_<stamp> [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def column_major_to_row_major_rows(m: list[float]) -> list[list[float]]:
    """ARCore Pose.toMatrix() returns a column-major float[16]; element (r, c) is
    m[c*4 + r]. Return the matrix as row-major rows (what np.loadtxt expects)."""
    if len(m) != 16:
        raise ValueError(f"pose matrix must have 16 elements, got {len(m)}")
    return [[m[c * 4 + r] for c in range(4)] for r in range(4)]


def format_pose_text(rows: list[list[float]]) -> str:
    return "\n".join(" ".join(f"{v:.9e}" for v in row) for row in rows) + "\n"


def object_pose_rows(meta: dict) -> list[list[float]] | None:
    """Extract the turntable object pose ``T_CO`` (camera<-object) as 4x4 rows.

    Returns None when the session is orbit-mode or the board wasn't seen this
    frame (``object_pose_valid`` false / missing). Accepts either a nested 4x4
    list or a flat row-major 16-element list under ``object_pose_T_CO``.

    Unlike the camera pose, ``T_CO`` comes from OpenCV ``solvePnP`` and is already
    a plain row-major matrix, so it is written straight through (no transpose).
    """
    if not meta.get("object_pose_valid", False):
        return None
    raw = meta.get("object_pose_T_CO")
    if raw is None:
        return None
    if isinstance(raw[0], (list, tuple)):
        rows = [[float(v) for v in row] for row in raw]
    else:
        flat = [float(v) for v in raw]
        if len(flat) != 16:
            raise ValueError(f"object_pose_T_CO must have 16 values, got {len(flat)}")
        rows = [flat[r * 4 : r * 4 + 4] for r in range(4)]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("object_pose_T_CO must be a 4x4 matrix")
    return rows


def format_intrinsics_text(intr: dict) -> str:
    # Tokens MUST be `key=value` with no comments: pipeline_float splits on
    # whitespace and rejects any token lacking '='.
    return (
        f"fx={float(intr['fx']):.9f} fy={float(intr['fy']):.9f} "
        f"cx={float(intr['cx']):.9f} cy={float(intr['cy']):.9f} "
        f"width={int(intr['width'])} height={int(intr['height'])}\n"
    )


def frame_index(json_path: Path) -> int:
    m = re.search(r"frame_(\d+)\.json$", json_path.name)
    if not m:
        raise ValueError(f"Unexpected metadata filename: {json_path.name}")
    return int(m.group(1))


def convert(session_dir: Path, out_dir: Path) -> int:
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    metas = sorted(session_dir.glob("frame_*.json"), key=frame_index)
    if not metas:
        raise FileNotFoundError(f"No frame_*.json files in {session_dir}")

    frames_dir = out_dir / "frames"
    arcore_dir = out_dir / "arcore"
    frames_dir.mkdir(parents=True, exist_ok=True)
    arcore_dir.mkdir(parents=True, exist_ok=True)

    # Carry the per-session capture sidecar (capture mode + board spec) so the
    # host can pick the object fusion frame for turntable sessions (plan §4.1).
    capture_sidecar = session_dir / "capture.json"
    if capture_sidecar.exists():
        shutil.copyfile(capture_sidecar, arcore_dir / "capture.json")

    intrinsics_written = False
    written = 0
    skipped = 0
    object_poses_written = 0

    for out_idx, meta_path in enumerate(metas):
        idx = frame_index(meta_path)
        suffix = f"{idx:04d}"
        rgb = session_dir / f"frame_{suffix}.png"
        depth = session_dir / f"depth_{suffix}.png"
        conf = session_dir / f"conf_{suffix}.png"

        missing = [p.name for p in (rgb, depth) if not p.exists()]
        if missing:
            print(f"  skip frame {suffix}: missing {missing}", file=sys.stderr)
            skipped += 1
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base = f"frame-{out_idx:06d}"

        # Color + depth: byte-for-byte copy (lossless; preserves 16-bit depth).
        shutil.copyfile(rgb, frames_dir / f"{base}.color.png")
        shutil.copyfile(depth, arcore_dir / f"{base}.depth.png")
        if conf.exists():
            shutil.copyfile(conf, arcore_dir / f"{base}.conf.png")

        # Pose: column-major (OpenGL) -> row-major camera-to-world text.
        rows = column_major_to_row_major_rows(meta["pose_matrix_column_major"])
        _sanity_check_translation(rows, meta, base)
        (arcore_dir / f"{base}.pose.txt").write_text(format_pose_text(rows), encoding="utf-8")

        # Turntable object pose (camera<-object), when the board was seen.
        obj_rows = object_pose_rows(meta)
        if obj_rows is not None:
            (arcore_dir / f"{base}.object_pose.txt").write_text(
                format_pose_text(obj_rows), encoding="utf-8"
            )
            object_poses_written += 1

        # Intrinsics: one file for the whole session (color-image intrinsics).
        if not intrinsics_written:
            (arcore_dir / "intrinsics.txt").write_text(
                format_intrinsics_text(meta["intrinsics"]), encoding="utf-8"
            )
            intrinsics_written = True

        written += 1

    print(
        f"Converted {written} frame(s)"
        + (f", skipped {skipped}" if skipped else "")
        + (f", {object_poses_written} object pose(s)" if object_poses_written else "")
    )
    print(f"  frames -> {frames_dir}")
    print(f"  arcore -> {arcore_dir}")
    disparities_dir = out_dir / "disparities"
    print()
    print("1) Generate disparities (Depth-Anything-3 Small; needs requirements-da3.txt):")
    print(
        f"  python3 depth_anything_v3.py --frames {frames_dir} "
        f"--output {disparities_dir}"
    )
    print("2) Run the float pipeline:")
    print(
        f"  python3 pipeline_float.py --frames {frames_dir} "
        f"--arcore {arcore_dir} --disparities {disparities_dir} "
        f"--output {out_dir / 'output' / 'mesh.glb'}"
    )
    return written


def _sanity_check_translation(rows: list[list[float]], meta: dict, base: str) -> None:
    """The last column of a camera-to-world matrix is the camera position, which
    must match the recorder's translation_xyz. Catches a pose transpose mistake."""
    tx = meta.get("translation_xyz")
    if not tx:
        return
    last_col = [rows[0][3], rows[1][3], rows[2][3]]
    if any(abs(a - b) > 1e-3 for a, b in zip(last_col, tx)):
        print(
            f"  WARNING {base}: pose translation {last_col} != translation_xyz {tx} "
            f"(possible matrix-order bug)",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path, help="Recorder session folder (session_<stamp>)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dataset root (default: ./<session_name>_dataset)",
    )
    args = parser.parse_args()

    out_dir = args.out or Path.cwd() / f"{args.session.name}_dataset"
    count = convert(args.session, out_dir)
    if count == 0:
        raise SystemExit("No frames converted.")


if __name__ == "__main__":
    main()
