"""Convert Lantern Recorder Android sessions into the host pipeline layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Android Lantern Recorder session for pipeline_float.py."
    )
    parser.add_argument("session", type=Path, help="sessions/session_<timestamp> folder")
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "frames")
    parser.add_argument("--arcore", type=Path, default=REPO_ROOT / "arcore")
    parser.add_argument(
        "--keep-rgb-resolution",
        action="store_true",
        help="Do not resize RGB frames to depth resolution or rescale intrinsics.",
    )
    return parser.parse_args()


def sorted_frame_jsons(session_dir: Path) -> list[Path]:
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory does not exist: {session_dir}")
    json_paths = sorted(session_dir.glob("frame_*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No frame_*.json files found in: {session_dir}")
    return json_paths


def frame_id(json_path: Path) -> str:
    return json_path.stem.split("_", 1)[1]


def pipeline_base(index: int) -> str:
    return f"frame-{index:06d}"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pose_from_column_major(values: list[float]) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64)
    if pose.size != 16:
        raise ValueError(f"pose_matrix_column_major must have 16 values, got {pose.size}")
    return pose.reshape((4, 4), order="F")


def scaled_intrinsics(metadata: dict, target_width: int, target_height: int) -> dict[str, float | int]:
    intrinsics = metadata["intrinsics"]
    source_width = float(intrinsics["width"])
    source_height = float(intrinsics["height"])
    x_scale = target_width / source_width
    y_scale = target_height / source_height
    return {
        "fx": float(intrinsics["fx"]) * x_scale,
        "fy": float(intrinsics["fy"]) * y_scale,
        "cx": float(intrinsics["cx"]) * x_scale,
        "cy": float(intrinsics["cy"]) * y_scale,
        "width": int(target_width),
        "height": int(target_height),
    }


def write_intrinsics(path: Path, intrinsics: dict[str, float | int]) -> None:
    path.write_text(
        " ".join(
            [
                f"fx={intrinsics['fx']}",
                f"fy={intrinsics['fy']}",
                f"cx={intrinsics['cx']}",
                f"cy={intrinsics['cy']}",
                f"width={intrinsics['width']}",
                f"height={intrinsics['height']}",
            ]
        ),
        encoding="utf-8",
    )


def read_depth_shape(depth_path: Path) -> tuple[int, int]:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {depth_path}")
    if depth.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth PNG, got {depth.dtype}: {depth_path}")
    return depth.shape[:2]


def write_color(src: Path, dst: Path, target_shape: tuple[int, int] | None) -> None:
    if target_shape is None:
        shutil.copy2(src, dst)
        return
    color = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"Failed to read RGB frame: {src}")
    height, width = target_shape
    resized = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(dst), resized):
        raise ValueError(f"Failed to write resized RGB frame: {dst}")


def convert_session(
    session_dir: Path,
    frames_dir: Path,
    arcore_dir: Path,
    keep_rgb_resolution: bool = False,
) -> int:
    json_paths = sorted_frame_jsons(session_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    arcore_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_written = False
    converted = 0
    for index, json_path in enumerate(json_paths):
        frame_num = frame_id(json_path)
        metadata = load_json(json_path)
        rgb_path = session_dir / f"frame_{frame_num}.png"
        depth_path = session_dir / f"depth_{frame_num}.png"
        conf_path = session_dir / f"conf_{frame_num}.png"
        for path in (rgb_path, depth_path, conf_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing Android session file: {path}")

        depth_height, depth_width = read_depth_shape(depth_path)
        target_shape = None if keep_rgb_resolution else (depth_height, depth_width)
        intrinsics = (
            metadata["intrinsics"]
            if keep_rgb_resolution
            else scaled_intrinsics(metadata, depth_width, depth_height)
        )

        base = pipeline_base(index)
        write_color(rgb_path, frames_dir / f"{base}.color.png", target_shape)
        shutil.copy2(depth_path, arcore_dir / f"{base}.depth.png")
        shutil.copy2(conf_path, arcore_dir / f"{base}.conf.png")
        np.savetxt(
            arcore_dir / f"{base}.pose.txt",
            pose_from_column_major(metadata["pose_matrix_column_major"]),
        )

        if not intrinsics_written:
            write_intrinsics(arcore_dir / "intrinsics.txt", intrinsics)
            intrinsics_written = True
        converted += 1

    return converted


def main() -> None:
    args = parse_args()
    try:
        count = convert_session(
            args.session.resolve(),
            args.frames.resolve(),
            args.arcore.resolve(),
            keep_rgb_resolution=args.keep_rgb_resolution,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Converted {count} Android frames to {args.frames} and {args.arcore}")


if __name__ == "__main__":
    main()
