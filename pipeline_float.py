"""End-to-end scale calibration and TSDF fusion runner."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scale_solver import global_scale, smooth_shifts, solve_affine
from tsdf_fuse import fuse, load_poses


SCRIPT_ROOT = Path(__file__).resolve().parent
DISP_UINT16_MAX = 65535.0
DEPTH_UINT16_MAX = 65535
DEPTH_MM_PER_M = 1000.0
DISPARITY_EPS = 1e-6
SMOOTHING_WINDOW = 5
SEVEN_SCENES_INTRINSICS = {
    "fx": 585.0,
    "fy": 585.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate relative disparity to metric depth and fuse a mesh."
    )
    parser.add_argument("--frames", type=Path, default=SCRIPT_ROOT / "frames")
    parser.add_argument("--disparities", type=Path, default=SCRIPT_ROOT / "disparities")
    parser.add_argument("--arcore", type=Path, default=SCRIPT_ROOT / "arcore")
    parser.add_argument("--output", type=Path, default=SCRIPT_ROOT / "output" / "mesh.glb")
    return parser.parse_args()


def load_intrinsics(intrinsics_path: Path) -> dict[str, float | int]:
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics file does not exist: {intrinsics_path}")

    values: dict[str, float | int] = {}
    for token in intrinsics_path.read_text(encoding="utf-8").split():
        if "=" not in token:
            raise ValueError(f"Invalid intrinsics token {token!r} in {intrinsics_path}")
        key, raw_value = token.split("=", 1)
        if key in {"width", "height"}:
            values[key] = int(float(raw_value))
        else:
            values[key] = float(raw_value)

    required = {"fx", "fy", "cx", "cy", "width", "height"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Intrinsics file is missing keys: {missing}")

    return values


def seven_scenes_intrinsics_for_frame(frame_path: Path) -> dict[str, float | int]:
    image = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read frame for default intrinsics: {frame_path}")

    height, width = image.shape[:2]
    x_scale = width / float(SEVEN_SCENES_INTRINSICS["width"])
    y_scale = height / float(SEVEN_SCENES_INTRINSICS["height"])
    return {
        "fx": float(SEVEN_SCENES_INTRINSICS["fx"]) * x_scale,
        "fy": float(SEVEN_SCENES_INTRINSICS["fy"]) * y_scale,
        "cx": float(SEVEN_SCENES_INTRINSICS["cx"]) * x_scale,
        "cy": float(SEVEN_SCENES_INTRINSICS["cy"]) * y_scale,
        "width": width,
        "height": height,
    }


def load_or_default_intrinsics(
    intrinsics_path: Path, frame_paths: list[Path]
) -> dict[str, float | int]:
    if intrinsics_path.exists():
        return load_intrinsics(intrinsics_path)

    if not frame_paths:
        raise FileNotFoundError(
            f"Intrinsics file does not exist and no frames are available: {intrinsics_path}"
        )

    print(
        f"Intrinsics file not found at {intrinsics_path}; "
        "using 7-Scenes Kinect defaults scaled to frame size."
    )
    return seven_scenes_intrinsics_for_frame(frame_paths[0])


def load_disp_maxes(disparities_dir: Path) -> dict[str, float]:
    disp_maxes_path = disparities_dir / "disp_maxes.json"
    if not disp_maxes_path.exists():
        raise FileNotFoundError(f"Disparity maxes file does not exist: {disp_maxes_path}")

    with disp_maxes_path.open("r", encoding="utf-8") as f:
        raw_maxes: dict[str, Any] = json.load(f)

    return {str(key): float(value) for key, value in raw_maxes.items()}


def sorted_paths(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    return sorted(directory.glob(pattern))


def validate_counts(
    frame_paths: list[Path],
    disp_paths: list[Path],
    depth_paths: list[Path],
    poses: list[np.ndarray],
) -> None:
    counts = {
        "frames": len(frame_paths),
        "disparities": len(disp_paths),
        "depths": len(depth_paths),
        "poses": len(poses),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(
            "Mismatched input counts by sorted index: "
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )
    if not frame_paths:
        raise ValueError("No input frames found")


def read_png(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read {label} PNG: {path}")
    if image.dtype != np.uint16:
        raise ValueError(f"Expected uint16 {label} PNG, got {image.dtype}: {path}")
    return image


def strip_known_suffix(path: Path) -> str:
    for suffix in (".color.png", ".disp.png", ".depth.png"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def disp_max_for_frame(
    disp_maxes: dict[str, float],
    disp_path: Path,
    frame_path: Path,
    depth_path: Path,
) -> float:
    base = strip_known_suffix(frame_path)
    candidates = [
        disp_path.name,
        disp_path.stem,
        strip_known_suffix(disp_path),
        frame_path.name,
        frame_path.stem,
        base,
        f"{base}.disp.png",
        depth_path.name,
        depth_path.stem,
        strip_known_suffix(depth_path),
    ]

    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key in disp_maxes:
            return disp_maxes[key]

    raise KeyError(
        f"No disp_maxes entry for {disp_path}; tried keys: {', '.join(seen)}"
    )


def load_pred_disp(disp_path: Path, disp_max: float) -> np.ndarray:
    raw_disp = read_png(disp_path, "disparity").astype(np.float32)
    return (raw_disp / DISP_UINT16_MAX) * np.float32(disp_max)


def mean_abs_residual(
    pred_disp: np.ndarray,
    metric_depth: np.ndarray,
    conf_mask: np.ndarray,
    s: float,
    t: float,
) -> float:
    valid = conf_mask & np.isfinite(pred_disp) & np.isfinite(metric_depth)
    if not np.any(valid):
        return float("nan")

    metric_disp = DEPTH_MM_PER_M / metric_depth[valid].astype(np.float64)
    fitted_disp = float(s) * pred_disp[valid].astype(np.float64) + float(t)
    return float(np.mean(np.abs(fitted_disp - metric_disp)))


def calibrated_depth_mm(
    pred_disp: np.ndarray,
    global_s: float,
    smoothed_t: float,
) -> np.ndarray:
    scaled_disp = float(global_s) * pred_disp.astype(np.float64) + float(smoothed_t)
    scaled_disp = np.maximum(scaled_disp, DISPARITY_EPS)
    depth_mm = (1.0 / scaled_disp) * DEPTH_MM_PER_M
    depth_mm = np.clip(depth_mm, 0.0, float(DEPTH_UINT16_MAX))
    return np.rint(depth_mm).astype(np.uint16)


def write_calibrated_inputs(
    frame_paths: list[Path],
    disp_paths: list[Path],
    depth_paths: list[Path],
    disp_maxes: dict[str, float],
    temp_dir: Path,
) -> list[str]:
    frame_records: list[tuple[Path, Path, np.ndarray, float, float, float]] = []

    for frame_path, disp_path, depth_path in zip(
        frame_paths, disp_paths, depth_paths, strict=True
    ):
        disp_max = disp_max_for_frame(disp_maxes, disp_path, frame_path, depth_path)
        pred_disp = load_pred_disp(disp_path, disp_max)
        metric_depth = read_png(depth_path, "metric depth")
        if pred_disp.shape != metric_depth.shape:
            raise ValueError(
                f"Shape mismatch for {frame_path.name}: "
                f"disparity={pred_disp.shape}, depth={metric_depth.shape}"
            )

        conf_mask = (metric_depth > 0) & (metric_depth < DEPTH_UINT16_MAX)
        s, t = solve_affine(pred_disp, metric_depth, conf_mask)
        residual = mean_abs_residual(pred_disp, metric_depth, conf_mask, s, t)
        print(
            f"{frame_path.name}: s={s:.6f}, t={t:.6f}, "
            f"mean_abs_residual={residual:.6f}"
        )
        frame_records.append((frame_path, depth_path, pred_disp, s, t, residual))

    raw_s_values = np.asarray([record[3] for record in frame_records], dtype=np.float64)
    raw_t_values = np.asarray([record[4] for record in frame_records], dtype=np.float64)
    global_s = global_scale(raw_s_values)
    smoothed_t_values = smooth_shifts(raw_t_values, SMOOTHING_WINDOW)
    print(f"global median s={global_s:.6f}")

    scaled_depth_paths: list[str] = []
    for idx, (frame_path, _depth_path, pred_disp, _s, _t, _residual) in enumerate(
        frame_records
    ):
        base = strip_known_suffix(frame_path)
        temp_color_path = temp_dir / f"{base}.color.png"
        temp_depth_path = temp_dir / f"{base}.depth.png"

        shutil.copy2(frame_path, temp_color_path)
        scaled_depth = calibrated_depth_mm(
            pred_disp, global_s, float(smoothed_t_values[idx])
        )
        if not cv2.imwrite(str(temp_depth_path), scaled_depth):
            raise ValueError(f"Failed to write calibrated depth PNG: {temp_depth_path}")

        scaled_depth_paths.append(str(temp_depth_path))

    return scaled_depth_paths


def main() -> None:
    args = parse_args()
    frames_dir = args.frames.resolve()
    disparities_dir = args.disparities.resolve()
    arcore_dir = args.arcore.resolve()
    output_path = args.output.resolve()

    frame_paths = sorted_paths(frames_dir, "*.color.png")
    disp_paths = sorted_paths(disparities_dir, "*.disp.png")
    depth_paths = sorted_paths(arcore_dir, "*.depth.png")
    poses = load_poses(str(arcore_dir))
    validate_counts(frame_paths, disp_paths, depth_paths, poses)

    intrinsics = load_or_default_intrinsics(arcore_dir / "intrinsics.txt", frame_paths)
    disp_maxes = load_disp_maxes(disparities_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lantern_scaled_depth_") as temp_name:
        temp_dir = Path(temp_name)
        scaled_depth_paths = write_calibrated_inputs(
            frame_paths,
            disp_paths,
            depth_paths,
            disp_maxes,
            temp_dir,
        )
        fuse(scaled_depth_paths, poses, intrinsics, output_path=str(output_path))

    print(f"Wrote mesh to {output_path}")


if __name__ == "__main__":
    main()
