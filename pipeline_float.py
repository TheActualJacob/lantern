"""End-to-end scale calibration and TSDF fusion runner."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from object_frame import (
    load_capture_meta,
    load_object_poses,
    resolve_fusion_frame,
    select_fusion_inputs,
)
from scale_solver import (
    global_scale,
    smooth_shifts,
    solve_affine,
    solve_affine_sequence,
)
from tsdf_fuse import fuse, load_poses


SCRIPT_ROOT = Path(__file__).resolve().parent
DISP_UINT16_MAX = 65535.0
DEPTH_UINT16_MAX = 65535
DEPTH_MM_PER_M = 1000.0
DISPARITY_EPS = 1e-6
SMOOTHING_WINDOW = 5
EDGE_DEPTH_THRESHOLD_MM = 100.0
DEFAULT_MIN_VALID_PIXELS = 500
DEFAULT_MAX_RESIDUAL = float("inf")
SEVEN_SCENES_INTRINSICS = {
    "fx": 585.0,
    "fy": 585.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480,
}


@dataclass
class FrameRecord:
    frame_path: Path
    depth_path: Path
    pred_disp: np.ndarray
    metric_depth: np.ndarray
    mask: np.ndarray
    raw_s: float
    raw_t: float
    raw_residual: float
    valid_pixels: int
    median_depth_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate relative disparity to metric depth and fuse a mesh."
    )
    parser.add_argument("--frames", type=Path, default=SCRIPT_ROOT / "frames")
    parser.add_argument("--disparities", type=Path, default=SCRIPT_ROOT / "disparities")
    parser.add_argument("--arcore", type=Path, default=SCRIPT_ROOT / "arcore")
    parser.add_argument("--output", type=Path, default=SCRIPT_ROOT / "output" / "mesh.glb")
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--use-metric-depth-only", action="store_true")
    parser.add_argument("--confidence-dir", type=Path, default=None)
    parser.add_argument("--min-confidence", type=int, default=0)
    parser.add_argument("--edge-erosion", type=int, default=1)
    parser.add_argument("--edge-threshold-mm", type=float, default=EDGE_DEPTH_THRESHOLD_MM)
    parser.add_argument("--bilateral-filter", action="store_true")
    parser.add_argument("--min-valid-pixels", type=int, default=DEFAULT_MIN_VALID_PIXELS)
    parser.add_argument("--max-residual", type=float, default=DEFAULT_MAX_RESIDUAL)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--sdf-trunc-mult", type=float, default=5.0)
    parser.add_argument("--depth-trunc", type=float, default=5.0)
    parser.add_argument("--pose-convention", choices=["arcore", "opencv"], default="arcore")
    parser.add_argument("--sequence-smooth-lambda", type=float, default=0.05)
    parser.add_argument(
        "--fusion-frame",
        choices=["auto", "world", "object"],
        default="auto",
        help=(
            "Frame the TSDF fuses in. 'world' = ARCore world (orbit, today's path); "
            "'object' = the board/object frame for turntable captures; 'auto' picks "
            "object for turntable sessions (capture.json) and world otherwise."
        ),
    )
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


def median_abs_residual(
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
    return float(np.median(np.abs(fitted_disp - metric_disp)))


def moving_average_trailing(values: np.ndarray, window: int) -> np.ndarray:
    """Backward-compatible alias for the S25-safe trailing shift smoother."""
    return smooth_shifts(values, window)


def confidence_path_for_frame(frame_path: Path, depth_path: Path, confidence_dir: Path) -> Path | None:
    base = strip_known_suffix(frame_path)
    candidates = [
        confidence_dir / f"{base}.conf.png",
        confidence_dir / f"{base}.confidence.png",
        confidence_dir / f"{strip_known_suffix(depth_path)}.conf.png",
        confidence_dir / depth_path.name.replace(".depth.png", ".conf.png"),
        confidence_dir / f"conf_{base.split('-')[-1]}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_conf_mask(
    metric_depth: np.ndarray,
    frame_path: Path,
    depth_path: Path,
    confidence_dir: Path | None,
    min_confidence: int,
    edge_erosion: int,
    edge_threshold_mm: float,
) -> np.ndarray:
    mask = (metric_depth > 0) & (metric_depth < DEPTH_UINT16_MAX)

    if confidence_dir is not None and min_confidence > 0:
        conf_path = confidence_path_for_frame(frame_path, depth_path, confidence_dir)
        if conf_path is not None:
            confidence = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
            if confidence is None:
                raise ValueError(f"Failed to read confidence PNG: {conf_path}")
            if confidence.shape != metric_depth.shape:
                confidence = cv2.resize(
                    confidence,
                    (metric_depth.shape[1], metric_depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask &= confidence.astype(np.int32) >= int(min_confidence)

    if edge_erosion > 0:
        mask = remove_depth_edges(mask, metric_depth, edge_erosion, edge_threshold_mm)

    return mask.astype(bool)


def remove_depth_edges(
    mask: np.ndarray,
    metric_depth: np.ndarray,
    edge_erosion: int,
    edge_threshold_mm: float,
) -> np.ndarray:
    depth = metric_depth.astype(np.float32)
    valid_depth = np.where(mask, depth, 0.0)
    grad_x = np.zeros_like(depth, dtype=bool)
    grad_y = np.zeros_like(depth, dtype=bool)
    both_x = mask[:, 1:] & mask[:, :-1]
    both_y = mask[1:, :] & mask[:-1, :]
    grad_x[:, 1:] |= both_x & (np.abs(valid_depth[:, 1:] - valid_depth[:, :-1]) > edge_threshold_mm)
    grad_x[:, :-1] |= both_x & (np.abs(valid_depth[:, 1:] - valid_depth[:, :-1]) > edge_threshold_mm)
    grad_y[1:, :] |= both_y & (np.abs(valid_depth[1:, :] - valid_depth[:-1, :]) > edge_threshold_mm)
    grad_y[:-1, :] |= both_y & (np.abs(valid_depth[1:, :] - valid_depth[:-1, :]) > edge_threshold_mm)
    edge_mask = grad_x | grad_y

    kernel = np.ones((2 * edge_erosion + 1, 2 * edge_erosion + 1), dtype=np.uint8)
    expanded_edges = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    eroded_valid = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return eroded_valid & ~expanded_edges


def calibrated_depth_mm(
    pred_disp: np.ndarray,
    global_s: float,
    smoothed_t: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    scaled_disp = float(global_s) * pred_disp.astype(np.float64) + float(smoothed_t)
    scaled_disp = np.maximum(scaled_disp, DISPARITY_EPS)
    depth_mm = (1.0 / scaled_disp) * DEPTH_MM_PER_M
    depth_mm = np.clip(depth_mm, 0.0, float(DEPTH_UINT16_MAX))
    depth_uint16 = np.rint(depth_mm).astype(np.uint16)
    if valid_mask is not None:
        depth_uint16 = np.where(valid_mask, depth_uint16, 0).astype(np.uint16)
    return depth_uint16


def maybe_filter_depth(depth_mm: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return depth_mm
    return cv2.bilateralFilter(depth_mm, d=5, sigmaColor=50, sigmaSpace=3)


def diagnostics_for_frame(
    record: FrameRecord,
    applied_s: float,
    applied_t: float,
    calibrated_depth: np.ndarray,
) -> dict[str, float | int | str]:
    scaled_disp = applied_s * record.pred_disp.astype(np.float64) + applied_t
    clipped_percent = float(np.mean(scaled_disp <= DISPARITY_EPS) * 100.0)
    valid_depth = calibrated_depth[calibrated_depth > 0]
    return {
        "frame": record.frame_path.name,
        "raw_s": record.raw_s,
        "raw_t": record.raw_t,
        "applied_s": applied_s,
        "applied_t": applied_t,
        "mean_abs_residual": mean_abs_residual(
            record.pred_disp, record.metric_depth, record.mask, applied_s, applied_t
        ),
        "median_abs_residual": median_abs_residual(
            record.pred_disp, record.metric_depth, record.mask, applied_s, applied_t
        ),
        "raw_mean_abs_residual": record.raw_residual,
        "valid_pixels": record.valid_pixels,
        "median_metric_depth_mm": record.median_depth_mm,
        "percent_clipped_disparity": clipped_percent,
        "output_depth_min_mm": int(valid_depth.min()) if valid_depth.size else 0,
        "output_depth_max_mm": int(valid_depth.max()) if valid_depth.size else 0,
    }


def write_frame_diagnostics(
    diagnostics_dir: Path | None,
    base: str,
    mask: np.ndarray,
    calibrated_depth: np.ndarray,
    residual_map: np.ndarray,
) -> None:
    if diagnostics_dir is None:
        return
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(diagnostics_dir / f"{base}.mask.png"), (mask.astype(np.uint8) * 255))
    cv2.imwrite(str(diagnostics_dir / f"{base}.calibrated_depth.png"), calibrated_depth)
    residual_vis = np.clip(residual_map * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(diagnostics_dir / f"{base}.residual.png"), residual_vis)


def write_calibrated_inputs(
    frame_paths: list[Path],
    disp_paths: list[Path],
    depth_paths: list[Path],
    disp_maxes: dict[str, float],
    temp_dir: Path,
    diagnostics_dir: Path | None = None,
    confidence_dir: Path | None = None,
    min_confidence: int = 0,
    edge_erosion: int = 1,
    edge_threshold_mm: float = EDGE_DEPTH_THRESHOLD_MM,
    bilateral_filter: bool = False,
    min_valid_pixels: int = DEFAULT_MIN_VALID_PIXELS,
    max_residual: float = DEFAULT_MAX_RESIDUAL,
    sequence_smooth_lambda: float = 0.05,
) -> tuple[list[str], list[dict[str, object]], list[int]]:
    frame_records: list[FrameRecord] = []

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

        conf_mask = build_conf_mask(
            metric_depth,
            frame_path,
            depth_path,
            confidence_dir,
            min_confidence,
            edge_erosion,
            edge_threshold_mm,
        )
        s, t = solve_affine(pred_disp, metric_depth, conf_mask)
        residual = mean_abs_residual(pred_disp, metric_depth, conf_mask, s, t)
        valid_pixels = int(np.count_nonzero(conf_mask))
        valid_depth = metric_depth[conf_mask]
        median_depth = float(np.median(valid_depth)) if valid_depth.size else float("nan")
        print(
            f"{frame_path.name}: s={s:.6f}, t={t:.6f}, "
            f"mean_abs_residual={residual:.6f}, valid_pixels={valid_pixels}"
        )
        frame_records.append(
            FrameRecord(
                frame_path=frame_path,
                depth_path=depth_path,
                pred_disp=pred_disp,
                metric_depth=metric_depth,
                mask=conf_mask,
                raw_s=s,
                raw_t=t,
                raw_residual=residual,
                valid_pixels=valid_pixels,
                median_depth_mm=median_depth,
            )
        )

    try:
        global_s, smoothed_t_values = solve_affine_sequence(
            [record.pred_disp for record in frame_records],
            [record.metric_depth for record in frame_records],
            [record.mask for record in frame_records],
            smooth_lambda=sequence_smooth_lambda,
        )
        print(f"sequence global s={global_s:.6f}")
    except ValueError:
        raw_s_values = np.asarray([record.raw_s for record in frame_records], dtype=np.float64)
        raw_t_values = np.asarray([record.raw_t for record in frame_records], dtype=np.float64)
        global_s = global_scale(raw_s_values)
        smoothed_t_values = smooth_shifts(raw_t_values, SMOOTHING_WINDOW)
        print(f"global median s={global_s:.6f}")

    scaled_depth_paths: list[str] = []
    diagnostics: list[dict[str, object]] = []
    accepted_indices: list[int] = []
    for idx, record in enumerate(frame_records):
        if record.valid_pixels < min_valid_pixels or record.raw_residual > max_residual:
            print(
                f"{record.frame_path.name}: skipped "
                f"(valid_pixels={record.valid_pixels}, residual={record.raw_residual:.6f})"
            )
            continue

        frame_path = record.frame_path
        base = strip_known_suffix(frame_path)
        temp_color_path = temp_dir / f"{base}.color.png"
        temp_depth_path = temp_dir / f"{base}.depth.png"

        shutil.copy2(frame_path, temp_color_path)
        scaled_depth = calibrated_depth_mm(
            record.pred_disp, global_s, float(smoothed_t_values[idx]), record.mask
        )
        scaled_depth = maybe_filter_depth(scaled_depth, bilateral_filter)
        if not cv2.imwrite(str(temp_depth_path), scaled_depth):
            raise ValueError(f"Failed to write calibrated depth PNG: {temp_depth_path}")

        metric_disp = np.zeros_like(record.pred_disp, dtype=np.float64)
        metric_disp[record.mask] = DEPTH_MM_PER_M / record.metric_depth[record.mask]
        fitted_disp = global_s * record.pred_disp.astype(np.float64) + float(
            smoothed_t_values[idx]
        )
        residual_map = np.where(record.mask, np.abs(fitted_disp - metric_disp), 0.0)
        write_frame_diagnostics(
            diagnostics_dir, base, record.mask, scaled_depth, residual_map
        )
        diagnostics.append(
            diagnostics_for_frame(
                record, global_s, float(smoothed_t_values[idx]), scaled_depth
            )
        )
        scaled_depth_paths.append(str(temp_depth_path))
        accepted_indices.append(idx)

    if not scaled_depth_paths:
        raise ValueError("All frames were rejected; relax thresholds or inspect diagnostics")

    return scaled_depth_paths, diagnostics, accepted_indices


def write_metric_depth_inputs(
    frame_paths: list[Path],
    depth_paths: list[Path],
    temp_dir: Path,
) -> list[str]:
    staged_depth_paths: list[str] = []
    for frame_path, depth_path in zip(frame_paths, depth_paths, strict=True):
        base = strip_known_suffix(frame_path)
        temp_color_path = temp_dir / f"{base}.color.png"
        temp_depth_path = temp_dir / f"{base}.depth.png"
        shutil.copy2(frame_path, temp_color_path)
        shutil.copy2(depth_path, temp_depth_path)
        staged_depth_paths.append(str(temp_depth_path))
    return staged_depth_paths


def write_diagnostics_json(
    diagnostics_dir: Path | None,
    frames: list[dict[str, object]],
    mesh_stats: dict[str, object] | None,
    args: argparse.Namespace,
) -> None:
    if diagnostics_dir is None:
        return
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "frames": frames,
        "mesh": mesh_stats or {},
    }
    with (diagnostics_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    frames_dir = args.frames.resolve()
    disparities_dir = args.disparities.resolve()
    arcore_dir = args.arcore.resolve()
    output_path = args.output.resolve()

    frame_paths = sorted_paths(frames_dir, "*.color.png")
    depth_paths = sorted_paths(arcore_dir, "*.depth.png")
    poses = load_poses(str(arcore_dir))

    intrinsics = load_or_default_intrinsics(arcore_dir / "intrinsics.txt", frame_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.diagnostics.resolve() if args.diagnostics else None
    confidence_dir = (
        args.confidence_dir.resolve()
        if args.confidence_dir
        else arcore_dir
    )

    # Pick the fusion frame: world (orbit) vs object (turntable). For object-frame
    # fusion we need the per-frame board poses (T_CO) written next to the depth.
    capture_meta = load_capture_meta(arcore_dir)
    fusion_frame = resolve_fusion_frame(args.fusion_frame, capture_meta)
    object_poses = load_object_poses(arcore_dir) if fusion_frame == "object" else None
    print(f"Fusion frame: {fusion_frame}")

    with tempfile.TemporaryDirectory(prefix="lantern_scaled_depth_") as temp_name:
        temp_dir = Path(temp_name)
        diagnostics: list[dict[str, object]] = []
        if args.use_metric_depth_only:
            validate_counts(frame_paths, frame_paths, depth_paths, poses)
            scaled_depth_paths = write_metric_depth_inputs(frame_paths, depth_paths, temp_dir)
            accepted_indices = list(range(len(frame_paths)))
            print("Using metric depth directly for TSDF baseline.")
        else:
            disp_paths = sorted_paths(disparities_dir, "*.disp.png")
            validate_counts(frame_paths, disp_paths, depth_paths, poses)
            disp_maxes = load_disp_maxes(disparities_dir)
            scaled_depth_paths, diagnostics, accepted_indices = write_calibrated_inputs(
                frame_paths,
                disp_paths,
                depth_paths,
                disp_maxes,
                temp_dir,
                diagnostics_dir=diagnostics_dir,
                confidence_dir=confidence_dir,
                min_confidence=args.min_confidence,
                edge_erosion=args.edge_erosion,
                edge_threshold_mm=args.edge_threshold_mm,
                bilateral_filter=args.bilateral_filter,
                min_valid_pixels=args.min_valid_pixels,
                max_residual=args.max_residual,
                sequence_smooth_lambda=args.sequence_smooth_lambda,
            )

        # Select world vs object poses and the matching axis convention. Object
        # frames without a valid board pose drop out here, so keep the depth
        # inputs aligned to the surviving frames.
        depth_by_index = dict(zip(accepted_indices, scaled_depth_paths, strict=True))
        fusion_poses, kept_indices, selected_convention = select_fusion_inputs(
            fusion_frame, poses, object_poses, accepted_indices
        )
        fusion_depth_paths = [depth_by_index[idx] for idx in kept_indices]
        pose_convention = (
            args.pose_convention if fusion_frame == "world" else selected_convention
        )
        if len(kept_indices) != len(accepted_indices):
            print(
                f"Object-frame fusion kept {len(kept_indices)}/{len(accepted_indices)} "
                "frames with a valid board pose."
            )

        mesh_stats = fuse(
            fusion_depth_paths,
            fusion_poses,
            intrinsics,
            voxel_size=args.voxel_size,
            sdf_trunc_mult=args.sdf_trunc_mult,
            depth_trunc=args.depth_trunc,
            pose_convention=pose_convention,
            output_path=str(output_path),
        )
        write_diagnostics_json(diagnostics_dir, diagnostics, mesh_stats, args)

    print(f"Wrote mesh to {output_path}")


if __name__ == "__main__":
    main()
