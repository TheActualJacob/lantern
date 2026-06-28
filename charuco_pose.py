"""Turntable T0 spike: estimate the object frame from a ChArUco board.

This is the host-side de-risking step for object-centric ("turntable") capture
(see ``OBJECT_TRACKING_PLAN.md`` §3, §7, §10). It does **one classical-geometry
job** and nothing neural:

    RGB frame  ->  detect ChArUco board  ->  solvePnP  ->  T_CO(t)
                                                           (camera <- object)

``T_CO(t)`` is the rigid transform that maps a point expressed in the board's
(object's) coordinate frame ``O`` into the camera frame ``C`` for that frame.
Its inverse ``T_OC = inv(T_CO)`` is the camera pose *in the object frame* — which
is exactly what host-side TSDF fusion needs to make a rotating object look static
(``object_frame.py`` / ``pipeline_float.py --fusion-frame object``).

The board's known square size also gives **metric scale** for free, so the object
frame is metric (millimetre-accurate corners), unlike a markerless tracker.

Yaw about the turntable axis
----------------------------
On a turntable the object spins about the board's local Z axis. With a (mostly)
fixed camera::

    T_CO(t) = T_CO(0) @ Rz(theta(t))

so the object's rotation between two frames is recovered from the relative
rotation ``R_CO(ref)^T @ R_CO(t)``, read as an angle about Z. That angle is the
"object azimuth" the on-device coverage ring will track instead of camera azimuth
(plan §3.4).

Usage
-----
Self-test (no data, no files — validates the math + the real detector path)::

    python charuco_pose.py --selftest

Process a recorded clip (a Lantern recorder session folder of ``frame_*.png`` +
``frame_*.json``, or any folder of images with a shared intrinsics file)::

    python charuco_pose.py --clip path/to/session_<stamp> \\
        --board DICT_5X5_100:5x7:0.03:0.022 --out yaw.json [--plot yaw.png]

The clip command writes per-frame ``T_CO``, validity, reprojection error, and the
object yaw vs. time so the T0 exit criterion (smooth monotonic yaw over a
hand-spin, reprojection error < ~1 px) can be eyeballed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ChArUco needs >= this many matched corners for a trustworthy solvePnP. A board
# (vs. a single marker) means partial occlusion by the object still leaves plenty.
MIN_CHARUCO_CORNERS = 6
# Frames whose reprojection error exceeds this (px) are flagged invalid: the board
# pose is too jittery to fuse against (plan §9, "board pose jitter smears TSDF").
DEFAULT_MAX_REPROJ_PX = 1.5


@dataclass(frozen=True)
class BoardSpec:
    """A ChArUco board definition shared by the phone (capture) and host (fuse).

    Mirrors the ``board`` block of the per-session ``capture.json`` sidecar
    (plan §4.1) so the same spec round-trips device -> disk -> host.
    """

    dict_name: str = "DICT_5X5_100"
    squares_x: int = 5
    squares_y: int = 7
    square_len_m: float = 0.03
    marker_len_m: float = 0.022

    @classmethod
    def from_dict(cls, data: dict) -> "BoardSpec":
        return cls(
            dict_name=str(data.get("dict", cls.dict_name)),
            squares_x=int(data.get("squares_x", cls.squares_x)),
            squares_y=int(data.get("squares_y", cls.squares_y)),
            square_len_m=float(data.get("square_len_m", cls.square_len_m)),
            marker_len_m=float(data.get("marker_len_m", cls.marker_len_m)),
        )

    @classmethod
    def parse(cls, text: str) -> "BoardSpec":
        """Parse ``DICT_5X5_100:5x7:0.03:0.022`` (dict:squaresXxsquaresY:sq:mk)."""
        parts = text.split(":")
        if len(parts) != 4:
            raise ValueError(
                "board spec must be DICT:SXxSY:square_len_m:marker_len_m, "
                f"got {text!r}"
            )
        dict_name, grid, square_len, marker_len = parts
        if "x" not in grid.lower():
            raise ValueError(f"grid must look like 5x7, got {grid!r}")
        sx, sy = grid.lower().split("x", 1)
        return cls(
            dict_name=dict_name,
            squares_x=int(sx),
            squares_y=int(sy),
            square_len_m=float(square_len),
            marker_len_m=float(marker_len),
        )

    def to_dict(self) -> dict:
        return {
            "dict": self.dict_name,
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_len_m": self.square_len_m,
            "marker_len_m": self.marker_len_m,
        }


@dataclass
class PoseResult:
    """Outcome of estimating the object frame for a single image."""

    valid: bool
    T_CO: Optional[np.ndarray]  # 4x4 camera<-object, OpenCV camera convention
    n_corners: int
    reproj_px: float


def _resolve_dictionary(dict_name: str) -> cv2.aruco.Dictionary:
    attr = getattr(cv2.aruco, dict_name, None)
    if attr is None:
        raise ValueError(
            f"Unknown ArUco dictionary {dict_name!r}; expected an attribute of "
            "cv2.aruco such as DICT_5X5_100."
        )
    return cv2.aruco.getPredefinedDictionary(attr)


def build_board(spec: BoardSpec) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
    """Construct the CharucoBoard + detector for a spec (OpenCV >= 4.7 API)."""
    dictionary = _resolve_dictionary(spec.dict_name)
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_len_m,
        spec.marker_len_m,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Pack a solvePnP (rvec, tvec) into a 4x4 ``T_CO`` (camera<-object).

    solvePnP returns the transform taking object-frame points into the camera
    frame: ``X_cam = R @ X_obj + t``. So ``[R|t]`` *is* ``T_CO``.
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def estimate_pose(
    gray: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    min_corners: int = MIN_CHARUCO_CORNERS,
) -> PoseResult:
    """Detect the board in ``gray`` and solve for ``T_CO`` (camera<-object)."""
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < min_corners:
        n = 0 if charuco_ids is None else int(len(charuco_ids))
        return PoseResult(valid=False, T_CO=None, n_corners=n, reproj_px=float("nan"))

    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if object_points is None or len(object_points) < min_corners:
        n = 0 if object_points is None else int(len(object_points))
        return PoseResult(valid=False, T_CO=None, n_corners=n, reproj_px=float("nan"))

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return PoseResult(
            valid=False, T_CO=None, n_corners=int(len(object_points)), reproj_px=float("nan")
        )

    reproj_px = reprojection_error(
        object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    return PoseResult(
        valid=True,
        T_CO=rvec_tvec_to_matrix(rvec, tvec),
        n_corners=int(len(object_points)),
        reproj_px=reproj_px,
    )


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(-1, 2)
    observed = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))


def yaw_about_z_deg(rotation: np.ndarray) -> float:
    """Read the rotation angle about the +Z axis (degrees) from a 3x3 matrix.

    Uses the upper-left 2x2 block, which isolates the Z component of rotation
    even when there is a small off-axis (handheld) wobble.
    """
    r = np.asarray(rotation, dtype=np.float64)
    angle = math.atan2(r[1, 0] - r[0, 1], r[0, 0] + r[1, 1])
    return math.degrees(angle)


def relative_object_yaw_deg(T_CO_ref: np.ndarray, T_CO: np.ndarray) -> float:
    """Object yaw (deg) at ``T_CO`` relative to a reference frame ``T_CO_ref``.

    The object spins about its board Z axis, so the relative rotation
    ``R_ref^T @ R`` is (ideally) a pure Z rotation; its angle is the turntable
    azimuth advanced since the reference frame.
    """
    r_ref = np.asarray(T_CO_ref, dtype=np.float64)[:3, :3]
    r_cur = np.asarray(T_CO, dtype=np.float64)[:3, :3]
    return yaw_about_z_deg(r_ref.T @ r_cur)


def unwrap_degrees(values: list[float]) -> list[float]:
    """Unwrap a sequence of degree angles so a hand-spin reads as monotonic."""
    if not values:
        return []
    unwrapped = np.degrees(np.unwrap(np.radians(np.asarray(values, dtype=np.float64))))
    return [float(v) for v in unwrapped]


# --------------------------------------------------------------------------- #
# Clip processing
# --------------------------------------------------------------------------- #


def _intrinsics_for_lantern_frame(meta: dict, image_shape: tuple[int, int]) -> np.ndarray:
    """Build K from a Lantern frame_*.json, rescaled to the actual image size.

    ARCore reports intrinsics for the full CPU image; rescale to whatever the
    saved RGB resolution actually is so principal point / focal length match.
    """
    intr = meta["intrinsics"]
    height, width = image_shape
    x_scale = width / float(intr["width"])
    y_scale = height / float(intr["height"])
    return intrinsics_matrix(
        float(intr["fx"]) * x_scale,
        float(intr["fy"]) * y_scale,
        float(intr["cx"]) * x_scale,
        float(intr["cy"]) * y_scale,
    )


def process_clip(
    clip_dir: Path,
    spec: BoardSpec,
    camera_matrix: np.ndarray | None = None,
    max_reproj_px: float = DEFAULT_MAX_REPROJ_PX,
) -> dict:
    """Run board-pose estimation across every frame of a recorded clip.

    Frames are discovered as ``frame_*.png`` (a Lantern recorder session) and
    matched to their ``frame_*.json`` for per-frame intrinsics; if no JSON is
    present, ``camera_matrix`` must be supplied.
    """
    board, detector = build_board(spec)
    frame_paths = sorted(clip_dir.glob("frame_*.png"))
    if not frame_paths:
        frame_paths = sorted(
            p for p in clip_dir.glob("*.png") if not p.name.startswith("depth_")
            and not p.name.startswith("conf_")
        )
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {clip_dir}")

    frames: list[dict] = []
    ref_pose: np.ndarray | None = None
    raw_yaws: list[float] = []

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        meta_path = frame_path.with_suffix(".json")
        if not meta_path.exists():
            meta_path = clip_dir / (frame_path.stem.replace("frame_", "frame_") + ".json")
        if camera_matrix is not None:
            K = camera_matrix
        elif meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            K = _intrinsics_for_lantern_frame(meta, gray.shape[:2])
        else:
            raise ValueError(
                f"No intrinsics for {frame_path.name}: provide --intrinsics or a frame JSON."
            )

        result = estimate_pose(gray, board, detector, K)
        valid = result.valid and (
            math.isnan(result.reproj_px) is False and result.reproj_px <= max_reproj_px
        )

        yaw: float | None = None
        if result.valid and result.T_CO is not None:
            if ref_pose is None:
                ref_pose = result.T_CO
                yaw = 0.0
            else:
                yaw = relative_object_yaw_deg(ref_pose, result.T_CO)
            raw_yaws.append(yaw)

        frames.append(
            {
                "frame": frame_path.name,
                "valid": bool(valid),
                "n_corners": result.n_corners,
                "reproj_px": result.reproj_px,
                "object_yaw_deg": yaw,
                "T_CO": result.T_CO.tolist() if result.T_CO is not None else None,
            }
        )

    yaw_unwrapped = unwrap_degrees(raw_yaws)
    yaw_idx = 0
    for frame in frames:
        if frame["object_yaw_deg"] is not None:
            frame["object_yaw_unwrapped_deg"] = yaw_unwrapped[yaw_idx]
            yaw_idx += 1

    valid_count = sum(1 for f in frames if f["valid"])
    reproj_values = [f["reproj_px"] for f in frames if f["valid"]]
    summary = {
        "board": spec.to_dict(),
        "n_frames": len(frames),
        "n_valid": valid_count,
        "median_reproj_px": float(np.median(reproj_values)) if reproj_values else None,
        "max_reproj_px": float(np.max(reproj_values)) if reproj_values else None,
        "yaw_span_deg": (
            float(yaw_unwrapped[-1] - yaw_unwrapped[0]) if len(yaw_unwrapped) >= 2 else None
        ),
        "frames": frames,
    }
    return summary


def maybe_plot(summary: dict, plot_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot.")
        return False

    xs = [i for i, f in enumerate(summary["frames"]) if f.get("object_yaw_unwrapped_deg") is not None]
    ys = [
        f["object_yaw_unwrapped_deg"]
        for f in summary["frames"]
        if f.get("object_yaw_unwrapped_deg") is not None
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker=".", lw=1)
    ax.set_xlabel("frame index")
    ax.set_ylabel("object yaw about board Z (deg, unwrapped)")
    ax.set_title("Turntable T0: object yaw vs. time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def _render_board_view(
    board: cv2.aruco.CharucoBoard,
    spec: BoardSpec,
    T_CO: np.ndarray,
    camera_matrix: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Render a synthetic camera view of the board at pose ``T_CO``.

    The board's ortho image is treated as the Z=0 plane of the object frame; a
    homography maps it into the camera image given the pose + intrinsics. This
    exercises the *real* detector path (not just the geometry), so the self-test
    matches the T0 exit criterion of sub-pixel reprojection from detection.
    """
    width_m = spec.squares_x * spec.square_len_m
    height_m = spec.squares_y * spec.square_len_m
    px_per_m = 2000.0
    board_w = int(round(width_m * px_per_m))
    board_h = int(round(height_m * px_per_m))
    board_img = board.generateImage((board_w, board_h))
    board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    # Object-frame 3D corners of the board plane (matches generateImage layout:
    # origin at top-left, +x right, +y down across the printed image).
    obj_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [width_m, 0.0, 0.0],
            [width_m, height_m, 0.0],
            [0.0, height_m, 0.0],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(T_CO[:3, :3])
    tvec = T_CO[:3, 3]
    projected, _ = cv2.projectPoints(
        obj_corners, rvec, tvec, camera_matrix, np.zeros((5, 1))
    )
    dst = projected.reshape(-1, 2).astype(np.float32)
    src = np.array(
        [[0, 0], [board_w - 1, 0], [board_w - 1, board_h - 1], [0, board_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((image_size[1], image_size[0], 3), 255, dtype=np.uint8)
    cv2.warpPerspective(
        board_img,
        H,
        image_size,
        dst=canvas,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    return canvas


def selftest() -> int:
    """Validate the math + the real detector path on synthetic spins.

    Returns a process exit code (0 = pass).
    """
    print("charuco_pose self-test")
    spec = BoardSpec()
    board, detector = build_board(spec)

    image_size = (1280, 960)
    fx = fy = 1100.0
    K = intrinsics_matrix(fx, fy, image_size[0] / 2.0, image_size[1] / 2.0)

    # Camera ~0.4 m above the board centre, looking straight down. We rotate the
    # OBJECT about its Z axis (turntable) and check we recover that yaw.
    width_m = spec.squares_x * spec.square_len_m
    height_m = spec.squares_y * spec.square_len_m
    center = np.array([width_m / 2.0, height_m / 2.0, 0.0])

    failures = 0
    spins = list(range(0, 200, 15))  # degrees of object spin
    recovered: list[float] = []
    reproj_list: list[float] = []
    n_detected = 0
    ref_pose = None

    for deg in spins:
        theta = math.radians(deg)
        Rz = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Object pose in a world where the camera looks down the -? Build T_CO:
        # place the board centre 0.4 m in front of the camera (+Z forward in
        # OpenCV convention), spun by Rz about its own centre.
        T = np.eye(4)
        T[:3, :3] = Rz
        # translate so the board centre maps to (0,0,0.4) in camera frame.
        T[:3, 3] = np.array([0.0, 0.0, 0.4]) - Rz @ center
        T_CO_true = T

        view = _render_board_view(board, spec, T_CO_true, K, image_size)
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        result = estimate_pose(gray, board, detector, K)
        if not result.valid:
            print(f"  spin {deg:3d}deg: NOT DETECTED (corners={result.n_corners})")
            failures += 1
            continue

        n_detected += 1
        reproj_list.append(result.reproj_px)
        if ref_pose is None:
            ref_pose_true = T_CO_true
            ref_pose = result.T_CO
            recovered.append(0.0)
            true_ref = deg
        else:
            yaw = relative_object_yaw_deg(ref_pose, result.T_CO)
            recovered.append(yaw)
            true_rel = deg - true_ref
            err = abs(((yaw - true_rel + 180) % 360) - 180)
            status = "ok" if err < 1.0 else "FAIL"
            if err >= 1.0:
                failures += 1
            print(
                f"  spin {deg:3d}deg: yaw={yaw:7.2f} true={true_rel:7.2f} "
                f"err={err:.3f}deg reproj={result.reproj_px:.3f}px [{status}]"
            )

    if n_detected < len(spins):
        print(f"  detection: {n_detected}/{len(spins)} views detected")

    max_reproj = max(reproj_list) if reproj_list else float("nan")
    print(f"  max reprojection error: {max_reproj:.3f} px (target < 1.0)")
    if not reproj_list or max_reproj >= 1.0:
        failures += 1

    # Pure-geometry check (no rendering): projecting known corners and solving
    # must recover the pose to numerical precision.
    geom_err = _selftest_geometry(spec, K)
    print(f"  geometry round-trip yaw error: {geom_err:.4e} deg (target < 1e-3)")
    if geom_err >= 1e-3:
        failures += 1

    print("PASS" if failures == 0 else f"FAIL ({failures} issue(s))")
    return 0 if failures == 0 else 1


def _selftest_geometry(spec: BoardSpec, K: np.ndarray) -> float:
    """Project board corners at a known yaw, solvePnP, return yaw recovery error."""
    board, _ = build_board(spec)
    obj_pts = board.getChessboardCorners().astype(np.float64)  # (N,3) in object frame
    center = obj_pts.mean(axis=0)

    def pose_for(deg: float) -> np.ndarray:
        theta = math.radians(deg)
        Rz = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        T = np.eye(4)
        T[:3, :3] = Rz
        T[:3, 3] = np.array([0.0, 0.0, 0.5]) - Rz @ center
        return T

    worst = 0.0
    ref_solved = None
    ref_true = 0.0
    for deg in (0.0, 20.0, 45.0, 90.0):
        T_true = pose_for(deg)
        rvec, _ = cv2.Rodrigues(T_true[:3, :3])
        tvec = T_true[:3, 3]
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, np.zeros((5, 1)))
        ok, rvec_s, tvec_s = cv2.solvePnP(
            obj_pts, img_pts, K, np.zeros((5, 1)), flags=cv2.SOLVEPNP_ITERATIVE
        )
        assert ok
        T_solved = rvec_tvec_to_matrix(rvec_s, tvec_s)
        if ref_solved is None:
            ref_solved = T_solved
            ref_true = deg
            continue
        yaw = relative_object_yaw_deg(ref_solved, T_solved)
        err = abs(((yaw - (deg - ref_true) + 180) % 360) - 180)
        worst = max(worst, err)
    return worst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Run the synthetic self-test and exit.")
    parser.add_argument("--clip", type=Path, default=None, help="Folder of frames to process.")
    parser.add_argument(
        "--board",
        type=str,
        default="DICT_5X5_100:5x7:0.03:0.022",
        help="Board spec DICT:SXxSY:square_len_m:marker_len_m, or a capture.json path.",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=None,
        help="Optional shared intrinsics 'fx,fy,cx,cy' if frames have no JSON.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write per-frame results JSON here.")
    parser.add_argument("--plot", type=Path, default=None, help="Write a yaw-vs-time PNG here.")
    parser.add_argument("--max-reproj-px", type=float, default=DEFAULT_MAX_REPROJ_PX)
    return parser.parse_args()


def _board_spec_from_arg(board_arg: str) -> BoardSpec:
    path = Path(board_arg)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        board_block = data.get("board", data)
        return BoardSpec.from_dict(board_block)
    return BoardSpec.parse(board_arg)


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest()

    if args.clip is None:
        raise SystemExit("Nothing to do: pass --selftest or --clip DIR.")

    spec = _board_spec_from_arg(args.board)
    camera_matrix = None
    if args.intrinsics:
        fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
        camera_matrix = intrinsics_matrix(fx, fy, cx, cy)

    summary = process_clip(
        args.clip.resolve(), spec, camera_matrix, max_reproj_px=args.max_reproj_px
    )
    print(
        f"frames={summary['n_frames']} valid={summary['n_valid']} "
        f"median_reproj={summary['median_reproj_px']} "
        f"yaw_span_deg={summary['yaw_span_deg']}"
    )
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    if args.plot:
        if maybe_plot(summary, args.plot):
            print(f"Wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
