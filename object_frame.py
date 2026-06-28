"""Object-frame fusion selection for turntable captures (plan §3.2, §4, §7 T4).

This is the *one* reconstruction-side change object-centric capture needs: the
TSDF fusion frame becomes selectable.

- **Orbit (today):** the object is fixed in ARCore's world; fuse using the
  camera-to-world poses (``frame-*.pose.txt``), ARCore (OpenGL) camera convention.
- **Turntable (new):** the object spins, so its surfaces land at different *world*
  coordinates every frame and the fused mesh would smear. Instead fuse in the
  object's own frame ``O`` (the ChArUco board frame). Each frame carries
  ``T_CO`` (camera<-object) from ``charuco_pose.py``; the camera pose *in the
  object frame* is its inverse ``T_OC = inv(T_CO)``. Feeding ``T_OC`` to the same
  TSDF integrator (which inverts it back to ``T_CO``) makes the rotating object
  look static, so TSDF/coverage work exactly as before.

Camera convention note
----------------------
The board pose comes from OpenCV ``solvePnP`` whose camera frame is +X right,
+Y down, +Z forward — i.e. the ``opencv`` convention already understood by
``tsdf_fuse._world_to_camera``. So object-frame fusion uses ``pose_convention=
"opencv"`` while orbit fusion keeps ``"arcore"``.

Everything here is pure NumPy + stdlib so it is trivially testable and pulls in
neither OpenCV nor Open3D.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


CAPTURE_SIDECAR = "capture.json"
OBJECT_POSE_SUFFIX = ".object_pose.txt"

ORBIT_MODE = "orbit"
TURNTABLE_MODE = "turntable"

# Which world-to-camera axis convention tsdf_fuse should apply per fusion frame.
WORLD_FUSION_CONVENTION = "arcore"
OBJECT_FUSION_CONVENTION = "opencv"


def load_capture_meta(arcore_dir: Path) -> Optional[dict]:
    """Load the per-session ``capture.json`` sidecar, or None if absent.

    Missing sidecar => a legacy/orbit session (plan §4.1: backward compatible).
    """
    path = Path(arcore_dir) / CAPTURE_SIDECAR
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def capture_mode(capture_meta: Optional[dict]) -> str:
    """Return the capture mode, defaulting to orbit when unknown/absent."""
    if not capture_meta:
        return ORBIT_MODE
    mode = str(capture_meta.get("capture_mode", ORBIT_MODE)).lower()
    return mode if mode in (ORBIT_MODE, TURNTABLE_MODE) else ORBIT_MODE


def resolve_fusion_frame(requested: str, capture_meta: Optional[dict]) -> str:
    """Resolve ``--fusion-frame {auto,world,object}`` into 'world' or 'object'.

    ``auto`` picks 'object' for turntable sessions and 'world' otherwise, so the
    default path keeps working unchanged for every existing orbit capture.
    """
    requested = (requested or "auto").lower()
    if requested == "world":
        return "world"
    if requested == "object":
        return "object"
    if requested != "auto":
        raise ValueError(
            f"fusion frame must be auto|world|object, got {requested!r}"
        )
    return "object" if capture_mode(capture_meta) == TURNTABLE_MODE else "world"


def read_pose_txt(path: Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Pose file must be 4x4, got {pose.shape}: {path}")
    return pose


def load_object_poses(arcore_dir: Path) -> list[Optional[np.ndarray]]:
    """Load ``frame-*.object_pose.txt`` (T_CO) aligned to sorted frame order.

    Returns one entry per ``frame-*.depth.png`` (the canonical frame ordering the
    pipeline uses). Frames where the board was not seen have no pose file and map
    to ``None`` (host skips or interpolates — plan §4.1).
    """
    arcore_dir = Path(arcore_dir)
    depth_paths = sorted(arcore_dir.glob("*.depth.png"))
    poses: list[Optional[np.ndarray]] = []
    for depth_path in depth_paths:
        base = depth_path.name[: -len(".depth.png")]
        pose_path = arcore_dir / f"{base}{OBJECT_POSE_SUFFIX}"
        poses.append(read_pose_txt(pose_path) if pose_path.exists() else None)
    return poses


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (uses the rotation/translation structure)."""
    T = np.asarray(T, dtype=np.float64)
    rotation = T[:3, :3]
    translation = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def object_pose_to_camera_pose(T_CO: np.ndarray) -> np.ndarray:
    """Camera-to-object pose ``T_OC = inv(T_CO)`` for the TSDF integrator.

    ``fuse()`` expects camera-to-"world" poses and inverts them internally, so
    handing it ``T_OC`` makes the volume's world the object frame ``O``.
    """
    return invert_pose(T_CO)


def select_fusion_inputs(
    fusion_frame: str,
    world_poses: list[np.ndarray],
    object_poses: Optional[list[Optional[np.ndarray]]],
    frame_indices: list[int],
) -> tuple[list[np.ndarray], list[int], str]:
    """Pick the pose list + convention for the chosen fusion frame.

    Args:
        fusion_frame: 'world' or 'object'.
        world_poses: camera-to-world poses for every accepted frame, index-aligned
            to the full frame ordering.
        object_poses: per-frame ``T_CO`` (or None) for the full frame ordering,
            required when ``fusion_frame == 'object'``.
        frame_indices: indices (into the full ordering) of the frames the depth
            calibration step accepted, in fuse order.

    Returns:
        (poses, kept_indices, pose_convention) where ``poses`` is camera-to-frame
        for the integrator, ``kept_indices`` is the subset of ``frame_indices``
        that survived (object frames need a valid ``T_CO``), and
        ``pose_convention`` is what to pass to ``tsdf_fuse.fuse``.
    """
    if fusion_frame == "world":
        poses = [world_poses[i] for i in frame_indices]
        return poses, list(frame_indices), WORLD_FUSION_CONVENTION

    if fusion_frame != "object":
        raise ValueError(f"fusion_frame must be 'world' or 'object', got {fusion_frame!r}")

    if object_poses is None:
        raise ValueError(
            "object-frame fusion requested but no object poses were loaded "
            "(missing frame-*.object_pose.txt). Is this a turntable session?"
        )

    poses: list[np.ndarray] = []
    kept: list[int] = []
    for idx in frame_indices:
        T_CO = object_poses[idx] if idx < len(object_poses) else None
        if T_CO is None:
            continue  # board not seen this frame -> skip (plan §4.1)
        poses.append(object_pose_to_camera_pose(T_CO))
        kept.append(idx)

    if not poses:
        raise ValueError(
            "object-frame fusion has no frames with a valid board pose; "
            "check the board was visible during capture."
        )
    return poses, kept, OBJECT_FUSION_CONVENTION


def write_object_pose_txt(path: Path, T_CO: np.ndarray) -> None:
    """Write a 4x4 ``T_CO`` (camera<-object) as whitespace text (np.loadtxt-ready)."""
    T = np.asarray(T_CO, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_CO must be 4x4, got {T.shape}")
    np.savetxt(path, T)
