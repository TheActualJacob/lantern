"""Offline in-hand object-frame tracker spike (LIVE_MESH_PLAN.md §3 / §11).

This is the de-risking prototype for board-free, in-hand capture: recover the
object's pose every frame by **frame-to-model registration** (point-to-plane
ICP of the masked object cloud against the model accumulated so far), with **no
board** at runtime. The model lives in a canonical object frame ``O`` defined by
the first frame; each frame's refined ``T_OC`` (camera->object) is both the
output and the tracking reference — exactly the machinery that replaces the
board's ``solvePnP`` (KinectFusion / BundleSDF style).

Why offline first (plan §11): validate the recovered trajectory against ground
truth on recorded clips before any on-device UI. Two ground-truth sources:

* **Board ``T_CO``** (turntable sessions): ``arcore/frame-*.object_pose.txt``.
  The gold standard; compared directly when present.
* **ARCore camera pose** (orbit / static-object sessions): for a *stationary*
  object the object frame is fixed in the world, so the tracker's relative
  camera motion must reproduce ARCore's. This lets us measure tracker drift on
  any ordinary orbit clip we already have, no board needed.

Reads the canonical pipeline layout produced by ``convert_session.py``:

    <session>/frames/frame-XXXXXX.color.png      # RGB at depth resolution
    <session>/arcore/frame-XXXXXX.depth.png      # uint16 millimetres
    <session>/arcore/frame-XXXXXX.pose.txt       # 4x4 ARCore camera->world (OpenGL)
    <session>/arcore/frame-XXXXXX.object_pose.txt# 4x4 board T_CO (optional GT)
    <session>/arcore/intrinsics.txt              # fx fy cx cy width height

Pure NumPy + Open3D + OpenCV (all already in requirements.txt).

Usage:
    python inhand_tracker.py <session_dir> [--fuse-out out/inhand.glb]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "inhand_tracker needs open3d + opencv (see requirements.txt): " + str(exc)
    )


DEPTH_SCALE_MM = 1000.0  # uint16 depth PNGs are millimetres (matches tsdf_fuse)
DEPTH_MAX_M = 5.0
# Camera-frame flip between ARCore's OpenGL camera (+Y up, -Z fwd) and the
# OpenCV camera (+Y down, +Z fwd) we back-project depth into. Self-inverse.
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class Frame:
    index: int
    depth_m: np.ndarray            # (H, W) float metres, 0 = invalid
    color: Optional[np.ndarray]    # (H, W, 3) uint8 RGB or None
    cam_to_world_gl: np.ndarray    # 4x4 ARCore camera->world (OpenGL)
    board_T_CO: Optional[np.ndarray]  # 4x4 board camera<-object, or None


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)


# --------------------------------------------------------------------------- IO

def read_intrinsics(path: Path) -> Intrinsics:
    tokens = dict(tok.split("=", 1) for tok in path.read_text().split())
    return Intrinsics(
        fx=float(tokens["fx"]), fy=float(tokens["fy"]),
        cx=float(tokens["cx"]), cy=float(tokens["cy"]),
        width=int(float(tokens["width"])), height=int(float(tokens["height"])),
    )


def _read_4x4(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    mat = np.loadtxt(path, dtype=np.float64)
    return mat if mat.shape == (4, 4) else None


def load_frames(session: Path, max_frames: Optional[int] = None) -> list[Frame]:
    arcore, frames = session / "arcore", session / "frames"
    depth_paths = sorted(arcore.glob("frame-*.depth.png"))
    if not depth_paths:
        raise SystemExit(f"No frame-*.depth.png in {arcore} (run convert_session.py first).")
    if max_frames:
        depth_paths = depth_paths[:max_frames]

    out: list[Frame] = []
    for i, depth_path in enumerate(depth_paths):
        base = depth_path.name[: -len(".depth.png")]
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None or depth_raw.dtype != np.uint16:
            raise SystemExit(f"Bad depth PNG (need uint16): {depth_path}")
        depth_m = depth_raw.astype(np.float32) / DEPTH_SCALE_MM

        color_path = frames / f"{base}.color.png"
        color = None
        if color_path.exists():
            bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if bgr is not None:
                color = cv2.cvtColor(
                    cv2.resize(bgr, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )

        c2w = _read_4x4(arcore / f"{base}.pose.txt")
        if c2w is None:
            raise SystemExit(f"Missing/invalid pose: {arcore / (base + '.pose.txt')}")
        out.append(Frame(i, depth_m, color, c2w, _read_4x4(arcore / f"{base}.object_pose.txt")))
    return out


# ---------------------------------------------------------------- masking + cloud

def depth_band_mask(depth_m: np.ndarray, center_frac: float = 0.6, band: float = 0.3) -> np.ndarray:
    """Cheap stand-in for the object mask: keep the central depth blob.

    Take the median depth in a central window (the object the user centres), then
    keep pixels within ``band`` fraction of it. Isolates a centred object from a
    farther background/floor without any ML — good enough to exercise tracking;
    swap in the real SAM mask later.
    """
    h, w = depth_m.shape
    cy, cx = h // 2, w // 2
    wy, wx = int(h * center_frac / 2), int(w * center_frac / 2)
    win = depth_m[cy - wy:cy + wy, cx - wx:cx + wx]
    valid = win[(win > 0) & (win < DEPTH_MAX_M)]
    if valid.size < 8:
        return np.zeros_like(depth_m, dtype=bool)
    med = float(np.median(valid))
    return (depth_m > 0) & (np.abs(depth_m - med) <= band * med)


def backproject(depth_m: np.ndarray, mask: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Masked depth -> (N,3) point cloud in the OpenCV camera frame (metres)."""
    ys, xs = np.nonzero(mask)
    z = depth_m[ys, xs]
    keep = (z > 0) & (z < DEPTH_MAX_M)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    x = (xs - intr.cx) * z / intr.fx
    y = (ys - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=1).astype(np.float64)


def to_o3d(points: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return pc


# ------------------------------------------------------------------- tracking

def icp_to_model(
    source: o3d.geometry.PointCloud,
    model: o3d.geometry.PointCloud,
    init: np.ndarray,
    max_corr: float,
) -> tuple[np.ndarray, float]:
    """Point-to-plane ICP aligning the current cloud (camera frame) to the model
    (object frame). Returns (T_OC, fitness)."""
    result = o3d.pipelines.registration.registration_icp(
        source, model, max_corr, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    return result.transformation.copy(), float(result.fitness)


@dataclass
class TrackResult:
    poses_T_OC: list[np.ndarray]   # camera->object per frame
    fitness: list[float]
    model: o3d.geometry.PointCloud


def track(frames: list[Frame], intr: Intrinsics, voxel: float, band: float,
          min_fitness: float = 0.5) -> TrackResult:
    """Frame-to-model ICP in the object frame, mirroring the on-device ``ObjectTracker``.

    Two robustness levers (validated to take a 28°+ catastrophic divergence down to <7° mean):
      * **ARCore-ego-motion seed** — carry the last *trusted* pose along the camera's relative
        motion since that frame. Far stronger than constant velocity, and free (we record pose).
      * **Fitness gate** — drop low-overlap registrations entirely (don't fuse, don't grow the
        model, don't advance the seed), so one bad frame can't poison everything downstream.
    """
    poses: list[np.ndarray] = []
    fitness: list[float] = []
    model: Optional[o3d.geometry.PointCloud] = None
    max_corr = voxel * 5.0
    c2w_cv = [f.cam_to_world_gl @ GL_TO_CV for f in frames]
    last_good_pose = np.eye(4)
    last_good_idx: Optional[int] = None

    for i, f in enumerate(frames):
        intr_d = intr.scaled_to(f.depth_m.shape[1], f.depth_m.shape[0])
        cloud = to_o3d(backproject(f.depth_m, depth_band_mask(f.depth_m, band=band), intr_d), voxel)

        if model is None or len(cloud.points) < 20:
            T = np.eye(4) if model is None else last_good_pose.copy()
            fit = 1.0 if model is None else 0.0
        else:
            # Seed from the last trusted frame + the camera ego-motion since then.
            rel = np.linalg.inv(c2w_cv[last_good_idx]) @ c2w_cv[i]  # cur cam -> last-good cam
            init = last_good_pose @ rel
            T, fit = icp_to_model(cloud, model, init, max_corr)

        poses.append(T)
        fitness.append(fit)

        accepted = model is None or fit >= min_fitness
        if not accepted:
            continue  # drop: don't fuse, grow, or advance the seed
        last_good_pose, last_good_idx = T.copy(), i

        merged = o3d.geometry.PointCloud(cloud).transform(T)  # now in object frame O
        if model is None:
            model = merged
        else:
            model += merged
            model = model.voxel_down_sample(voxel)
            model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))

    return TrackResult(poses, fitness, model if model is not None else o3d.geometry.PointCloud())


# ------------------------------------------------------------------ validation

def rot_trans_error(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    err = np.linalg.inv(truth) @ estimate
    cos = (np.trace(err[:3, :3]) - 1.0) / 2.0
    rot_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    trans_m = float(np.linalg.norm(err[:3, 3]))
    return rot_deg, trans_m


def arcore_T_OC(frames: list[Frame]) -> list[np.ndarray]:
    """ARCore ground-truth proxy: relative camera motion in OpenCV frame, with the
    object frame pinned to the first camera (so it's comparable to the tracker)."""
    c2w_cv = [f.cam_to_world_gl @ GL_TO_CV for f in frames]
    w2cv0 = np.linalg.inv(c2w_cv[0])
    return [w2cv0 @ c for c in c2w_cv]


def board_T_OC(frames: list[Frame]) -> Optional[list[np.ndarray]]:
    """Board ground truth T_OC = inv(T_CO), pinned to the first valid board frame."""
    if any(f.board_T_CO is None for f in frames):
        return None
    T_OC = [np.linalg.inv(f.board_T_CO) for f in frames]  # camera->object
    ref = np.linalg.inv(T_OC[0])
    return [ref @ t for t in T_OC]


def report(frames: list[Frame], result: TrackResult) -> None:
    board = board_T_OC(frames)
    truth = board if board is not None else arcore_T_OC(frames)
    label = "board T_CO" if board is not None else "ARCore (static-object proxy)"
    ref = np.linalg.inv(result.poses_T_OC[0])
    est = [ref @ t for t in result.poses_T_OC]

    print(f"\n  frame | fitness | rot_err(deg) | trans_err(m)   [GT: {label}]")
    print("  ------+---------+--------------+-------------")
    rots, trans = [], []
    for f, e, fit in zip(frames, est, result.fitness):
        r, t = rot_trans_error(e, truth[f.index])
        rots.append(r); trans.append(t)
        print(f"  {f.index:5d} | {fit:6.3f}  | {r:11.2f}  | {t:.4f}")
    if rots:
        print(f"\n  mean rot_err={np.mean(rots):.2f} deg  max={np.max(rots):.2f} deg  "
              f"mean trans_err={np.mean(trans):.4f} m  max={np.max(trans):.4f} m")
        print(f"  model points={len(result.model.points)}")


def fuse_with_poses(frames: list[Frame], result: TrackResult, intr: Intrinsics,
                    voxel: float, out_path: Path) -> None:
    """Fuse masked depth in the object frame using the *tracked* poses, to eyeball
    whether the recovered trajectory yields a coherent (non-smeared) shell."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * 5.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for f, T_OC in zip(frames, result.poses_T_OC):
        h, w = f.depth_m.shape
        intr_d = intr.scaled_to(w, h)
        mask = depth_band_mask(f.depth_m)
        depth_masked = np.where(mask, f.depth_m, 0.0).astype(np.float32)
        color = f.color if f.color is not None else np.full((h, w, 3), 160, np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(depth_masked),
            depth_scale=1.0, depth_trunc=DEPTH_MAX_M, convert_rgb_to_intensity=False,
        )
        cam_intr = o3d.camera.PinholeCameraIntrinsic(w, h, intr_d.fx, intr_d.fy, intr_d.cx, intr_d.cy)
        vol.integrate(rgbd, cam_intr, np.linalg.inv(T_OC))  # extrinsic = object->camera
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"\n  fused mesh -> {out_path}  (verts={len(mesh.vertices)}, tris={len(mesh.triangles)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline in-hand object-frame tracker spike.")
    ap.add_argument("session", type=Path, help="converted session dir (frames/ + arcore/)")
    ap.add_argument("--voxel", type=float, default=0.004, help="model/ICP voxel size (m)")
    ap.add_argument("--band", type=float, default=0.3, help="depth-band mask half-width (fraction)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fuse-out", type=Path, default=None, help="also fuse with tracked poses to this GLB")
    args = ap.parse_args()

    intr = read_intrinsics(args.session / "arcore" / "intrinsics.txt")
    frames = load_frames(args.session, args.max_frames)
    print(f"Loaded {len(frames)} frames; depth {frames[0].depth_m.shape[1]}x{frames[0].depth_m.shape[0]}, "
          f"color={'yes' if frames[0].color is not None else 'no'}")

    result = track(frames, intr, args.voxel, args.band)
    report(frames, result)
    if args.fuse_out:
        fuse_with_poses(frames, result, intr, args.voxel, args.fuse_out)


if __name__ == "__main__":
    main()
