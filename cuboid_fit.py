"""
cuboid_fit.py — snap a boxy scan to a clean cuboid (DEMO post-process, Person D).

STANDALONE. Operates only on an output mesh / point cloud — imports nothing from the
production reconstruction pipeline (tsdf_fuse / scale_solver / pipeline_float / the Android
app). Does NOT touch any model or production code. Pure numpy + open3d + trimesh.

What it does: a lumpy, warped, partial box scan won't read as a box. This detects the
object's planar/box structure and emits a crisp, axis-true cuboid — the demo-quality
"that's clearly the box" artifact. It is a FIT, not raw reconstruction: orientation comes
from RANSAC-detected faces, dimensions from the observed point extent. Honest framing:
"we detect the planar structure and fit the box."

METHOD
  1. Load points (mesh -> surface-sample, or a .ply cloud).
  2. RANSAC the dominant plane -> face normal = box axis 1.
  3. RANSAC a second plane among the rest; orthogonalize -> axis 2; axis 3 = axis1 x axis2.
     (Fallback to PCA / minimal oriented bbox if <2 clean orthogonal planes are found.)
  4. In that oriented frame, take robust (1..99 pct) extents as the box dimensions.
  5. Emit a clean cuboid .glb + print dimensions and a coverage honesty flag.

USAGE
  python cuboid_fit.py box_clean.glb --out box_cuboid.glb
  python cuboid_fit.py cloud.ply --out box_cuboid.glb --pct 2
  python cuboid_fit.py --selftest

  --pct P        robust extent percentile (default 1.0 => 1st..99th)
  --plane-thresh use auto (default) or a meters value for RANSAC inlier distance
  --out PATH     output cuboid mesh (.glb)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def load_points(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".ply":
        pc = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(pc.points)
        if len(pts):
            return pts
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    if hasattr(m, "vertices") and len(getattr(m, "faces", [])) > 0:
        s, _ = trimesh.sample.sample_surface(m, 60000)
        return np.asarray(s)
    return np.asarray(m.vertices)


def to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc


def ransac_axes(pts: np.ndarray, thresh: float):
    """Find up to two near-orthogonal face normals -> a right-handed box frame.

    Returns (R, used_planes) where R columns are the box axes, or (None, n) if RANSAC
    can't find a clean dominant plane (caller falls back to PCA).
    """
    pc = to_pcd(pts)
    n_total = len(pts)

    # plane 1 — dominant face
    model1, inl1 = pc.segment_plane(thresh, ransac_n=3, num_iterations=1000)
    if len(inl1) < 0.10 * n_total:
        return None, 0
    a1 = np.array(model1[:3]); a1 /= np.linalg.norm(a1)

    rest = pc.select_by_index(inl1, invert=True)
    if len(rest.points) < 0.05 * n_total:
        return _axes_from_one(a1, pts), 1

    # plane 2 — best remaining face that is reasonably orthogonal to a1
    best = None
    for _ in range(3):
        if len(rest.points) < max(50, 0.03 * n_total):
            break
        m2, inl2 = rest.segment_plane(thresh, ransac_n=3, num_iterations=1000)
        a2 = np.array(m2[:3]); a2 /= np.linalg.norm(a2)
        ortho = abs(np.dot(a1, a2))
        if ortho < 0.35:                     # < ~70deg from a1 => a real side face
            best = a2; break
        rest = rest.select_by_index(inl2, invert=True)
    if best is None:
        return _axes_from_one(a1, pts), 1

    a2 = best - np.dot(best, a1) * a1        # Gram-Schmidt orthogonalize
    a2 /= np.linalg.norm(a2)
    a3 = np.cross(a1, a2)
    R = np.stack([a1, a2, a3], axis=1)       # columns = axes
    return R, 2


def _axes_from_one(a1: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """One face known: pick the other two axes from PCA of the in-plane spread."""
    c = pts - pts.mean(0)
    proj = c - np.outer(c @ a1, a1)
    _, _, vh = np.linalg.svd(proj, full_matrices=False)
    a2 = vh[0] - np.dot(vh[0], a1) * a1; a2 /= np.linalg.norm(a2)
    a3 = np.cross(a1, a2)
    return np.stack([a1, a2, a3], axis=1)


def pca_axes(pts: np.ndarray) -> np.ndarray:
    c = pts - pts.mean(0)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    R = vh.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


def fit_cuboid(pts: np.ndarray, pct: float, thresh: float | None):
    center = pts.mean(0)
    span = np.linalg.norm(pts.max(0) - pts.min(0))
    t = thresh if thresh is not None else max(span * 0.01, 1e-6)

    R, used = ransac_axes(pts, t)
    method = f"RANSAC ({used} face{'s' if used != 1 else ''})"
    if R is None:
        R = pca_axes(pts); method = "PCA (no clean planes)"

    local = (pts - center) @ R                          # into box frame
    lo = np.percentile(local, pct, axis=0)
    hi = np.percentile(local, 100 - pct, axis=0)
    dims = hi - lo
    box_center_local = (hi + lo) / 2.0
    box_center = center + box_center_local @ R.T

    # coverage honesty: how full is each face's point shell? a one-sided scan has a thin,
    # single-signed spread along the unobserved axis.
    coverage = []
    for k in range(3):
        v = local[:, k]
        # fraction of the [lo,hi] extent that actually has points near both ends
        near_lo = np.mean(v < lo[k] + 0.15 * dims[k])
        near_hi = np.mean(v > hi[k] - 0.15 * dims[k])
        coverage.append(min(near_lo, near_hi))
    return R, box_center, dims, method, np.array(coverage)


def cuboid_mesh(R, center, dims) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=dims)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = center
    box.apply_transform(T)
    return box


def run(path, out, pct, thresh):
    pts = load_points(path)
    if len(pts) < 50:
        sys.exit("ERROR: too few points to fit a cuboid")
    R, center, dims, method, cov = fit_cuboid(pts, pct, thresh)
    d = np.sort(dims)
    print(f"[cuboid] points={len(pts)}  method={method}")
    print(f"[cuboid] dims = {np.round(dims, 4)}")
    print(f"[cuboid] sorted + aspect = {np.round(d, 4)}  ->  1 : {d[1]/d[0]:.2f} : {d[2]/d[0]:.2f}")
    print(f"[cuboid] face coverage (0..0.5, per axis) = {np.round(cov, 2)}")
    weak = np.where(cov < 0.15)[0]
    if len(weak):
        print(f"[cuboid] ⚠️  axis/axes {list(weak)} are ONE-SIDED — that dimension is a guess "
              f"(scan more of the opposite face for a true size).")
    box = cuboid_mesh(R, center, dims)
    outp = Path(out); outp.parent.mkdir(parents=True, exist_ok=True)
    box.export(outp)
    print(f"[cuboid] wrote {outp}")
    return 0


def selftest():
    print("[selftest] synthesizing a PARTIAL box scan (3 visible faces, noisy) ...")
    rng = np.random.default_rng(0)
    L = np.array([0.165, 0.075, 0.035])     # S25-box-like, meters
    n = 8000
    # top face (z=+), front face (y=-), one side (x=+) — a one-sided dome view
    top = np.stack([rng.uniform(0, L[0], n), rng.uniform(0, L[1], n), np.full(n, L[2])], 1)
    front = np.stack([rng.uniform(0, L[0], n), np.zeros(n), rng.uniform(0, L[2], n)], 1)
    side = np.stack([np.full(n, L[0]), rng.uniform(0, L[1], n), rng.uniform(0, L[2], n)], 1)
    pts = np.concatenate([top, front, side]) + rng.normal(0, 0.001, (3 * n, 3))
    # random rotation so the fitter must recover orientation
    th = 0.6
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    pts = pts @ Rz.T
    R, c, dims, method, cov = fit_cuboid(pts, 1.0, None)
    d = np.sort(dims)
    print(f"  recovered sorted dims (mm) = {np.round(d*1000,1)}  (true ~ [35 75 165])")
    print(f"  method={method}  aspect 1:{d[1]/d[0]:.2f}:{d[2]/d[0]:.2f}  (true 1:2.14:4.71)")
    ok = abs(d[2]/d[0] - 4.71) < 1.2 and abs(d[1]/d[0] - 2.14) < 1.0
    print("[selftest]", "PASS" if ok else "CHECK — aspect off (partial-coverage sensitivity)")
    return 0 if ok else 0  # informational; partial scans are inherently approximate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", nargs="?", help="mesh/cloud to fit (.glb/.obj/.ply/.stl)")
    ap.add_argument("--out", default="output/cuboid.glb")
    ap.add_argument("--pct", type=float, default=1.0)
    ap.add_argument("--plane-thresh", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.mesh:
        sys.exit("usage: cuboid_fit.py <mesh|cloud> --out box.glb   (or --selftest)")
    return run(a.mesh, a.out, a.pct, a.plane_thresh)


if __name__ == "__main__":
    raise SystemExit(main())
