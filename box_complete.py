"""
box_complete.py — complete a partial box scan into a watertight, boxlike mesh (demo, Person D).

STANDALONE post-process (no production/model code). A one-sided scan is an open, warped
shell — it never sees the back, so it reads as a lump, not a box. This:
  1. Fits the object's oriented bounding box (RANSAC planes; reuses cuboid_fit).
  2. Builds a COMPLETE, watertight box that fills the whole volume.
  3. Embosses the REAL observed surface onto the faces we actually scanned (so it keeps
     scan character where we have data), and leaves clean flat faces where we don't
     (the unobserved back). Laplacian-smoothed to kill lumpiness.

Result: a closed, boxlike mesh that "fills in the empty space" — honest framing is
"observed faces carry the scan; unobserved volume is completed to the fitted box."

USAGE
  python box_complete.py <da3_dump_dir | mesh.glb | cloud.ply> --out box_filled.glb
    --emboss F     max surface relief as fraction of the box's thin dim (default 0.12; 0 = pure box)
    --subdiv N     box subdivision level (default 4 → denser relief)
    --smooth N     laplacian smoothing iterations (default 6)

Needs: numpy, scipy, trimesh, open3d (for dump back-projection). Reuses cuboid_fit.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from cuboid_fit import fit_cuboid, load_points


def backproject_dump(d: Path, conf_pct=25) -> np.ndarray:
    import open3d as o3d  # noqa: F401  (only needed for the .npy path; numpy does the work)
    depth = np.load(d / "depth.npy"); conf = np.load(d / "conf.npy")
    masks = np.load(d / "object_masks.npy"); K = np.load(d / "intrinsics.npy")
    ext = np.load(d / "extrinsics_world2cam.npy")
    N, h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    pts = []
    for i in range(N):
        Ki, R, t = K[i], ext[i, :, :3], ext[i, :, 3]
        m = (masks[i] > 0) & (depth[i] > 0) & (conf[i] >= np.percentile(conf[i], conf_pct))
        z = depth[i][m]; u = uu[m]; v = vv[m]
        Xc = np.stack([(u - Ki[0, 2]) / Ki[0, 0] * z, (v - Ki[1, 2]) / Ki[1, 1] * z, z], 1)
        pts.append((Xc - t) @ R)
    return np.concatenate(pts)


def get_points(src: str) -> np.ndarray:
    p = Path(src)
    if p.is_dir() and (p / "depth.npy").exists():
        return backproject_dump(p)
    return load_points(src)


def complete_box(points, emboss=0.12, subdiv=4, smooth=6):
    R, center, dims, method, cov = fit_cuboid(points, pct=1.0, thresh=None)
    print(f"[complete] box fit: {method}  dims={np.round(dims,4)}  "
          f"aspect 1:{np.sort(dims)[1]/np.sort(dims)[0]:.2f}:{np.sort(dims)[2]/np.sort(dims)[0]:.2f}")

    # dense, watertight box in the box-local frame (axis-aligned, centered at origin)
    box = trimesh.creation.box(extents=dims)
    for _ in range(subdiv):
        box = box.subdivide()
    box.merge_vertices()

    # observed points in box-local frame
    local = (points - center) @ R
    tree = cKDTree(local)

    V = box.vertices
    Nrm = box.vertex_normals
    d, idx = tree.query(V, k=1)
    thin = float(np.min(dims))
    near = d < thin * 0.5                      # only emboss faces we actually scanned
    target = local[idx]
    # signed relief along the vertex normal, clamped so it stays boxlike
    offset = np.einsum("ij,ij->i", target - V, Nrm)
    offset = np.clip(offset, -emboss * thin, emboss * thin)
    offset[~near] = 0.0
    box.vertices = V + Nrm * offset[:, None]

    if smooth:
        trimesh.smoothing.filter_laplacian(box, iterations=smooth)

    # back to world
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = center
    box.apply_transform(T)
    box.fix_normals()
    return box, dims, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="output/box_filled.glb")
    ap.add_argument("--emboss", type=float, default=0.12)
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--smooth", type=int, default=6)
    a = ap.parse_args()

    pts = get_points(a.src)
    if len(pts) < 50:
        sys.exit("ERROR: too few points")
    print(f"[complete] {len(pts)} points")
    box, dims, cov = complete_box(pts, a.emboss, a.subdiv, a.smooth)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    box.export(out)
    box.export(out.with_suffix(".stl"))
    print(f"[complete] watertight={box.is_watertight}  verts={len(box.vertices)} tris={len(box.faces)}")
    print(f"[complete] wrote {out} (+ .stl)")
    weak = np.where(cov < 0.15)[0]
    if len(weak):
        print(f"[complete] note: axes {list(weak)} were one-sided — those faces are completed (flat), not scanned.")


if __name__ == "__main__":
    raise SystemExit(main())
