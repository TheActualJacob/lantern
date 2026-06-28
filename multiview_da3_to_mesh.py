"""
multiview_da3_to_mesh.py — turn a multi-view DA3 dump into a clean mesh (Person D).

Consumes a `da3_outputs/<session>/` dump (the multi-view DepthAnything3 pass: per-view
depth + conf + object masks + intrinsics + world->cam extrinsics, all in one consistent
frame) and fuses it into a single object mesh.

Why this path: per-frame mono DA3 + ARCore poses is multi-view-INCONSISTENT and drifts.
The multi-view DA3 pass shares one coherent frame, so it fuses cleanly (see
docs/RECON_QUALITY_DIAGNOSIS.md). DA3 depth is up-to-scale; pass --metric-scale (or
--metric-from-arcore later) to put it in meters.

Pipeline: back-project masked, confidence-gated depth -> world cloud -> outlier trim ->
TSDF (Open3D) OR ball-pivoting -> .glb/.ply. Feed the .glb to import_and_clean.py /
process_scan.sh for the CAD-ready + accuracy + hero chain.

USAGE
  python multiview_da3_to_mesh.py da3_outputs/session_XXXX --out output/box.glb
  python multiview_da3_to_mesh.py <dir> --method tsdf --voxel 0.006 --metric-scale 0.27
  python multiview_da3_to_mesh.py <dir> --method bpa --conf-pct 25

  --method tsdf|bpa   surface method (default tsdf; bpa handles partial/open scans)
  --voxel F           TSDF voxel size in DA3 units (default 0.006)
  --conf-pct P        drop depths below this confidence percentile per view (default 20)
  --metric-scale F    multiply DA3 units by F to get meters (object's true long-axis / measured)
  --out PATH          output mesh (.glb)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def load_dump(d: Path):
    g = lambda n: np.load(d / n)
    return (g("depth.npy"), g("conf.npy"), g("object_masks.npy"),
            g("intrinsics.npy"), g("extrinsics_world2cam.npy"),
            np.load(d / "processed_images.npy"))


def backproject(depth, conf, masks, K, ext, rgb, conf_pct):
    """All views -> one world-space colored point cloud (DA3's shared frame)."""
    N, h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    pts, cols = [], []
    for i in range(N):
        Ki, R, t = K[i], ext[i, :, :3], ext[i, :, 3]
        thr = np.percentile(conf[i], conf_pct)
        m = (masks[i] > 0) & (depth[i] > 0) & (conf[i] >= thr)
        if not m.any():
            continue
        z = depth[i][m]
        u, v = uu[m], vv[m]
        Xc = np.stack([(u - Ki[0, 2]) / Ki[0, 0] * z,
                       (v - Ki[1, 2]) / Ki[1, 1] * z, z], 1)
        # X_world = R^T (X_cam - t)   (world->cam is X_cam = R X_world + t)
        pts.append((Xc - t) @ R)
        cols.append(rgb[i][m] / 255.0)
    P = np.concatenate(pts)
    C = np.clip(np.concatenate(cols), 0, 1)
    return P, C


def make_cloud(P, C, voxel_hint):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector(C)
    pc = pc.voxel_down_sample(voxel_hint)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_hint * 5, max_nn=30))
    pc.orient_normals_consistent_tangent_plane(15)
    return pc


def mesh_tsdf(P, C, voxel):
    """Poisson surface (smooth, closes gaps). Cropped to the cloud bbox."""
    pc = make_cloud(P, C, voxel)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=9)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))  # trim low-density spray
    mesh = mesh.crop(pc.get_axis_aligned_bounding_box())
    mesh.compute_vertex_normals()
    return mesh


def mesh_bpa(P, C, voxel):
    """Ball-pivoting (faithful to points, leaves real openings — good for partial scans)."""
    pc = make_cloud(P, C, voxel)
    d = np.mean(pc.compute_nearest_neighbor_distance())
    radii = o3d.utility.DoubleVector([d * 1.5, d * 3, d * 6])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, radii)
    mesh.compute_vertex_normals()
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="da3_outputs/<session> dir")
    ap.add_argument("--out", default="output/multiview_mesh.glb")
    ap.add_argument("--method", choices=["tsdf", "bpa"], default="tsdf")
    ap.add_argument("--voxel", type=float, default=0.006)
    ap.add_argument("--conf-pct", type=float, default=20)
    ap.add_argument("--metric-scale", type=float, default=1.0,
                    help="multiply DA3 units to get meters (e.g. measured long-axis / DA3 long-axis)")
    args = ap.parse_args()

    d = Path(args.dump)
    depth, conf, masks, K, ext, rgb = load_dump(d)
    print(f"[mv] {depth.shape[0]} views @ {depth.shape[2]}x{depth.shape[1]}")

    P, C = backproject(depth, conf, masks, K, ext, rgb, args.conf_pct)
    ext_xyz = P.max(0) - P.min(0)
    print(f"[mv] cloud {len(P)} pts, extents(DA3) {np.round(ext_xyz, 3)} "
          f"aspect {np.round(np.sort(ext_xyz) / np.sort(ext_xyz)[0], 2)}")

    if args.metric_scale != 1.0:
        P = P * args.metric_scale
        args.voxel *= args.metric_scale
        print(f"[mv] scaled to meters x{args.metric_scale}: extents(m) {np.round(P.max(0)-P.min(0),3)}")

    mesh = (mesh_tsdf if args.method == "tsdf" else mesh_bpa)(P, C, args.voxel)
    print(f"[mv] mesh ({args.method}): verts={len(mesh.vertices)} tris={len(mesh.triangles)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"[mv] wrote {out}")
    print(f"[mv] next: blender --background --python import_and_clean.py -- {out} "
          f"{out.with_name(out.stem + '_clean.glb')}"
          + ("" if args.metric_scale != 1.0 else "   (mesh is in DA3 units — set --metric-scale for mm-correct CAD)"))


if __name__ == "__main__":
    main()
