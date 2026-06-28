"""
ground_truth.py — honest accuracy check for a reconstructed mesh (Person D).

Produces the defensible "mean surface error = X mm" number for the demo. Two modes,
per roadmap Phase 3c:

  1. REFERENCE mode — surface distance vs a trusted reference scan (iPad LiDAR / Object
     Capture, or any mesh/point cloud). Reports symmetric nearest-neighbor error
     (accuracy + completeness), median, RMS, p95, and Hausdorff; writes a color-coded
     error point cloud (.ply, openable in CloudCompare/MeshLab for the visual overlay).

  2. KNOWN-DIMS mode — compare the mesh's bounding-box extents to a calibration object
     of known size (e.g. a 3D print measured with calipers). Orientation-agnostic
     (extents are sorted), so no alignment needed. Simplest, very robust for a demo.

USAGE
  # vs a reference scan (units converted to a common mm basis; optional ICP align)
  python ground_truth.py --mesh cleaned.glb --reference lidar.ply \
      --mesh-unit mm --ref-unit m --align --tol 2.0 --out ground_truth_report.txt

  # vs a known-dimension calibration object (WxHxD in mm)
  python ground_truth.py --mesh cleaned.glb --known-dims 100x200x300 \
      --mesh-unit mm --out ground_truth_report.txt

  # offline self-test (no data needed)
  python ground_truth.py --selftest

Deps: numpy, trimesh, open3d (all in requirements.txt). matplotlib optional (for a
histogram PNG); skipped with a note if absent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}


# --------------------------------------------------------------------------- IO
def load_as_trimesh(path: str) -> trimesh.Trimesh:
    """Load a mesh/scene/point cloud from any trimesh-supported format -> Trimesh."""
    obj = trimesh.load(path, force="mesh")
    if isinstance(obj, trimesh.Scene):
        obj = obj.to_geometry()
    if not isinstance(obj, trimesh.Trimesh) or obj.vertices.size == 0:
        raise ValueError(f"No usable geometry in: {path}")
    return obj


def surface_samples(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    """Even surface sampling (falls back to vertices for degenerate/point-only inputs)."""
    if mesh.faces is not None and len(mesh.faces) > 0:
        pts, _ = trimesh.sample.sample_surface(mesh, n)
        return np.asarray(pts, dtype=np.float64)
    return np.asarray(mesh.vertices, dtype=np.float64)


def to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# -------------------------------------------------------------------- distances
def nn_distances(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Nearest-neighbor distance from every src point to the tgt set (via Open3D KDTree)."""
    return np.asarray(to_pcd(src).compute_point_cloud_distance(to_pcd(tgt)))


def align_to_reference(src_pts: np.ndarray, tgt_pts: np.ndarray, tol_mm: float):
    """Global registration (FPFH+RANSAC) for arbitrary pose, then ICP refine.

    A real reference scan (iPad LiDAR etc.) is in an unknown rotation+translation
    relative to our mesh, so a local ICP from the centroid can't recover it. We first
    do feature-based global registration to get the gross pose, then refine with ICP.
    Returns (aligned_src, fitness, rmse).
    """
    src, tgt = to_pcd(src_pts.copy()), to_pcd(tgt_pts)

    # Voxel size from the data scale (~1% of the reference bbox diagonal).
    diag = float(np.linalg.norm(tgt_pts.max(0) - tgt_pts.min(0)))
    voxel = max(diag * 0.01, tol_mm)

    def prep(p):
        d = p.voxel_down_sample(voxel)
        d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
        return d, fpfh

    src_d, src_f = prep(src); tgt_d, tgt_f = prep(tgt)
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, True, voxel * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max(tol_mm * 5.0, voxel), ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    src.transform(reg.transformation)
    return np.asarray(src.points), float(reg.fitness), float(reg.inlier_rmse)


def stats(d: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(d)), "median": float(np.median(d)),
        "rms": float(np.sqrt(np.mean(d ** 2))), "p95": float(np.percentile(d, 95)),
        "max": float(np.max(d)),
    }


# ---------------------------------------------------------------------- reports
def ascii_hist(d: np.ndarray, width: int = 40, bins: int = 10) -> str:
    counts, edges = np.histogram(d, bins=bins)
    peak = max(int(counts.max()), 1)
    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / peak))
        lines.append(f"  {lo:7.2f}–{hi:7.2f} mm | {bar} {c}")
    return "\n".join(lines)


def write_color_ply(points: np.ndarray, dists: np.ndarray, tol_mm: float, out: Path) -> None:
    """Error-colored point cloud: green (on target) -> red (>= 2*tol). Open in CloudCompare."""
    scale = max(tol_mm * 2.0, 1e-6)
    f = np.clip(dists / scale, 0.0, 1.0)
    colors = np.stack([f, 1.0 - f, np.zeros_like(f)], axis=1)  # R up, G down
    pcd = to_pcd(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out), pcd)


def maybe_histogram_png(d: np.ndarray, out: Path) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return "  (matplotlib not installed — skipped PNG; see the .ply overlay instead)"
    plt.figure(figsize=(6, 3.5))
    plt.hist(d, bins=40, color="#3b7dd8")
    plt.xlabel("surface error (mm)"); plt.ylabel("points"); plt.title("Reconstruction error")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    return f"  histogram PNG : {out}"


# ------------------------------------------------------------------------ modes
def run_reference(args) -> int:
    mesh = load_as_trimesh(args.mesh); ref = load_as_trimesh(args.reference)
    m_pts = surface_samples(mesh, args.samples) * UNIT_TO_MM[args.mesh_unit]
    r_pts = surface_samples(ref, args.samples) * UNIT_TO_MM[args.ref_unit]

    align_note = "none (assumed pre-aligned)"
    if args.align:
        m_pts, fitness, rmse = align_to_reference(m_pts, r_pts, args.tol)
        align_note = f"global (FPFH+RANSAC) + ICP refine (fitness={fitness:.3f}, inlier_rmse={rmse:.2f} mm)"

    d_acc = nn_distances(m_pts, r_pts)   # accuracy: our surface -> reference
    d_comp = nn_distances(r_pts, m_pts)  # completeness: reference -> our surface
    a, c = stats(d_acc), stats(d_comp)
    hausdorff = max(a["max"], c["max"])
    within = float(np.mean(d_acc <= args.tol) * 100.0)

    out = Path(args.out)
    ply = out.with_suffix(".error.ply"); png = out.with_suffix(".hist.png")
    write_color_ply(m_pts, d_acc, args.tol, ply)
    png_note = maybe_histogram_png(d_acc, png)

    report = f"""GROUND-TRUTH VALIDATION REPORT  (reference mode)
============================================================
mesh        : {args.mesh}  [{args.mesh_unit}]
reference   : {args.reference}  [{args.ref_unit}]
samples     : {len(m_pts)} (mesh) / {len(r_pts)} (reference)
alignment   : {align_note}
tolerance   : {args.tol:.2f} mm

ACCURACY  (our surface -> nearest reference point)
  mean   = {a['mean']:.3f} mm      <-- headline accuracy number
  median = {a['median']:.3f} mm
  rms    = {a['rms']:.3f} mm
  p95    = {a['p95']:.3f} mm
  max    = {a['max']:.3f} mm
  within {args.tol:.2f} mm: {within:.1f}% of surface

COMPLETENESS  (reference -> nearest reconstructed point)
  mean   = {c['mean']:.3f} mm
  median = {c['median']:.3f} mm
  max    = {c['max']:.3f} mm

Hausdorff (symmetric, max of both) = {hausdorff:.3f} mm

ERROR DISTRIBUTION (accuracy)
{ascii_hist(d_acc)}

ARTIFACTS
  error point cloud : {ply}   (open in CloudCompare/MeshLab; green=on-target, red>=2*tol)
{png_note}
"""
    out.write_text(report)
    print(report)
    print(f"[ground_truth] wrote {out}")
    return 0


def run_known_dims(args) -> int:
    mesh = load_as_trimesh(args.mesh)
    extents = np.sort(np.asarray(mesh.extents, dtype=np.float64) * UNIT_TO_MM[args.mesh_unit])
    target = np.sort(np.array([float(x) for x in args.known_dims.lower().split("x")]))
    if len(target) != 3:
        sys.exit("ERROR: --known-dims must be WxHxD, e.g. 100x200x300")

    abs_err = np.abs(extents - target)
    pct_err = abs_err / target * 100.0
    out = Path(args.out)
    report = f"""GROUND-TRUTH VALIDATION REPORT  (known-dimension mode)
============================================================
mesh        : {args.mesh}  [{args.mesh_unit}]
method      : bounding-box extents vs caliper-measured object (sorted, orientation-agnostic)

  axis (sorted) | measured |  mesh   |  abs err | pct err
  --------------+----------+---------+----------+--------
     smallest   | {target[0]:7.2f} | {extents[0]:7.2f} | {abs_err[0]:7.2f} | {pct_err[0]:5.2f}%
      middle    | {target[1]:7.2f} | {extents[1]:7.2f} | {abs_err[1]:7.2f} | {pct_err[1]:5.2f}%
     largest    | {target[2]:7.2f} | {extents[2]:7.2f} | {abs_err[2]:7.2f} | {pct_err[2]:5.2f}%

  mean abs error = {abs_err.mean():.2f} mm   ({pct_err.mean():.2f}% mean)
  max  abs error = {abs_err.max():.2f} mm
"""
    out.write_text(report)
    print(report)
    print(f"[ground_truth] wrote {out}")
    return 0


# --------------------------------------------------------------------- selftest
def run_selftest() -> int:
    print("[selftest] reference mode: sphere vs same sphere scaled +2% ...")
    a = trimesh.creation.icosphere(subdivisions=4, radius=100.0)   # 100 mm sphere
    b = trimesh.creation.icosphere(subdivisions=4, radius=102.0)   # +2 mm radius
    pa = surface_samples(a, 20000); pb = surface_samples(b, 20000)
    d = nn_distances(pa, pb)
    s = stats(d)
    print(f"  mean={s['mean']:.3f} mm (expect ~2.0)  max={s['max']:.3f}")
    assert 1.5 < s["mean"] < 2.5, f"unexpected mean error {s['mean']}"

    print("[selftest] known-dims mode: 100x200x300 box ...")
    box = trimesh.creation.box(extents=(100.0, 200.0, 300.0))
    ext = np.sort(box.extents)
    assert np.allclose(ext, [100, 200, 300], atol=1e-6), ext
    print(f"  extents={np.round(ext,2)}  OK")

    print("[selftest] alignment: recover a known rotation+translation ...")
    base = surface_samples(box, 8000)
    R = trimesh.transformations.rotation_matrix(np.radians(50), [0.3, 1.0, 0.2])[:3, :3]
    moved = base @ R.T + np.array([400.0, -250.0, 150.0])
    aligned, fitness, rmse = align_to_reference(moved, base, tol_mm=2.0)
    err = float(np.mean(nn_distances(aligned, base)))
    print(f"  post-align mean err={err:.3f} mm  fitness={fitness:.3f}")
    assert fitness > 0.9 and err < 3.0, f"alignment failed (fit={fitness}, err={err})"
    print("[selftest] PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Ground-truth accuracy validation for a mesh.")
    p.add_argument("--mesh", help="reconstructed mesh (.glb/.ply/.obj)")
    p.add_argument("--reference", help="reference scan for surface-distance mode")
    p.add_argument("--known-dims", help="calibration object WxHxD, e.g. 100x200x300")
    p.add_argument("--mesh-unit", default="mm", choices=UNIT_TO_MM)
    p.add_argument("--ref-unit", default="m", choices=UNIT_TO_MM)
    p.add_argument("--align", action="store_true", help="ICP-align mesh to reference first")
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--tol", type=float, default=2.0, help="tolerance band in mm")
    p.add_argument("--out", default="ground_truth_report.txt")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return run_selftest()
    if not args.mesh:
        sys.exit("ERROR: --mesh is required (or use --selftest)")
    if args.reference:
        return run_reference(args)
    if args.known_dims:
        return run_known_dims(args)
    sys.exit("ERROR: provide either --reference <file> or --known-dims WxHxD")


if __name__ == "__main__":
    raise SystemExit(main())
