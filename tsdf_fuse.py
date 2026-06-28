"""Fuse metric depth frames into a TSDF mesh and export GLB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 5.0
INTRINSIC_KEYS = {"fx", "fy", "cx", "cy", "width", "height"}
ARCORE_TO_OPEN3D_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
POSE_CONVENTIONS = {"arcore", "opencv"}


def load_poses(arcore_dir: str) -> list[np.ndarray]:
    """Load sorted 4x4 camera-to-world pose matrices from ``*.pose.txt`` files."""
    pose_dir = Path(arcore_dir)
    if not pose_dir.exists():
        raise FileNotFoundError(f"Pose directory does not exist: {pose_dir}")
    if not pose_dir.is_dir():
        raise ValueError(f"Pose path is not a directory: {pose_dir}")

    pose_paths = sorted(pose_dir.glob("*.pose.txt"))
    if not pose_paths:
        raise FileNotFoundError(f"No .pose.txt files found in: {pose_dir}")

    poses: list[np.ndarray] = []
    for pose_path in pose_paths:
        pose = np.loadtxt(pose_path, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(
                f"Pose file must contain a 4x4 matrix, got {pose.shape}: {pose_path}"
            )
        poses.append(pose)

    return poses


def fuse(
    depth_paths: list[str],
    poses: list[np.ndarray],
    intrinsics: dict,
    voxel_size: float = 0.01,
    sdf_trunc_mult: float = 5.0,
    depth_trunc: float = DEPTH_TRUNC,
    pose_convention: str = "arcore",
    output_path: str = "output/mesh.glb",
) -> dict[str, object]:
    """Fuse 16-bit millimeter depth frames into a TSDF and export a GLB mesh.

    Args:
        depth_paths: Paths to 16-bit PNG depth images in millimeters.
        poses: Camera-to-world 4x4 pose matrices, one per depth frame.
        intrinsics: Dict with fx, fy, cx, cy, width, and height.
        voxel_size: TSDF voxel length in meters.
        sdf_trunc_mult: TSDF truncation multiplier relative to voxel_size.
        depth_trunc: Maximum depth integrated, in meters.
        pose_convention: ``arcore`` for OpenGL-style ARCore camera poses
            (+Y up, -Z forward), or ``opencv`` for Open3D/OpenCV poses
            (+Y down, +Z forward).
        output_path: Destination GLB path.
    """
    _validate_inputs(
        depth_paths,
        poses,
        intrinsics,
        voxel_size,
        sdf_trunc_mult,
        depth_trunc,
        pose_convention,
    )

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics["width"]),
        int(intrinsics["height"]),
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size),
        sdf_trunc=float(voxel_size) * float(sdf_trunc_mult),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for depth_path_str, pose in zip(depth_paths, poses, strict=True):
        depth_path = Path(depth_path_str)
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image does not exist: {depth_path}")

        depth = o3d.io.read_image(str(depth_path))
        color_path = _color_path_for_depth(depth_path)
        if color_path.exists():
            color = o3d.io.read_image(str(color_path))
        else:
            # No RGB available (e.g. C's ARCore sessions are depth+pose only).
            # Geometry comes entirely from depth; color is cosmetic for TSDF.
            d = np.asarray(depth)
            color = o3d.geometry.Image(
                np.full((d.shape[0], d.shape[1], 3), 160, dtype=np.uint8)
            )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=DEPTH_SCALE,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )

        world_to_camera = _world_to_camera(pose, pose_convention)
        volume.integrate(rgbd, camera_intrinsic, world_to_camera)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    _keep_largest_triangle_cluster(mesh)
    _export_glb(mesh, output_path)
    return mesh_stats(mesh)


def _validate_inputs(
    depth_paths: list[str],
    poses: list[np.ndarray],
    intrinsics: dict,
    voxel_size: float,
    sdf_trunc_mult: float,
    depth_trunc: float,
    pose_convention: str,
) -> None:
    if not depth_paths:
        raise ValueError("depth_paths must contain at least one depth image path")
    if len(depth_paths) != len(poses):
        raise ValueError(
            f"depth_paths and poses must have matching lengths, got "
            f"{len(depth_paths)} and {len(poses)}"
        )
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if sdf_trunc_mult <= 0:
        raise ValueError(f"sdf_trunc_mult must be positive, got {sdf_trunc_mult}")
    if depth_trunc <= 0:
        raise ValueError(f"depth_trunc must be positive, got {depth_trunc}")
    if pose_convention not in POSE_CONVENTIONS:
        raise ValueError(
            f"pose_convention must be one of {sorted(POSE_CONVENTIONS)}, "
            f"got {pose_convention!r}"
        )

    missing_keys = sorted(INTRINSIC_KEYS - set(intrinsics))
    if missing_keys:
        raise ValueError(f"intrinsics is missing required keys: {missing_keys}")

    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    if width <= 0 or height <= 0:
        raise ValueError(
            f"intrinsics width and height must be positive, got {width}x{height}"
        )

    for key in ("fx", "fy"):
        if float(intrinsics[key]) <= 0:
            raise ValueError(f"intrinsics {key} must be positive, got {intrinsics[key]}")

    for idx, pose in enumerate(poses):
        pose_array = np.asarray(pose)
        if pose_array.shape != (4, 4):
            raise ValueError(f"poses[{idx}] must have shape (4, 4), got {pose_array.shape}")
        if not np.all(np.isfinite(pose_array)):
            raise ValueError(f"poses[{idx}] contains non-finite values")


def _world_to_camera(pose: np.ndarray, pose_convention: str) -> np.ndarray:
    """Convert a camera-to-world pose to Open3D's world-to-camera extrinsic."""
    camera_to_world = np.asarray(pose, dtype=np.float64)
    world_to_camera = np.linalg.inv(camera_to_world)
    if pose_convention == "arcore":
        # ARCore poses use an OpenGL camera frame: +X right, +Y up, -Z forward.
        # Open3D RGB-D integration expects +X right, +Y down, +Z forward.
        return ARCORE_TO_OPEN3D_CAMERA @ world_to_camera
    return world_to_camera


def _color_path_for_depth(depth_path: Path) -> Path:
    if depth_path.name.endswith(".depth.png"):
        return depth_path.with_name(
            depth_path.name[: -len(".depth.png")] + ".color.png"
        )
    return depth_path.with_name(f"{depth_path.stem}.color.png")


def _keep_largest_triangle_cluster(mesh: o3d.geometry.TriangleMesh) -> None:
    if len(mesh.triangles) == 0:
        return

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_counts = np.asarray(cluster_n_triangles)
    if cluster_counts.size == 0:
        return

    largest_cluster = int(np.argmax(cluster_counts))
    triangles_to_remove = np.asarray(triangle_clusters) != largest_cluster
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()


def _export_glb(mesh: o3d.geometry.TriangleMesh, output_path: str) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    vertex_colors = None
    colors = np.asarray(mesh.vertex_colors)
    if colors.size and len(colors) == len(vertices):
        vertex_colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    trimesh_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    trimesh_mesh.export(output)


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict[str, object]:
    """Return lightweight mesh stats for host-side diagnostics."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    stats: dict[str, object] = {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "has_vertex_colors": bool(len(np.asarray(mesh.vertex_colors)) == len(vertices)),
    }
    if len(vertices):
        stats["bounds_min"] = vertices.min(axis=0).tolist()
        stats["bounds_max"] = vertices.max(axis=0).tolist()
    else:
        stats["bounds_min"] = None
        stats["bounds_max"] = None

    if len(faces):
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        try:
            edge_counts = np.bincount(trimesh_mesh.edges_unique_inverse)
            stats["boundary_edges"] = int(np.count_nonzero(edge_counts == 1))
            stats["nonmanifold_edges"] = int(np.count_nonzero(edge_counts > 2))
        except Exception:
            stats["boundary_edges"] = None
            stats["nonmanifold_edges"] = None
        try:
            stats["components"] = int(len(trimesh_mesh.split(only_watertight=False)))
        except Exception:
            stats["components"] = None
    else:
        stats["components"] = 0
        stats["boundary_edges"] = 0
        stats["nonmanifold_edges"] = 0
    return stats


if __name__ == "__main__":
    raise SystemExit(
        "Import tsdf_fuse and call fuse(depth_paths, poses, intrinsics, output_path=...)."
    )
