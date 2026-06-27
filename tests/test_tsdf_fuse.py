from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import fresh_import


def write_pose(path: Path, value: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = value
    np.savetxt(path, pose)
    return pose


def test_load_poses_sorts_pose_files(tmp_path, monkeypatch) -> None:
    tsdf_fuse = fresh_import("tsdf_fuse", monkeypatch, fake_open3d=True)
    later_pose = write_pose(tmp_path / "frame-002.pose.txt", 2.0)
    earlier_pose = write_pose(tmp_path / "frame-001.pose.txt", 1.0)

    poses = tsdf_fuse.load_poses(str(tmp_path))

    assert len(poses) == 2
    np.testing.assert_allclose(poses[0], earlier_pose)
    np.testing.assert_allclose(poses[1], later_pose)


def test_load_poses_rejects_missing_and_malformed_inputs(tmp_path, monkeypatch) -> None:
    tsdf_fuse = fresh_import("tsdf_fuse", monkeypatch, fake_open3d=True)

    with pytest.raises(FileNotFoundError, match="No .pose.txt files"):
        tsdf_fuse.load_poses(str(tmp_path))

    malformed = tmp_path / "bad.pose.txt"
    np.savetxt(malformed, np.ones((3, 4)))
    with pytest.raises(ValueError, match="4x4 matrix"):
        tsdf_fuse.load_poses(str(tmp_path))


def test_color_path_for_depth_handles_depth_suffix(monkeypatch) -> None:
    tsdf_fuse = fresh_import("tsdf_fuse", monkeypatch, fake_open3d=True)

    assert tsdf_fuse._color_path_for_depth(Path("frame-000.depth.png")) == Path(
        "frame-000.color.png"
    )
    assert tsdf_fuse._color_path_for_depth(Path("frame-000.png")) == Path(
        "frame-000.color.png"
    )


def test_export_glb_writes_parent_directory(tmp_path, monkeypatch) -> None:
    pytest.importorskip("trimesh")
    tsdf_fuse = fresh_import("tsdf_fuse", monkeypatch, fake_open3d=True)

    mesh = type(
        "Mesh",
        (),
        {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            "triangles": np.array([[0, 1, 2]], dtype=np.int64),
            "vertex_colors": np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        },
    )()
    output_path = tmp_path / "nested" / "mesh.glb"

    tsdf_fuse._export_glb(mesh, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
