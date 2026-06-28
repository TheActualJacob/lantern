from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import REPO_ROOT  # noqa: F401  (ensures repo root on sys.path)

import object_frame as of


def _rigid(yaw_deg: float, t: tuple[float, float, float]) -> np.ndarray:
    theta = np.radians(yaw_deg)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def test_capture_mode_defaults_to_orbit() -> None:
    assert of.capture_mode(None) == of.ORBIT_MODE
    assert of.capture_mode({}) == of.ORBIT_MODE
    assert of.capture_mode({"capture_mode": "turntable"}) == of.TURNTABLE_MODE
    assert of.capture_mode({"capture_mode": "nonsense"}) == of.ORBIT_MODE


def test_resolve_fusion_frame_auto_and_explicit() -> None:
    assert of.resolve_fusion_frame("auto", None) == "world"
    assert of.resolve_fusion_frame("auto", {"capture_mode": "turntable"}) == "object"
    assert of.resolve_fusion_frame("world", {"capture_mode": "turntable"}) == "world"
    assert of.resolve_fusion_frame("object", None) == "object"
    with pytest.raises(ValueError):
        of.resolve_fusion_frame("bogus", None)


def test_invert_pose_round_trips() -> None:
    T = _rigid(37.0, (0.1, -0.2, 0.5))
    np.testing.assert_allclose(of.invert_pose(T) @ T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(
        of.object_pose_to_camera_pose(T), np.linalg.inv(T), atol=1e-12
    )


def test_load_capture_meta_missing_returns_none(tmp_path) -> None:
    assert of.load_capture_meta(tmp_path) is None
    (tmp_path / "capture.json").write_text(json.dumps({"capture_mode": "turntable"}))
    assert of.load_capture_meta(tmp_path)["capture_mode"] == "turntable"


def test_load_object_poses_aligns_to_depth_order(tmp_path) -> None:
    # Two frames; only the first has a board pose.
    for base in ("frame-000000", "frame-000001"):
        (tmp_path / f"{base}.depth.png").write_bytes(b"\x00")
    pose0 = _rigid(10.0, (0.0, 0.0, 0.4))
    of.write_object_pose_txt(tmp_path / "frame-000000.object_pose.txt", pose0)

    poses = of.load_object_poses(tmp_path)
    assert len(poses) == 2
    np.testing.assert_allclose(poses[0], pose0, atol=1e-9)
    assert poses[1] is None


def test_select_fusion_inputs_world_passthrough() -> None:
    world = [_rigid(0, (i, 0, 0)) for i in range(3)]
    poses, kept, conv = of.select_fusion_inputs("world", world, None, [0, 2])
    assert kept == [0, 2]
    assert conv == of.WORLD_FUSION_CONVENTION
    np.testing.assert_allclose(poses[0], world[0])
    np.testing.assert_allclose(poses[1], world[2])


def test_select_fusion_inputs_object_drops_invalid() -> None:
    world = [np.eye(4) for _ in range(3)]
    T_CO_0 = _rigid(15, (0, 0, 0.4))
    T_CO_2 = _rigid(45, (0, 0, 0.4))
    object_poses = [T_CO_0, None, T_CO_2]

    poses, kept, conv = of.select_fusion_inputs(
        "object", world, object_poses, [0, 1, 2]
    )
    assert kept == [0, 2]  # frame 1 has no board pose -> dropped
    assert conv == of.OBJECT_FUSION_CONVENTION
    # Pose passed to fuse() must be camera<-object = inv(T_CO).
    np.testing.assert_allclose(poses[0], of.invert_pose(T_CO_0), atol=1e-9)
    np.testing.assert_allclose(poses[1], of.invert_pose(T_CO_2), atol=1e-9)


def test_select_fusion_inputs_object_requires_poses() -> None:
    with pytest.raises(ValueError, match="no object poses"):
        of.select_fusion_inputs("object", [np.eye(4)], None, [0])


def test_select_fusion_inputs_object_all_invalid_errors() -> None:
    with pytest.raises(ValueError, match="no frames with a valid board pose"):
        of.select_fusion_inputs("object", [np.eye(4)], [None], [0])
