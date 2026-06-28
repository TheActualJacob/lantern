from __future__ import annotations

import json
import math

import numpy as np
import pytest

from conftest import REPO_ROOT  # noqa: F401  (ensures repo root on sys.path)

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

import charuco_pose as cp


def test_board_spec_parse_round_trip() -> None:
    spec = cp.BoardSpec.parse("DICT_5X5_100:5x7:0.03:0.022")
    assert spec.squares_x == 5 and spec.squares_y == 7
    assert spec.square_len_m == pytest.approx(0.03)
    assert spec.marker_len_m == pytest.approx(0.022)
    again = cp.BoardSpec.from_dict(spec.to_dict())
    assert again == spec


def test_board_spec_parse_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        cp.BoardSpec.parse("DICT_5X5_100:5x7:0.03")  # too few fields
    with pytest.raises(ValueError):
        cp.BoardSpec.parse("DICT_5X5_100:57:0.03:0.022")  # no grid 'x'


def test_resolve_dictionary_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
        cp._resolve_dictionary("DICT_NOT_REAL")


def test_rvec_tvec_to_matrix_identity() -> None:
    T = cp.rvec_tvec_to_matrix(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(T, expected, atol=1e-12)


def test_yaw_about_z_recovers_angle() -> None:
    for deg in (0.0, 30.0, -45.0, 90.0):
        theta = math.radians(deg)
        Rz = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert cp.yaw_about_z_deg(Rz) == pytest.approx(deg, abs=1e-6)


def test_relative_object_yaw_is_difference() -> None:
    T0 = cp.rvec_tvec_to_matrix(
        cv2.Rodrigues(_rz(20.0))[0], np.array([0, 0, 0.4])
    )
    T1 = cp.rvec_tvec_to_matrix(
        cv2.Rodrigues(_rz(65.0))[0], np.array([0, 0, 0.4])
    )
    assert cp.relative_object_yaw_deg(T0, T1) == pytest.approx(45.0, abs=1e-4)


def test_unwrap_degrees_monotonic() -> None:
    wrapped = [170.0, 175.0, -179.0, -170.0]
    unwrapped = cp.unwrap_degrees(wrapped)
    # Differences should be small and positive after unwrapping (monotonic spin).
    diffs = np.diff(unwrapped)
    assert np.all(diffs > 0)
    assert unwrapped[-1] > 180.0


def test_geometry_selftest_recovers_yaw() -> None:
    spec = cp.BoardSpec()
    K = cp.intrinsics_matrix(1100.0, 1100.0, 640.0, 480.0)
    err = cp._selftest_geometry(spec, K)
    assert err < 1e-3


def test_estimate_pose_on_rendered_board() -> None:
    spec = cp.BoardSpec()
    board, detector = cp.build_board(spec)
    image_size = (1280, 960)
    K = cp.intrinsics_matrix(1100.0, 1100.0, image_size[0] / 2, image_size[1] / 2)

    width_m = spec.squares_x * spec.square_len_m
    height_m = spec.squares_y * spec.square_len_m
    center = np.array([width_m / 2, height_m / 2, 0.0])
    T = np.eye(4)
    T[:3, :3] = _rz(25.0)
    T[:3, 3] = np.array([0.0, 0.0, 0.4]) - _rz(25.0) @ center

    view = cp._render_board_view(board, spec, T, K, image_size)
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    result = cp.estimate_pose(gray, board, detector, K)

    assert result.valid
    assert result.n_corners >= cp.MIN_CHARUCO_CORNERS
    assert result.reproj_px < 1.0


def test_process_clip_reports_yaw_span(tmp_path) -> None:
    spec = cp.BoardSpec()
    board, _ = cp.build_board(spec)
    image_size = (1280, 960)
    fx = fy = 1100.0
    K = cp.intrinsics_matrix(fx, fy, image_size[0] / 2, image_size[1] / 2)

    width_m = spec.squares_x * spec.square_len_m
    height_m = spec.squares_y * spec.square_len_m
    center = np.array([width_m / 2, height_m / 2, 0.0])

    intr = {
        "fx": fx,
        "fy": fy,
        "cx": image_size[0] / 2,
        "cy": image_size[1] / 2,
        "width": image_size[0],
        "height": image_size[1],
    }

    for i, deg in enumerate((0.0, 20.0, 40.0)):
        T = np.eye(4)
        T[:3, :3] = _rz(deg)
        T[:3, 3] = np.array([0.0, 0.0, 0.4]) - _rz(deg) @ center
        view = cp._render_board_view(board, spec, T, K, image_size)
        cv2.imwrite(str(tmp_path / f"frame_{i:04d}.png"), view)
        (tmp_path / f"frame_{i:04d}.json").write_text(
            json.dumps({"intrinsics": intr})
        )

    summary = cp.process_clip(tmp_path, spec)
    assert summary["n_valid"] == 3
    assert summary["yaw_span_deg"] == pytest.approx(40.0, abs=1.0)
    assert summary["median_reproj_px"] < 1.0


def _rz(deg: float) -> np.ndarray:
    theta = math.radians(deg)
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
