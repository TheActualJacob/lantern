from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import fresh_import, install_fake_cv2, install_fake_open3d


def import_pipeline_float(monkeypatch):
    if importlib.util.find_spec("cv2") is None:
        install_fake_cv2(monkeypatch)
    install_fake_open3d(monkeypatch)
    sys.modules.pop("tsdf_fuse", None)
    return fresh_import("pipeline_float", monkeypatch)


def test_load_intrinsics_parses_values_and_dimensions(tmp_path, monkeypatch) -> None:
    pipeline_float = import_pipeline_float(monkeypatch)
    intrinsics_path = tmp_path / "intrinsics.txt"
    intrinsics_path.write_text(
        "fx=501.5 fy=502.5 cx=320.25 cy=240.75 width=640.0 height=480",
        encoding="utf-8",
    )

    intrinsics = pipeline_float.load_intrinsics(intrinsics_path)

    assert intrinsics == {
        "fx": 501.5,
        "fy": 502.5,
        "cx": 320.25,
        "cy": 240.75,
        "width": 640,
        "height": 480,
    }


def test_load_intrinsics_rejects_missing_keys(tmp_path, monkeypatch) -> None:
    pipeline_float = import_pipeline_float(monkeypatch)
    intrinsics_path = tmp_path / "intrinsics.txt"
    intrinsics_path.write_text("fx=1 fy=1 cx=0 cy=0 width=2", encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        pipeline_float.load_intrinsics(intrinsics_path)


def test_moving_average_trailing_uses_only_past_values(monkeypatch) -> None:
    pipeline_float = import_pipeline_float(monkeypatch)

    values = np.array([10.0, 20.0, 40.0, 80.0])
    smoothed = pipeline_float.moving_average_trailing(values, window=3)

    np.testing.assert_allclose(smoothed, np.array([10.0, 15.0, 70.0 / 3.0, 140.0 / 3.0]))


def test_disp_max_for_frame_accepts_common_keys(monkeypatch) -> None:
    pipeline_float = import_pipeline_float(monkeypatch)

    disp_maxes = {"frame-000": 3.25}
    value = pipeline_float.disp_max_for_frame(
        disp_maxes,
        Path("frame-000.disp.png"),
        Path("frame-000.color.png"),
        Path("frame-000.depth.png"),
    )

    assert value == pytest.approx(3.25)


def test_load_pred_disp_reconstructs_float_disparity(monkeypatch) -> None:
    pipeline_float = import_pipeline_float(monkeypatch)
    raw = np.array([[0, 32768, 65535]], dtype=np.uint16)
    monkeypatch.setattr(pipeline_float, "read_png", lambda path, label: raw)

    pred_disp = pipeline_float.load_pred_disp(Path("frame-000.disp.png"), disp_max=10.0)

    np.testing.assert_allclose(
        pred_disp,
        np.array([[0.0, 32768.0 / 65535.0 * 10.0, 10.0]], dtype=np.float32),
        rtol=1e-6,
    )


def test_calibrated_depth_mm_clips_and_rounds(monkeypatch) -> None:
    pipeline_float = import_pipeline_float(monkeypatch)

    pred_disp = np.array([[0.5, 1.5, -10.0]], dtype=np.float32)
    depth_mm = pipeline_float.calibrated_depth_mm(pred_disp, global_s=2.0, smoothed_t=0.0)

    assert depth_mm.dtype == np.uint16
    np.testing.assert_array_equal(depth_mm, np.array([[1000, 333, 65535]], dtype=np.uint16))
