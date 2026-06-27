from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from android_session_to_pipeline import convert_session


def test_convert_session_writes_pipeline_layout(tmp_path: Path) -> None:
    session = tmp_path / "session_001"
    frames = tmp_path / "frames"
    arcore = tmp_path / "arcore"
    session.mkdir()

    color = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = np.full((2, 3), 1000, dtype=np.uint16)
    confidence = np.full((2, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(session / "frame_0001.png"), color)
    assert cv2.imwrite(str(session / "depth_0001.png"), depth)
    assert cv2.imwrite(str(session / "conf_0001.png"), confidence)

    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 1.5
    metadata = {
        "pose_matrix_column_major": pose.reshape(-1, order="F").tolist(),
        "intrinsics": {
            "fx": 600.0,
            "fy": 500.0,
            "cx": 300.0,
            "cy": 200.0,
            "width": 6,
            "height": 4,
        },
    }
    (session / "frame_0001.json").write_text(json.dumps(metadata), encoding="utf-8")

    count = convert_session(session, frames, arcore)

    assert count == 1
    assert (frames / "frame-000000.color.png").exists()
    assert (arcore / "frame-000000.depth.png").exists()
    assert (arcore / "frame-000000.conf.png").exists()
    np.testing.assert_allclose(
        np.loadtxt(arcore / "frame-000000.pose.txt"), pose
    )
    intrinsics_text = (arcore / "intrinsics.txt").read_text(encoding="utf-8")
    assert "fx=300.0" in intrinsics_text
    assert "fy=250.0" in intrinsics_text
    assert "width=3" in intrinsics_text
    assert "height=2" in intrinsics_text
