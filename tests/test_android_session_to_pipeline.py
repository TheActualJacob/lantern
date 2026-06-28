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


def test_convert_session_carries_turntable_object_pose(tmp_path: Path) -> None:
    session = tmp_path / "session_tt"
    frames = tmp_path / "frames"
    arcore = tmp_path / "arcore"
    session.mkdir()

    color = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = np.full((2, 3), 1000, dtype=np.uint16)
    confidence = np.full((2, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(session / "frame_0001.png"), color)
    cv2.imwrite(str(session / "depth_0001.png"), depth)
    cv2.imwrite(str(session / "conf_0001.png"), confidence)
    cv2.imwrite(str(session / "frame_0002.png"), color)
    cv2.imwrite(str(session / "depth_0002.png"), depth)
    cv2.imwrite(str(session / "conf_0002.png"), confidence)

    pose = np.eye(4, dtype=np.float64)
    intr = {"fx": 600.0, "fy": 500.0, "cx": 300.0, "cy": 200.0, "width": 6, "height": 4}
    T_CO = np.eye(4, dtype=np.float64)
    T_CO[:3, 3] = [0.01, 0.02, 0.4]

    # Frame 1 saw the board; frame 2 did not (object_pose_valid false).
    (session / "frame_0001.json").write_text(
        json.dumps(
            {
                "pose_matrix_column_major": pose.reshape(-1, order="F").tolist(),
                "intrinsics": intr,
                "object_pose_valid": True,
                "object_pose_T_CO": T_CO.tolist(),
            }
        )
    )
    (session / "frame_0002.json").write_text(
        json.dumps(
            {
                "pose_matrix_column_major": pose.reshape(-1, order="F").tolist(),
                "intrinsics": intr,
                "object_pose_valid": False,
            }
        )
    )

    # Per-session capture sidecar.
    (session / "capture.json").write_text(
        json.dumps({"capture_mode": "turntable", "object_frame": "charuco"})
    )

    count = convert_session(session, frames, arcore)
    assert count == 2

    # Sidecar carried through.
    carried = json.loads((arcore / "capture.json").read_text())
    assert carried["capture_mode"] == "turntable"

    # Frame 0 has an object pose; frame 1 does not.
    assert (arcore / "frame-000000.object_pose.txt").exists()
    assert not (arcore / "frame-000001.object_pose.txt").exists()
    np.testing.assert_allclose(
        np.loadtxt(arcore / "frame-000000.object_pose.txt"), T_CO, atol=1e-9
    )
