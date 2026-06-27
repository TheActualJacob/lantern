from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def install_fake_open3d(monkeypatch):
    """Install just enough Open3D surface for helper-only imports."""
    fake = types.ModuleType("open3d")
    fake.camera = types.SimpleNamespace(PinholeCameraIntrinsic=object)
    fake.geometry = types.SimpleNamespace(
        TriangleMesh=object,
        RGBDImage=types.SimpleNamespace(
            create_from_color_and_depth=lambda *args, **kwargs: object()
        ),
    )
    fake.io = types.SimpleNamespace(read_image=lambda path: object())
    fake.pipelines = types.SimpleNamespace(
        integration=types.SimpleNamespace(
            ScalableTSDFVolume=object,
            TSDFVolumeColorType=types.SimpleNamespace(RGB8=object()),
        )
    )
    monkeypatch.setitem(sys.modules, "open3d", fake)
    return fake


def install_fake_cv2(monkeypatch):
    fake = types.ModuleType("cv2")
    fake.IMREAD_UNCHANGED = -1
    fake.imread = lambda *args, **kwargs: None
    fake.imwrite = lambda *args, **kwargs: True
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return fake


def fresh_import(module_name: str, monkeypatch, *, fake_open3d: bool = False):
    if fake_open3d:
        install_fake_open3d(monkeypatch)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)
