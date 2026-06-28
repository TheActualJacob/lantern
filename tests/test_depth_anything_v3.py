from __future__ import annotations

import numpy as np
import pytest

from conftest import REPO_ROOT  # noqa: F401  (ensures repo root on sys.path)

import depth_anything_v3 as da3


def test_depth_to_disparity_is_inverse_depth() -> None:
    depth = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    disp = da3.depth_to_disparity(depth)
    np.testing.assert_allclose(disp, [[1.0, 0.5, 0.25]], rtol=1e-6)


def test_depth_to_disparity_floors_invalid_to_zero() -> None:
    depth = np.array(
        [[0.0, -1.0, np.nan, np.inf, 2.0]],
        dtype=np.float32,
    )
    disp = da3.depth_to_disparity(depth)
    # Non-positive / non-finite depths -> 0 disparity (treated as invalid downstream).
    np.testing.assert_array_equal(disp[0, :4], np.zeros(4, dtype=np.float32))
    assert disp[0, 4] == pytest.approx(0.5)
    assert np.all(np.isfinite(disp))


def test_depth_to_disparity_preserves_shape_and_dtype() -> None:
    depth = np.random.default_rng(0).uniform(0.5, 5.0, size=(7, 11)).astype(np.float32)
    disp = da3.depth_to_disparity(depth)
    assert disp.shape == depth.shape
    assert disp.dtype == np.float32


def test_hf_repo_for_maps_preset_to_uppercase_repo() -> None:
    assert da3._hf_repo_for("da3-small") == "depth-anything/DA3-SMALL"
    assert da3._hf_repo_for("da3-base") == "depth-anything/DA3-BASE"


def test_hf_repo_for_passes_through_full_repo_id() -> None:
    assert da3._hf_repo_for("depth-anything/DA3-SMALL") == "depth-anything/DA3-SMALL"
    assert da3._hf_repo_for("my-org/custom-da3") == "my-org/custom-da3"


def test_default_model_is_apache_phone_sized() -> None:
    assert da3.DEFAULT_MODEL == "da3-small"
