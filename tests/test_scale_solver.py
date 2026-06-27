from __future__ import annotations

import numpy as np
import pytest

from scale_solver import solve_affine, solve_affine_ransac


def depth_mm_from_affine(pred_disp: np.ndarray, scale: float, shift: float) -> np.ndarray:
    metric_disp = scale * pred_disp + shift
    return (1000.0 / metric_disp).astype(np.float32)


def test_solve_affine_recovers_synthetic_affine() -> None:
    pred = np.linspace(0.2, 2.5, 64, dtype=np.float32).reshape(8, 8)
    true_s = 1.75
    true_t = 0.4
    depth_mm = depth_mm_from_affine(pred, true_s, true_t)
    mask = np.ones_like(pred, dtype=bool)

    recovered_s, recovered_t = solve_affine(pred, depth_mm, mask)

    assert recovered_s == pytest.approx(true_s, rel=1e-5)
    assert recovered_t == pytest.approx(true_t, rel=1e-5)


def test_solve_affine_filters_invalid_and_saturated_depths() -> None:
    pred = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, np.nan]], dtype=np.float32)
    true_s = 2.0
    true_t = 0.25
    depth_mm = depth_mm_from_affine(pred, true_s, true_t)
    depth_mm[0, 0] = 0.0
    depth_mm[0, 1] = 65535.0
    depth_mm[1, 2] = 1200.0
    mask = np.ones_like(pred, dtype=bool)
    mask[1, 2] = False

    recovered_s, recovered_t = solve_affine(pred, depth_mm, mask)

    assert recovered_s == pytest.approx(true_s, rel=1e-5)
    assert recovered_t == pytest.approx(true_t, rel=1e-5)


def test_solve_affine_rejects_shape_mismatch() -> None:
    pred = np.ones((2, 2), dtype=np.float32)
    depth_mm = np.ones((2, 3), dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="matching shapes"):
        solve_affine(pred, depth_mm, mask)


def test_solve_affine_ransac_recovers_with_outliers() -> None:
    rng = np.random.default_rng(7)
    pred = rng.uniform(0.2, 4.0, size=(25, 20)).astype(np.float32)
    true_s = 1.35
    true_t = 0.55
    depth_mm = depth_mm_from_affine(pred, true_s, true_t)

    outlier_indices = rng.choice(pred.size, size=20, replace=False)
    depth_flat = depth_mm.reshape(-1)
    depth_flat[outlier_indices] = rng.uniform(4000.0, 12000.0, size=outlier_indices.size)
    mask = np.ones_like(pred, dtype=bool)

    recovered_s, recovered_t = solve_affine_ransac(pred, depth_mm, mask, n_iter=500)

    assert recovered_s == pytest.approx(true_s, rel=0.02)
    assert recovered_t == pytest.approx(true_t, abs=0.03)
