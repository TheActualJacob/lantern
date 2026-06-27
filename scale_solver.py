"""Fit an affine scale and shift from relative to metric disparity."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


SATURATED_DEPTH_MM = 65535.0


def _valid_samples(
    pred_disp: np.ndarray, metric_depth: np.ndarray, conf_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite relative disparity and metric disparity samples."""
    if pred_disp.shape != metric_depth.shape or pred_disp.shape != conf_mask.shape:
        raise ValueError("pred_disp, metric_depth, and conf_mask must have matching shapes")

    pred = np.asarray(pred_disp, dtype=np.float64)
    depth_mm = np.asarray(metric_depth, dtype=np.float64)
    mask = np.asarray(conf_mask, dtype=bool)

    valid = (
        mask
        & np.isfinite(pred)
        & np.isfinite(depth_mm)
        & (depth_mm > 0.0)
        & (depth_mm < SATURATED_DEPTH_MM)
    )

    if np.count_nonzero(valid) < 2:
        raise ValueError("Need at least two valid depth samples to solve affine fit")

    x = pred[valid]
    depth_m = depth_mm[valid] / 1000.0
    y = 1.0 / depth_m
    return x, y


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2:
        raise ValueError("Need at least two samples to solve affine fit")

    design = np.column_stack((x, np.ones_like(x)))
    s, t = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(s), float(t)


def solve_affine(
    pred_disp: np.ndarray, metric_depth: np.ndarray, conf_mask: np.ndarray
) -> tuple[float, float]:
    """Fit metric disparity ~= s * pred_disp + t using Huber loss.

    Args:
        pred_disp: HxW float32 relative disparity from DA-V2.
        metric_depth: HxW float32 metric depth in millimeters, where 0 is invalid.
        conf_mask: HxW bool mask indicating valid metric depth pixels.

    Returns:
        The affine scale and shift as ``(s, t)``.
    """
    x, y = _valid_samples(pred_disp, metric_depth, conf_mask)
    initial = np.asarray(_linear_fit(x, y), dtype=np.float64)

    def residual(params: np.ndarray) -> np.ndarray:
        s, t = params
        return s * x + t - y

    result = least_squares(residual, initial, loss="huber", f_scale=0.1)
    if not result.success:
        return solve_affine_ransac(pred_disp, metric_depth, conf_mask)

    s, t = result.x
    if not np.isfinite(s) or not np.isfinite(t):
        return solve_affine_ransac(pred_disp, metric_depth, conf_mask)

    return float(s), float(t)


def solve_affine_ransac(
    pred_disp: np.ndarray,
    metric_depth: np.ndarray,
    conf_mask: np.ndarray,
    n_iter: int = 100,
) -> tuple[float, float]:
    """RANSAC fallback for fitting metric disparity ~= s * pred_disp + t."""
    x, y = _valid_samples(pred_disp, metric_depth, conf_mask)
    sample_size = min(50, x.size)
    threshold = 0.1
    rng = np.random.default_rng(0)

    best_inliers: np.ndarray | None = None
    best_count = -1
    best_error = np.inf
    best_params = _linear_fit(x, y)

    for _ in range(n_iter):
        sample_idx = rng.choice(x.size, size=sample_size, replace=False)
        try:
            s, t = _linear_fit(x[sample_idx], y[sample_idx])
        except np.linalg.LinAlgError:
            continue

        residuals = np.abs(s * x + t - y)
        inliers = residuals <= threshold
        count = int(np.count_nonzero(inliers))
        mean_error = float(np.mean(residuals[inliers])) if count else np.inf

        if count > best_count or (count == best_count and mean_error < best_error):
            best_inliers = inliers
            best_count = count
            best_error = mean_error
            best_params = (s, t)

    if best_inliers is not None and np.count_nonzero(best_inliers) >= 2:
        best_params = _linear_fit(x[best_inliers], y[best_inliers])

    return float(best_params[0]), float(best_params[1])


def smooth_shifts(t_values: np.ndarray, window: int = 5) -> np.ndarray:
    """Temporally smooth per-frame shift values with a trailing moving average.

    Frame i is smoothed using only frames <= i, so no future information leaks
    in when the pipeline processes frames sequentially.

    Args:
        t_values: 1-D array of per-frame shift estimates.
        window: Number of frames to average over (default 5).

    Returns:
        Smoothed shift array of the same length.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    t = np.asarray(t_values, dtype=np.float64)
    smoothed = np.empty_like(t)
    for i in range(t.size):
        start = max(0, i - window + 1)
        smoothed[i] = float(np.mean(t[start : i + 1]))
    return smoothed


def global_scale(s_values: np.ndarray) -> float:
    """Return the global scale as the median of per-frame scale estimates.

    Using the median rather than the mean makes this robust to frames where
    the affine solver has poor sparse-depth coverage.

    Args:
        s_values: 1-D array of per-frame scale estimates.

    Returns:
        A single float to use as the global scale across the sequence.
    """
    s = np.asarray(s_values, dtype=np.float64)
    if s.size == 0:
        raise ValueError("s_values must not be empty")
    return float(np.median(s))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    true_s = 2.5
    true_t = 0.3

    pred = rng.uniform(0.1, 2.0, size=(80, 100)).astype(np.float32)
    metric_disp = true_s * pred + true_t
    metric_disp += rng.normal(0.0, 0.02, size=pred.shape).astype(np.float32)

    depth_m = 1.0 / metric_disp
    depth_mm = (depth_m * 1000.0).astype(np.float32)
    mask = np.ones_like(pred, dtype=bool)

    recovered_s, recovered_t = solve_affine(pred, depth_mm, mask)

    assert abs(recovered_s - true_s) / true_s < 0.05, (recovered_s, true_s)
    assert abs(recovered_t - true_t) / true_t < 0.05, (recovered_t, true_t)
    print(f"Recovered s={recovered_s:.4f}, t={recovered_t:.4f}")
