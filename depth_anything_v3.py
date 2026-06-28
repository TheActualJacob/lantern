"""Run Depth Anything 3 (DA3) inference and save pipeline-compatible disparities.

Drop-in successor to ``depth_model.py`` (DA-V2). It emits the **exact same output
contract** that ``pipeline_float.py`` already consumes:

    <output>/<base>.disp.png   uint16, per-frame normalized disparity
    <output>/disp_maxes.json   {output_name: disp_max_float}

so the host fusion pipeline is unchanged. DA-V2 (``depth_model.py``) stays as the
documented fallback.

Why DA3 (see ``LIVE_MESH_PLAN.md`` §5.0a, resolved 2026-06-27):
  - ``da3-small`` is Apache-2.0, 24.7 M params, 518x518, and Qualcomm AI Hub already
    ships it NPU-optimized for the S25's SoC (SM8750) at ~43 ms/frame float on the
    Hexagon NPU. It beats DA-V2 on monocular depth.
  - The ExecuTorch ``.pte`` export for on-device use lives in
    ``export_da3_executorch.py``.

Convention note: DA3 returns **metric-ish depth** (distance), whereas the lantern
pipeline works in **disparity** (inverse depth, ``s*disp + t`` affine fit). We convert
``disp = 1 / depth`` here so the existing affine solver in ``pipeline_float.py`` needs
no changes. da3-small depth is *relative* (up to scale); scale is recovered downstream
exactly as it was for DA-V2.

Install the extra deps with ``pip install -r requirements-da3.txt`` (heavy: torch +
the ``depth-anything-3`` package). For large batches without a local GPU, run on a
hosted GPU (see FEATURES.md), same as DA-V2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

# Apache-2.0, phone-sized, and the variant Qualcomm AI Hub exports for the S25 NPU.
DEFAULT_MODEL = "da3-small"
DISP_UINT16_MAX = 65535.0
DISPARITY_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 on color frames and save disparity PNGs."
    )
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "frames")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "disparities")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "DA3 checkpoint preset. da3-small/da3-base are Apache-2.0 and phone-sized; "
            "da3-large/da3-giant/da3nested-* are stronger but several are CC BY-NC."
        ),
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=518,
        help="DA3 working resolution (multiple of 14). 518 matches the Qualcomm NPU export.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    return parser.parse_args()


def select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if requested == "mps" and not _mps_available(torch):
        raise RuntimeError("MPS was requested, but torch.backends.mps is not available")
    return requested


def _mps_available(torch) -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def load_model(model_name: str, device: str):
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "The 'depth-anything-3' package is not installed. "
            "Install the DA3 extras: pip install -r requirements-da3.txt"
        ) from exc

    model = DepthAnything3.from_pretrained(_hf_repo_for(model_name))
    return model.to(device).eval()


def _hf_repo_for(model_name: str) -> str:
    """Map a DA3 preset name to its Hugging Face repo id.

    DA3 presets are lowercase (``da3-small``); the HF repos are uppercase
    (``depth-anything/DA3-SMALL``). Pass a full ``org/name`` through untouched.
    """
    if "/" in model_name:
        return model_name
    return f"depth-anything/{model_name.upper()}"


def depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    """Convert DA3 depth (distance) to the pipeline's disparity (inverse depth).

    Non-finite and non-positive depths map to 0 disparity (treated as invalid by the
    downstream affine fit / TSDF integration), matching how ``encode_disparity`` floors
    bad values.
    """
    depth = np.asarray(depth, dtype=np.float32)
    disp = np.zeros_like(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > DISPARITY_EPS)
    disp[valid] = 1.0 / depth[valid]
    return disp


def infer_disparity(model, image_bgr: np.ndarray, process_res: int) -> np.ndarray:
    """Run DA3 on one BGR frame, return disparity at the frame's native resolution."""
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    import torch

    with torch.no_grad():
        prediction = model.inference(
            image=[image_rgb],
            process_res=process_res,
            process_res_method="upper_bound_resize",
        )

    depth = np.asarray(prediction.depth)
    if depth.ndim == 3:  # (N, H, W) -> single view
        depth = depth[0]

    disp = depth_to_disparity(depth)
    if disp.shape != (height, width):
        # DA3 works at process_res; resize disparity back to the color frame size so
        # the output matches DA-V2's convention (disparity at native resolution).
        disp = cv2.resize(disp, (width, height), interpolation=cv2.INTER_CUBIC)
    return disp


def run_inference(
    frames_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    process_res: int,
) -> None:
    # Reuse depth_model's IO/encoding as the single source of truth (it imports torch,
    # so keep the dependency lazy and out of the pure-logic import path).
    from depth_model import (
        encode_disparity,
        output_name_for_frame,
        sorted_color_frames,
    )

    frame_paths = sorted_color_frames(frames_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_name, device)

    disp_maxes: dict[str, float] = {}
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read frame image: {frame_path}")

        start = time.perf_counter()
        disp = infer_disparity(model, image, process_res)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        output_name = output_name_for_frame(frame_path)
        output_path = output_dir / output_name
        encoded_disp, disp_max = encode_disparity(disp)
        if not cv2.imwrite(str(output_path), encoded_disp):
            raise ValueError(f"Failed to write disparity PNG: {output_path}")

        disp_maxes[output_name] = disp_max
        print(f"{frame_path.name}: done ({device}, {elapsed_ms:.1f} ms)")

    disp_maxes_path = output_dir / "disp_maxes.json"
    with disp_maxes_path.open("w", encoding="utf-8") as f:
        json.dump(disp_maxes, f, indent=2, sort_keys=True)

    print(f"Wrote {len(frame_paths)} disparity maps to {output_dir} (DISP max {DISP_UINT16_MAX:.0f})")


def main() -> None:
    args = parse_args()
    try:
        device = select_device(args.device)
        run_inference(
            frames_dir=args.frames.resolve(),
            output_dir=args.output.resolve(),
            model_name=args.model,
            device=device,
            process_res=args.process_res,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
