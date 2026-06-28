"""Run Depth Anything V2 inference and save pipeline-compatible disparities.

For large batches without a local GPU, run on Google Colab with a T4 runtime
instead (see FEATURES.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose


REPO_ROOT = Path(__file__).resolve().parent
DEPTH_ANYTHING_ROOT = REPO_ROOT / "Depth-Anything-V2"
DISP_UINT16_MAX = 65535.0

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}

HF_CHECKPOINTS = {
    "vits": "depth-anything/Depth-Anything-V2-Small",
    "vitb": "depth-anything/Depth-Anything-V2-Base",
    "vitl": "depth-anything/Depth-Anything-V2-Large",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything V2 on color frames and save disparity PNGs."
    )
    parser.add_argument("--frames", type=Path, default=REPO_ROOT / "frames")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "disparities")
    parser.add_argument("--encoder", choices=sorted(MODEL_CONFIGS), default="vits")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if requested == "mps" and not _mps_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps is not available")
    return requested


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def add_depth_anything_to_path() -> None:
    if not DEPTH_ANYTHING_ROOT.exists():
        raise FileNotFoundError(
            f"Depth Anything V2 clone not found at {DEPTH_ANYTHING_ROOT}"
        )
    sys.path.insert(0, str(DEPTH_ANYTHING_ROOT))


def checkpoint_path(encoder: str) -> Path:
    return DEPTH_ANYTHING_ROOT / "checkpoints" / f"depth_anything_v2_{encoder}.pth"


def load_model(encoder: str, device: str):
    add_depth_anything_to_path()

    from depth_anything_v2.dpt import DepthAnythingV2

    weights_path = checkpoint_path(encoder)
    if not weights_path.exists():
        raise FileNotFoundError(
            "\n".join(
                [
                    f"Missing Depth Anything V2 checkpoint: {weights_path}",
                    f"Download it from HuggingFace: {HF_CHECKPOINTS[encoder]}",
                    "Then place it in Depth-Anything-V2/checkpoints/.",
                ]
            )
        )

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # The local DA-V2 helper chooses a device internally. Bind an equivalent
    # helper so explicit --device values are honored while still using
    # model.infer_image(img) for inference.
    _bind_image2tensor_for_device(model, device)
    return model


def _bind_image2tensor_for_device(model, device: str) -> None:
    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

    def image2tensor(self, raw_image, input_size: int = 518):
        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                PrepareForNet(),
            ]
        )

        height, width = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        return image, (height, width)

    model.image2tensor = MethodType(image2tensor, model)


def sorted_color_frames(frames_dir: Path) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")
    if not frames_dir.is_dir():
        raise ValueError(f"Frames path is not a directory: {frames_dir}")

    frame_paths = sorted(frames_dir.glob("*.color.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No *.color.png frames found in {frames_dir}")
    return frame_paths


def output_name_for_frame(frame_path: Path) -> str:
    if frame_path.name.endswith(".color.png"):
        return frame_path.name[: -len(".color.png")] + ".disp.png"
    return frame_path.stem + ".disp.png"


def encode_disparity(disp: np.ndarray) -> tuple[np.ndarray, float]:
    disp = np.asarray(disp, dtype=np.float32)
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    disp = np.maximum(disp, 0.0)
    disp_max = float(disp.max())

    if disp_max <= 0.0:
        return np.zeros(disp.shape, dtype=np.uint16), disp_max

    encoded = np.clip((disp / disp_max) * DISP_UINT16_MAX, 0.0, DISP_UINT16_MAX)
    return encoded.astype(np.uint16), disp_max


def run_inference(
    frames_dir: Path,
    output_dir: Path,
    encoder: str,
    device: str,
) -> None:
    frame_paths = sorted_color_frames(frames_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(encoder, device)

    disp_maxes: dict[str, float] = {}
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read frame image: {frame_path}")

        start = time.perf_counter()
        with torch.no_grad():
            disp = model.infer_image(image)
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

    print(f"Wrote {len(frame_paths)} disparity maps to {output_dir}")


def main() -> None:
    args = parse_args()
    try:
        device = select_device(args.device)
        run_inference(
            frames_dir=args.frames.resolve(),
            output_dir=args.output.resolve(),
            encoder=args.encoder,
            device=device,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
