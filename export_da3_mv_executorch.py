"""Export Depth Anything 3 (DA3-Small) **multi-view** to an ExecuTorch ``.pte``.

The mono counterpart is ``export_da3_executorch.py`` (one image -> relative depth). This one
exports DA3's *native multi-view* forward: N images in one pass -> per-view depth that is
**mutually consistent in one shared frame**, plus DA3's own estimated **extrinsics + intrinsics**
and a confidence map. That is the geometry that makes a scanned box come out as a coherent box
instead of a bent, convex "banana" (the mono per-frame + affine path can't do this — see
docs/RECON_QUALITY_DIAGNOSIS.md and MESH_ENHANCEMENT_PLAN.md).

Input contract (fixed at export, ExecuTorch ranks are immutable):
    image : (1, N, 3, RES, RES) float32, ImageNet-normalized RGB, letterboxed to square.
Outputs (a 4-tuple, all batched B=1):
    depth       : (1, N, RES, RES) float32   # per-view, shared-frame consistent (DA3 units, z=distance)
    depth_conf  : (1, N, RES, RES) float32
    extrinsics  : (1, N, 3, 4)     float32   # world->camera [R|t], OpenCV
    intrinsics  : (1, N, 3, 3)     float32   # pinhole at RES x RES

On-device: letterbox the N selected keyframes to RES, stack, run, then back-project each view's
masked depth with its own (intrinsics, extrinsics) into one cloud (X_world = R^T (X_cam - t)).

Usage:
    python export_da3_mv_executorch.py --views 8 --res 350 -o models/da3_mv_n8_r350.pte
    # flatc must be on PATH (it ships in the executorch wheel, e.g. .venv-da3/bin/flatc):
    #   export PATH="$PWD/.venv-da3/bin:$PATH"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = "da3-small"
DEFAULT_VIEWS = 8
DEFAULT_RES = 350  # multiple of 14 (350/14 = 25 patch tokens per side)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export multi-view DA3-Small to an ExecuTorch .pte.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--views", type=int, default=DEFAULT_VIEWS, help="Fixed number of input views N.")
    p.add_argument("--res", type=int, default=DEFAULT_RES, help="Square input size (mult of 14); used if --width/--height unset.")
    p.add_argument("--width", type=int, default=None, help="Input width (mult of 14). DA3-native 16:9 = 504.")
    p.add_argument("--height", type=int, default=None, help="Input height (mult of 14). DA3-native 16:9 = 280.")
    p.add_argument("-o", "--output", type=Path, default=None)
    return p.parse_args()


def build_mv_module(net):
    """Wrap the DA3 net so forward(image) -> (depth, depth_conf, extrinsics, intrinsics).

    extrinsics/intrinsics are passed as None so DA3 **self-estimates** the cameras (the whole
    point of multi-view). The other args are fixed constants the tracer specializes on.
    """
    import torch.nn as nn

    class MV(nn.Module):
        def __init__(self, net: nn.Module):
            super().__init__()
            self.net = net

        def forward(self, image):
            out = self.net(image, None, None, [], False, False, "saddle_balanced")
            return out["depth"], out["depth_conf"], out["extrinsics"], out["intrinsics"]

    return MV(net).eval()


def main() -> None:
    args = parse_args()
    import torch

    from depth_anything_3.api import DepthAnything3
    from export_da3_executorch import patch_rope_for_export

    model = DepthAnything3.from_pretrained(_hf_repo(args.model)).eval()
    patch_rope_for_export()  # make DINOv2 2D-RoPE export-safe (data-dependent symint)
    wrap = build_mv_module(model.model)

    width = args.width or args.res
    height = args.height or args.res
    example = (torch.randn(1, args.views, 3, height, width, dtype=torch.float32),)
    print(f"Tracing multi-view DA3: N={args.views} {width}x{height} (WxH) ...")
    t0 = time.time()
    exported = torch.export.export(wrap, example)
    print(f"  torch.export OK ({len(list(exported.graph.nodes))} nodes, {time.time() - t0:.0f}s)")

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    print("Lowering to XNNPACK ...")
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    program = edge.to_executorch()

    out = args.output or REPO_ROOT / "models" / f"da3_mv_n{args.views}_w{width}_h{height}.pte"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(program.buffer)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, N={args.views}, {width}x{height}, {time.time() - t0:.0f}s total)")


def _hf_repo(name: str) -> str:
    if "/" in name:
        return name
    return f"depth-anything/{name.upper().replace('DA3-', 'DA3-')}"


if __name__ == "__main__":
    main()
