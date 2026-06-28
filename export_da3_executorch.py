"""Export Depth Anything 3 (DA3-Small) to an ExecuTorch ``.pte`` for the S25 NPU.

This is the on-device counterpart to ``depth_anything_v3.py`` (host inference). It
lowers the DA3-Small **monocular depth** network to:

  * ``--backend xnnpack`` — portable CPU baseline. Always available; use it to validate
    the export plumbing before touching the QNN toolchain.
  * ``--backend qnn`` — Qualcomm AI Engine (QNN/HTP) delegate for the **Snapdragon 8
    Elite for Galaxy = SM8750** (the Galaxy S25 SoC; see ``roadmap.md`` / ``AGENT.md``).
    Float fp16 on the Hexagon NPU; Qualcomm AI Hub measures DA3-Small at ~43 ms/frame
    in this configuration, so int8 PT2E quantization is an *optional* later optimization
    (roadmap Phase 3), not a prerequisite.

Background / why this is a short hop, not a research risk (LIVE_MESH_PLAN.md §5.0a):
Qualcomm AI Hub already publishes DA3-Small exported + NPU-optimized for SM8750, but
only as QNN_DLC / TFLITE / ONNX — *not* ExecuTorch. This script produces the ``.pte``
we actually need, reusing the QNN lowering recipe already documented for DA-V2.

IMPORTANT — what is and isn't accelerated:
Only the **single-image depth** path is exported here (and only depth is the
NPU-optimized DA3 mode). DA3's any-view *pose* estimation is NOT exported / not the
accelerated path — the markerless tracker stays hand-rolled (LIVE_MESH_PLAN.md §3, §9.1).

Prereqs:
  pip install -r requirements-da3.txt           # torch + depth-anything-3 + executorch
  # for --backend qnn you additionally need the Qualcomm AI Engine Direct (QNN) SDK and
  # an ExecuTorch build with the Qualcomm backend; see:
  # https://docs.pytorch.org/executorch/stable/backends-qualcomm.html

Usage:
  python export_da3_executorch.py --backend xnnpack -o da3_small_xnnpack.pte
  python export_da3_executorch.py --backend qnn --soc SM8750 -o da3_small_sm8750.pte
  python export_da3_executorch.py --inspect          # print DA3 module tree to find the net

After a QNN export, read the partitioner fallback/fragmentation report it prints: the
gate is *few contiguous QNN partitions, no per-frame CPU<->NPU thrash* — not raw op
coverage (roadmap §Phase 2, Card B).
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = "da3-small"
# DA3-Small native working resolution (multiple of 14). Matches the Qualcomm AI Hub
# export and depth_anything_v3.py's default --process-res.
DEFAULT_RES = 518
DEFAULT_SOC = "SM8750"  # Galaxy S25 = Snapdragon 8 Elite for Galaxy.

# Common attribute names the underlying torch nn.Module hides behind on the DA3 wrapper.
_CANDIDATE_MODULE_ATTRS = ("model", "net", "network", "da3", "_model", "backbone")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DA3-Small monocular depth to an ExecuTorch .pte."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DA3 preset (default da3-small).")
    parser.add_argument(
        "--backend",
        choices=["xnnpack", "qnn"],
        default="xnnpack",
        help="Delegate: xnnpack (CPU baseline) or qnn (Qualcomm NPU/HTP).",
    )
    parser.add_argument(
        "--soc",
        default=DEFAULT_SOC,
        help="QcomChipset enum name for --backend qnn (S25 = SM8750).",
    )
    parser.add_argument("--res", type=int, default=DEFAULT_RES, help="Fixed square input size.")
    parser.add_argument(
        "--module-attr",
        default=None,
        help="Override the attribute that holds the underlying torch nn.Module on the DA3 wrapper.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .pte path (default: da3_small_<backend>.pte).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the DA3 module's named children/attributes and exit (helps locate the net).",
    )
    return parser.parse_args()


def load_da3(model_name: str):
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError(
            "The 'depth-anything-3' package is not installed. "
            "Install the DA3 extras: pip install -r requirements-da3.txt"
        ) from exc

    from depth_anything_v3 import _hf_repo_for

    return DepthAnything3.from_pretrained(_hf_repo_for(model_name)).eval()


def find_core_module(wrapper, override: str | None):
    """Locate the underlying ``torch.nn.Module`` to export from the DA3 API wrapper.

    The public ``DepthAnything3`` object wraps preprocessing + the network. ``torch.export``
    needs the bare nn.Module. We try the common attribute names; ``--module-attr`` overrides.
    """
    import torch.nn as nn

    if isinstance(wrapper, nn.Module) and override is None:
        # The wrapper may itself be the nn.Module; prefer a known sub-net if present.
        for attr in _CANDIDATE_MODULE_ATTRS:
            sub = getattr(wrapper, attr, None)
            if isinstance(sub, nn.Module):
                return sub
        return wrapper

    attrs = [override] if override else list(_CANDIDATE_MODULE_ATTRS)
    for attr in attrs:
        if attr is None:
            continue
        sub = getattr(wrapper, attr, None)
        if isinstance(sub, nn.Module):
            return sub

    raise RuntimeError(
        "Could not locate the underlying torch nn.Module on the DA3 wrapper. "
        "Run with --inspect to see its structure, then pass --module-attr <name>."
    )


def describe(wrapper) -> None:
    import torch.nn as nn

    print(f"DA3 wrapper type: {type(wrapper).__name__}")
    print(f"  is nn.Module: {isinstance(wrapper, nn.Module)}")
    print("  candidate sub-module attributes:")
    for attr in _CANDIDATE_MODULE_ATTRS:
        sub = getattr(wrapper, attr, None)
        kind = type(sub).__name__ if sub is not None else "—"
        is_mod = isinstance(sub, nn.Module)
        print(f"    .{attr:<10} -> {kind}{'  (nn.Module)' if is_mod else ''}")
    if isinstance(wrapper, nn.Module):
        print("  named_children:")
        for name, child in wrapper.named_children():
            print(f"    {name}: {type(child).__name__}")


def patch_rope_for_export(max_pos: int = 4096) -> None:
    """Make DINOv2 2D-RoPE export-safe.

    ``RotaryPositionEmbedding2D.forward`` sizes its frequency table from
    ``int(positions.max()) + 1`` — a data-dependent value ``torch.export`` cannot
    specialize (it raises on the unbacked symint ``u0``). The table is only *indexed*
    by ``positions``, so any fixed size >= max position + 1 yields identical numerics
    (the extra rows are never read). For our fixed export resolution the patch grid is
    tiny (518/14 = 37), so a constant ``max_pos`` is exact, not an approximation.
    """
    import torch

    from depth_anything_3.model.dinov2.layers import rope as _rope

    cls = _rope.RotaryPositionEmbedding2D

    def forward(self, tokens: "torch.Tensor", positions: "torch.Tensor") -> "torch.Tensor":
        feature_dim = tokens.size(-1) // 2
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_pos, tokens.device, tokens.dtype
        )
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp
        )
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp
        )
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    cls.forward = forward


def build_export_module(core_module):
    """Wrap the core DA3 net so it has a clean, single-tensor in / single-tensor out forward.

    Input:  (1, 3, H, W) float32, ImageNet-normalized RGB.
    Output: (1, H, W) or (H, W) depth (whatever the net returns is passed through; the
            export harness only needs a deterministic, traceable forward).
    """
    import torch
    import torch.nn as nn

    class _Wrapper(nn.Module):
        def __init__(self, net: nn.Module):
            super().__init__()
            self.net = net

        def forward(self, image: "torch.Tensor"):
            out = self.net(image)
            # DA3 nets may return a tuple/dict/Prediction-like object. Reduce to the
            # depth tensor for a clean export signature.
            if isinstance(out, dict):
                for key in ("depth", "pred_depth", "disparity"):
                    if key in out:
                        return out[key]
                return next(iter(out.values()))
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    return _Wrapper(core_module).eval()


def example_input(res: int):
    import torch

    # DA3's net forward expects a multi-view batch (B, N, 3, H, W). For single-image
    # depth N=1. The on-device loader must feed the same 5D rank (ExecuTorch input
    # ranks are immutable: a 4D tensor triggers "Error resizing tensor at input 0").
    return (torch.randn(1, 1, 3, res, res, dtype=torch.float32),)


def lower_xnnpack(module, example_inputs):
    """Lower to the portable XNNPACK CPU backend (baseline; always available)."""
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    exported = torch.export.export(module, example_inputs)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    return edge.to_executorch()


def lower_qnn(module, example_inputs, soc: str):
    """Lower to the Qualcomm QNN/HTP backend for the given SoC (S25 = SM8750), float fp16.

    Mirrors roadmap.md §Phase 2 / Card B. After this returns, inspect the partitioner
    report: the real gate is *few contiguous QNN partitions, no per-frame CPU<->NPU thrash*.
    """
    import torch
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from executorch.backends.qualcomm.utils.utils import (
        QcomChipset,
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
    )
    from executorch.exir import to_edge_transform_and_lower

    soc_model = getattr(QcomChipset, soc, None)
    if soc_model is None:
        available = [n for n in dir(QcomChipset) if not n.startswith("_")]
        raise RuntimeError(
            f"QcomChipset.{soc} not found in this ExecuTorch build. "
            f"The enum lags new SoCs; available: {available}. "
            "Patch/mirror the nearest entry per roadmap.md §Phase 2."
        )

    backend_options = generate_htp_compiler_spec(use_fp16=True)  # float; quantize later (Phase 3)
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=soc_model,
        backend_options=backend_options,
    )
    exported = torch.export.export(module, example_inputs)
    edge = to_edge_transform_and_lower(
        exported, partitioner=[QnnPartitioner(compile_spec)]
    )
    return edge.to_executorch()


def main() -> None:
    args = parse_args()
    try:
        wrapper = load_da3(args.model)

        if args.inspect:
            describe(wrapper)
            return

        core = find_core_module(wrapper, args.module_attr)
        patch_rope_for_export()
        export_module = build_export_module(core)
        example_inputs = example_input(args.res)

        if args.backend == "xnnpack":
            program = lower_xnnpack(export_module, example_inputs)
        else:
            program = lower_qnn(export_module, example_inputs, args.soc)

        output = args.output or REPO_ROOT / f"da3_small_{args.backend}.pte"
        with open(output, "wb") as f:
            f.write(program.buffer)

        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"Wrote {output} ({size_mb:.1f} MB, backend={args.backend}, res={args.res})")
        if args.backend == "qnn":
            print(
                "Next: push to device, run via the ExecuTorch Android runtime, and diff a "
                "single-frame output against host depth_anything_v3.py (numerics gate)."
            )
    except (RuntimeError, ImportError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
