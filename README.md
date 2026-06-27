# Lantern — Phone Scan → 3D

Point a phone at a real object, get a clean, correctly-scaled, **CAD-ready `.glb` mesh** — reconstructed **on-device**, with the depth network running on the Snapdragon Hexagon NPU via ExecuTorch.

Built for the **Qualcomm × Meta ExecuTorch Hackathon** (Jun 27–28, 2026).

## How it works

Faithful reconstruction (not generative): **TSDF fusion of ARCore-aligned monocular depth maps.**

```
RGB + ARCore pose + raw depth + intrinsics
   │
   ├─► Depth-Anything-V2 Small ........ dense RELATIVE depth (affine-invariant)
   ├─► affine scale/shift solver ...... fit metric ≈ s·pred + t  vs ARCore sparse
   │                                     metric depth → real meters (solve s AND t)
   ├─► TSDF fusion (Open3D) ........... average many views → one clean surface
   ├─► marching cubes ................. raw .glb mesh
   └─► import_and_clean.py (Blender) .. watertight + normals + m→mm → CAD-ready .glb
```

Two depth sources by design: **DA-V2** gives a dense, smooth depth in unknown units; **ARCore** gives sparse-but-metric depth. The affine solver marries them so the mesh is both dense *and* correctly sized. See [`roadmap.md`](roadmap.md) for the full execution plan, decision log, and risk register.

## Target device

**Galaxy S25 / S25+ / S25 Ultra → Snapdragon 8 Elite = `SM8750`** (Adreno 830 + Hexagon NPU).

> ⚠️ The **S25 FE is Exynos**, not Snapdragon — the QNN lane (`QnnPartitioner`/`QnnQuantizer`/`SM8750`) does **not** target it. On an FE unit, use ExecuTorch's **Samsung ENN backend**, or fall back to a CPU/GPU (XNNPACK) `.pte`.

## Repo contents

| File | What |
|---|---|
| `roadmap.md` | Full execution roadmap — DAG, 6 phases, 5 decisions, risk register, module cards |
| `import_and_clean.py` | Host-side mesh cleanup: raw TSDF mesh → watertight, scaled, CAD-ready `.glb` |
| `test_harness.sh` | Offline smoke test (generates a sphere fixture, runs the script, validates) |
| `orientation_test.sh` | Distinct-dims box test — verifies the pipeline is orientation-preserving |

## Mesh cleanup — `import_and_clean.py`

Takes the reconstruction's output mesh (`.glb`, also `.obj/.ply/.stl`) and produces a clean, CAD-ready `.glb`: import → join parts → voxel-remesh to watertight → consistent normals → scale m→mm → export. Returns a non-zero exit on failure so the pipeline can gate on it.

```bash
blender --background --python import_and_clean.py -- input.glb output.glb

# options:
#   --scale  FLOAT   uniform scale (default 1000 = m→mm for CAD)
#   --voxel  FLOAT   voxel remesh size in meters (default 0.005); smaller = finer
#   --no-remesh      keep raw topology (normals still fixed)
#   --rotate-x DEG   optional frame correction (default 0 — usually unneeded)
#   --rotate-z DEG
```

**Orientation note:** Blender's glTF importer and exporter apply inverse +Y-up↔+Z-up conversions, so `import → clean → export` is orientation-preserving with no manual rotation (verified by `orientation_test.sh`). Use `--rotate-x/--rotate-z` only if a mesh comes in tipped.

**Contract** (input from the float pipeline): meters (1.0 = 1 m, ARCore world), glTF +Y-up. The script logs `imported dims (pre-transform, m)` every run — the fastest check that units are right (a coffee mug ≈ `0.10`).

## Run the tests

```bash
./test_harness.sh        # end-to-end smoke test
./orientation_test.sh    # axis-swap / orientation check
```

Both are offline (they generate their own fixtures) and exit non-zero on failure. Verified on **Blender 5.1**.

## Status

Host-side (~80% of de-risking runs before the phone is unboxed): float pipeline + quantization in progress; **mesh cleanup + validation complete and tested.** Phone integration is the final phase, not the first.
