# Live Mesh Feed — Scoping

**Goal:** show the reconstruction **growing on the phone screen in real time as you sweep**,
instead of today's record → offline-reconstruct → view. This is the demo's "wow" (deck slide 5).

**The chip we actually have:** loaner is the **Galaxy S25 FE = Exynos** (Exynos 2400-class:
**Xclipse GPU**, AMD RDNA-based + a **Samsung NPU**) — **NOT** the Snapdragon SM8750/Hexagon.
This decides everything below: the Qualcomm **QNN/Hexagon** path does **not** run here. On-device
NN must use ExecuTorch's **Samsung ENN** (NPU), **Vulkan GPU** delegate (Xclipse), or **XNNPACK**
(CPU). That's a hard constraint and an Aiyan (Role B) dependency for any model-based live feed.

---

## The real-time loop (what "live" requires)

Per ARCore frame:
1. Acquire RGB + 6-DoF pose + raw depth + intrinsics (ARCore already gives this at ~30 fps).
2. **Keyframe-gate** — only do heavy work when the camera has moved enough (translation
   threshold). Pure rotation = no parallax = skip. Target ~3–8 keyframes/sec, not 30.
3. (model path) Run **Depth Anything V2** on the keyframe → dense depth.
4. **Scale-lock** that depth to ARCore's metric points (incremental affine solve: one global `s`,
   smoothed per-frame `t`).
5. **Integrate** the metric depth into a running **TSDF voxel volume** at the camera pose.
6. **Incrementally extract mesh** for the voxels touched this keyframe (only the dirty blocks —
   never re-mesh the whole volume).
7. **Render** the growing mesh/points as an overlay on the camera feed.

The expensive, novel parts on mobile are **3 (NN inference)**, **5 (TSDF integrate)**, and
**6 (incremental meshing)**. Everything else ARCore/GPU already do cheaply.

---

## Compute budget on the Exynos (rough, per keyframe)

| Stage | Cost | Where it runs | Notes |
|---|---|---|---|
| DA-V2 Small inference | ~30–100 ms (quantized, small input) | NPU (ENN) or GPU (Vulkan) | the gating cost; needs Aiyan's Exynos `.pte` |
| Affine scale solve | <2 ms | CPU | tiny least-squares on confident points |
| TSDF integrate | ~3–15 ms | GPU compute ideal, CPU ok | voxel-hash; cost ∝ touched voxels |
| Incremental marching cubes | ~5–20 ms | GPU/CPU | **only dirty blocks**, or coarse re-mesh every N keyframes |
| Render overlay | <3 ms | GPU | trivial |

**Verdict:** a **~3–6 Hz live mesh update** with **30 fps camera/pose passthrough** is realistic
*if* depth is quantized + small-input and meshing is incremental. Do **not** target 30 fps meshing.
**Thermal** is the real ceiling (roadmap R8): sustained NN + GPU heats the phone — keyframe-gating
and capping the depth rate are mandatory, not optional.

---

## Three ways to build it (easiest → hardest)

### Option A — Live POINT-CLOUD feed (recommended first; partly model-free)
Accumulate **scaled depth points in ARCore world space** and render as a growing colored point
cloud. No meshing at all.
- **Cheapest, most robust, visually compelling** ("watch the object materialize").
- **A sparse version needs only ARCore raw depth — NO DA-V2, NO Aiyan, NO host.** Fully on-device,
  buildable today. Densify with DA-V2 later for a fuller cloud.
- Great fallback even if the mesh path slips.

### Option B — Hybrid live mesh (edge-assisted)
Phone streams keyframes (depth + pose) over wifi/USB → **laptop runs the existing host TSDF**
(`pipeline_float.py` / Open3D) → streams the mesh back to the phone to render live.
- Gets a real **live growing mesh on the phone screen with ZERO new on-device meshing code** —
  reuses Sneha's proven pipeline.
- Caveat: not "100% on-device" — frame it as edge-assisted, or keep as a demo fallback.

### Option C — Full on-device incremental TSDF mesh (the real thing)
TSDF + incremental marching cubes running on the phone (GPU compute shaders on Xclipse, or a
C++/JNI voxel-hash KinectFusion-style core).
- The honest "100% on-device live mesh" — strongest claim.
- **Most work + highest risk**: on-device meshing is hard, Open3D-Android is heavy/experimental,
  and it still needs Aiyan's Exynos depth `.pte`.

---

## Recommended path for the hackathon

1. **Phase A now:** live **point cloud from ARCore raw depth** — on-device, no model, no host.
   This alone makes the demo feel alive and is essentially unblocked (Robert owns it; ARCore
   already exposes per-frame depth + pose). Densify with DA-V2 once Aiyan's Exynos model lands.
2. **Phase B in parallel:** wire the **hybrid laptop mesh** as the "live mesh" beauty shot and as
   demo insurance — reuses our working host pipeline, no new recon code.
3. **Phase C only if time:** true on-device incremental meshing. Treat as stretch; don't let it
   block the demo.

**The unlock:** a live point-cloud feed gives 80% of the "wow" for ~20% of the effort and is the
only live option that doesn't wait on the Exynos model. Build that first.

## Dependencies & risks
- **Exynos backend (Aiyan):** any DA-V2-based live feed needs an ENN/Vulkan/XNNPACK `.pte`, not QNN.
  The ARCore-only point cloud sidesteps this entirely.
- **Thermal throttling:** cap depth rate, keyframe-gate, consider running TSDF on GPU so it doesn't
  contend with NN on the NPU.
- **Live scale stability:** the incremental affine solve must be temporally smoothed or the cloud
  "breathes." Lock global `s` early, smooth `t`.
- **On-device meshing complexity:** the reason A and B exist — don't gate the demo on C.
