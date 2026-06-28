# Phase A — Live Point-Cloud Feed (build spec)

**What:** the reconstruction materializing **on the phone screen in real time** as you sweep —
a growing colored point cloud, accumulated in ARCore world space. **No neural model, no laptop,
no Aiyan dependency.** Pure ARCore raw depth + pose + GPU render. This is the demo's "it's alive"
moment (deck slide 5) and works on the Exynos S25 FE today.

**Owner:** Robert (Android/ARCore). Builds on the existing recorder app.
**Why it's unblocked:** everything here comes from ARCore APIs we already use in `FrameRecorder.kt`.
DA-V2 densification is a later upgrade, not required for Phase A.

---

## The loop (per frame, target 30 fps passthrough / accumulate ~5–10 keyframes/s)

```
ARCore Frame
  ├─ acquireRawDepthImage16Bits()   → 16-bit depth (mm) + confidence (already in recorder)
  ├─ getCameraPose()                → 4x4 cam→world (meters)
  ├─ CameraIntrinsics               → fx,fy,cx,cy for the depth image
  │
  ├─ KEYFRAME GATE: skip unless camera translated > THRESH since last keyframe
  │     (pure rotation adds no new surface; also caps thermal load)
  │
  ├─ UNPROJECT depth → 3D points (camera space):
  │     for each pixel (u,v) with conf >= CONF_MIN and 0 < d < D_MAX:
  │       z = d / 1000.0                        # mm → m
  │       x = (u - cx) * z / fx
  │       y = (v - cy) * z / fy
  │     (subsample: every Nth pixel — a few thousand pts/keyframe is plenty)
  │
  ├─ TRANSFORM to world: p_world = camPose * [x,y,z,1]
  │
  ├─ ACCUMULATE into a persistent world-space buffer (voxel-downsample to dedupe:
  │     hash by round(p / VOXEL) so revisited surface doesn't pile up)
  │
  └─ RENDER all accumulated points as an overlay on the camera feed (GPU)
```

## Key parameters (tune on device)
| Param | Start | Purpose |
|---|---|---|
| `TRANS_THRESH` | 2–3 cm | keyframe gate; lower = denser/hotter |
| `CONF_MIN` | ARCore conf ≥ ~0.5 | drop noisy/edge-bleed depth |
| `D_MAX` | 1.5–2.0 m | ignore background; object-scale only |
| `PIXEL_STRIDE` | 4–8 | subsample for perf (1=every pixel) |
| `VOXEL` (dedupe) | 3–5 mm | merge revisited points; caps buffer growth |
| `MAX_POINTS` | ~300k | hard cap; ring-buffer or stop adding |

## Rendering (Android)
- Reuse the app's GLES/Compose-GL surface (the recorder already draws the camera background via
  `BackgroundRenderer.kt`). Add a second pass:
  - VBO of accumulated world points (position + confidence-as-color, or a flat accent color).
  - Draw `GL_POINTS` with the ARCore view+projection matrices (so points lock to the world).
  - Point size ~4–6 px; color by confidence (green=high→red=low) or a clean accent for a slick look.
- Update the VBO incrementally — append new keyframe points; don't rebuild every frame.

## Coverage feedback (ties into Robert's dome work)
- Reuse the **azimuth+elevation dome** coverage guide (commits `7e25c70`/`d1e37b0`): light up a
  dome wedge once enough points land from that viewing angle. The live cloud + the dome together =
  "scan until the dome is full," which is exactly the full-coverage capture we need for the hero/accuracy.

## Performance & thermal (Exynos)
- All math is cheap (unproject + matrix mult); the only real costs are pixel count and point count.
- Keyframe-gate + `PIXEL_STRIDE` + `VOXEL` dedupe keep it bounded. Run unprojection on a worker
  thread (or a GPU compute/shader pass) so the render thread stays at 30 fps.
- No NN = far cooler than the full pipeline; sustained scanning should hold without throttling.

## Definition of done (Phase A)
1. Sweep the phone → a colored point cloud grows on screen, locked in world space.
2. Cloud is metric (a 20 cm object reads ~20 cm) and stops piling up on revisit (dedupe works).
3. Holds ~30 fps camera with smooth point updates for a 60–90 s scan, no overheating.
4. **Bonus:** "export cloud" button writes the accumulated world points to `.ply` → feeds
   `import_and_clean.py` / `ground_truth.py` for a mesh + accuracy number **with no DA-V2 at all**
   (sparser than the fused mesh, but a fully on-device end-to-end artifact).

## Upgrade path (later, not Phase A)
- Swap/augment ARCore raw depth with **DA-V2 dense depth** (once Aiyan's Exynos `.pte` lands) for a
  denser cloud → then Phase C (on-device TSDF meshing) turns the cloud into a live surface.
- The Phase-A render surface and accumulation buffer are reused as-is; only the depth *source* upgrades.

## Why this is the right first move
- **On-device, today, on the chip we have** — no QNN/Exynos blocker, no laptop.
- **80% of the "live" wow for ~20% of the effort** vs full on-device meshing.
- **Doubles as demo insurance** and as a model-free end-to-end artifact (export → mesh → mm number).
- Sets up Phases B/C without rework.
