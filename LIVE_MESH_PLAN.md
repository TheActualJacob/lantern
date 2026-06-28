# Live Markerless Object Capture (hold-in-hand) — Implementation Handoff

**Audience:** the next agent building board-free, real-time 3D capture.
**Goal in one line:** the user holds **any** object in their hand, turns it in front of
the phone (or moves the phone around it), and a 3D model **grows live and fills in** —
**no ChArUco board, no markers, nothing printed**.
**Status:** Not started, research-heavy. The landed turntable/board work (T2/T3/T4) is a
**different, board-based** path; this document is the markerless replacement and does **not**
depend on a board at runtime.
**Date:** 2026-06-27 · **Related:** `OBJECT_TRACKING_PLAN.md` (§8 names this the "markerless
successor"), `roadmap.md`.

---

## 0. Read this first — the honest reality

This is the hard problem the board was invented to avoid. With a board, the object's pose
each frame is *given* by `solvePnP`. With **no board**, you must *recover* the object's pose
every frame from the pixels/geometry alone, while the object (and the hand holding it) move.
That is **object-centric SLAM / in-hand scanning**. It is doable on device, but:

- It is a **multi-week** effort, not a tweak. Budget for real research spikes.
- It needs **on-device ML that the app does not have yet**: monocular depth and object
  segmentation (no ExecuTorch runtime is wired into the app today — confirm in `roadmap.md`).
- **It will not work on every object.** The failure cases are not bugs, they are math:
  - **Rotationally symmetric + untextured objects are nearly impossible.** A smooth
    cylinder (e.g. a plain salt shaker) looks identical as it spins about its axis, so
    geometry gives the tracker **nothing** to lock the rotation onto. Only surface
    **texture/print** can disambiguate the spin. If the object is both smooth *and*
    featureless, markerless in-hand tracking of the spin is fundamentally unobservable —
    you must fall back (orbit the phone instead, §14) or accept failure on that object.
  - **The hand moves *with* the object**, so you can't separate it by motion — you need
    actual hand/object segmentation, or the hand fuses into the model.

Set expectations accordingly. The plan below maximizes the set of objects that work and
fails gracefully (with guidance) on the ones that can't.

---

## 1. What "works on anything" actually requires

Per frame, board-free, you need three things and then a tracker that ties them together:

1. **RGB** — already have it (ARCore camera image / `BackgroundRenderer` texture, and the
   CPU image via `acquireCameraImage`).
2. **Metric-ish depth of the object** — **monocular** (Depth-Anything **V3** via ExecuTorch;
   V2 is the repo incumbent but V3 is newer/better — see §5.0), because ARCore's depth assumes
   a *static* scene and will be wrong/unstable for an object being turned in the hand. (Plain
   monocular depth is relative; see scale, §5/§11.)
3. **Object mask** — isolate the object from the **hand** and the background every frame.
   This is mandatory here (unlike the board path, where "above the board plane" sufficed).

Then the core new component:

4. **Frame-to-model tracker** — estimate camera-from-object pose `T_OC(t)` by registering
   the current masked frame against the **model accumulated so far** (not a marker). This is
   what replaces `solvePnP`. The model is both the output *and* the tracking reference.

---

## 2. Pipeline overview

```
          ┌─────────── per frame (camera) ───────────┐
RGB ──┐   │                                           │
      ├──►│ 1. Object mask (SAM/MobileSAM + hand rm)  │── masked RGB+depth
Depth─┘   │ 2. Monocular depth (DA3), metric-scaled   │
(DA3)     └───────────────────────────────────────────┘
                         │ masked object point cloud (camera frame)
                         ▼
            3. TRACK: register cloud → current model  ⇒  T_OC(t)   (ICP + photometric/features)
                         │
                         ▼
            4. FUSE: integrate masked depth into TSDF in object frame O at T_OC(t)
                         │
                         ▼
            5. MESH: incremental marching cubes  ─► live growing mesh
                         │
                         ▼
            6. RENDER: inset 3D preview (and/or AR overlay), drive surface-coverage UI
            7. (loop closure / pose-graph to fight drift over a full turn)
```

Steps 4–6 are the same machinery the board plan would have used; the **new, hard** parts are
**1–3 (the ML inputs)** and **3/7 (markerless tracking + drift control)**.

---

## 3. The tracker (the heart of board-free capture)

Replace "pose from board" with **frame-to-model registration**. Maintain the model in a
canonical **object frame `O`** (defined by the first frame). For each new frame:

- **Predict** an initial pose from the previous frame (constant-velocity on the rotation),
  or identity for the first frame.
- **Align** the current masked object cloud to the model and refine the pose:
  - **Geometric:** point-to-plane **ICP** of the masked depth against the model surface
    (extract a point cloud / use the TSDF gradient). Robust for shaped objects; **degenerate
    on symmetric ones** (§0).
  - **Photometric / feature:** match RGB features (e.g. ORB) or do direct image alignment
    against nearby **keyframes**. **Required** to track spin on smooth/symmetric objects and
    to reduce drift. Combine with ICP (joint or cascaded) for robustness.
- **Integrate** the masked depth into the TSDF at the refined `T_OC(t)`.
- **Keyframe** periodically (store pose + RGB + mask) for loop closure.

**Bootstrap:** first masked frame seeds the model and defines `O` (pose = identity). Small
inter-frame rotation + high frame rate lets ICP/feature tracking converge incrementally.

**Drift & loop closure:** errors accumulate over a full rotation → the start and end won't
meet (visible seam / double wall). Add a **pose graph** over keyframes with a **loop-closure**
constraint when the view returns near the start (appearance/feature match), then re-integrate.
Without this, a 360° scan will not close cleanly. This is the second hard research chunk.

> Pragmatic staging: get **incremental** frame-to-model tracking working first (good for a
> partial scan / “fills in as I turn it”), then add loop closure for a clean full turn.

> **Possible shortcut (evaluate first):** Depth Anything 3's **any-view** model estimates
> camera pose + consistent geometry across frames directly (§5.0). If it runs in budget on
> device (or as an async refinement over keyframes), it could **replace or seed** this whole
> ICP/feature stack — including the symmetric-object case it learns priors for. Treat this as
> the first research spike (§9.1) before hand-building the full tracker.

---

## 4. What already exists to build on

| Capability | Where | Use |
|---|---|---|
| GL render loop + camera background | `rendering/BackgroundRenderer.kt`, `MainActivity.onDrawFrame` | Add mesh/point render after the camera draw; per-frame entry point. |
| Camera image + intrinsics access | `FrameRecorder.capture` (`acquireCameraImage`, `camera.imageIntrinsics`) | RGB input + pinhole model. |
| Depth plane reading (reference) | `FrameRecorder.extractGray16BigEndian`, `MainActivity.estimateCenterDepthMeters` | Pattern for reading 16-bit depth (only if you ever use ARCore depth; default is monocular DA3). |
| Host TSDF (parity target) | `tsdf_fuse.py` `fuse(...)`, `pipeline_float.py` | Match voxel size + **opencv** convention so the live preview predicts a final host build. |
| Host depth reference | `depth_model.py` | Current DA-V2 model/IO to mirror for the `.pte` export; **update to DA3** (§5.0). |
| 3D viewer styling + stats | `recon/ModelViewerScreen.kt`, `recon/ModelStats.kt`, `recon/SessionDetailScreen.kt` | Reuse for the live inset preview; replace simulated stages with real progress. |
| Capture mode switch | `ui/CaptureUiState.CaptureMode`, `MainActivity.poseProviderForMode` | Add a new board-free mode (e.g. `CaptureMode.Handheld`) alongside Orbit; keep Orbit/Turntable intact. |

> The board detector (`OpenCvBoardDetector.kt`) and turntable path are **not** used at
> runtime here. Recorded board sessions are still handy as **offline ground truth** to
> validate the tracker/geometry math (you have `T_CO` truth in their `frame_*.json`), but the
> shipping path touches no board.

---

## 5. The ML you must stand up (no board ⇒ this is now mandatory)

> ### 5.0 Depth Anything **V3** vs V2 — and a possible shortcut
> The repo currently vendors **DA-V2** (`depth_model.py`, `Depth-Anything-V2/`); that's the
> only reason earlier drafts said "V2". **Prefer Depth Anything 3** (ByteDance Seed, Nov 2025,
> arXiv:2511.10647): it beats DA2 on monocular depth, and its **any-view** models also do
> **camera-pose estimation + spatially-consistent multi-view geometry**, with a **Nested**
> variant that adds **metric scale**. That overlaps heavily with the custom tracker in §3 —
> **DA3 could subsume much of the markerless tracking**, not just supply depth. Practical
> constraints to evaluate (see research §9):
> - **Variants/licenses:** DA3-Small (0.08B) / DA3-Base (0.12B) are **Apache-2.0** and
>   phone-sized; the strong pose/any-view + metric (Large/Giant 0.35–1.4B, `DA3NESTED-*`,
>   `DA3METRIC-LARGE`) are bigger and several are **CC BY-NC 4.0** (non-commercial — matters
>   if this ships). (`DA3METRIC-LARGE`/`DA3MONO-LARGE` are Apache-2.0 but 0.35B.)
> - **Streaming:** DA3 is built for *batched any-view* inference; using it incrementally
>   (current frame + keyframes) at interactive rates on device is unproven — spike it.
> - **ExecuTorch export** of DA3 is unverified. If DA3 won't export/run in budget, fall back
>   to DA3-Small/DA2 **for depth only** and keep the §3 hand-rolled tracker.
> Keep depth/pose behind the `DepthSource`/`Tracker` interfaces so DA3 (depth-only, or
> depth+pose) is a drop-in either way.
>
> ### 5.0a RESEARCH RESOLVED (2026-06-27) — DA3 *depth* is already NPU-optimized for the S25
> Findings from researching DA3 on ExecuTorch / the S25 Snapdragon NPU. **Sources:**
> Qualcomm AI Hub model card `qualcomm/Depth-Anything-V3` (HF), the DA3 repo/PyPI
> (`depth-anything-3`), DA3 paper arXiv:2511.10647, ExecuTorch QNN backend docs.
>
> - **DA3-Small is the phone target and it's already exported + benchmarked on the S25 SoC.**
>   Qualcomm AI Hub publishes a pre-exported, NPU-optimized **DA3-Small** (checkpoint
>   `da3-small`, 24.7 M params, **518×518** input, ~94 MB float). It runs on the **Hexagon
>   NPU** of the **Snapdragon 8 Elite for Galaxy = SM8750** — the exact S25 chip
>   (`AGENT.md`/`roadmap.md`) — measured at **~43 ms QNN** / **~39 ms TFLITE** / **~37 ms ONNX**
>   per frame (≈ **23–27 fps, NPU, float**). This **resolves** "DA3 is phone-sized?" (yes) and
>   "does it run in budget on the S25 NPU?" (yes, for depth, comfortably async at 5–10 Hz, even
>   real-time). Peak memory ~0.5 GB.
> - **CRUCIAL CAVEAT — only the *monocular depth* path is the optimized one.** The Qualcomm
>   asset is a **single-image → depth map** export. It is **NOT** the any-view /
>   pose-estimating mode. So the §3/§9.1 "DA3 replaces the tracker" shortcut is **not** what's
>   NPU-accelerated: there is no off-the-shelf, on-device, real-time DA3 *pose* path. Treat
>   DA3 as a **fast, high-quality on-device depth source** and keep the §3 hand-rolled tracker
>   (ICP + features) for pose. Any-view pose remains an *offline / async-over-keyframes* spike
>   (§9.1), not the live tracker.
> - **ExecuTorch `.pte` is a short hop, not a research risk.** ExecuTorch's **Qualcomm (QNN)
>   backend** supports SM8750 (`QcomChipset.SM8750`) and lowers via
>   `to_edge_transform_and_lower` + `QnnPartitioner` (recipe already in `roadmap.md` §Phase 2
>   for DA-V2; identical for DA3). Qualcomm AI Hub ships QNN_DLC/TFLITE/ONNX, **not** `.pte`,
>   so we export the `.pte` ourselves from the DA3-Small torch module — see
>   **`export_da3_executorch.py`** (XNNPACK CPU baseline + QNN/HTP SM8750 fp16). **float fp16
>   already hits ~43 ms on SM8750**, so PT2E int8 quantization (roadmap Phase 3) is an
>   *optional* latency/power optimization, not a prerequisite.
> - **Host side updated to DA3.** `depth_anything_v3.py` runs DA3-Small and emits the **same
>   disparity contract** as `depth_model.py` (`*.disp.png` + `disp_maxes.json`), so
>   `pipeline_float.py` consumes it unchanged. DA3 returns *depth*; we convert to the pipeline's
>   *disparity* convention (`disp = 1/depth`) so the existing affine `s·disp + t` solver is
>   untouched. DA-V2 stays as the documented fallback.
> - **Net decision:** Depth = **DA3-Small via ExecuTorch (QNN/SM8750, float fp16)**, run async.
>   Pose = **§3 tracker** (DA3 any-view pose is offline-only). Scale = still §11.2 (da3-small is
>   relative; `DA3METRIC-LARGE` is metric but 0.35B and not in the NPU-optimized asset set).

1. **Monocular depth — Depth-Anything-V3 (fallback V2) via ExecuTorch.** Export a phone-sized
   model to `.pte` (XNNPACK CPU first; QNN/NPU per `roadmap.md` later). Mirror the host IO in
   `depth_model.py` (update it to DA3). Plain monocular depth is **relative/affine-invariant**
   → see scale (§11.2); DA3's metric variant may remove that step. Target: usable latency
   async (run at e.g. 5–10 Hz, track between frames with features).
2. **Object segmentation — MobileSAM / EfficientSAM via ExecuTorch.** Per-frame object mask,
   prompted from the previous mask centroid or a center box. **Plus hand removal** — either a
   hand-segmentation model, skin heuristic, or train/choose a mask that excludes hands.
   Without hand removal the model grows fingers.
3. **(Optional) feature extractor.** ORB/classic features are fine on CPU (OpenCV is already
   a dependency) for the photometric side of tracking and for loop closure — no ML needed.

Keep depth/mask behind interfaces (`DepthSource`, `ObjectMask`) so they can be stubbed/swapped
and so the tracker can be built and tested before the models are perfect.

---

## 6. Geometry backend, frames, and rendering

- **TSDF in object frame `O`** — dense fixed grid is fine (object roughly centered; default a
  ~0.3–0.4 m cube at 2–4 mm voxels ⇒ ~80–160³). Integrate masked depth at `T_OC(t)`;
  incremental **marching cubes** on a throttle (2–4 Hz) on a worker thread.
- **Convention:** keep everything in the **OpenCV camera frame** (+X right, +Y down, +Z fwd)
  and `pose_convention="opencv"` to match `object_frame`/`tsdf_fuse`. Back-project a pixel
  `(u,v)` with metric depth `z`: `x=(u-cx)z/fx`, `y=(v-cy)z/fy`, `Z=z`; world(object) point
  `X_obj = T_OC · [x,y,z,1]`. Scale intrinsics to the depth resolution you actually run at.
- **`T_OC` now comes from the tracker (§3), not a board.** Everything downstream is identical
  to a board-based fuse — that's why the board path was a useful dry run.
- **Render:** start with an **inset, orbitable preview** (reuse `ModelViewerScreen` look) — no
  AR-alignment math needed. AR overlay (mesh skinned onto the real object via `T_OC` + display
  rotation) is a polish stretch.

---

## 7. "Filling in of regions scanned"

- **Accumulate-as-you-go (primary):** each tracked frame fuses newly seen surface; prior
  surface persists; the shell visibly fills in. Color by recency/confidence so fresh fill-in
  pops. Drive a **surface**-coverage metric (occupied vs expected area) — a real upgrade over
  the angle-only ring.
- **Hole closing (stretch / host):** TSDF interpolates thin gaps for free; true watertight
  filling belongs in host cleanup (`import_and_clean.py`), not the live loop.

---

## 8. Milestones & exit criteria

| Milestone | Deliverable | Exit criteria |
|---|---|---|
| **M0 — Inputs** | `DepthSource` (DA3 `.pte`, fallback DA2) + `ObjectMask` (SAM + hand removal) producing a masked object point cloud per frame; ExecuTorch runtime stood up. | On a handheld object, masked cloud is object-only (no hand/table) at ≥ ~5 Hz. |
| **M1 — Incremental tracker** | Frame-to-model ICP (+feature/photometric) → `T_OC(t)`; TSDF fuse. | Turning a **textured** object ~90° builds a coherent partial shell that doesn't smear; tracking holds. |
| **M2 — Live mesh + preview + coverage** | Incremental marching cubes; inset 3D preview; surface-coverage UI. | Model grows/fills live ≥ 20 fps camera; coverage reflects real surface. |
| **M3 — Loop closure** | Keyframe pose-graph + loop closure for a full 360°. | A full turn closes without a seam/double wall on a textured object. |
| **M4 — Robustness + graceful failure** | Tracking-lost detection, “add texture / try orbit” guidance, symmetric-object detection. | On a smooth/untextured object it **detects** it can't track and guides the user instead of producing garbage. |
| **M5 — Polish (stretch)** | AR overlay skinned on the object; light hole fill; export handoff to host. | Mesh visibly skins the real object; export produces a usable model. |

---

## 9. Open research questions (resolve in-place; record answers here)

1. **[RESEARCH — PARTIALLY RESOLVED 2026-06-27] Can Depth Anything 3 replace/seed the tracker?**
   DA3's any-view model outputs camera pose + consistent multi-view geometry (Nested = metric).
   **Resolved for the *live* path: no.** The only NPU-optimized DA3 asset for the S25 (Qualcomm
   AI Hub `da3-small`, SM8750 NPU, ~43 ms/frame) is the **monocular depth** export, not the
   any-view/pose mode (§5.0a). There is no off-the-shelf real-time on-device DA3 *pose*. So DA3
   = on-device **depth** source; keep the §3 hand-rolled tracker for pose. **Still open (lower
   priority):** DA3 any-view pose as an **offline / async-over-keyframes** refinement to seed or
   correct the tracker — spike on recorded clips (measure pose quality vs board ground truth,
   §11). Mind **CC BY-NC** on the capable any-view variants (Large/Giant) if shipping; da3-small
   any-view is Apache-2.0 but its pose head is not the optimized/accelerated path.
1b. **[RESEARCH — blocking] Markerless in-hand tracking method (fallback if DA3 doesn't fit).**
   Which combination is robust on device: point-to-plane ICP + ORB feature matching to
   keyframes, dense direct alignment, or a learned pose/flow net? Survey **in-hand scanning /
   object SLAM** (e.g. KinectFusion object mode, InfiniTAM object reconstruction, BundleSDF,
   “in-hand scanning with online loop closure”). Pick the lightest that handles moderate
   texture. *Spike offline on recorded clips first.*
2. **[RESEARCH — fundamental] Symmetric/textureless objects (the salt-shaker case).** Confirm
   the degeneracy, and decide the product answer: detect low-texture/symmetry and route to a
   fallback (orbit, §14) or tell the user to add surface texture. Can photometric tracking
   alone carry a *lightly* textured cylinder? Quantify where it breaks.
3. **[RESEARCH — blocking] Hand removal.** Object and hand move together, so motion can't
   separate them. Options: hand-seg model, skin-tone + depth heuristic, SAM with negative
   prompts on hand regions, or “contact-aware” masking. Which is reliable on device?
4. **[RESEARCH — RESOLVED 2026-06-27] Monocular depth via ExecuTorch.** **DA3-Small**
   (Apache-2.0) is the pick: Qualcomm AI Hub already exports it NPU-optimized for the S25
   (SM8750) at **~43 ms QNN float** (§5.0a). We own the `.pte` export via ExecuTorch's QNN
   backend — **`export_da3_executorch.py`** (XNNPACK CPU baseline + QNN/HTP SM8750 fp16);
   DA2-Small remains the fallback. Host side runs DA3 via **`depth_anything_v3.py`** (mirrors
   `depth_model.py`'s disparity contract; DA-V2 kept as fallback). **Remaining to measure on
   real hardware:** sustained/thermal fps under the live loop, and whether int8 PT2E (roadmap
   Phase 3) is worth it (float fp16 may already be enough). See §5.0a and
   `OBJECT_TRACKING_PLAN.md` §11.
5. **[RESEARCH] Metric scale without any reference.** Monocular + free object ⇒ absolute scale
   is unobservable from relative depth. Acceptable to reconstruct **up-to-scale** and set scale
   later? Or use a **metric** variant (**`DA3METRIC-LARGE`** / DA3 Nested, or DA2-metric), or
   anchor scale once from ARCore depth of the static background before pickup? Decide and
   document.
6. **[RESEARCH] Drift & loop closure.** Pose-graph backend (e.g. g2o-style, or a hand-rolled
   small solver) feasible on device? Loop-closure detection (appearance/bag-of-words/feature).
7. **[RESEARCH] Geometry/perf backend.** Kotlin vs NDK C++ vs GPU for TSDF + ICP at frame
   budget on an S25-class device. Likely **NDK C++** for ICP/TSDF; OpenCV (already vendored)
   helps. Decide early — it shapes the whole module.
8. **[OPEN] Initialization & relocalization.** Recover when tracking is lost mid-scan
   (re-detect against keyframes) without restarting the model.

---

## 10. Risks & honest limitations

| Risk / limitation | Mitigation |
|---|---|
| **Smooth, symmetric, untextured objects can't be tracked** (math, not a bug) | Detect low texture/symmetry; guide user to add texture or use **orbit fallback** (§14); never silently produce garbage. |
| Hand fuses into the model | Hand removal is M0-mandatory; don't ship without it. |
| ARCore depth wrong for moving object | Use **monocular** DA3 (fallback DA2) as the depth source, not ARCore depth. |
| No ExecuTorch in the app yet | M0 includes standing it up; gate honestly. Build the tracker against stubbed inputs in parallel. |
| Drift over a full turn | Loop closure (M3); until then, scope to partial scans. |
| Tracking jitter smears TSDF | Robust ICP (point-to-plane, outlier rejection), feature priors, temporal smoothing. |
| Perf (ML + ICP + TSDF + mesh) blows frame budget | Async depth/mask at low Hz; track between with cheap features; NDK; throttle meshing; double-buffer. |
| Scope creep | This is genuinely a research project; ship partial-scan “fills in as I turn it” early, perfect 360° later. |

---

## 11. Test / validation plan

- **Offline first.** Build a desktop harness (Python) that runs the **tracker** on recorded
  clips and checks recovered `T_OC(t)` against ground truth. You can generate ground truth two
  cheap ways: (a) **synthetic** renders with known rotation (extend `charuco_pose` selftest /
  `demo/render_turntable.py`), and (b) **recorded board sessions** whose `frame_*.json` already
  contain true `T_CO` — board only as a *measuring stick*, never at runtime. Tracker error vs
  truth is your core metric.
- **Geometry parity:** fused cloud/mesh from tracker poses should approximate
  `pipeline_float.py` output on the same clip.
- **Pure-logic unit tests** (JVM) for TSDF indexing, coverage %, pose math — like
  `ObjectAngleCoverage`’s tests.
- **On device:** textured object (passes) and a smooth cylinder (must fail *gracefully*).
- Keep Orbit/Turntable modes and the host suite **green** — this is additive.

---

## 12. Suggested file layout

```
app/.../recon/DepthSource.kt        # iface; DaV2DepthSource (.pte) impl
app/.../recon/ObjectMask.kt         # iface; SAM + hand-removal impl
app/.../recon/Tracker.kt            # frame-to-model ICP + features → T_OC(t)
app/.../recon/LiveModel.kt          # TSDF in O, integrate + marching cubes (NDK-backed?)
app/.../recon/PoseGraph.kt          # keyframes + loop closure (M3)
app/.../rendering/MeshRenderer.kt   # live mesh (+ point splats early)
app/.../recon/LivePreviewPanel.kt   # inset Compose 3D preview
app/src/main/cpp/ (optional)        # NDK ICP/TSDF kernels if Kotlin is too slow
```

Add a `CaptureMode.Handheld` (board-free) so Orbit/Turntable are untouched.

---

## 13. TL;DR for the next agent

1. This is **markerless in-hand object SLAM**. No board, ever. It's a real research build.
2. Stand up the inputs (M0): **DA3 depth + object/hand mask via ExecuTorch** → masked
   object cloud.
3. Build the **frame-to-model tracker** (M1) — ICP + features → `T_OC(t)`; fuse into a TSDF in
   `O`; this *replaces* the board's `solvePnP`.
4. Live mesh + preview + surface coverage (M2); **loop closure** for clean 360° (M3).
5. **Fail gracefully** on smooth/symmetric/untextured objects (M4) — detect and guide.
6. Validate the tracker **offline against ground truth** before any on-device UI.

---

## 14. Fallback for objects that can't be tracked markerless

Be upfront in the UX: some objects (smooth, symmetric, untextured) **cannot** be scanned by
turning them in hand without markers — there's no signal to track. For those, the working,
board-free option is **Orbit mode**: set the object down and move the **phone** around it
(ARCore tracks the phone; works on any stationary object). Detect the tracking-failure case
and offer to switch to Orbit rather than letting the user fight it.

---

## 15. SHIPPED (2026-06-27) — on-device live mesh, end to end on the device

The board-free live mesh is **implemented and building into the debug APK**. It runs the
phone-moves-around-a-stationary-object path (the M2 "live mesh + surface coverage" target via
ARCore world tracking, not yet the in-hand tracker of §3/M1), with the §5.0a DA3-on-NPU depth
source wired in and degrading gracefully to ARCore depth when no model is present.

### What was built (all under `app/src/main/java/com/lantern/recorder/`)
- **`recon/ReconTypes.kt`** — `CameraIntrinsics`, `DepthMap`, `MeshData`, and a tiny row-major
  `Mat4` (column↔row major, rigid invert, point transform).
- **`recon/ImageUtils.kt`** — copies out of ARCore `Image`s: raw depth16→meters,
  confidence8→0..1, YUV_420_888→ARGB (matches `FrameRecorder` conventions).
- **`recon/DepthSource.kt`** — `ArCoreRawDepthSource` (metric, always available) and
  `Da3DepthModel` (ExecuTorch `Module.load` of the DA3 `.pte`; ImageNet-normalized
  518×518 NCHW in, depth out → relative disparity `1/depth`). Missing/failed model → `null`.
- **`recon/AffineScaleSolver.kt`** — on-device port of `scale_solver.py` (Huber-IRLS `s·disp+t`,
  the **both** scale *and* shift fit from roadmap Decision 3); turns DA3 relative disparity into
  dense metric depth using ARCore depth as the metric reference.
- **`recon/TsdfVolume.kt`** — dense world-space TSDF (default 160³ @ 4 mm), ARCore→OpenCV
  camera flip matching `tsdf_fuse.py`, confidence-weighted integration.
- **`recon/MarchingCubes.kt`** — canonical Bourke edge/tri tables; only meshes fully-observed
  cells so the surface fills in as you scan.
- **`recon/LiveReconstructor.kt`** — keyframe gate (2 cm), acquires depth/RGB on the GL thread,
  fuses + re-meshes on a worker thread, publishes the newest mesh lock-free.
- **`rendering/MeshRenderer.kt`** — GLES2 two-sided-Lambert mesh draw over the camera feed,
  using ARCore's view/projection.
- **UI:** new `CaptureMode.LiveMesh` (segmented toggle), a live `verts · frames · backend`
  readout, and full `MainActivity` wiring (start/stop on mode select, per-frame draw, lifecycle
  cleanup). Orbit/Turntable paths untouched.
- **Gradle:** `org.pytorch:executorch-android` dependency + `jniLibs` `pickFirst` for the
  `libc++_shared.so` clash between OpenCV and fbjni.

### Build / run
```
# Build the debug APK (needs JDK 17+; Android Studio's bundled JBR works):
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
  ./gradlew :app:assembleDebug
# Install:
adb install -r app/build/outputs/apk/debug/app-debug.apk
```
The live mesh works **with no model** (ARCore depth only). For dense DA3-on-NPU depth, export
the `.pte` and push it to the app's files dir:
```
python export_da3_executorch.py --backend qnn --soc SM8750 --out da3_small_sm8750.pte
adb push da3_small_sm8750.pte \
  /sdcard/Android/data/com.lantern.recorder/files/da3_small_sm8750.pte
```
In the app: pick **Live Mesh**, then slowly move the phone around a stationary object — the
mesh grows live. The readout shows `DA3 NPU` when the model loaded, else `ARCore depth`.

### Known gaps / next
- This is the **phone-orbits-object** live mesh (ARCore world pose). The §3/M1 **in-hand
  markerless tracker** (ICP + features → `T_OC`) is still the open research item; the
  reconstruction modules here are reused by it once the tracker lands.
- TSDF is pure Kotlin (160³). Fine at keyframe rate; move to NDK (§12) if 30 Hz fusion is
  needed.
- Verify the exact DA3 `.pte` output semantics (depth vs disparity, output rank) against the
  actual export and adjust `Da3DepthModel.depthToDisparity` if needed.
