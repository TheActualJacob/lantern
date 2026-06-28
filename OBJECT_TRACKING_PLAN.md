# Object-Centric Capture — Migration Plan

**Status:** In progress (T0, T1, T2, T3, T4 landed; T3.5/T5 remaining) · **Owner:** Person C (Android) + Person A (host recon) · **Date:** 2026-06-27

> **Implementation status** — see §11 for what is in the repo. Landed: the host
> ChArUco T0 spike, the phone `PoseProvider` seam (T1) + data contract, the
> on-device OpenCV ChArUco board detector + object-rotation keyframe gating (T2),
> the live object-angle coverage + capture-mode UI (T3), and the host object-frame
> fusion (T4). Gated/next: on-device DA-V2/SAM via ExecuTorch (T3.5), live mesh (T5).

Switch Lantern from **camera-tracked** capture (walk around a stationary object) to
**object-tracked** capture (rotate the object in front of a mostly-fixed phone), and lay
the first rail toward a **real-time on-device mesh preview**.

---

## 1. Why the current system requires walking around the object

The whole pipeline assumes **camera moves, object is fixed in ARCore's world frame**:

1. **ARCore raw depth needs parallax.** Depth is triangulated from the camera moving
   between frames. A near-stationary phone gives no baseline → depth degenerates.
2. **TSDF fuses in ARCore world space.** If the object rotates, its surfaces land at
   *different world coordinates* every frame → the fused mesh smears/ghosts.
3. **Coverage tracks camera azimuth** around the object. No camera motion → azimuth never
   changes → coverage can't advance. This is exactly the "rotating the object does nothing"
   symptom.

None of these are bugs; they're consequences of fusing in the **camera/world** frame.
To rotate the object, we must fuse in the **object's own frame**.

> Key idea: introduce an **object frame** `O`. Track the rigid transform `T_OW(t)`
> (object←world) per frame, and fuse every frame's depth into `O`. If the object is on a
> turntable, `T_OW` is a pure rotation about the turntable axis. Once everything lives in
> `O`, a rotating object is "stationary" again and TSDF/coverage work as before.

---

## 2. Target capture modes

| Mode | Object | Phone | Pose source | Notes |
|---|---|---|---|---|
| **Orbit** (today) | still | orbits | ARCore world pose | Works now; keep as default for big/awkward objects. |
| **Turntable** (new) | rotates | ~fixed (small handheld orbit OK) | **fiducial board** under/around object | The "rotate the object" mode. Robust, deterministic. |
| **Markerless object** (stretch) | rotates | ~fixed | object SLAM / dense tracking | Research-grade; fragile. Only if turntable proves out. |

We build **Turntable** next. Markerless is explicitly a later stretch.

---

## 3. Turntable mode — how it works

### 3.1 Fiducial board defines the object frame
- Print a **ChArUco / ArUco board** (OpenCV `cv::aruco`, board > single marker so partial
  occlusion by the object is tolerated; recommend ChArUco for sub-pixel corner accuracy).
- Place the object **on/at the center** of the board; the board's coordinate system *is*
  the object frame `O`. As the user spins the board, the object spins with it.
- Per RGB frame: detect markers → `solvePnP` (board corners + camera intrinsics, which we
  already record) → `T_CO(t)` (camera←object). Invert to get the object's pose.

### 3.2 Fuse in the object frame
For each keyframe we already capture (RGB + raw depth + intrinsics + ARCore pose):
- Back-project depth → points in **camera** space.
- *(Optional but recommended)* mask points to the **object only** using an on-device
  segmentation model (see §6), so the ChArUco board and table are discarded and never
  enter the mesh.
- Transform points by `T_OC(t) = T_CO(t)⁻¹` → points in the **object** frame.
- Integrate into a TSDF volume anchored to `O`. The rotating object is now static in `O`.

### 3.3 Why depth still needs a little camera motion
Raw depth quality drops with zero baseline. Mitigations (in priority order):
1. **Small handheld orbit on top of the rotation** — a few cm of phone sway restores
   parallax while the turntable provides the angular coverage. Easiest, biggest win.
2. **Lean on the neural depth model** (DA-V2) for dense depth, using ARCore sparse/raw
   depth only where confident (already the host pipeline's fallback strategy).
3. Accept lower per-frame depth, rely on TSDF averaging across many angles.

### 3.4 Coverage redefined as object-angle
Coverage becomes **how much of the turntable rotation has been observed**: derive the
object's yaw about the turntable axis from `T_CO(t)` and light the dome ring by *object
angle* instead of *camera azimuth*. The existing dome/ring UI carries over unchanged —
only the angle source swaps. The flip flow (capture the bottom) also carries over: flip the
object on the board, the board re-localizes, pass two fuses into the same `O`.

---

## 4. Architecture changes

```
                 ┌────────────────────────── PHONE (this repo) ──────────────────────────┐
 ARCore frame ──▶│ RGB + rawDepth + intrinsics + ARCore pose  (unchanged FrameRecorder)   │
                 │                                                                         │
                 │  NEW: PoseProvider abstraction                                          │
                 │   ├─ OrbitPoseProvider     → camera world pose (today's path)           │
                 │   └─ TurntablePoseProvider → ArUco board detect + solvePnP → T_CO(t)    │
                 │                                                                         │
                 │  Recorder writes per-frame object pose T_CO into frame_NNNN.json        │
                 │  + capture_mode + board spec in a session sidecar                       │
                 └─────────────────────────────────────────────────────────────────────────┘
                                              │  session files (+ object poses)
                                              ▼
                 ┌────────────────────────── HOST (Person A) ────────────────────────────┐
                 │  pipeline picks frame: world pose (orbit) OR object pose (turntable)    │
                 │  fuse TSDF in the chosen frame → mesh → Blender clean → .glb            │
                 └─────────────────────────────────────────────────────────────────────────┘
```

The phone stays a **recorder**; the only new on-device work is **per-frame board pose
estimation**. The host **fusion frame** becomes selectable — this is the one real
reconstruction-side change and must be coordinated with Person A.

### 4.1 Data contract additions (no breaking changes)
- `frame_NNNN.json`: add `object_pose_T_CO` (4×4, camera←object) + `object_pose_valid`
  (bool; false when the board wasn't seen that frame → host skips or interpolates).
- New `capture.json` sidecar per session:
  ```json
  {
    "capture_mode": "turntable",        // or "orbit"
    "object_frame": "charuco",
    "board": { "dict": "DICT_5X5_100", "squares_x": 5, "squares_y": 7,
               "square_len_m": 0.03, "marker_len_m": 0.022 },
    "axis_hint": "board_z"
  }
  ```
- Backward compatible: missing fields ⇒ host treats the session as `orbit` (today's path).

---

## 5. On-device fiducial detection options

| Option | Pros | Cons |
|---|---|---|
| **OpenCV Android (`opencv-android` aruco)** | Battle-tested, ChArUco + solvePnP built in | ~adds OpenCV dep (mitigate: aruco/calib3d modules only, or a slim build) |
| **Hand-rolled ArUco** | No heavy dep | Re-implementing robust detection is a trap; don't |
| **ARCore Augmented Images** | Already in ARCore, no new dep | Tracks *image targets*, not a metric board pose for fusion; weaker for precise `T_CO` |

**Recommendation:** OpenCV aruco/ChArUco. Detection runs on the RGB frame we already
acquire; budget it on a worker thread keyed to keyframes, not every GL frame.

---

## 6. ExecuTorch / on-device models — where they fit

**Key distinction: ExecuTorch runs neural networks, not classical geometry.** It is the
on-device inference runtime for exported PyTorch models (`.pte`) on CPU/GPU/NPU. It does
**not** run marker detection, `solvePnP`, TSDF, or ICP — those stay classical (OpenCV /
geometry). So ExecuTorch does **not** help with the turntable's *pose* core, but it has
three high-value roles around it.

| Workflow piece | Neural? | ExecuTorch? |
|---|---|---|
| Marker/board pose (turntable core) | No | ❌ OpenCV `solvePnP` |
| Dense depth on a near-fixed phone | **Yes** | ✅ **DA-V2 `.pte` (already the project spine)** |
| Object segmentation (mask out board/table) | **Yes** | ✅ **EfficientSAM-Ti / MobileSAM** |
| TSDF / ICP merge | No | ❌ classical geometry |
| Live-mesh per-frame depth source | **Yes** | ✅ feeds T5 |
| Markerless object tracking | Yes | ⚠️ possible, research-grade |

### 6.1 Dense depth (DA-V2) — fixes the turntable's biggest weakness
The zero-parallax problem in §3.3 is exactly what a **monocular** depth net solves: it needs
no stereo baseline, estimating depth from a single image. On a turntable, on-device DA-V2
(already being lowered to `.pte` for the Hexagon NPU, see roadmap P2/P3) therefore becomes
the **primary** depth source rather than a fallback. ExecuTorch is what runs it live.

### 6.2 Segmentation (EfficientSAM-Ti / MobileSAM) — the legitimate "object detection"
The recurring "we need object detection" instinct is really a need for an **object mask**,
not a classifier. Segmentation (already Decision 1 in the roadmap) runs well on-device via
ExecuTorch and is especially valuable in turntable mode:
- **Isolates the object** from the ChArUco board and table, so markers/background never land
  in the mesh (§3.2).
- **Robust centroid + coverage:** masking gives a far more reliable object centroid than the
  depth-center heuristic the current capture UI uses, and lets coverage track real object
  surface rather than whatever sits at frame center.
- Run it keyed to keyframes on a worker thread, like the board detector — not every GL frame.

### 6.3 The live mesh (T5) is downstream of ExecuTorch
The on-device TSDF preview needs a depth map (and ideally a mask) per frame. DA-V2 + the
masker, both via ExecuTorch, are exactly what supply it. ExecuTorch is the inference engine
under the live-mesh stretch — object tracking gives the *frame*, ExecuTorch gives the
*geometry source* that fills it.

### 6.4 Honest "no": markerless object tracking
Dropping the board and tracking the object directly could use learned components ExecuTorch
*can* run (learned feature matchers, learned 6-DoF pose nets), but they are heavy and
fragile on texture-less objects. ExecuTorch makes them deployable, not reliable — keep
markerless a research stretch (§2), unchanged.

> Dependency note: DA-V2 and the masker are **models exported to `.pte`**, run through the
> ExecuTorch Android runtime the project already plans to integrate (roadmap P4) — they do
> **not** pull in OpenCV. OpenCV is only for the classical board pose (§5). The two new
> on-device costs (board pose via OpenCV, depth+mask via ExecuTorch) are independent.

---

> Constraint check: the original UX task said "no new heavyweight dependencies." OpenCV is
> heavy, so Turntable mode is a **new feature behind a mode switch**, gated and documented —
> not bundled into the default orbit path. Confirm the dependency before Phase T2.

---

## 7. Phased delivery

Named **T-phases** to slot beside the existing roadmap (P0–P6).

| Phase | Deliverable | Side | Exit criteria |
|---|---|---|---|
| **T0 — Spike** | Offline: feed a recorded turntable clip through OpenCV ChArUco + solvePnP; plot `T_CO(t)` yaw vs. time | Host | Smooth, monotonic yaw over a hand-spin; reprojection error < ~1 px |
| **T1 — PoseProvider seam** | Refactor capture to a `PoseProvider` interface; `OrbitPoseProvider` = today's behavior, no functional change | Phone | Orbit mode byte-identical; build green |
| **T2 — Board pose on device** | `TurntablePoseProvider` (OpenCV aruco) detects the board, writes `object_pose_T_CO` + `capture.json` | Phone | On-device overlay draws the board axes locked to the turntable while spinning |
| **T3 — Object-angle coverage** | Coverage ring driven by object yaw from `T_CO`; flip flow works on the board | Phone | Rotating the object advances coverage; flip prompt fires; no camera walk needed |
| **T3.5 — On-device depth + mask** | Run DA-V2 (+ EfficientSAM/MobileSAM) via ExecuTorch as the turntable depth source and object isolator (§6) | Phone | Live monocular depth on a near-fixed phone; object masked from board/table |
| **T4 — Host fuse in object frame** | `pipeline_float.py` selects world/object frame; TSDF fuses turntable session into one mesh | Host | Clean mesh from a rotate-the-object capture on ≥2 objects |
| **T5 — Live mesh preview (stretch)** | Coarse on-device TSDF in `O`, render the growing shell as the user spins | Phone | Low-res mesh grows live; ties into "true surface coverage" |

T1 is pure-refactor insurance. T0 + T4 de-risk the host side before committing phone work.
T3.5 reuses the project's existing ExecuTorch lane (roadmap P2–P4) rather than new modeling.

---

## 8. The bridge to real-time mesh

This plan is the on-ramp to the live mesh, because the hard prerequisite for an on-device
mesh is a **stable frame to fuse into**:

- **Today:** no persistent geometry; coverage is a pose-direction proxy.
- **T2–T3:** every keyframe carries a metric `T_CO` → frames are co-registered in `O`.
- **T5:** stream those object-frame point clouds into a coarse voxel/TSDF grid and render
  the partial shell. That *is* a minimal live mesh — and it finally gives **true surface
  coverage** (which faces actually have geometry) instead of viewing-angle coverage.

So object tracking (T2–T3) and the live mesh (T5) are the same effort staged: get the
frame right first, accumulate geometry into it second.

Markerless object tracking (drop the board, track the object directly via dense/ICP
frame-to-model alignment) is the natural successor to T5 once a live model exists to align
against — but it stays a research stretch and never blocks the turntable demo.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Object **occludes** the board markers | Use a **board** (many markers) not one; ChArUco tolerates partial views; require ≥N markers for a valid `T_CO`. |
| **Zero-parallax** depth on a fixed phone | Coach a small handheld orbit on top of rotation; **lean on on-device DA-V2 monocular depth via ExecuTorch (§6.1)** — it needs no baseline. |
| OpenCV **dependency weight** | Gate behind Turntable mode; ship aruco/calib3d modules only; keep orbit path dep-free. Depth/mask models are `.pte` via ExecuTorch, separate from OpenCV (§6). |
| Board/table **bleeds into the mesh** | On-device segmentation mask (EfficientSAM-Ti / MobileSAM via ExecuTorch, §6.2) keeps only object points before fusion. |
| Board pose **jitter** smears TSDF | Temporal smoothing of `T_CO`; reject high-reprojection-error frames; sub-pixel ChArUco corners. |
| **Scale**: board gives metric scale | Bonus — the board's known square size yields metric scale directly, cross-checking the host affine solver. |
| Host **fusion-frame** change is cross-team | T0/T4 prove it offline first; data contract is backward compatible so orbit keeps working throughout. |

---

## 10. First concrete steps (no repo-wide commitment)

1. **T0 spike (host, ~half day):** record one hand-spin of an object on a printed ChArUco
   board; run OpenCV detect + solvePnP offline; confirm clean yaw and reprojection error.
2. **T1 refactor (phone):** introduce `PoseProvider`, move today's logic into
   `OrbitPoseProvider`. Zero behavior change — safe to land immediately.
3. Decide OpenCV dependency + mode-switch UX, then start **T2**.

T0 and T1 are independent and low-risk; everything heavier waits on the T0 result and an
explicit OK to (a) add OpenCV and (b) change the host fusion frame.

---

## 11. Implementation status (this repo)

What has actually landed, file by file. The data contract is backward compatible
throughout: a session with no `capture.json` / no object poses is treated as
today's orbit capture and fuses in the world frame exactly as before.

### Landed

| Phase | Status | Where | Notes |
|---|---|---|---|
| **T0 — Spike** | ✅ Done | `charuco_pose.py` | ChArUco detect → `solvePnP` → `T_CO(t)` + object-yaw-about-Z, with a synthetic `--selftest` (renders + detects real boards) and a `--clip` CLI that dumps per-frame `T_CO`, reprojection error, and yaw-vs-time (optional `--plot`). Self-test result: yaw recovery err < 0.002°, reprojection < 0.1 px, geometry round-trip ~3e-9° — exceeds the "< ~1 px" exit bar. |
| **T1 — PoseProvider seam** | ✅ Done | `app/.../recording/PoseProvider.kt` | `PoseProvider` interface + `ObjectPose` + `BoardSpec` + `BoardDetector` seam. `OrbitPoseProvider` = today's behaviour (no object pose). `TurntablePoseProvider` delegates to a (currently null) detector → records metadata-only turntable sessions without pulling in OpenCV. |
| **Data contract** | ✅ Done | `app/.../recording/FrameRecorder.kt` | Holds a `poseProvider` (default orbit). Writes a per-session `capture.json` sidecar (mode + board spec) and stamps each keyframe JSON with `object_pose_valid` and, when valid, `object_pose_T_CO` (row-major 4×4) + corner count + reprojection error. Orbit frame bytes unchanged. |
| **T2 — Board pose on device** | ✅ Done | `app/.../recording/OpenCvBoardDetector.kt`, `app/build.gradle.kts` | OpenCV `org.opencv:opencv:4.13.0` (objdetect ChArUco + calib3d `solvePnP`), `OpenCVLoader.initLocal()` at startup. `OpenCvBoardDetector` grabs the camera luma plane, runs `CharucoDetector.detectBoard` → match ids to board object points → `solvePnP` with ARCore intrinsics → row-major `T_CO`, rejecting < 6 corners or > 2 px reprojection. Plugged into `TurntablePoseProvider`. Classical geometry — **not** ExecuTorch. |
| **T3 — Object-angle coverage** | ✅ Done (logic + wired) | `app/.../scanning/ObjectAngleCoverage.kt`, `FrameRecorder.kt`, `CaptureUiState.kt`, `MainActivity.kt` | Turntable keyframes gate on **object yaw** (≥ 5° about board Z) instead of camera translation — so spinning the object in front of a fixed phone advances capture. The coverage ring is now driven live by `T_CO` (12 object-yaw sectors mapped onto the existing dome), and a full spin auto-completes a two-pass scan. Capture-mode toggle (Orbit/Turntable) + board-lock status in the overlay. |
| **T4 — Host fuse in object frame** | ✅ Done | `object_frame.py`, `pipeline_float.py` | New `--fusion-frame {auto,world,object}` (auto picks `object` for turntable sessions via `capture.json`). Object frames feed `T_OC = inv(T_CO)` to the unchanged TSDF integrator with the `opencv` axis convention; frames with no board pose are dropped. Verified end-to-end through real Open3D on a synthetic spin. |
| **Converters** | ✅ Done | `convert_session.py`, `android_session_to_pipeline.py` | Carry `capture.json` to the host and write `frame-*.object_pose.txt` (only for valid frames). |
| **Tests** | ✅ Done | `tests/test_charuco_pose.py`, `tests/test_object_frame.py`, `tests/test_android_session_to_pipeline.py` | Full host suite: **37 passing** (was 17). Android `:app:compileDebugKotlin` is green. |

### Gated / not yet started (need explicit go-ahead per §6, §7)

- **T3.5 — On-device depth + mask (DA-V2 / EfficientSAM via ExecuTorch):** reuses
  the project's existing ExecuTorch lane (roadmap P2–P4); independent of OpenCV.
- **T5 — Live mesh preview:** downstream of T2/T3.5 — coarse on-device TSDF in `O`.

### How to exercise what's landed

```sh
# T0: validate the board-pose math + detector end-to-end (no data needed)
python charuco_pose.py --selftest

# T0 on a real hand-spin clip (a recorder session folder of frame_*.png + frame_*.json)
python charuco_pose.py --clip path/to/session_<stamp> --out yaw.json --plot yaw.png

# T4: fuse a converted turntable session in the object frame
python convert_session.py path/to/session_<stamp>           # carries capture.json + object poses
python pipeline_float.py --frames .../frames --arcore .../arcore \
    --disparities .../disparities --fusion-frame auto --output out/mesh.glb
```
