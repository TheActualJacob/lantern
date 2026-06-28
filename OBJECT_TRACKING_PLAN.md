# Object-Centric Capture — Migration Plan

**Status:** Proposed · **Owner:** Person C (Android) + Person A (host recon) · **Date:** 2026-06-27

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

> Constraint check: the original UX task said "no new heavyweight dependencies." OpenCV is
> heavy, so Turntable mode is a **new feature behind a mode switch**, gated and documented —
> not bundled into the default orbit path. Confirm the dependency before Phase T2.

---

## 6. Phased delivery

Named **T-phases** to slot beside the existing roadmap (P0–P6).

| Phase | Deliverable | Side | Exit criteria |
|---|---|---|---|
| **T0 — Spike** | Offline: feed a recorded turntable clip through OpenCV ChArUco + solvePnP; plot `T_CO(t)` yaw vs. time | Host | Smooth, monotonic yaw over a hand-spin; reprojection error < ~1 px |
| **T1 — PoseProvider seam** | Refactor capture to a `PoseProvider` interface; `OrbitPoseProvider` = today's behavior, no functional change | Phone | Orbit mode byte-identical; build green |
| **T2 — Board pose on device** | `TurntablePoseProvider` (OpenCV aruco) detects the board, writes `object_pose_T_CO` + `capture.json` | Phone | On-device overlay draws the board axes locked to the turntable while spinning |
| **T3 — Object-angle coverage** | Coverage ring driven by object yaw from `T_CO`; flip flow works on the board | Phone | Rotating the object advances coverage; flip prompt fires; no camera walk needed |
| **T4 — Host fuse in object frame** | `pipeline_float.py` selects world/object frame; TSDF fuses turntable session into one mesh | Host | Clean mesh from a rotate-the-object capture on ≥2 objects |
| **T5 — Live mesh preview (stretch)** | Coarse on-device TSDF in `O`, render the growing shell as the user spins | Phone | Low-res mesh grows live; ties into "true surface coverage" |

T1 is pure-refactor insurance. T0 + T4 de-risk the host side before committing phone work.

---

## 7. The bridge to real-time mesh

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

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Object **occludes** the board markers | Use a **board** (many markers) not one; ChArUco tolerates partial views; require ≥N markers for a valid `T_CO`. |
| **Zero-parallax** depth on a fixed phone | Coach a small handheld orbit on top of rotation; lean on DA-V2 dense depth. |
| OpenCV **dependency weight** | Gate behind Turntable mode; ship aruco/calib3d modules only; keep orbit path dep-free. |
| Board pose **jitter** smears TSDF | Temporal smoothing of `T_CO`; reject high-reprojection-error frames; sub-pixel ChArUco corners. |
| **Scale**: board gives metric scale | Bonus — the board's known square size yields metric scale directly, cross-checking the host affine solver. |
| Host **fusion-frame** change is cross-team | T0/T4 prove it offline first; data contract is backward compatible so orbit keeps working throughout. |

---

## 9. First concrete steps (no repo-wide commitment)

1. **T0 spike (host, ~half day):** record one hand-spin of an object on a printed ChArUco
   board; run OpenCV detect + solvePnP offline; confirm clean yaw and reprojection error.
2. **T1 refactor (phone):** introduce `PoseProvider`, move today's logic into
   `OrbitPoseProvider`. Zero behavior change — safe to land immediately.
3. Decide OpenCV dependency + mode-switch UX, then start **T2**.

T0 and T1 are independent and low-risk; everything heavier waits on the T0 result and an
explicit OK to (a) add OpenCV and (b) change the host fusion frame.
