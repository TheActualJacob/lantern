# Prompt: Scanning UX & Real-Time Coverage Guidance for Lantern Recorder

Paste this whole file as the opening message of a new agent session. It is self-contained.

---

## Role

You are working on **Lantern**, an Android app (Person C's component) that records synced
ARCore sessions (RGB + raw depth + confidence + 6-DoF pose + camera intrinsics) for an
off-device 3D reconstruction pipeline. Your job in this session is to design and implement
the **object-scanning user experience**: clear capture instructions, a real-time coverage
guide, and correctness guardrails for how users physically scan objects.

Do **not** touch the Python host pipeline, the ARCore session config, or the capture/threading
logic in `FrameRecorder`. This is a UI + on-screen-guidance task only.

---

## First: fetch up-to-date docs before writing code

Use the available doc/web tools to pull **current (2026)** guidance for:

- Jetpack Compose + Material 3 (Material 3 Expressive) — components, theming, motion.
- Compose `AndroidView` interop for overlaying UI on a `GLSurfaceView` (the ARCore camera
  feed MUST stay on the existing `GLSurfaceView`; your UI is a Compose overlay on top).
- ARCore `Camera.getPose()` / `Frame` pose usage for deriving a viewing-direction coverage
  visualization on the host display.
- Compose Canvas / Graphics for drawing a lightweight 2D/3D coverage gizmo overlay.
- Accessibility for camera-overlay UI (content descriptions, contrast scrims, 48dp targets).

Confirm and pin the latest stable Compose BOM + Material3 versions in
`gradle/libs.versions.toml` if you add anything.

---

## Current architecture (do not break)

- Capture feed renders via `BackgroundRenderer` into a `GLSurfaceView`. The UI was recently
  migrated to Jetpack Compose (see `app/src/main/java/com/lantern/recorder/ui/` and
  `MainActivity`). Keep GL rendering + ARCore session lifecycle intact.
- Recording is driven by `FrameRecorder`
  (`app/src/main/java/com/lantern/recorder/recording/FrameRecorder.kt`), which exposes:
  `onStatus`, `onFrameSaved`, `isRecording`, `savedFrameCount`, `sessionName`,
  `sessionPath`. Wire new UI to these callbacks; do not change the recorder's logic.
- A frame is only saved after the camera **translates ≥ ~3 cm** since the last saved frame
  (keyframe gating). Per-frame camera pose, intrinsics, and raw depth are already captured.
- Brand color `#0B1F33`; record red `#E53935`; theme `Theme.Lantern` (Material3 DayNight).

---

## Why these constraints exist (ground the UX in the pipeline)

- ARCore **raw depth needs translation/parallax** — standing still or pure rotation produces
  no usable depth and degenerates the downstream scale solver.
- **TSDF fusion assumes the object is stationary** in ARCore's world frame. If the object
  moves (e.g. held in a moving hand, or rotated in place), its surfaces land in different
  world positions each frame and the mesh smears/ghosts/destroys.
- TSDF **only reconstructs surfaces the camera actually saw** — unseen faces become holes.
- Raw depth is low-resolution (e.g. 160×90 on the test device) and unreliable very close
  (<~15 cm) or far; objects should fill the frame at roughly 20–60 cm.

---

## Deliverable 1 — Scanning instructions (coaching UI)

Surface concise, friendly guidance. Do NOT wall-of-text it.

- A one-time **first-run coach overlay** (dismissible, re-openable from a help affordance)
  with the core rules, each as a short line + small icon:
  - "Put the object on a flat surface — don't hold or rotate it."
  - "Move the phone slowly around the object."
  - "Orbit around it — don't spin in place." (translation, not rotation)
  - "Keep it centered, about 20–60 cm away."
  - "Cover all sides — top, sides, and angles."
  - "Use good, even lighting; avoid glare."
- A persistent, low-key **hint line** during capture (reuse the existing status chip), e.g.
  "Move slowly around the object" that yields to real recording status once frames flow.
- Copy belongs in `strings.xml` (no hardcoded strings), with content descriptions.

## Deliverable 2 — Real-time coverage guide (the high-value feature)

Goal: show the user **where they still need to scan**, live, on the phone.

Implementation guidance — **do the cheap pose-based version, not a live mesh:**

- The phone is a recorder; there is no on-device mesh yet, so do NOT attempt live TSDF.
- Instead, build a **viewing-direction coverage gizmo**: using the per-keyframe camera poses
  (already available from ARCore), compute the direction from the object's estimated
  centroid to the camera for each saved frame, and fill in that sector/marker on a
  ring or dome gizmo rendered as a Compose overlay.
- Estimate the object centroid pragmatically (e.g. a point a fixed distance in front of the
  first keyframe along the camera's forward axis, or the running median of raw-depth points
  near frame center). Keep it simple and robust; it only needs to be good enough to tell
  "you covered the front but not the back/top."
- Visual: a small orbit ring or hemisphere in a screen corner with covered angles lit and
  uncovered angles dim, updating as `onFrameSaved` fires. Add a coverage % readout.
- Keep it cheap (no heavy 3D engine). Compose Canvas or a minimal GL overlay is fine.

State whether you implemented the ring (2D azimuth coverage) or dome (azimuth+elevation);
ring is acceptable for v1.

## Deliverable 3 — Correctness guardrails (not just polish)

- Prominent, hard-to-miss guidance to **place the object on a surface and move the phone
  around it — do not hold or rotate the object.** This is a reconstruction-correctness rule,
  not a nicety; moving objects break fusion.
- If feasible from existing signals, show a gentle warning state when motion looks like
  **pure rotation / no translation** (recorder isn't saving frames) — e.g. "Move around the
  object, not just turning." Derive this only from data already available (pose deltas /
  `onStatus`); do not modify the recorder.
- Note in the UI that **hands/fingers in frame may appear in the scan** — encourage table
  placement so hands stay out of view.

---

## Constraints

- Edge-to-edge, gesture-nav friendly, respects `WindowInsets`; portrait capture.
- Accessibility: content descriptions, ≥48dp touch targets, contrast scrims over the feed.
- No new heavyweight dependencies. Keep the build green: `./gradlew :app:assembleDebug`.
- UI/guidance layer only — do not change Python, ARCore config, or capture logic.

## Done means

- App builds and runs on device with the live feed + recording fully functional.
- Coach overlay + persistent hint line present, all copy in `strings.xml`.
- Live pose-based coverage gizmo updates as the user orbits and shows coverage %.
- Clear "place on a surface, don't hold/rotate" guardrail surfaced prominently.
- A short note listing the doc versions you pinned and whether coverage is ring or dome.
