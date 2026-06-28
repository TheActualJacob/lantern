# Live Mesh → Demo & Proof — Handoff

**Audience:** Robert (Android/live mesh) + Charles (D, integration/proof/demo).
**Date:** 2026-06-28.

> **Read `LIVE_MESH_PLAN.md` §15 first.** The live mesh feed is **already SHIPPED** —
> `LiveReconstructor` + `TsdfVolume` + `MarchingCubes` + `AffineScaleSolver` + on-device `.obj`
> export + live `MeshRenderer`, running the phone-orbits-object path on ARCore world pose, with
> DA3-on-NPU depth degrading to ARCore depth. **This handoff is NOT "build live mesh" — that's
> done.** It's the remaining seam to make the live mesh *win the demo*: connect the on-device
> export to the proof/CAD chain, and close the small demo-readiness gaps.

---

## The one missing seam (highest value, my lane × yours)

The phone now writes a live-mesh **`.obj`** (`MeshExport.writeObj` → app `models/` dir, adb-pullable).
My host tooling already turns a mesh into the demo's proof artifacts — but **nobody has wired the
two ends together.** Once joined, one phone scan produces: clean CAD mesh + STL + a real accuracy
number. That's deck slides 4/5/6 in one pull.

```
[phone] Live Mesh → MeshExport.writeObj → models/lantern_*.obj
   │  adb pull  (or in-app share)
   ▼
[host] import_and_clean.py  → watertight, scaled .glb + .stl   (CAD handoff)
       cad_check.py         → "imports as solid" verdict (OpenCASCADE)
       ground_truth.py      → mean error __ mm  + color error .ply  (vs caliper/LiDAR)
       demo/render_turntable.py → polished hero clip
```

**Action (one command once a scan exists):**
```bash
adb pull /sdcard/Android/data/com.lantern.recorder/files/models/<latest>.obj  scan.obj
.venv/bin/... blender --background --python import_and_clean.py -- scan.obj scan_clean.glb
.venv/bin/python cad_check.py scan_clean.stl
.venv/bin/python ground_truth.py --mesh scan_clean.glb --known-dims <W>x<H>x<D> --mesh-unit mm
```
I'll wrap this into one `scripts/process_scan.sh` so it's copy-paste. **Robert: confirm the `.obj`
units are meters** (my cleanup assumes meters in → ×1000 mm; the `imported dims` log line will
tell us instantly — a 20 cm object must read ~0.20).

---

## Demo-readiness gaps in the live mesh (small, worth closing)

Ordered by demo impact. None are research; the hard research item (in-hand tracker) is explicitly
**out of scope** for the demo — the phone-orbits-object path already shipped is enough.

1. **In-app "Export / Share" is stubbed** (`ModelViewerScreen.kt:58` — "export/share actions
   (stubbed)"). Wire the button to `MeshExport.writeObj` + an Android share intent so the mesh
   leaves the phone live on stage (and so I can pull it without adb). *Owner: Robert.*
2. **NPU depth is one version-bump from live** (`QNN_SETUP.md`): QAIRT **2.44** installed vs DLC
   exported at **2.45** → `dlc handle 1002`, falls back to ARCore. Fix = install QAIRT 2.45 +
   re-copy the 5 `.so`s, **or** re-export the DLC at 2.44. Unlocks the "dense DA3 on the Hexagon
   NPU" claim for the pitch. *Owner: Aiyan; Robert verifies on device.*
3. **DA3 `.pte` output semantics unverified** (§15 known-gaps): confirm depth-vs-disparity / output
   rank against the actual export, adjust `Da3DepthModel.depthToDisparity` if the mesh looks
   inside-out or scaled wrong. *Owner: Aiyan + Robert.*
4. **Capture the demo footage** the moment a clean orbit scan works: screen-record the live mesh
   growing (deck slide 5) — this is the fallback-video insurance. *Owner: Robert; I assemble via
   `demo/build_reel.py`.*

## What I (D) will do off this handoff
- Write `scripts/process_scan.sh` (the pull→clean→cad→accuracy one-liner above).
- Run a pulled `.obj` through it end-to-end, tune `--voxel`, and report the real mm number.
- Render the hero turntable from the **real** live-mesh scan (replaces the synthetic placeholder).
- Update the deck on-device/demo slides to claim the **shipped live on-device mesh** (it's real now).

## Notes
- The earlier `docs/LIVE_MESH_FEED.md` / `LIVE_POINTCLOUD_PHASE_A.md` were written before I saw the
  shipped pipeline — they're **superseded** by `LIVE_MESH_PLAN.md` §15. Keep only the point-cloud
  *export* idea there as a model-free fallback if DA3/NPU ever regresses.
- Device is a **Snapdragon S25 (Hexagon NPU)** — the earlier "S25 FE/Exynos, QNN won't work" note
  was wrong; `QNN_SETUP.md` proves QNN builds/loads on the connected S25.
