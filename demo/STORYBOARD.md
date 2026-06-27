# Lantern — Demo Reel Storyboard

Target: **~45–60s** reel for Sunday demos. The reel is the **fallback insurance** — if the
live demo dies on stage, this plays. Build it early with placeholders; swap real footage in
as it lands.

## Shot list

| # | Shot | Source | Caption / VO | Proves |
|---|---|---|---|---|
| 1 | Title card | generated | "Lantern — Phone Scan → CAD-ready 3D, 100% on-device" | hook |
| 2 | **Live on-device scan** | 📱 screen-record on the S25 (Robert) | "Step 1 — Scan. Galaxy S25, Snapdragon NPU." | it's real + on-device |
| 3 | Raw → clean | `before/` + `after/` turntables | "Step 2 — Clean. Raw fusion → watertight mesh." | the cleanup value |
| 4 | Hero turntable | `after/` (hi-res) | "Step 3 — Export. CAD-ready in ~30s." | the money shot |
| 5 | CAD import | screen-record opening the STL in Fusion/FreeCAD | "Imports straight into CAD." | usable output |
| 6 | Accuracy | `gt_overlay.png` (color error map + number) | "Mean error __ mm vs ground truth." | credibility |
| 7 | Closing card | generated | "Measure reality. Export it. On-device." | CTA |

## The two clips humans must capture (everything else auto-generates)

- **Shot 2 — live scan:** screen-record a clean scan→export on the S25. *Record the first good one immediately* — this is the non-negotiable fallback.
- **Shot 5 — CAD import:** screen-record dragging the exported `.stl` into Fusion 360 / FreeCAD. (We've already verified it imports via the OpenCASCADE kernel; this is just the visual.)

## Build commands

```bash
# 1. render turntables from a cleaned mesh (Blender)
blender --background --python demo/render_turntable.py -- after.glb  frames/after  --frames 60 --res 1920x1080
blender --background --python demo/render_turntable.py -- raw.glb    frames/before --frames 60 --res 1920x1080 --clay

# 2. assemble (any missing piece becomes a labelled placeholder, so it always builds)
python demo/build_reel.py --out lantern_reel.mp4 \
    --before frames/before --after frames/after --turntable frames/after \
    --live assets/live_scan.mp4 --gt assets/gt_overlay.png --accuracy "2.3 mm"
```

The ground-truth overlay image comes from `ground_truth.py` (its `.error.ply` opened in
CloudCompare, or the `.hist.png`). Caliper/LiDAR number from Robert → `--accuracy`.

## Status
- ✅ turntable renderer + reel assembler working; full reel builds from the real S25 scan with placeholders.
- ⬜ Shot 2 (live scan) + Shot 5 (CAD import) — need screen-records once Robert has a clean on-device scan.
- ⬜ real accuracy number (needs caliper/LiDAR).
