# demo/ — Lantern demo reel toolkit (Person D)

Auto-assembles the demo/fallback reel. Designed so the reel **always builds** (missing
footage → labelled placeholders) and the **first good on-device scan assembles itself**.

## Files
- `render_turntable.py` — Blender headless: mesh → studio turntable PNG sequence.
- `build_reel.py` — stitches title cards + turntables + live-scan + accuracy into `reel.mp4`.
- `STORYBOARD.md` — the shot list and what humans must capture.

## Setup (one-time)
```bash
uv pip install --python .venv Pillow imageio-ffmpeg   # ffmpeg ships with imageio-ffmpeg
```
`render_turntable.py` runs in Blender's own Python (no venv needed); `build_reel.py` runs
in the project `.venv`. These are **demo-only deps** — intentionally not in the main
`requirements.txt`.

## Quick build (with placeholders — verify it works)
```bash
blender --background --python demo/render_turntable.py -- cleaned.glb frames/after --frames 60 --res 1920x1080
python demo/build_reel.py --out lantern_reel.mp4 --turntable frames/after --after frames/after
```

## Final cut (once real footage exists)
Drop `assets/live_scan.mp4` (Robert's screen-record) and `assets/gt_overlay.png`
(from `ground_truth.py`), then:
```bash
python demo/build_reel.py --out lantern_reel.mp4 \
    --before frames/before --after frames/after --turntable frames/after \
    --live assets/live_scan.mp4 --gt assets/gt_overlay.png --accuracy "2.3 mm"
```

See `STORYBOARD.md` for the full shot list. Reel output is gitignored (`output/`-style);
keep source meshes + scripts in git, not the rendered video.
