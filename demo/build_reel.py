"""
build_reel.py — assemble the Lantern demo reel (Person D).

Stitches the reel from whatever assets exist, filling missing pieces with labelled
placeholder cards so the reel ALWAYS builds — the moment Robert's live on-device scan
lands, drop it in and re-run for the final cut. This is the "fallback video" insurance.

Reel structure (in order):
  1. Title card
  2. Live on-device scan        (--live clip, else a "DROP IN" placeholder)
  3. Before / after cleanup     (--before + --after frame dirs, else placeholder)
  4. Hero turntable             (--turntable frame dir, else placeholder)
  5. Ground-truth accuracy      (--gt image, else placeholder; caption via --accuracy)
  6. Closing card

USAGE
  python demo/build_reel.py --out reel.mp4 \
      --turntable frames/hero --before frames/raw --after frames/clean \
      --live assets/live_scan.mp4 --gt assets/gt_overlay.png --accuracy "2.3 mm"

Everything is optional — run with no args to build the all-placeholder skeleton.
Needs: Pillow, imageio-ffmpeg (both in the venv).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1280, 720, 30
BG = (12, 14, 20)
FG = (235, 238, 245)
ACCENT = (90, 150, 240)


def font(size: int):
    for p in ("/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _centered(draw, y, text, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text(((W - (box[2] - box[0])) / 2, y), text, font=fnt, fill=fill)


def card_png(path: Path, title: str, subtitle: str = "", accent: str = "", placeholder=False):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if accent:
        _centered(d, H * 0.30, accent, font(34), ACCENT)
    _centered(d, H * 0.40, title, font(64), FG)
    if subtitle:
        _centered(d, H * 0.56, subtitle, font(32), (150, 156, 168))
    if placeholder:
        d.rectangle([40, 40, W - 40, H - 40], outline=(200, 120, 60), width=4)
        _centered(d, H * 0.80, "⟮ PLACEHOLDER — drop in real footage ⟯", font(26), (200, 120, 60))
    img.save(path)


def run(args):
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


SCALE = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}"


def seg_from_still(png: Path, seconds: float, out: Path):
    run(["-loop", "1", "-t", f"{seconds}", "-i", str(png),
         "-vf", SCALE, "-pix_fmt", "yuv420p", "-c:v", "libx264", str(out)])


def seg_from_frames(frames_dir: Path, out: Path, loops: int = 2):
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if not pngs:
        return False
    # concat the sequence onto itself a couple times so the spin reads on screen
    listing = out.with_suffix(".txt")
    with open(listing, "w") as f:
        for _ in range(loops):
            for p in pngs:
                f.write(f"file '{p.resolve()}'\nduration {1.0/FPS}\n")
        f.write(f"file '{pngs[-1].resolve()}'\n")
    run(["-f", "concat", "-safe", "0", "-i", str(listing),
         "-vf", SCALE, "-pix_fmt", "yuv420p", "-c:v", "libx264", str(out)])
    return True


def seg_from_clip(clip: Path, out: Path):
    run(["-i", str(clip), "-vf", SCALE, "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-an", str(out)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reel.mp4")
    ap.add_argument("--turntable", help="hero turntable frame dir")
    ap.add_argument("--before", help="raw-mesh frame dir")
    ap.add_argument("--after", help="cleaned-mesh frame dir")
    ap.add_argument("--live", help="live on-device scan clip (mp4)")
    ap.add_argument("--gt", help="ground-truth overlay image")
    ap.add_argument("--accuracy", default="__ mm", help="accuracy caption")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="reel_"))
    segs: list[Path] = []
    n = 0

    def add_still(title, sub="", accent="", placeholder=False, secs=2.5):
        nonlocal n
        png = work / f"card_{n}.png"; mp4 = work / f"seg_{n}.mp4"; n += 1
        card_png(png, title, sub, accent, placeholder)
        seg_from_still(png, secs, mp4); segs.append(mp4)

    def add_frames(d, fallback_title):
        nonlocal n
        mp4 = work / f"seg_{n}.mp4"; n += 1
        if d and seg_from_frames(Path(d), mp4):
            segs.append(mp4)
        else:
            add_still(fallback_title, "(no frames yet)", placeholder=True)

    # 1. title
    add_still("Lantern", "Phone Scan → CAD-ready 3D, 100% on-device", accent="QUALCOMM × META · ExecuTorch", secs=3)
    # 2. live scan
    if args.live and Path(args.live).is_file():
        mp4 = work / f"seg_{n}.mp4"; n += 1
        seg_from_clip(Path(args.live), mp4); segs.append(mp4)
    else:
        add_still("Live on-device scan", "Galaxy S25 · Snapdragon NPU", accent="STEP 1 — SCAN", placeholder=True, secs=3)
    # 3. before / after
    add_still("Raw fusion → clean mesh", accent="STEP 2 — CLEAN")
    add_frames(args.before, "Before: raw TSDF mesh")
    add_frames(args.after, "After: watertight, CAD-ready")
    # 4. hero turntable
    add_still("The result", accent="STEP 3 — EXPORT")
    add_frames(args.turntable, "Hero turntable")
    # 5. ground truth
    add_still("Measured against reality", accent="ACCURACY")
    if args.gt and Path(args.gt).is_file():
        mp4 = work / f"seg_{n}.mp4"; n += 1
        seg_from_still(Path(args.gt), 3.0, mp4); segs.append(mp4)
    else:
        add_still(f"Mean error {args.accuracy}", "vs LiDAR / caliper ground truth", placeholder=True, secs=3)
    # 6. closing
    add_still("Lantern", "measure reality. export it. on-device.", accent="CAD-READY 3D IN ~30s", secs=3)

    listing = work / "concat.txt"
    with open(listing, "w") as f:
        for s in segs:
            f.write(f"file '{s.resolve()}'\n")
    run(["-f", "concat", "-safe", "0", "-i", str(listing),
         "-pix_fmt", "yuv420p", "-c:v", "libx264", args.out])
    shutil.rmtree(work, ignore_errors=True)
    print(f"[reel] built {args.out}  ({len(segs)} segments)")


if __name__ == "__main__":
    main()
