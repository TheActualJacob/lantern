#!/usr/bin/env bash
# make_demo_video.sh — build the narratable slide demo video from real pipeline artifacts.
#
# Story (no baked captions — you narrate live):
#   1. SCAN GROWING   colored point cloud accumulates as the "camera" orbits (scan_animation.py)
#   2. FUSED MESH     the reconstructed mesh turntable (render_turntable.py)
#   3. CLEAN BOX      the cuboid-fit result turntable (the crisp demo output)
# Segments are held long + slow with cross-fades so they're easy to talk over.
#
# USAGE
#   demo/make_demo_video.sh <da3_dump_dir> <mesh.glb> <cuboid.glb> [out.mp4]
# e.g.
#   demo/make_demo_video.sh da3_outputs/session_XXXX out/box.glb out/box_cuboid.glb demo/demo_video.mp4
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DUMP="${1:?need da3 dump dir}"; MESH="${2:?need mesh .glb}"; CUBOID="${3:?need cuboid .glb}"
OUT="${4:-demo/demo_video.mp4}"
RES="1280x720"

PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
if command -v blender >/dev/null 2>&1; then BLENDER="$(command -v blender)"
else BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"; fi
FF="$("$PY" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")"

W="$CLAUDE_JOB_DIR/tmp/demovid"; mkdir -p "$W" 2>/dev/null || W="$(mktemp -d)"
echo "[demo] work=$W"

frames_to_mp4 () {  # <frames_dir> <fps> <out.mp4>
  "$FF" -y -hide_banner -loglevel error -framerate "$2" -i "$1/frame_%04d.png" \
    -vf "fps=30,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p" \
    -c:v libx264 -crf 18 "$3"
}

# --- 1. scan-growing animation (slow: 90 frames @ 20fps ~= 4.5s) ----------
echo "[demo] 1/3 scan-growing animation ..."
"$PY" demo/scan_animation.py "$DUMP" --out "$W/scan" --frames 90 --res "$RES" 2>&1 | grep -E "scan-anim" | tail -2
frames_to_mp4 "$W/scan" 20 "$W/seg1.mp4"

# --- 2. fused mesh turntable ----------------------------------------------
echo "[demo] 2/3 fused-mesh turntable ..."
"$BLENDER" --background --python demo/render_turntable.py -- "$MESH" "$W/mesh" --frames 48 --res "$RES" --clay 2>&1 | grep -E "turntable done" || true
frames_to_mp4 "$W/mesh" 24 "$W/seg2.mp4"

# --- 3. clean cuboid turntable --------------------------------------------
echo "[demo] 3/3 clean-box turntable ..."
"$BLENDER" --background --python demo/render_turntable.py -- "$CUBOID" "$W/cuboid" --frames 48 --res "$RES" --clay 2>&1 | grep -E "turntable done" || true
frames_to_mp4 "$W/cuboid" 24 "$W/seg3.mp4"

# --- stitch with 0.5s cross-fades -----------------------------------------
echo "[demo] stitching with cross-fades ..."
# `ffmpeg -i` (no output) exits non-zero; `|| true` keeps the pipeline from
# aborting the script under `set -e`+pipefail. Strip awk's trailing comma too.
dur () { { "$FF" -i "$1" 2>&1 || true; } | awk '/Duration/{gsub(",","",$2);split($2,a,":");print a[1]*3600+a[2]*60+a[3]}'; }
d1="$(dur "$W/seg1.mp4")"; d2="$(dur "$W/seg2.mp4")"
o1="$(echo "$d1 - 0.5" | bc)"; o2="$(echo "$d1 + $d2 - 1.0" | bc)"
mkdir -p "$(dirname "$OUT")"
"$FF" -y -hide_banner -loglevel error -i "$W/seg1.mp4" -i "$W/seg2.mp4" -i "$W/seg3.mp4" \
  -filter_complex "[0][1]xfade=transition=fade:duration=0.5:offset=${o1}[a];[a][2]xfade=transition=fade:duration=0.5:offset=${o2}[v]" \
  -map "[v]" -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT"

echo "=============================================================="
echo "[demo] DONE -> $OUT"
"$FF" -hide_banner -i "$OUT" 2>&1 | grep -E "Duration|Stream.*Video" | sed 's/^ *//'
echo "  (no captions — narrate the 3 beats: scan -> fuse -> clean box)"
echo "=============================================================="
