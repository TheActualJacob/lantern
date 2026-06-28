#!/usr/bin/env bash
# process_scan.sh — one command: live-mesh scan -> clean CAD mesh + STL + accuracy + hero clip.
#
# Takes a mesh exported by the on-device live mesh (MeshExport.writeObj -> .obj), or any
# .obj/.glb/.ply/.stl, and runs the full Person-D proof chain:
#   import_and_clean.py  -> watertight, scaled .glb + .stl   (CAD handoff)
#   cad_check.py         -> OpenCASCADE "imports as solid?" verdict
#   ground_truth.py      -> mean error mm        (only if --known-dims / --reference given)
#   render_turntable.py  -> hero turntable PNGs + mp4   (unless --no-hero)
#
# USAGE
#   scripts/process_scan.sh <mesh.obj> [options]
#   scripts/process_scan.sh --adb-pull            # pull newest live-mesh .obj off the phone first
#
# OPTIONS
#   --out DIR            output dir (default: output/scan_<name>)
#   --voxel F            cleanup voxel size in meters (default 0.003)
#   --known-dims WxHxD   caliper dims (mm) -> runs ground_truth known-dims mode
#   --reference FILE     reference scan (e.g. iPad LiDAR) -> ground_truth reference mode
#   --adb-pull          pull the newest .obj from the app's models/ dir via adb, then process
#   --no-hero           skip the turntable render
#   --frames N          hero frames (default 60)
#   --res WxH           hero resolution (default 1920x1080)
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
ROOT="$(pwd)"

# --- locate tools ---------------------------------------------------------
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "ERROR: no python (.venv/bin/python or python3)"; exit 1; }
if command -v blender >/dev/null 2>&1; then BLENDER="$(command -v blender)"
elif [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
else echo "ERROR: Blender not found"; exit 1; fi

# --- defaults / args ------------------------------------------------------
MESH=""; OUT=""; VOXEL="0.003"; KNOWN=""; REF=""; ADB_PULL=0; HERO=1; FRAMES=60; RES="1920x1080"
APP_MODELS="/sdcard/Android/data/com.lantern.recorder/files/models"

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --voxel) VOXEL="$2"; shift 2;;
    --known-dims) KNOWN="$2"; shift 2;;
    --reference) REF="$2"; shift 2;;
    --adb-pull) ADB_PULL=1; shift;;
    --no-hero) HERO=0; shift;;
    --frames) FRAMES="$2"; shift 2;;
    --res) RES="$2"; shift 2;;
    --*) echo "unknown option $1"; exit 1;;
    *) MESH="$1"; shift;;
  esac
done

# --- optional: pull newest live-mesh .obj off the phone -------------------
if [ "$ADB_PULL" = "1" ]; then
  command -v adb >/dev/null 2>&1 || { echo "ERROR: adb not on PATH"; exit 1; }
  NEWEST="$(adb shell "ls -t $APP_MODELS/*.obj 2>/dev/null | head -1" | tr -d '\r')"
  [ -n "$NEWEST" ] || { echo "ERROR: no .obj found in $APP_MODELS on device"; exit 1; }
  mkdir -p output/pulled
  MESH="output/pulled/$(basename "$NEWEST")"
  echo "[scan] pulling $NEWEST -> $MESH"
  adb pull "$NEWEST" "$MESH" >/dev/null
fi

[ -n "$MESH" ] && [ -f "$MESH" ] || { echo "ERROR: mesh not found (give a path or --adb-pull)"; exit 1; }
NAME="$(basename "${MESH%.*}")"
[ -n "$OUT" ] || OUT="output/scan_${NAME}"
mkdir -p "$OUT"
CLEAN="$OUT/${NAME}_clean.glb"
STL="$OUT/${NAME}_clean.stl"

echo "=============================================================="
echo "[scan] mesh : $MESH"
echo "[scan] out  : $OUT   (voxel=$VOXEL)"
echo "=============================================================="

# --- 1. clean -> CAD-ready .glb + .stl ------------------------------------
echo "[scan] 1/4 cleanup ..."
"$BLENDER" --background --python import_and_clean.py -- "$MESH" "$CLEAN" --voxel "$VOXEL" 2>&1 | grep -E "^\[lantern\]"
[ -f "$STL" ] || { echo "ERROR: cleanup did not produce $STL"; exit 1; }

# --- 2. CAD-import verdict ------------------------------------------------
echo "[scan] 2/4 CAD kernel check ..."
if "$PY" -c "import OCP" 2>/dev/null; then
  "$PY" cad_check.py "$STL" 2>&1 | grep -E "RESULT|bounding box|watertight|volume" || true
else
  echo "  (skip: cadquery-ocp not installed — 'uv pip install cadquery-ocp' to enable)"
fi

# --- 3. accuracy (only if a reference was given) --------------------------
echo "[scan] 3/4 ground truth ..."
if [ -n "$KNOWN" ]; then
  "$PY" ground_truth.py --mesh "$CLEAN" --known-dims "$KNOWN" --mesh-unit mm --out "$OUT/ground_truth_report.txt" 2>&1 | grep -E "mean abs|max  abs|smallest|middle|largest" || true
elif [ -n "$REF" ]; then
  "$PY" ground_truth.py --mesh "$CLEAN" --reference "$REF" --mesh-unit mm --ref-unit m --align --out "$OUT/ground_truth_report.txt" 2>&1 | grep -E "mean |median|p95|within|Hausdorff" || true
else
  echo "  (skip: pass --known-dims WxHxD or --reference FILE for a real accuracy number)"
fi

# --- 4. hero turntable ----------------------------------------------------
if [ "$HERO" = "1" ]; then
  echo "[scan] 4/4 hero turntable ($RES, $FRAMES frames) ..."
  "$BLENDER" --background --python demo/render_turntable.py -- "$CLEAN" "$OUT/hero_frames" --frames "$FRAMES" --res "$RES" 2>&1 | grep -E "turntable done" || true
  FF="$("$PY" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || true)"
  if [ -n "$FF" ] && ls "$OUT/hero_frames"/frame_*.png >/dev/null 2>&1; then
    "$FF" -y -hide_banner -loglevel error -framerate 30 -i "$OUT/hero_frames/frame_%04d.png" \
      -vf "fps=30,format=yuv420p" -c:v libx264 -crf 18 "$OUT/hero.mp4"
    echo "[scan] hero -> $OUT/hero.mp4"
  fi
else
  echo "[scan] 4/4 hero skipped (--no-hero)"
fi

echo "=============================================================="
echo "[scan] DONE -> $OUT"
echo "  clean mesh : $CLEAN"
echo "  CAD STL    : $STL"
[ -f "$OUT/ground_truth_report.txt" ] && echo "  accuracy   : $OUT/ground_truth_report.txt"
[ -f "$OUT/hero.mp4" ] && echo "  hero clip  : $OUT/hero.mp4"
echo "=============================================================="
