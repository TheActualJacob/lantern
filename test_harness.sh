#!/usr/bin/env bash
# test_harness.sh — smoke-test import_and_clean.py end-to-end, offline.
#
# 1. Locates the Blender binary (PATH or the macOS .app).
# 2. Generates a fixture mesh (a UV sphere) and exports it as test_in.glb,
#    so the test has zero network dependency.
# 3. Runs import_and_clean.py on it.
# 4. Verifies the output .glb exists, is non-trivial, and re-imports with faces.
#
# Usage:  ./test_harness.sh
set -euo pipefail
cd "$(dirname "$0")"

# --- locate Blender -------------------------------------------------------
if command -v blender >/dev/null 2>&1; then
  BLENDER="$(command -v blender)"
elif [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
  BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
else
  echo "FAIL: Blender not found (not on PATH, not in /Applications)." >&2
  exit 1
fi
echo "[test] blender: $BLENDER"
"$BLENDER" --version | head -1

WORK="$(mktemp -d)"
IN="$WORK/test_in.glb"
OUT="$WORK/test_out.glb"
trap 'rm -rf "$WORK"' EXIT

# --- 1. generate a fixture mesh (UV sphere, ~0.1 m radius) -----------------
echo "[test] generating fixture -> $IN"
"$BLENDER" --background --python-expr "
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1)   # 0.1 m, mimics a small scanned object
bpy.ops.export_scene.gltf(filepath=r'$IN', export_format='GLB')
" >/dev/null

[ -f "$IN" ] || { echo "FAIL: fixture not created." >&2; exit 1; }

# --- 2. run the script under test -----------------------------------------
echo "[test] running import_and_clean.py ..."
"$BLENDER" --background --python import_and_clean.py -- "$IN" "$OUT" --scale 1000 --voxel 0.01

# --- 3. verify output ------------------------------------------------------
[ -f "$OUT" ] || { echo "FAIL: output .glb was not written." >&2; exit 1; }
BYTES=$(wc -c < "$OUT" | tr -d ' ')
echo "[test] output size: ${BYTES} bytes"
[ "$BYTES" -gt 1024 ] || { echo "FAIL: output suspiciously small (${BYTES}B)." >&2; exit 1; }

# Re-import and assert it has geometry; print dims (should be ~200 mm after x1000).
echo "[test] re-importing output to validate geometry + scale ..."
"$BLENDER" --background --python-expr "
import bpy, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r'$OUT')
m=[o for o in bpy.context.scene.objects if o.type=='MESH']
assert m, 'no mesh in output'
o=m[0]; d=o.dimensions
print(f'[verify] faces={len(o.data.polygons)} dims_mm=({d.x:.1f},{d.y:.1f},{d.z:.1f})')
assert len(o.data.polygons) > 0, 'output has no faces'
assert 150 < d.x < 250, f'unexpected scale: {d.x:.1f} mm (expected ~200)'
print('[verify] OK')
"

echo "[test] PASS ✅  import_and_clean.py is working end-to-end."
