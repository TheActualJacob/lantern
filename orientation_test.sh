#!/usr/bin/env bash
# orientation_test.sh — verify import_and_clean.py is ORIENTATION-PRESERVING.
#
# A sphere can't catch an axis bug (it's symmetric). This builds a box with three
# DISTINCT dims (0.1 x 0.2 x 0.3 m in Blender), exports it as a proper glTF (+Y-up)
# file, runs the cleanup script (no remesh, so dims stay exact), then re-imports the
# output and asserts the dims are unchanged (x100,200,300 mm after the m->mm scale).
# Any 90-degree rotation permutes the axes and the assertion catches it.
set -euo pipefail
cd "$(dirname "$0")"

if command -v blender >/dev/null 2>&1; then BLENDER="$(command -v blender)"
elif [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
else echo "FAIL: Blender not found." >&2; exit 1; fi
echo "[orient] blender: $BLENDER"

WORK="$(mktemp -d)"; IN="$WORK/box_in.glb"; OUT="$WORK/box_out.glb"
trap 'rm -rf "$WORK"' EXIT

# Fixture: distinct-dims box, exported as standard glTF (+Y-up).
"$BLENDER" --background --python-expr "
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=2)            # dims 2,2,2
o=bpy.context.active_object
o.scale=(0.05,0.10,0.15)                            # -> dims 0.1, 0.2, 0.3 m (X,Y,Z)
bpy.ops.object.transform_apply(scale=True)
bpy.ops.export_scene.gltf(filepath=r'$IN', export_format='GLB')
" >/dev/null
echo "[orient] fixture: box X=0.1 Y=0.2 Z=0.3 m (Blender frame)"

echo "[orient] running import_and_clean.py (--no-remesh to keep dims exact) ..."
"$BLENDER" --background --python import_and_clean.py -- "$IN" "$OUT" --scale 1000 --no-remesh >/dev/null

# Re-import output, read Blender-frame dims, assert orientation preserved.
"$BLENDER" --background --python-expr "
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r'$OUT')
o=[x for x in bpy.context.scene.objects if x.type=='MESH'][0]
d=o.dimensions
print(f'[orient] output dims (mm): ({d.x:.1f}, {d.y:.1f}, {d.z:.1f})')
exp=(100.0,200.0,300.0); tol=15.0
ok=all(abs(a-b)<tol for a,b in zip((d.x,d.y,d.z),exp))
print('[orient] expected (100.0, 200.0, 300.0) mm  ->  ' + ('PASS' if ok else 'FAIL — axes permuted, rotation bug'))
import sys; sys.exit(0 if ok else 2)
"
echo "[orient] PASS ✅  orientation preserved."
