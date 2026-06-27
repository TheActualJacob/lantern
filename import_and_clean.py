"""
import_and_clean.py — Lantern mesh cleanup contract (Person D / Integration).

Takes a raw TSDF-extracted mesh and produces a watertight, correctly-scaled,
CAD-ready .glb. Device-agnostic: this is host-side Blender, nothing here depends
on the phone's SoC.

USAGE (Blender headless — args go AFTER the `--`):

    blender --background --python import_and_clean.py -- input.glb output.glb
    blender --background --python import_and_clean.py -- in.glb out.glb --scale 1000 --voxel 0.5

Args after `--`:
    input            path to input mesh (.glb / .gltf / .obj / .ply / .stl)
    output           path to output .glb
    --scale    FLOAT   uniform scale applied to the mesh (default 1000.0 = m -> mm for CAD)
    --voxel    FLOAT   voxel remesh size in *input* units, applied BEFORE scale (default 0.005 m = 5 mm)
    --no-remesh        skip the voxel remesh (keep raw topology; normals are still fixed)
    --rotate-x FLOAT   OPTIONAL extra rotation about X, degrees (default 0)
    --rotate-z FLOAT   OPTIONAL extra rotation about Z, degrees (default 0)

Contract (see roadmap.md, Decision 4):
    - input units: meters (1.0 = 1 m, ARCore world space)
    - input orientation: +Y up, -Z forward (glTF). Blender's glTF importer and exporter
      apply INVERSE +Y-up<->+Z-up conversions, so import->clean->export is orientation-
      PRESERVING with no manual rotation (verified by orientation_test.sh). Only reach for
      --rotate-x / --rotate-z if A's exporter writes a non-standard frame and the mesh
      comes in tipped.
    - TSDF output is NOT guaranteed watertight -> voxel remesh + consistent normals
      are mandatory before any CAD export (CAD tools reject flipped/inconsistent normals).

Exit codes: 0 = clean export, non-zero = failure (so the pipeline can gate on it).
"""

import math
import os
import sys

import bpy  # provided by the Blender Python runtime


# ---------------------------------------------------------------------------
# Arg parsing (Blender hands us everything after the literal `--`)
# ---------------------------------------------------------------------------
def parse_args(argv):
    if "--" not in argv:
        sys.exit("ERROR: pass script args after `--`, e.g.  ... -- input.glb output.glb")
    argv = argv[argv.index("--") + 1:]

    positionals, scale, voxel = [], 1000.0, 0.005
    remesh, rotate_x, rotate_z = True, 0.0, 0.0

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--scale":
            scale = float(argv[i + 1]); i += 2
        elif a == "--voxel":
            voxel = float(argv[i + 1]); i += 2
        elif a == "--no-remesh":
            remesh = False; i += 1
        elif a == "--rotate-x":
            rotate_x = float(argv[i + 1]); i += 2
        elif a == "--rotate-z":
            rotate_z = float(argv[i + 1]); i += 2
        elif a.startswith("--"):
            sys.exit(f"ERROR: unknown flag {a}")
        else:
            positionals.append(a); i += 1

    if len(positionals) != 2:
        sys.exit("ERROR: expected exactly <input> <output>, got: " + " ".join(positionals))

    inp, out = positionals
    if not os.path.isfile(inp):
        sys.exit(f"ERROR: input not found: {inp}")
    return inp, out, scale, voxel, remesh, rotate_x, rotate_z


# ---------------------------------------------------------------------------
# Import — dispatch on file extension. Blender 4.x op names where they changed.
# ---------------------------------------------------------------------------
def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        # glTF importer applies the +Y-up -> +Z-up rotation on the object transform.
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        # 4.0+ removed import_scene.obj in favour of wm.obj_import.
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            bpy.ops.import_mesh.stl(filepath=path)
    else:
        sys.exit(f"ERROR: unsupported input extension: {ext}")


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def join_meshes(objs):
    """Collapse all imported mesh parts into one object so remesh/export are single-object."""
    if not objs:
        sys.exit("ERROR: no mesh objects after import (empty or non-mesh file).")
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def dims_str(obj):
    d = obj.dimensions
    return f"({d.x:.4f}, {d.y:.4f}, {d.z:.4f})"


# ---------------------------------------------------------------------------
def main():
    inp, out, scale, voxel, remesh, rotate_x, rotate_z = parse_args(sys.argv)
    print(f"[lantern] in={inp} out={out} scale={scale} voxel={voxel} "
          f"remesh={remesh} rotate_x={rotate_x} rotate_z={rotate_z}")

    # Clean slate so leftover default cube/camera/light never reach the export.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    import_mesh(inp)
    obj = join_meshes(mesh_objects())
    print(f"[lantern] imported dims (pre-transform, m): {dims_str(obj)}")

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)

    # Orientation: the glTF importer already converts +Y-up -> Blender +Z-up, and the
    # exporter converts back, so the round-trip is orientation-PRESERVING on its own
    # (verified by orientation_test.sh — a manual +90 here was a no-op). We only bake the
    # importer's rotation into the geometry so downstream transforms are clean, plus any
    # explicit, opt-in --rotate-x / --rotate-z correction for a non-standard input frame.
    if rotate_x:
        obj.rotation_euler[0] += math.radians(rotate_x)
    if rotate_z:
        obj.rotation_euler[2] += math.radians(rotate_z)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Watertightness — TSDF extraction does NOT guarantee it. Voxel remesh closes
    # small holes and yields manifold topology; run BEFORE scaling so voxel size
    # stays in the natural input (meter) units.
    if remesh:
        mod = obj.modifiers.new(name="VoxelRemesh", type="REMESH")
        mod.mode = "VOXEL"
        mod.voxel_size = voxel
        bpy.ops.object.modifier_apply(modifier="VoxelRemesh")
        print(f"[lantern] voxel remeshed @ {voxel} m -> {len(obj.data.polygons)} faces")

    # Consistent outward normals — CAD importers reject flipped/inconsistent normals.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Scale last (e.g. x1000 m -> mm for CAD) and bake it in.
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print(f"[lantern] final dims (post-scale): {dims_str(obj)}")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)
    print(f"[lantern] OK -> {out}")

    # Also emit an STL next to the GLB — .glb is a graphics format that CAD tools
    # (Fusion 360, FreeCAD, SolidWorks) cannot import; STL is the universal mesh
    # handoff for CAD. (Use Insert/Import Mesh, then mesh->solid if watertight.)
    stl_out = os.path.splitext(out)[0] + ".stl"
    try:
        if hasattr(bpy.ops.wm, "stl_export"):       # Blender 4.x+
            bpy.ops.wm.stl_export(filepath=stl_out, export_selected_objects=True)
        else:                                        # older Blender
            bpy.ops.export_mesh.stl(filepath=stl_out, use_selection=True)
        print(f"[lantern] CAD STL -> {stl_out}")
    except Exception as exc:
        print(f"[lantern] WARN: STL export failed ({exc}); GLB still written", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # surface a clean non-zero exit for pipeline gating
        print(f"[lantern] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
