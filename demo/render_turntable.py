"""
render_turntable.py — studio turntable render of a mesh (Blender headless, Person D).

Produces the visual centerpiece of the demo reel: the scanned object rotating 360 on
a clean studio background. Renders a PNG frame sequence (so the reel assembler can
re-time / caption it); pass --mp4 to also encode a movie with Blender's bundled ffmpeg.

USAGE (args after `--`):
  blender --background --python demo/render_turntable.py -- mesh.glb out_dir/
  blender --background --python demo/render_turntable.py -- mesh.glb frames/ --frames 90 --res 1920x1080 --mp4

  mesh          .glb/.gltf/.obj/.ply/.stl
  out_dir       directory for the PNG sequence (created if missing)
  --frames N    turntable frames (default 60 = 2s @30fps)
  --res WxH     resolution (default 1280x720)
  --clay        force neutral clay material (ignore vertex colors)

Outputs a PNG sequence (frame_0000.png ...). Encode to mp4 with demo/build_reel.py
(uses a bundled ffmpeg) — kept separate so the assembler can re-time/caption.
"""

import glob
import math
import os
import sys

import bpy
from mathutils import Vector


def args_after_dashes():
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def parse(argv):
    if len(argv) < 2:
        sys.exit("usage: ... -- <mesh> <out_dir> [--frames N] [--res WxH] [--mp4] [--clay]")
    mesh, out_dir = argv[0], argv[1]
    frames, res, clay = 60, (1280, 720), False
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--frames":
            frames = int(argv[i + 1]); i += 2
        elif a == "--res":
            w, h = argv[i + 1].lower().split("x"); res = (int(w), int(h)); i += 2
        elif a == "--clay":
            clay = True; i += 1
        else:
            sys.exit(f"unknown arg {a}")
    if not os.path.isfile(mesh):
        sys.exit(f"mesh not found: {mesh}")
    return mesh, out_dir, frames, res, clay


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        (bpy.ops.wm.obj_import if hasattr(bpy.ops.wm, "obj_import") else bpy.ops.import_scene.obj)(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".stl":
        (bpy.ops.wm.stl_import if hasattr(bpy.ops.wm, "stl_import") else bpy.ops.import_mesh.stl)(filepath=path)
    else:
        sys.exit(f"unsupported mesh ext {ext}")
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        sys.exit("no mesh in file")
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def center_and_normalize(obj):
    """Center geometry at origin and scale so the largest dim ~= 2 units (stable framing)."""
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    dims = max(obj.dimensions)
    if dims > 0:
        s = 2.0 / dims
        obj.scale = (s, s, s)
        bpy.ops.object.transform_apply(scale=True)
    # drop it so it sits on z=0 (ground)
    zmin = min((obj.matrix_world @ Vector(c)).z for c in obj.bound_box)
    obj.location.z -= zmin


def make_material(obj, clay):
    has_colors = bool(obj.data.color_attributes) and not clay
    mat = bpy.data.materials.new("lantern_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.55
    if has_colors:
        vc = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = obj.data.color_attributes[0].name
        mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.6, 0.62, 0.66, 1.0)  # clay grey
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def build_studio(res, frames):
    scene = bpy.context.scene
    # neutral studio world
    scene.world = scene.world or bpy.data.worlds.new("W")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.05, 0.055, 0.07, 1.0)
    bg.inputs[1].default_value = 1.0

    # ground plane
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    gmat = bpy.data.materials.new("ground"); gmat.use_nodes = True
    gmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.02, 0.02, 0.025, 1)
    bpy.context.active_object.data.materials.append(gmat)

    # key + fill + rim
    for loc, energy in [((4, -4, 6), 800), ((-5, -2, 3), 250), ((0, 5, 4), 400)]:
        bpy.ops.object.light_add(type="AREA", location=loc)
        light = bpy.context.active_object
        light.data.energy = energy
        light.data.size = 5

    # camera with track-to constraint on origin
    target = bpy.data.objects.new("target", None)
    bpy.context.collection.objects.link(target)
    target.location = (0, 0, 1.0)
    bpy.ops.object.camera_add(location=(0, -6, 3.5))
    cam = bpy.context.active_object
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    scene.camera = cam

    # render settings
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, frames
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng; break
        except TypeError:
            continue


def main():
    mesh, out_dir, frames, res, clay = parse(args_after_dashes())
    bpy.ops.wm.read_factory_settings(use_empty=True)

    obj = import_mesh(mesh)
    center_and_normalize(obj)
    make_material(obj, clay)
    build_studio(res, frames)

    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    obj.rotation_mode = "XYZ"
    print(f"[reel] rendering {frames} frames @ {res[0]}x{res[1]} -> {out_dir}")

    # Manual per-frame turntable — version-proof (no Action/fcurve API, which churns
    # across Blender releases). Set Z rotation, render one still, repeat.
    for i in range(frames):
        obj.rotation_euler = (0, 0, 2.0 * math.pi * i / frames)
        scene.render.filepath = os.path.join(out_dir, f"frame_{i:04d}")
        bpy.ops.render.render(write_still=True)

    print(f"[reel] turntable done — {frames} PNG frames in {out_dir}")


if __name__ == "__main__":
    main()
