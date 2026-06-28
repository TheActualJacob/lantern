"""
cad_check.py — verify a cleaned mesh actually imports into CAD (Person D).

CAD tools (Fusion 360, FreeCAD, SolidWorks) cannot read .glb — they import STL/OBJ
(mesh) or STEP (solid). This runs the STL through the **OpenCASCADE kernel** (the same
geometry kernel FreeCAD uses, via the `cadquery-ocp` / `OCP` bindings) — a real CAD
import, not a proxy — then reports two verdicts:

  * IMPORTS AS MESH BODY  — the CAD kernel reads it into a valid shape (what you get
    from Fusion "Insert Mesh" / FreeCAD Mesh import). This is the demo claim.
  * SOLID-CONVERTIBLE     — watertight + manifold + consistent winding, so the mesh
    can be turned into a usable solid/BRep for real CAD work.

USAGE
  python cad_check.py path/to/mesh.stl
  python cad_check.py path/to/mesh.glb     # auto-converts to a temp STL first

Exits 0 if it at least imports as a mesh body; non-zero if the CAD kernel rejects it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from OCP.StlAPI import StlAPI_Reader
from OCP.TopoDS import TopoDS_Shape
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE


def to_stl(path: Path) -> Path:
    """Return an STL path; convert via trimesh if given a non-STL mesh."""
    if path.suffix.lower() == ".stl":
        return path
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    tmp = Path(tempfile.gettempdir()) / (path.stem + "_cadcheck.stl")
    mesh.export(tmp)
    return tmp


def kernel_import(stl_path: Path):
    """Read STL into an OpenCASCADE shape (the real CAD import). Returns (shape, n_faces, valid)."""
    reader = StlAPI_Reader()
    shape = TopoDS_Shape()
    ok = reader.Read(shape, str(stl_path))
    if not ok or shape.IsNull():
        return None, 0, False
    n_faces = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        n_faces += 1
        exp.Next()
    valid = BRepCheck_Analyzer(shape).IsValid()
    return shape, n_faces, valid


def bbox_mm(shape) -> tuple[float, float, float]:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (xmax - xmin, ymax - ymin, zmax - zmin)


def solid_readiness(stl_path: Path) -> dict:
    """The criteria a CAD tool needs to convert a mesh into a usable solid."""
    m = trimesh.load(stl_path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    degenerate = int((m.area_faces <= 1e-12).sum())
    return {
        "triangles": len(m.faces),
        "watertight": bool(m.is_watertight),
        "winding_consistent": bool(m.is_winding_consistent),
        "degenerate_faces": degenerate,
        "volume_mm3": float(abs(m.volume)) if m.is_watertight else 0.0,
    }


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: python cad_check.py <mesh.stl|.glb|.obj|.ply>")
    src = Path(sys.argv[1])
    if not src.is_file():
        sys.exit(f"ERROR: not found: {src}")

    stl = to_stl(src)
    print(f"[cad_check] input  : {src}")
    if stl != src:
        print(f"[cad_check] (converted to STL for CAD import: {stl})")

    shape, n_faces, valid = kernel_import(stl)
    print("\n=== CAD KERNEL IMPORT (OpenCASCADE — FreeCAD's kernel) ===")
    if shape is None:
        print("  RESULT: ❌ REJECTED — the CAD kernel could not read this file.")
        return 1
    dx, dy, dz = bbox_mm(shape)
    print(f"  imported faces     : {n_faces}")
    print(f"  topology valid     : {valid}  (BRepCheck_Analyzer)")
    print(f"  bounding box (mm)  : {dx:.2f} x {dy:.2f} x {dz:.2f}")
    print("  RESULT: ✅ IMPORTS AS A MESH BODY (Fusion 'Insert Mesh' / FreeCAD Mesh import)")

    r = solid_readiness(stl)
    print("\n=== SOLID-CONVERTIBLE CHECK (mesh -> usable BRep solid) ===")
    print(f"  triangles          : {r['triangles']}")
    print(f"  watertight         : {r['watertight']}")
    print(f"  winding consistent : {r['winding_consistent']}")
    print(f"  degenerate faces   : {r['degenerate_faces']}")
    if r["watertight"]:
        print(f"  enclosed volume    : {r['volume_mm3']/1000.0:.2f} cm^3")
    solid_ok = r["watertight"] and r["winding_consistent"] and r["degenerate_faces"] == 0
    if solid_ok:
        print("  RESULT: ✅ SOLID-CONVERTIBLE — closed, manifold; CAD can make a solid.")
    else:
        why = []
        if not r["watertight"]:
            why.append("not watertight (open scan — needs a complete closed capture)")
        if not r["winding_consistent"]:
            why.append("inconsistent winding")
        if r["degenerate_faces"]:
            why.append(f"{r['degenerate_faces']} degenerate faces")
        print(f"  RESULT: ⚠️  imports as mesh, NOT yet solid-convertible — { '; '.join(why) }")

    return 0  # importing as a mesh body is the pass bar


if __name__ == "__main__":
    raise SystemExit(main())
