# Reconstruction Quality — Diagnosis (2026-06-28)

Investigated the "final mesh is super inaccurate / doesn't look like the object" issue using
the committed `da3_outputs/session_20260629_000233/` dump (12-view DA3-Small, "the box").
Back-projected the raw DA3 depth into a world cloud and inspected it. Findings, ranked.

## Finding 1 — DA3 outputs DEPTH (distance), not disparity. **Likely the on-device bug.**
- Confirmed empirically: object depths are min **0.86**, max **1.65** — *near = small* ⇒ distance,
  not inverse depth. `meta.json` agrees (`is_metric:false`, README back-projects `z = depth`).
- The host pipeline (`depth_anything_v3.py`) **explicitly converts `disp = 1/depth`** before the
  affine solver. The exported `.pte` (`export_da3_executorch.py`) **passes the raw depth through**.
- **⇒ The on-device `Da3DepthModel` MUST apply the same `1/depth` inversion before the affine
  solve.** If it treats the model output as disparity directly, the `s·disp+t` fit is garbage and
  the mesh won't resemble the object. **This is the §15 known-gap. Check this first — one-line fix.**
- Demonstrated: back-projecting the same data with depth mis-read as `1/depth` distorts the box
  aspect (long axis 1.49 → 1.87) and warps it.

## Finding 2 — The DA3 multi-view data is USABLE; reconstructs a coherent box-sized cloud.
- All 12 views fused with DA3's **own** poses → a tight, compact cloud: extents 0.49×0.62×0.72
  (DA3 units), aspect **1 : 1.28 : 1.49**, no smear. So **the data + poses are fine; the failure is
  downstream in how the on-device pipeline consumes them.**
- **Visual confirmation** (`recon_diagnosis_imgs/da3_fused_box_3views.png`): the fused cloud is a
  recognizable box-shaped slab with correct proportions and tan/cardboard color — NOT garbage.

## Finding 3 — Even best-case is SOFT and WARPED (monocular-depth limit). Set expectations.
- The fused cloud is a rounded slab with a visible **saddle/banana warp** on the flat top and
  rounded edges — not a crisp box (see `recon_diagnosis_imgs/da3_fused_box_3views.png`, az=30
  view). This is inherent to monocular depth (DA bends flats / rounds edges), present even in
  DA3's multi-view-consistent output with no affine fit.
- ⇒ Two implications:
  1. **ARCore raw depth is a true metric sensor** — for rigid objects it may give *cleaner*
     geometry than DA3. Consider weighting ARCore depth as the skeleton and using DA3 only to fill
     low-confidence regions (roadmap Decision 3, "depth completion" escalation), rather than
     trusting DA3 depth wholesale.
  2. Don't expect CAD-crisp edges from monocular depth alone; the watertight remesh + a
     measured-accuracy claim is the honest framing.

## Finding 4 — On-device per-frame + ARCore-pose is the HARDEST path; multi-view DA3 is better.
- This dump used **multi-view** DA3 (all frames at once → one shared consistent frame). The
  on-device live path uses **per-frame mono** DA3 + ARCore pose + affine scale solve + TSDF, which
  is NOT multi-view consistent (roadmap R5) and compounds frame-to-frame scale drift.
- ⇒ **Strongest single fix for the FINAL artifact:** run the **offline multi-view DA3
  reconstruction** (this exact path) for the exported mesh, even if the live preview stays the
  rough per-frame version. It already produces a coherent result — wire its cloud → Poisson/TSDF →
  `import_and_clean.py`.

## Recommended fix order
1. **Confirm/repair the depth→disparity inversion in `Da3DepthModel`** (Finding 1). Fastest, highest-probability.
2. **Log the affine fit residual per frame** (`AffineScaleSolver.fitAffine`) — if it's large, the
   scale solve is failing (often a symptom of #1). A healthy fit has small residual + positive `s`.
3. **For the final/export mesh, use the multi-view DA3 path** (Finding 4), not per-frame fusion.
4. **Bias toward ARCore metric depth** for geometry; DA3 to fill (Finding 3 / depth completion).
5. Scan a **textured, asymmetric** object while debugging (rules out the unobservable-object confound).

## Reproduce
```python
# from da3_outputs/session_20260629_000233/ — see README.md back-projection block.
# CORRECT: z = depth[i][m]      → coherent box (aspect 1:1.28:1.49)
# WRONG:   z = 1/depth[i][m]    → warped (aspect long axis 1.49→1.87)
```
