# Learned Mesh Enhancement — make the reconstruction *pretty* and *faithful*

**Goal in one line:** take the captured **frames (RGB), DA3 dense depth, SAM masks, ARCore
poses,** and the **raw TSDF mesh**, and run a machine-learning stage during processing that
**fills in missing detail, adds real color/material, and cleans the surface** — so the final
render and exported mesh look like the *actual object*, not a gray blob.

**Status:** Research + plan. Nothing here is built. The live capture path
(`LiveReconstructor` → `TsdfVolume` → `MarchingCubes`) and host cleanup (`import_and_clean.py`)
are the inputs this stage consumes; it does **not** change them.
**Date:** 2026-06-28 · **Related:** `roadmap.md` (Decision 0 — faithful vs generative),
`LIVE_MESH_PLAN.md` (§7 "filling in"), `README.md` (pipeline).

---

## 0. Read this first — the honest framing

The repo's spine is **faithful reconstruction**: TSDF fusion *averages what you actually
scanned* into a metrically-correct surface. The roadmap's **Decision 0** deliberately fences
off "generative" image-to-3D (TripoSR / TRELLIS class) as a *separate fallback lane* because
those models **hallucinate a plausible object from sparse views** — they will happily invent a
handle that isn't there, or round off a corner that is. That is the opposite of what a
CAD-ready scan promises.

So the request — "extrapolate and fill in details to make it prettier and like the actual
object" — has a built-in tension:

- **"Like the actual object"** = faithfulness. The output must stay anchored to the measured
  geometry and the captured pixels. Metric scale (the affine solver's whole job) must survive.
- **"Pretty / fill in / extrapolate"** = exactly where learned priors *invent* things.

**The resolution is conditioning, not generation-from-scratch.** Every model we add must be
**conditioned on the real capture** (the mesh, the masked RGB frames, the depth) and used as a
*refiner / completer / texturer*, never as a free image-to-3D generator that ignores the scan.
The win is biggest, and the hallucination risk lowest, when we **separate the two things people
actually mean by "ugly":**

1. **It's gray / untextured.** Today `MeshData` carries no color at all
   (`recon/ReconTypes.kt`: verts + normals + faces only). *The single largest perceived-quality
   jump is just putting the real photographed color on the surface* — and that needs **no
   generative ML at all** at first. Do this before anything fancy.
2. **It's holey / blobby / noisy.** Unseen backs, depth noise, TSDF over-thickening, jagged
   marching-cubes facets. This is where learned completion/refinement earns its place — but
   tightly conditioned so it polishes rather than re-imagines.

Build #1 first (cheap, huge visual payoff, zero faithfulness risk), then layer #2 (learned)
on top with guardrails. The plan below is staged that way.

---

## 1. What "pretty + faithful" decomposes into

| Sub-problem | What's wrong today | Fix class | Hallucination risk |
|---|---|---|---|
| **A. Color / texture** | Mesh is untextured gray Lambert | Project captured frames onto the mesh (classic) → then learned texture *inpainting* for unseen faces | Low (classic) → Med (inpaint) |
| **B. Surface noise / facets** | Monocular-depth multi-view inconsistency thickens + roughens the surface; marching-cubes stair-steps | Learned **mesh/point denoising & refinement** conditioned on the raw mesh | Low–Med |
| **C. Holes / unseen regions** | Backs and occluded parts never scanned → open boundaries | Learned **shape completion** conditioned on the partial mesh + masks | **Med–High** (this is where it invents) |
| **D. Missing fine detail** | DA3 depth is smooth; engraving/relief is lost | **Detail/normal super-resolution** from the high-res RGB frames (the pixels saw detail the depth didn't) | Med |
| **E. Materials / relighting** | Baked-in capture lighting looks flat/wrong under viewer light | **PBR delighting** (albedo/roughness/metallic estimation) | Med |

Most "wow" per unit risk: **A (color) → B (denoise) → D (detail from RGB) → E (PBR) → C
(completion)**. Notably the riskiest one (C, inventing unseen geometry) is the *least*
important for "looks like the actual object" — push it last and make it optional.

---

## 2. The inputs we already have to feed a model

Everything a learned enhancer wants is already produced or trivially producible:

| Input | Where it lives today | Notes |
|---|---|---|
| **Masked RGB frames** | `ImageUtils.yuvToArgb` in `LiveReconstructor.process`; SAM mask from `QnnSegmentationModel` | High-res, object-isolated. The richest signal — pixels carry color + detail the depth lost. |
| **DA3 dense depth (metric)** | `ExecuTorchDepthModel`/`QnnDlcDepthModel` → `AffineScaleSolver.buildMetricDepth` | Already scaled to meters against ARCore. |
| **Per-frame camera pose** | ARCore `cameraToWorld`; for in-hand, `T_OC` from `ObjectTracker` | Lets us project any frame onto the mesh and back. |
| **Object mask** | `MaskStabilizer` output | Keeps hand/background out of the texture + completion. |
| **Raw TSDF mesh** | `TsdfVolume.extractMesh()` → `MeshData` | The geometry to refine. Also a TSDF grid available for volumetric conditioning. |
| **Intrinsics** | `CameraIntrinsics.scaledTo(...)` | Needed for projective texturing / re-rendering. |

This is *exactly* the input set of modern conditioned 3D refiners (multi-view images + coarse
geometry + cameras). We are not missing data — we're missing the consumer.

---

## 3. Where it runs — on-device vs host post-process (decide first)

This is the load-bearing decision; it shapes everything else.

- **On-device (ExecuTorch / Hexagon).** On-brand for the hackathon ("all on the phone"). But
  the strong enhancers (TRELLIS.2 ~4B, Hunyuan3D 2.x, multi-view diffusion) are **far too big**
  to run on an S25 at interactive rates, and none have a vetted `.pte` export. On-device is
  realistic only for the *small, surgical* nets: depth/normal super-res, a lightweight mesh
  denoiser, texture inpainting at modest resolution.
- **Host / cloud post-process.** "Tap *Enhance* → upload the capture bundle (frames + mesh +
  poses) → a beefier model returns a textured, completed `.glb` in ~10–60 s." This is where the
  SOTA models actually live and where "pretty" really happens. It also fits the existing
  **host cleanup** step (`import_and_clean.py` already runs off-device in Blender).

**Recommendation: hybrid, staged.**
- **Stage 1 (on-device, ships in the demo):** projective vertex-color / texture bake — no big
  model — so the live preview and export immediately show real color.
- **Stage 2 (host "Enhance" button):** the learned geometry refine + texture inpaint +
  optional completion + PBR. Keep it as a clearly-labeled *post-process* the user opts into,
  so the faithful on-device mesh is always available untouched.

Keep the boundary clean: the device emits a **capture bundle** (already most of what
`FrameRecorder` writes for offline sessions — RGB + pose + intrinsics + depth, plus the
masks + raw mesh), and the host enhancer consumes it. This means the enhancer can be built and
iterated **entirely offline against recorded sessions** before any device integration — the
same front-loading philosophy as `roadmap.md`.

---

## 4. Candidate model families (2025–2026 landscape)

Researched current options and how each maps onto our "condition on the real scan" constraint.

### 4a. Geometry refinement / denoising (Sub-problems B, partial C) — *recommended core*
Take the coarse TSDF mesh (or its point cloud / TSDF grid) + the multi-view images and produce
a cleaner, sharper surface **anchored to the input**. This is the faithful sweet spot.
- **Conditioned mesh/SDF refiners** (e.g. DiMeR-style disentangled mesh reconstruction, and the
  geometry stage of TRELLIS.2's sparse-voxel "O-Voxel" representation) take coarse geometry +
  views and regress a refined surface. Use the **geometry branch only**, conditioned hard on
  our mesh, so it denoises/sharpens rather than re-generates.
- **Point-cloud denoising nets** (score-based / displacement nets) are small enough to consider
  on-device and operate directly on the fused cloud before meshing.

### 4b. Texture / material generation (Sub-problems A, D, E) — *highest visual ROI*
Our object is *seen*, so this is mostly **back-projection + inpaint the gaps**, not full
synthesis:
- **Neural back-projection texturing** (Im2SurfTex / MVD²-style): fuse multi-view image features
  onto the surface into a UV texture, handling view conflicts and seams better than naive
  averaging. Strongly faithful because it's literally re-projecting our photos.
- **Multi-view-consistent texture inpainting** (FlexPainter / MD-ProjTex / MV2UV class): fill
  *only* the never-seen patches, conditioned on the observed texture so the invented bits match.
- **PBR delighting / material estimation** (LumiTex / Hunyuan3D 2.1 PBR branch class): split the
  baked color into albedo + roughness + metallic so the model relights correctly in the viewer.
  Optional polish.

### 4c. Full image-to-3D as a *prior*, not a *generator* (Sub-problem C, optional) — *use with guardrails*
TRELLIS.2 / Hunyuan3D / Step1X-3D produce gorgeous complete textured meshes from images. The
**trap** is using their output *as* the model — it won't be metric and won't match the measured
geometry. The **safe** use is as a **completion prior**: run image-to-3D to get a plausible full
shape, then **non-rigidly register it to our faithful mesh and only borrow geometry where we
have no measurements** (the unseen back). Measured regions always win. This keeps scale and the
seen surface faithful while letting a strong prior close the back prettily. Treat as a stretch.

### 4d. Radiance-field / Gaussian-splat rendering (alternative *rendering* path)
If the goal is a beautiful *render* (not a CAD mesh), training a per-object **3D Gaussian
splatting** model from the captured frames + poses gives photoreal novel views fast, and recent
work (DiffSplat, feed-forward splat predictors) does it in seconds. This is a *parallel output*:
"pretty render" via splats, "CAD mesh" via TSDF. Worth a spike for the demo's hero shot, but it
doesn't itself produce the clean mesh — keep it as an additional viewer mode, not a replacement.

> Licensing watch-out (matters if this ever ships, not for a hackathon demo): several strong
> models are **non-commercial** (research-only) checkpoints. TRELLIS/Hunyuan terms vary by
> release. Note licenses per model the moment one is chosen — same discipline `LIVE_MESH_PLAN`
> §5.0 applied to DA3 variants.

---

## 5. Recommended architecture

```
        CAPTURE BUNDLE  (from device, or a recorded session)
        ├─ masked RGB frames + intrinsics + poses (ARCore world, or T_OC in-hand)
        ├─ DA3 metric depth per frame
        ├─ object masks
        └─ raw TSDF mesh (.glb, meters)
                       │
   ┌───────────────────┼─────────────────────────────────────────────┐
   │ STAGE 1 — on-device, no big model (ships in demo)                │
   │  projective texture bake:                                         │
   │   for each mesh vertex, gather color from the frames that see it  │
   │   (visibility + view-angle + depth-consistency weighted), write   │
   │   vertex colors / a UV texture  ──►  real-colored mesh NOW        │
   └───────────────────┬─────────────────────────────────────────────┘
                       │ (faithful colored mesh always available)
   ┌───────────────────┼─────────────────────────────────────────────┐
   │ STAGE 2 — host "Enhance" (opt-in post-process)                    │
   │  B. geometry refine/denoise  (condition: mesh + views)           │
   │  D. detail/normal super-res  (condition: high-res RGB)           │
   │  A. texture back-project + multi-view-consistent inpaint of gaps  │
   │  E. PBR delight  → albedo/roughness/metallic   (optional)        │
   │  C. shape completion of unseen back  (optional, prior-as-fill,    │
   │      measured surface always wins)                                │
   └───────────────────┬─────────────────────────────────────────────┘
                       ▼
        enhanced textured .glb  ──►  import_and_clean.py (watertight, m→mm, STL)
                       │                     └─ untouched faithful path still exists
                       ▼
        viewer: faithful mesh ◄toggle► enhanced mesh ◄toggle► splat render
```

The toggle is the product point: **never silently replace the faithful scan with an enhanced
one.** Show both; let the user (and the CAD handoff) choose. "Enhanced" is the pretty hero;
"faithful" is the measured truth.

---

## 6. Milestones & exit criteria

| Milestone | Deliverable | Exit criteria |
|---|---|---|
| **E0 — Color the mesh (no ML)** | Projective texturing: bake captured frames onto `MeshData` as vertex colors (then UV atlas). Carry color through `MeshExport` + `import_and_clean.py`. | The exported `.glb` shows the object's *real* colors; viewer renders it. Biggest visual jump, zero hallucination. |
| **E1 — Offline enhance harness** | Host script: capture-bundle in → enhanced `.glb` out, run on recorded sessions. Stubs for each stage so they slot in independently. | Round-trips a real session; faithful mesh preserved as baseline for diffing. |
| **E2 — Geometry refine + detail SR** | Wire a conditioned mesh-denoise/refine net (4a) + RGB-driven detail/normal super-res (4b/D). | Surface is visibly smoother/sharper **and** chamfer-distance to the raw mesh stays within a bound (proves it refined, didn't re-imagine). |
| **E3 — Texture inpaint + PBR** | Multi-view-consistent inpaint of unseen patches; optional delight to PBR. | Seams/holes in the texture are filled coherently; relights correctly in viewer. |
| **E4 — Completion (stretch)** | Image-to-3D prior (4c) registered to the faithful mesh; fills only unmeasured regions. | Back closes plausibly; **measured regions provably unchanged** (masked diff = 0). |
| **E5 — On-device slice (stretch)** | Push the smallest net(s) (detail SR or point denoise) to ExecuTorch so *some* enhancement is on-device. | One enhancement stage runs in `.pte` on the S25 within latency budget. |

Ship E0 + E1 + as much of E2/E3 as time allows; E4/E5 are bonus.

---

## 7. Faithfulness guardrails (the part that keeps it "the actual object")

Bake these in from E1, or the enhancer becomes a hallucination machine:

1. **Measured-region lock.** Mark every vertex/voxel as *observed* (≥1 frame saw it with
   consistent depth) or *unobserved*. Learned geometry edits are **clamped** on observed
   regions (bounded displacement) and only free on unobserved ones. Texture from real pixels on
   observed faces; inpaint only unobserved faces.
2. **Metric scale is sacred.** The enhancer works in the mesh's existing metric frame; no model
   output is allowed to rescale it. Re-run the dimensional check (`ground_truth.py` /
   `cad_check.py`) after enhancement — error bound must not regress vs the faithful mesh.
3. **Diff gate, not eyeball.** Every stage reports chamfer/Hausdorff distance and a masked
   "did-not-touch-observed-geometry" metric against the faithful input. Regressions block.
4. **Always keep the faithful mesh.** Enhanced is an *additional* artifact. The CAD/STL handoff
   defaults to faithful unless the user explicitly exports the enhanced one.
5. **Honest UI.** Label enhanced output as "AI-enhanced (cosmetic)" so nobody mistakes invented
   detail for measured truth — same honesty posture as `LIVE_MESH_PLAN` §0/§14.

---

## 8. Validation plan

- **Offline-first** (mirrors `roadmap.md`): build the enhancer against **recorded sessions** and
  the existing **board sessions with ground-truth `T_CO`**. No device needed to iterate.
- **Faithfulness metrics:** chamfer/Hausdorff vs faithful mesh (observed regions ≈ 0 change);
  dimensional error vs `ground_truth.py` reference must not regress.
- **Appearance metrics:** re-render the enhanced mesh from each held-out capture pose and
  compare to the real frame (PSNR/SSIM/LPIPS). This directly measures "looks like the actual
  object" — the model should *reproduce held-out real views*, the cleanest faithfulness test for
  texture/detail.
- **Ablations:** color-only (E0) vs +refine vs +inpaint, so we know each stage's real
  contribution and can drop any that mostly adds risk.
- **Keep the existing host suite green** (`test_harness.sh`, `orientation_test.sh`,
  `cad_check.py`) — enhancement is additive.

---

## 9. Risks & limitations

| Risk | Mitigation |
|---|---|
| Model invents geometry/texture that isn't the real object | §7 measured-region lock + diff gates; completion is opt-in and back-only |
| Loses metric scale → CAD mesh wrong size | Scale frozen; post-enhance `ground_truth`/`cad_check` gate |
| Big models won't run on-device | Hybrid: on-device color bake (E0) + host enhance (E2–E4); on-device nets only for small slices (E5) |
| Non-commercial model licenses | Record license per chosen model before integrating; prefer permissive (cf. DA3 Apache vs CC-BY-NC) |
| Texture seams / view-conflict artifacts | Use neural back-projection (conflict-aware) over naive averaging; multi-view-consistent inpaint |
| Capture lighting baked into texture looks flat | Optional PBR delight (E3); acceptable to ship lit-texture for the demo |
| Scope creep into a research project | E0 alone (real color) is a huge, safe win — ship it first; treat E2+ as earned upside |

---

## 10. TL;DR

1. The single biggest "pretty" win needs **no generative model**: the mesh is gray today — put
   the **real photographed color** on it (E0, on-device, projective bake). Do this first.
2. Then add **learned refinement as a host "Enhance" post-process**: geometry denoise + RGB
   detail super-res + multi-view texture inpaint of unseen gaps (+ optional PBR). These are
   **conditioned on the real capture** — refiners/completers, *never* free image-to-3D.
3. Use a full image-to-3D model (TRELLIS.2 / Hunyuan3D class) only as a **completion prior for
   the unseen back**, non-rigidly registered to the faithful mesh — measured geometry always wins.
4. **Guardrails make it faithful:** lock observed regions, freeze metric scale, diff-gate every
   stage, and always keep the untouched faithful mesh. "Enhanced" is the hero render; "faithful"
   is the CAD truth — show both, never silently swap.
5. Validate **offline against recorded + board-ground-truth sessions**, scoring *held-out real
   views* (LPIPS/PSNR) for appearance and chamfer/dimension for faithfulness.

---

**Sources (model landscape):**
[TRELLIS.2 (Microsoft)](https://github.com/microsoft/TRELLIS.2) ·
[Hunyuan3D 2.1](https://trellis3d.co/hunyuan3d) ·
[Step1X-3D](https://arxiv.org/pdf/2505.07747) ·
[DiMeR — disentangled mesh reconstruction](https://arxiv.org/pdf/2504.17670) ·
[FlexPainter — multi-view texture](https://arxiv.org/pdf/2506.02620) ·
[MD-ProjTex](https://arxiv.org/pdf/2504.02762) ·
[Im2SurfTex — neural back-projection texturing](https://arxiv.org/pdf/2502.14006) ·
[MVD² — multiview recon](https://dl.acm.org/doi/10.1145/3641519.3657403) ·
[LumiTex — PBR texture](https://arxiv.org/pdf/2511.19437) ·
[DiffSplat — image-diffusion → Gaussian splats](https://arxiv.org/pdf/2501.16764)
