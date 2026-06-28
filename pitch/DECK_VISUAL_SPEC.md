# Lantern — Deck Visual Spec

Per-slide build sheet. Pairs with `DECK_NARRATIVE.md` (the spoken script). Each slide:
**Layout · On-slide text · Visual asset · Notes · Asset status.** Rule: one idea per slide,
big visual, minimal text (the words are *spoken*, not read).

## Global style
- **Mood:** dark studio — matches our renders. Background `#0C0E14`, panels `#14161E`.
- **Text:** off-white `#EBEEF5`; muted `#969CA8`; **accent blue `#5A96F0`** (use sparingly for the one key word per slide).
- **Type:** one clean sans (Inter / Helvetica Neue). Headline 48–64pt, kicker 24pt caps, body ≤28pt. Max ~8 words on screen.
- **Motion:** subtle. Let the turntable/scan footage move; keep text still. No slide transitions fancier than a cut/fade.
- **Consistency:** every render already uses this palette (`demo/build_reel.py` cards, turntable studio) — reuse it so slides + reel feel like one piece.

---

## Slide 1 — Hook
- **Layout:** full-bleed split, left/right.
- **On-slide:** headline "AI doesn't measure. **It guesses.**" (only "It guesses" in accent).
- **Visual:** LEFT a glossy prompt-to-3D result; RIGHT a caliper on a real part. Thin caption under: "plausible ≠ correct."
- **Notes:** the contrast IS the slide. No bullet points.
- **Asset:** ⬜ grab one generative-3D sample image + a caliper/real-part photo (stock or phone snap).

## Slide 2 — Stakes
- **Layout:** 2×2 quadrant of icons/photos.
- **On-slide:** kicker "WHO NEEDS THE TRUTH" + four labels: Field repair · Prosthetics · Replacement parts · Insurance.
- **Visual:** four real-world photos (mechanic, clinician, machinist, adjuster). Bottom strip: "$20,000 scanner → or the phone in your pocket."
- **Asset:** ⬜ 4 stock photos.

## Slide 3 — Insight
- **Layout:** centered diagram, two inputs merging into one.
- **On-slide:** "Don't generate. **Measure — and fuse.**"
- **Visual:** two boxes → one. LEFT "Depth Anything — dense, no scale" (smooth depth map). RIGHT "ARCore — sparse, real meters" (dotted points). Arrow → "Dense geometry, locked to real scale."
- **Notes:** this is the intellectual core — make the diagram crisp and self-explanatory.
- **Asset:** ⬜ build diagram (Figma/Canva/Keynote). Depth-map thumb can come from `depth_model.py` output later.

## Slide 4 — How it works
- **Layout:** horizontal 4-step pipeline strip.
- **On-slide:** "Sweep → Scale-lock → Fuse → CAD-ready" + badge "Runs on the Snapdragon NPU · ~30s · no cloud."
- **Visual:** 4 thumbnails L→R: phone sweeping · scaled depth · TSDF volume forming · clean mesh. End thumb = our turntable still.
- **Asset:** 🟡 mesh thumb = `demo/assets/hero_complete/frame_0012.png` (have). Steps 1–3 = simple icons/screens (build).

## Slide 5 — DEMO ⭐
- **Layout:** full-screen video, no text (or tiny corner kicker "LIVE").
- **On-slide:** nothing — let it breathe.
- **Visual:** LIVE on-device scan → mesh forming → export. **Fallback:** the rendered reel.
- **Notes:** climax. Rehearse cold; fallback one keystroke away.
- **Asset:** ⬜ live screen-record (Robert, once model lands) · ✅ fallback `demo/build_reel.py` output exists now.

## Slide 6 — Proof
- **Layout:** three stat panels across.
- **On-slide:** big numbers — "**__ mm** mean error" · "Imports as a **solid** in CAD" · "**100%** on-device."
- **Visual:** panel 1 = ground-truth error heatmap (`ground_truth.py` → `.error.ply` screenshot or `.hist.png`); panel 2 = STL open in FreeCAD/Fusion screenshot; panel 3 = NPU badge.
- **Asset:** 🟡 heatmap (have tooling — needs a real scan) · ⬜ CAD screenshot (drag `cad_check`-verified STL into FreeCAD, screenshot) · ✅ NPU badge.

## Slide 7 — Why on-device
- **Layout:** centered chip render + 3 icon callouts.
- **On-slide:** "The phone **is** the scanner." + Offline · Private · Instant.
- **Visual:** Snapdragon/Hexagon NPU graphic; subtle "no cloud" crossed-out server icon.
- **Asset:** ⬜ chip graphic (Qualcomm brand asset / icon).

## Slide 8 — Copilot (MANDATORY)
- **Layout:** left = stack list, right = one real before/after code snippet.
- **On-slide:** kicker "HOW WE BUILT IT" + "Copilot = our 4½th teammate" + stack: Kotlin/ARCore · PyTorch/ExecuTorch · OpenCASCADE · Blender.
- **Visual:** a genuine Copilot moment — e.g. the affine scale-solver or the CAD export — shown as a small code card.
- **Notes:** GitHub PM judges this; concrete > generic. Show an actual snippet, name the task it unblocked.
- **Asset:** ⬜ screenshot one real Copilot suggestion from the build.

## Slide 9 — Vision
- **Layout:** center line + fan-out of market icons.
- **On-slide:** "A 3D scanner in **every pocket.**"
- **Visual:** 5 market icons: repair · medical/prosthetics · insurance · e-commerce · reverse engineering.
- **Asset:** ⬜ 5 icons.

## Slide 10 — Close
- **Layout:** centered, minimal.
- **On-slide:** "**Lantern** — measure reality. On your phone." + team names/roles + repo/QR.
- **Visual:** the hero turntable looping softly behind, dimmed.
- **Asset:** 🟡 `demo/assets/hero_complete.mp4` (placeholder) → swap for real-scan hero when available.

---

## Asset checklist (what humans still must produce)
| Asset | Slide | Owner | Status |
|---|---|---|---|
| Live scan screen-record | 5 | Robert | ⬜ blocked on full capture |
| CAD-import screenshot (STL in FreeCAD/Fusion) | 6 | Charles | ⬜ ~5 min, do anytime |
| Ground-truth heatmap + real mm number | 6 | Charles | 🟡 tooling ready, needs real scan |
| Real-scan hero turntable | 4,10 | Charles | 🟡 one command after full capture |
| Copilot before/after snippet | 8 | anyone | ⬜ screenshot from build history |
| Insight + pipeline diagrams | 3,4 | deck builder | ⬜ Figma/Keynote |
| Stock photos + market/chip icons | 1,2,7,9 | deck builder | ⬜ |

**Auto-generated already / reusable:** turntable stills + clips (`demo/assets/`), the fallback reel
(`demo/build_reel.py`), the studio palette. The single highest-value missing asset is the **full
dome capture** — it unlocks slides 4, 5, 6, and 10 at once.
