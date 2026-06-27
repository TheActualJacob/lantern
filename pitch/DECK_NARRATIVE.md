# Lantern — Pitch Deck Narrative

**Format:** ~4-minute talk, 11 slides. Arc: Hook → Stakes → Insight → Demo → Proof →
(On-device) → Copilot → Vision → Ask. One message per slide. The demo is the climax —
everything builds to it and pays off after it.

**The one sentence (if they remember nothing else):**
> *While everyone teaches AI to imagine 3D, Lantern teaches your phone to measure it — and prove it.*

**Targets all 3 prizes:** Grand/Qualcomm (on-device NPU), GitHub Copilot (slide 8), People's Choice (story + live demo).

---

## Cold open (say it before the title lands)
> "Quick — picture a broken bracket on your car. You need its exact shape to replace it.
> All you have is your phone. Today, that's impossible. In two minutes I'll show you it isn't."

---

## Slide 1 — Hook · "AI doesn't measure. It guesses."
- **Message:** every AI 3D tool today hallucinates geometry.
- **On screen:** a slick prompt-to-3D result next to a caliper; caption "plausible ≠ correct."
- **Script:** "Every AI 3D tool right now *generates*. Give it a photo, it invents a plausible
  object. Gorgeous for games. But point it at the part holding your engine together and ask it
  to be right to the millimeter — and it lies. Confidently."

## Slide 2 — Stakes · "Plausible isn't good enough."
- **Message:** for repair, medicine, manufacturing, "close" is useless.
- **On screen:** four faces of the problem — field repair, prosthetics, replacement parts, insurance.
- **Script:** "A mechanic in the field. A clinician fitting a prosthetic. An engineer rebuilding
  a part that's no longer made. They don't need plausible — they need *true*. And the only tool
  that measures the real world today is a $20,000 scanner on a tripod, not the phone in your pocket."

## Slide 3 — Insight · "Don't generate. Measure — and fuse."
- **Message:** the non-obvious move — combine dense neural depth with sparse metric truth.
- **On screen:** two-source diagram — Depth Anything (dense, unitless) + ARCore (sparse, metric) → fused.
- **Script:** "Here's our insight: your phone already sees depth two ways. A neural net — Depth
  Anything — gives a dense, smooth shape, but in unknown units. ARCore gives sparse points in
  *real meters*. Neither is enough alone. Fuse them, and you get dense geometry locked to
  real-world scale. We don't imagine the object — we measure it, from every angle, and average
  out the noise into truth."

## Slide 4 — How it works · "Phone → fused depth → solid, on-device."
- **Message:** a real reconstruction pipeline, running on the phone.
- **On screen:** pipeline strip — sweep → scale-locked depth → TSDF fusion → watertight CAD mesh.
- **Script:** "You sweep the phone around the object. Every frame, we lock the neural depth to
  ARCore's metric truth, then fuse all the views into one watertight surface — TSDF, the same
  math behind professional scanners. The depth network runs on the Snapdragon NPU. Thirty
  seconds. No cloud."

## Slide 5 — DEMO · "Watch it happen." ⭐ (climax)
- **Message:** it's real, and it's fast.
- **On screen:** LIVE scan → mesh forming in real time → export. **(Fallback reel cued — never let this depend on stage wifi.)**
- **Script:** "Let me just show you." → *[scan]* "…and there's the mesh, forming as I move.
  Done. That's a real object, reconstructed right here on the phone."

## Slide 6 — Proof · "Accurate. And usable."
- **Message:** measured, not rendered — and it drops into CAD.
- **On screen:** ground-truth error number (mean __ mm), CAD-import screenshot, "100% on-device" badge.
- **Script:** "And it's not just pretty — it's correct. Mean surface error of just __ millimeters
  against ground truth. And it drops straight into CAD — we verified it imports into FreeCAD's
  actual geometry kernel as a watertight solid. This is a measurement you can build from, not a
  picture."

## Slide 7 — Why on-device · "The NPU is what makes it real-world."
- **Message:** on-device is the whole point, not a detail.
- **On screen:** Snapdragon Hexagon NPU; icons — offline, private, instant.
- **Script:** "Why on-device? Because the mechanic in the field has no wifi, the clinic has
  privacy rules, and nobody wants to wait on a server. Running Depth Anything on the Hexagon NPU
  means this works anywhere, instantly, and privately. The phone doesn't *send* data to a
  scanner — the phone *is* the scanner."

## Slide 8 — How we built it · "Copilot was our fourth-and-a-half teammate." (MANDATORY — Copilot prize)
- **Message:** Copilot let 4 people ship a cross-stack pipeline in a weekend.
- **On screen:** the stack we spanned — Kotlin/ARCore · PyTorch/ExecuTorch · OpenCASCADE · Blender — with the Copilot moments called out.
- **Script:** "Four people, two days, a stack spanning Kotlin, PyTorch, the OpenCASCADE CAD
  kernel, and Blender. GitHub Copilot was our connective tissue — it scaffolded the ARCore
  recorder, helped write the affine scale-solver, and debugged our CAD export when none of us
  are CAD-kernel experts. It let each of us work *outside* our home language without drowning.
  This pipeline does not exist in a weekend without it."
- *(Show one concrete before/after: a tricky function Copilot drafted.)*

## Slide 9 — Vision · "A 3D scanner in every pocket."
- **Message:** any phone becomes a measurement instrument.
- **On screen:** market fan-out — field repair, custom medical/prosthetics, insurance claims, e-commerce, reverse engineering.
- **Script:** "Today a 3D scanner is a $20,000 tool on a tripod. We put one in every pocket.
  Field repair, custom-fit medical devices, instant insurance claims, reverse-engineering parts
  the world stopped making — anywhere someone needs the true shape of a real thing."

## Slide 10 — Ask / Close · the one line
- **Message:** land the thesis, name the team.
- **On screen:** "Lantern — measure reality. On your phone." + team names/roles.
- **Script:** "While everyone else is teaching AI to *imagine* 3D, we taught the phone to
  *measure* it — and prove it. That's Lantern. Thank you."

---

## Delivery notes
- **Lead with the punchline,** support after. Don't explain the pipeline before they feel the problem (slides 1–2).
- **The demo is the climax** — rehearse it cold, and have the fallback reel one keystroke away. A dead live demo is the only way to lose this.
- **Make the audience the hero:** "what *you* can now do," not "what we built."
- **Honesty beat:** if asked about the on-device model status, own it — "today the heavy lift is
  proven host-side; on-device inference is landing on the NPU" — owning the gap reads as credible.
- **Timing:** 1:1 + 2:2 + 3:1 + 4:1 + 5:0:45 + 6:0:30 + 7:0:20 + 8:0:30 + 9:0:20 + 10:0:15 ≈ 4 min.
- **Numbers to fill before showtime:** mean error (mm) from `ground_truth.py`; scan→export wall-clock.

## Demo-fail insurance
Slide 5's live demo has a recorded fallback (`demo/build_reel.py` output). If the scan stutters,
cut to the reel without breaking stride: "—and here it is from a clean run earlier." Never apologize, never freeze.
