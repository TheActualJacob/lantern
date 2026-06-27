# Currently Working — session claim board

> ⚠️ **Multiple Claude/agent sessions work this repo in parallel.** This file prevents two
> sessions from doing the same task (or editing the same file) at once. **Read it before you
> start, claim before you touch code, release when you're done.** This is a coordination
> board, not a status report — for the post-merge snapshot see `REPO_STATUS.md`.

## How to use it (every session, every time)

1. **Before starting:** scan **Active claims** below. If your task — or a file you'll edit —
   is already claimed `🟡 IN PROGRESS`, pick something else or coordinate; don't double-work it.
2. **Claim it:** add a row to **Active claims** with your session label, the task, the files
   you'll touch, the start time, and status `🟡 IN PROGRESS`. Keep the file list specific so
   others can see overlap.
3. **When done:** move the row to **Recently completed** (newest on top) with the finish time
   and a one-line result. If you abandon it, delete the row so it frees up.
4. **Stale?** A `🟡 IN PROGRESS` row older than its expected window with no commits is probably
   dead — confirm, then clear it.

Session label = anything that disambiguates (e.g. `S-vault-personD`, `S-gt-script`, a tmux
name, or initials + what you're on). Times in local PDT.

---

## Active claims

| Started | Session | Task | Files / area | Status |
|---|---|---|---|---|
| 2026-06-27 14:24 | _(other active session)_ | Person-D code tasks: run/tune `clean_ondevice.py` on real TSDF mesh, before/after deck renders, live-metrology math | `clean_ondevice.py`, `output/` renders | 🟡 IN PROGRESS |
| 2026-06-27 14:24 | S-vault-personD | Vault planning + coordination only (win-plan, pitch-outline, Claude Design prompt, this board). **Not** touching pipeline code while the row above is active. | vault `projects/executorch-hackathon/*`, this file | 🟡 IN PROGRESS |

> The first row is a placeholder for the session Charles said is already working these tasks —
> have that session refine it to the exact subset it owns so the rest free up.

## Recently completed

| Finished | Session | Task | Result |
|---|---|---|---|
| 2026-06-27 ~14:30 | S-personD-integration | Ground-truth validator (Phase 3c) + env + integration | `ground_truth.py` pushed; reference mode (FPFH+RANSAC→ICP, color `.ply` + histogram) & known-dims mode; `--selftest` PASSED (sphere +2%≈2 mm, align recovery 0.000 mm). `requirements.txt` pushed (open3d≥0.19). Verified Sneha's pipeline end-to-end on py3.12. **Note:** built as `ground_truth.py` (not `ground_truth_report.py`) — claim above trimmed to avoid dup. |
| 2026-06-27 ~13:50 | S-vault-personD | On-device mesh cleanup prototype | `clean_ondevice.py` added; `--selftest` PASSED (sphere → 200.0 mm; Poisson path runs) |
| 2026-06-27 ~13:50 | S-vault-personD | Win plan + pitch outline + 5-min Claude Design prompt | written to vault `projects/executorch-hackathon/` |

## Don't-collide notes

- **A owns** `pipeline_float.py`, `scale_solver.py`, `tsdf_fuse.py`, `depth_model.py` — coordinate before editing; keep them pristine (see vault `repos/lantern/notes.md`).
- **C owns** the Android app under `app/` + the ARCore capture format.
- **D (Charles)** owns integration/cleanup/validation/demo: `import_and_clean.py`, `clean_ondevice.py`, `ground_truth.py` (the validator — NOT `ground_truth_report.py`; that name is retired to avoid a duplicate), env manifest, demo assets.
