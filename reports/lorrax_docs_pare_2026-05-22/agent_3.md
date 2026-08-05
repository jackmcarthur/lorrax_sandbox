# Agent 3 — Plans / WIP / Status / Drift

Slice: `docs/index.md`, `docs/AGENT_TODO.md`, `docs/PLAN_zeta_g_flat_migration.md`,
`docs/FREQ_INTEGRATION_REWRITE_PLAN.md`, `docs/FREQ_INTEGRATION_PROGRESS.md`,
`docs/SIGMA_FREQ_AUDIT_STATUS.md`, `docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`,
`docs/NEW_WINDOW_MINIMAX_GUIDELINES.md`, `docs/PROFILING_SUGGESTIONS.md`,
`docs/plans/phdf5_cray_mpich_migration.md`, `docs/plans/unified_slab_io.md`,
`docs/archive/` (8 files + subdirs, already archived), `docs/advanced/` (3 files).

Out of scope: the `*_COMPREHENSIVE.md` set, `MEMORY_MODEL.md`,
`MINIMAX_QUADRATURE.md`, `SYMMETRY_COMPREHENSIVE.md`, `ZETA_V_Q_ALGORITHMS.md`
(agents 1, 2, 4).

Branch read: `sources/lorrax_D` on `agent/install-blitz-integration` (HEAD: `3079a1f`).

---

## 1 — Per-doc verdict table

| Doc | Lines | Verdict | One-sentence justification |
|---|---:|---|---|
| `docs/index.md` | 58 | **DELETE** | Three stale facts (wrong module paths, dead `examples/` link, dead `formalism.md` link); shorter than the README section it duplicates; repair cost exceeds value since AGENTS.md and README already route readers. |
| `docs/AGENT_TODO.md` | 292 | **DELETE** | Header explicitly disowns the content ("NOT the user's current priorities"); suggestions were written by a prior agent about code that has since been heavily refactored; nothing actionable survives a code diff. |
| `docs/PLAN_zeta_g_flat_migration.md` | 373 | **ARCHIVE** | All four phases (A–D) have landed: Phase A (`accumulate_rchunk_to_gflat` / `gflat_to_rchunk` in `wfn_transforms.py`, commit `f0af7c2`), Phase B (IBZ solve gather, commit `feb1342`), Phase C (G-flat on-disk, `isdf_header.py` `zeta_layout` field, commits `63c5eac` + `93fe8d1`), Phase D (bispinor V_q unfold, commit `882ed4a`). Archive; the reference line-number table is useful archaeology. |
| `docs/FREQ_INTEGRATION_REWRITE_PLAN.md` | 651 | **ARCHIVE** | The `freqint/` package it planned (Stages 1-3) was built and the sigma path was left as a `NotImplementedError` placeholder; however the package itself (`src/gw/freqint/`) no longer exists in the integration branch — it was never merged into the production pipeline. The plan is for work that stalled and was superseded; archive it alongside the PROGRESS doc. |
| `docs/FREQ_INTEGRATION_PROGRESS.md` | 71 | **ARCHIVE** | Companion status to the REWRITE_PLAN; both stalled at Stage 3 with sigma unimplemented and `freqint/` not present on integration branch; no reader benefit to keeping it live. |
| `docs/SIGMA_FREQ_AUDIT_STATUS.md` | 121 | **ARCHIVE** | Date-stamped 2026-03-31; contains absolute paths to `tests_isdf/` directories outside the repo and MAE numbers from early March runs. The CO agreement it describes has since been stabilized and the note in KNOWN_SANDBOX_ERRORS / memory confirms GN-PPM works for both 2D and 3D. Archive; the decomposition-mapping notes (§2) are the only content worth salvaging — they belong as a code comment in `ppm_sigma.py`, not a separate doc. |
| `docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` | 459 | **KEEP + RENAME** | Actively referenced by `PHYSICS_COMPREHENSIVE.md` and `CODEBASE_COMPREHENSIVE.md`; the un-revised original (`GN_PPM_MINIMAX_SIGMA_GUIDE.md`) was committed in the same commit and was presumably deleted when the revision superseded it (it does not exist in the working tree or any branch); the `_REVISED` suffix is therefore noise. Rename to `GN_PPM_MINIMAX_SIGMA_GUIDE.md`. Content is current (the window-edge derivation and `_SigmaWindow` / `_build_three_sigma_windows` references match production code in `ppm_sigma.py`). |
| `docs/NEW_WINDOW_MINIMAX_GUIDELINES.md` | 478 | **MERGE INTO PHYSICS_COMPREHENSIVE** | Referenced once from `PHYSICS_COMPREHENSIVE.md` as the windowing-strategy source. Content is derivation-level theory (why coarse six-window beats fine-grained, sin vs exp quadrature handoff margin) that fits naturally in §"CTSP / minimax" of PHYSICS_COMPREHENSIVE. The practical routing tables at the end are already duplicated there. At ~478 lines it is large for a merged section; a leaner version (~80 lines keeping the six-window rationale and handoff-margin formula) replaces the current placeholder sentence. |
| `docs/PROFILING_SUGGESTIONS.md` | 342 | **DELETE** | Written by an agent circa early blitz sessions; describes `psi_coh_rtot_Y` as "~20 GB wasted" but those call paths were eliminated during the G-flat refactor. The memory model it proposes (§8) was superseded by the production `gflat_memory_model.py` + HLO-calibration work through May. No live code paths match the file anymore; the analysis is archaeologically interesting but wrong. Delete. |
| `docs/plans/phdf5_cray_mpich_migration.md` | 197 | **ARCHIVE** | Migration is done: commit `c3f29bf` (2026-04-17) landed the Cray MPICH stack and user memory (`project_phdf5_mpich_default`) confirms "2026-04-20: new default stack". Keep the "Upshot" paragraph (the GCC-12 ABI mismatch insight) as a comment in `CMakeLists.txt` instead; the full plan is archaeology. |
| `docs/plans/unified_slab_io.md` | 291 | **ARCHIVE** | Unified `SlabIO` landed in commit `b11cf98` (2026-04-17): `src/file_io/slab_io.py`, `_slab_io_allgather.py`, `_slab_io_ffi.py` all exist. Plan fulfilled. Archive. |
| `docs/archive/` (8 files ~1640 lines) | various | **KEEP as-is** | Already archived. Spot-checked: none of the archived files (cohsex_jax_physics.md, ctsp_revised.md, formalism.md, ZETA_FITTING_ALGORITHM.md, isdf_context.md, isdf_spin_galerkin_derivation.md, Kim-2020, nufft/) are referenced from any active doc except via the `archive/README.md` supersession table, which is correct. No un-archiving detected. |
| `docs/advanced/README.md` | 19 | **KEEP** | Correct directory-level router; 2 sentences per file. |
| `docs/advanced/jax_multihost.md` | 870 | **ARCHIVE** | Third-party tutorial content (JeanZay-specific, from github.com/ASKabalan/Jax-multihost); 870 lines of generic JAX SPMD pedagogy. Blitz #3 consolidated distributed init into `runtime.init_jax_distributed()` with a post-init assert — the distributed-launch complexity this guide describes is now LORRAX-abstracted. No active doc cross-references it. It's useful background reading but not LORRAX docs. Move to archive. |
| `docs/advanced/HL_GPP_derivation.md` | 186 | **KEEP** | Short (186 lines), correct, and genuinely useful reference for anyone implementing or debugging GPP mode. Not duplicated elsewhere in the corpus. No stale facts detected. |

---

## 2 — Per-doc detail notes

### A. `docs/PLAN_zeta_g_flat_migration.md` — verification of "all phases landed"

All four phases are confirmed complete on the integration branch:

- **Phase A**: `gflat_to_rchunk` and `accumulate_rchunk_to_gflat` exist in
  `src/common/wfn_transforms.py`; `wfn_transforms.py` commit history shows
  `f0af7c2` ("wfn_transforms: phase-after-slice + accumulate_rchunk_to_gflat
  (Phase A)") and subsequent consolidation.
- **Phase B**: `feb1342` ("perf(zeta): factor Cholesky/LU only at IBZ q-points")
  matches the IBZ-only gather described in §B1.
- **Phase C**: `isdf_header.py` has `zeta_layout` field (commit `63c5eac`); the
  r-space-on-disk path was deleted in `93fe8d1` ("isdf_fitting + gw_init: bake
  G-flat ζ writer on; delete r-space-on-disk path"). `zeta_reader.py` and
  `zeta_loader.py` both reference `zeta_layout` in the working tree.
- **Phase D**: bispinor V_q unfold landed in `882ed4a` ("feat(sym): bispinor IBZ
  cascade with 3-vector Lorentz mixing on TT tiles").

Conclusion: archive, do not delete — the file-line reference table in §"Key
file/line references" is still correct archaeology for anyone reading the Cholesky
/IBZ design.

### B. `docs/FREQ_INTEGRATION_REWRITE_PLAN.md` + `FREQ_INTEGRATION_PROGRESS.md` — freqint stalled

The PROGRESS doc says Stages 1–3 for chi are complete and sigma remains
`NotImplementedError`. However `src/gw/freqint/` does not exist anywhere on the
integration branch. The tests it references are in `tests/archive/`. The production
sigma path runs through `ppm_sigma.py` + `WindowExecutionPlan` (from `gw_config.py`)
— a parallel, independent implementation. The freqint engine never graduated to
production. Both docs should be archived together as a design detour that didn't
merge.

### C. `docs/SIGMA_FREQ_AUDIT_STATUS.md` — stale audit

The key salvageable insight (§2, "Do Not Mix These") — that `Sigma_cor+` /
`Sigma_cor-` is not the same partition as BGW's `SX-X` + `CH` — is a one-paragraph
code comment that belongs in `ppm_sigma.py` near the `Sigma^(+) + Sigma^(-)`
decomposition. Move it there; archive the doc.

### D. `docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` — rename, no content changes needed

The un-revised original (`GN_PPM_MINIMAX_SIGMA_GUIDE.md`) was added and presumably
replaced in commit `186ac1b` (2026-03-31); it does not appear in the working tree or
any branch tip. The `_REVISED` suffix is vestigial noise. Two active references in
`PHYSICS_COMPREHENSIVE.md` and one in `CODEBASE_COMPREHENSIVE.md` all use the
`_REVISED` filename — update those links after renaming. Content was last touched in
commit `7a23d55` ("Rename minimax regeneration input flag") and matches production
`ppm_sigma.py`.

### E. `docs/index.md` — three concrete stale facts

1. `src/isdf/common/wfnreader.py` — actual path is `src/common/wfnreader.py`
   (package renamed from `isdf` to flat layout).
2. `src/gw/gw_jax.py` — exists but the module listing is abbreviated/wrong
   (e.g. no mention of `ppm_sigma.py`, `v_q_tile.py`, etc.).
3. `examples/` — directory does not exist.
4. `formalism.md` — linked as `formalism.md`; actually at
   `docs/archive/formalism.md`.
5. `docs/api/` — generated API docs; `gen_api_docs.sh` exists but the directory
   is not checked in.

All five errors in 58 lines. AGENTS.md and README already serve as routers. The
"Key modules" list in `index.md` would need continuous maintenance as modules
evolve. Delete rather than repair.

### F. `docs/PROFILING_SUGGESTIONS.md` — superseded by gflat_memory_model

The central finding (`psi_coh_rtot_Y` is UNUSED, ~20 GB wasted) was fixed during
the G-flat refactor: those `psi_*_rtot_Y` calls no longer appear in `gw_jax.py`.
The proposed three-phase memory model in §8 was superseded by
`src/gw/gflat_memory_model.py` and its HLO-calibration work through May. The
`FFT_BUFFERS=2 vs 4` debate is also resolved in the planner.

Nothing survives a line-by-line diff against the current code. Delete.

### G. `docs/advanced/jax_multihost.md` — upstream tutorial, not LORRAX docs

This is a verbatim copy of the JeanZay multihost tutorial
(github.com/ASKabalan/Jax-multihost). It predates LORRAX's runtime module. With
`runtime.init_jax_distributed()` centralizing init and Blitz #3 landing the
post-init assert, the LORRAX-relevant portion of JAX multihost is a few lines in
`src/runtime/__init__.py`, not 870 lines of pedagogy. If background reading is
desired, link to the upstream repo. Archive.

---

## 3 — Cross-cutting recommendations

### R1. No more plan docs in `docs/`

This slice contained 5 plan docs (G-flat, freqint rewrite, phdf5, unified_slab,
zeta G-flat) totaling ~1,500 lines. All are finished or abandoned work. The pattern
of "write a plan doc in `docs/`" should stop. Plans belong in:
- A GitHub issue or GitHub Discussion while the work is open.
- A comment in the first commit that starts the feature branch.
- `reports/` (the sandbox already has this convention).

After the current paring, the only plan-like content that should survive is the
pending-work section of `AGENTS.md`.

### R2. No more status/audit docs in `docs/`

`SIGMA_FREQ_AUDIT_STATUS.md` and `FREQ_INTEGRATION_PROGRESS.md` are the status
pattern. Both have immediate-half-life (weeks). If project-wide status is valuable,
one `docs/PROJECT_STATUS.md` maintained as a short "current known issues + active
work" list is better than accumulating per-initiative status files. Keep it under 80
lines with a "last updated" date at the top so staleness is obvious.

### R3. `_REVISED` suffix convention is harmful

`GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` is the current canonical guide; the
original is gone. The `_REVISED` suffix implies a pair that doesn't exist, confuses
readers looking for the "real" version, and makes in-repo search ugly. Policy: when
a revision supersedes the original, delete the original and commit the new content
without the `_REVISED` suffix. Rename the surviving guide in this pass.

### R4. `docs/archive/README.md` references `AGENT_TODO.md`

Line 33 of `docs/archive/README.md` says: "Agent suggestions: [`../AGENT_TODO.md`]".
If `AGENT_TODO.md` is deleted (recommended), remove that line.

### R5. `CODEBASE_COMPREHENSIVE.md` and `ENVIRONMENT_COMPREHENSIVE.md` link to `AGENT_TODO.md`

Both reference `AGENT_TODO.md` as "agent todos." After deletion, remove those links.
The `PROFILING_SUGGESTIONS.md` doc is not cross-referenced from any active doc so
deletion leaves no dangling links.

---

## 4 — What's missing (fresh writing needed)

### 4a. A paragraph in `ppm_sigma.py` on the BGW decomposition mapping

The best content in `SIGMA_FREQ_AUDIT_STATUS.md` §2 is a 4-sentence note:
`Sigma_cor+` / `Sigma_cor-` ≠ BGW `SX-X` / `CH`; only the sum is a stable
cross-code target; BGW near-pole redistribution between the two channels is another
reason. This belongs as a block comment at the `Sigma^(+) + Sigma^(-)` definition
in `ppm_sigma.py`, not as a separate doc. Target: ~10 lines of comment.

### 4b. Six-window rationale in `PHYSICS_COMPREHENSIVE.md`

`NEW_WINDOW_MINIMAX_GUIDELINES.md` §§1-3 (motivation + consequence for windowing +
the coarsest-decomposition principle) are worth ~80 lines in
`PHYSICS_COMPREHENSIVE.md`'s CTSP section. Currently that section has one sentence
("Windowing strategy in `docs/NEW_WINDOW_MINIMAX_GUIDELINES.md`") as a
placeholder — the merge would make it self-contained.

### 4c. Docs index in `README.md` vs `AGENTS.md` reconciliation (out of scope for this agent)

The two indices don't agree. Not fixing here — flagging for Agent 4 who owns
synthesis.

---

## 5 — Open questions

1. **`docs/advanced/HL_GPP_derivation.md`**: the first line says "Yeah, your memory
   is basically right" — this is an AI chat response, not prose documentation. Is
   the author happy with that tone for in-repo docs, or should the first line be
   trimmed to start at "The Hybertsen-Louie GPP model..."? (Author judgment only.)

2. **`docs/FREQ_INTEGRATION_REWRITE_PLAN.md`**: if the `freqint/` engine is
   intended to be revived later (sigma channel still needs it), archive is the right
   call but the author should confirm the engine was intentionally abandoned vs.
   merely deferred.

3. **`docs/NEW_WINDOW_MINIMAX_GUIDELINES.md` merge**: merging ~80 lines into
   `PHYSICS_COMPREHENSIVE.md` creates an editing task for agent 4 (who owns
   PHYSICS_COMPREHENSIVE). Flag as a dependency — do not merge independently.

---

## 6 — Summary: what moves where

| Action | Docs | Lines freed |
|---|---|---:|
| DELETE | `index.md`, `AGENT_TODO.md`, `PROFILING_SUGGESTIONS.md` | 692 |
| ARCHIVE | `PLAN_zeta_g_flat_migration.md`, `FREQ_INTEGRATION_REWRITE_PLAN.md`, `FREQ_INTEGRATION_PROGRESS.md`, `SIGMA_FREQ_AUDIT_STATUS.md`, `plans/phdf5_cray_mpich_migration.md`, `plans/unified_slab_io.md`, `advanced/jax_multihost.md` | 2,175 |
| RENAME (keep) | `GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` → `GN_PPM_MINIMAX_SIGMA_GUIDE.md` | 0 |
| MERGE INTO PHYSICS_COMPREHENSIVE | `NEW_WINDOW_MINIMAX_GUIDELINES.md` (condensed to ~80 lines) | ~400 |
| KEEP as-is | `archive/` (all), `advanced/HL_GPP_derivation.md`, `advanced/README.md` | — |
| **Net freed** | | **~3,267 lines** |

The entire slice (~3,350 lines before archive items) reduces to ~270 lines of
surviving content across two kept files plus one rename.
