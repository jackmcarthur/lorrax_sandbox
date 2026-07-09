# GW refactor map — feature catalog + gate audit

**Date:** 2026-07-01 · **Checkout:** `sources/lorrax_D` @ `agent/docs-tighten` (== main `e7b6c7d`) ·
**Status:** audit complete (read-only); no source modified.

## Goal
Produce the orientation artifacts for a major cleanup of `gw.gw_jax` and its dependencies: an
exhaustive, teleologically-sorted feature catalog + interaction map, a verified dead-code / suspected-bug
ledger, the complete input-flag surface, and an honest regression-gate coverage audit — so a
stage-by-stage refactor can proceed with a green gate per commit, and so future models can one-shot new
features in-style.

## Method
Agent fan-out over ~150 source files (67 reader groups → structured digest + per-file `files/*.md`),
each suspect adjudicated by a high-effort adversarial verifier (enumerate all call mechanisms before
"dead"; explicit index/sign math before "bug" — per memory `feedback_agent_audit_failure_modes`). In
parallel: a full flag reconciliation (`FLAGS.md`) and a GPU gate audit (`GATE_AUDIT.md`, run on 1×A100
via the `lorrax_agent` pool). Then two independent teleological sortings, merged by hand in the main
loop (the third sorter + auto-synthesis were dropped to save tokens — see Cost). 1049 verdicts,
128 flags, 250/255 unit tests green.

## Deliverables
| File | What |
|---|---|
| `MAP.md` | The spine: dataflow diagram, concern×stage matrix, category interaction cross-map, ranked refactor targets, attack order. **Start here.** |
| `FEATURES.md` | Exhaustive feature catalog by teleology tier; per feature: physics, files, flags, interactions, refactor note. The "load before adding a feature" doc. |
| `DEAD_CODE.md` | 74 suspected bugs (top), 212 confirmed-dead (delete pass), 48 test-only, 185 refactor targets — with evidence. |
| `FLAGS.md` | All 128 input flags: default, category, consumers, dead/undocumented/BGW-gotcha status. |
| `GATE_AUDIT.md` | Test inventory, GPU run, stage×axis coverage matrix, recommended golden gates. |
| `files/*.md` | 67 per-file function-level catalogs. |
| `_raw_verdicts.json`, `_raw_sorts.json` | Machine-readable verdicts + the two full taxonomies. |

## Key findings (top 10)
1. **The end-to-end gate is RED on main, and nobody noticed.** Two failures: (a) `write_qp_wfn_h5`
   crashes on *every IBZ run* (`debug.write_wfn_h5` default true, full-BZ U vs `wfn.nkpts`); (b)
   bypassing it, SX/COH have drifted a plateau-shaped **~3.5 meV** since the reference was frozen (VH
   bit-stable) — a W-path change that never got the reference re-frozen. **The gate is not being run.**
2. **Coverage is a single 2D/charge/IBZ/1-GPU COHSEX fixture.** Ungated end-to-end: all dynamic Σ,
   bispinor beyond bookkeeping, all 3D/truncation-off, head-off, self-consistency, all multi-GPU.
   `w_isdf.solve_w`, `sc_iteration`, `sigma_dispatch`, `compute_vcoul` slab truncation, `degen_average`
   have no dedicated test at all.
3. **Two concerns are smeared across the whole pipeline via parallel helpers**: the memory planner
   (C6 — 3 stacked planners in `gw_init`, plus copies in `v_q_tile`, `isdf_fitting`, `fft_helpers`) and
   IBZ symmetry (B4 — ≥6 parallel unfold helpers). These two drive the refactor.
4. **`gw_init.py`, `isdf_fitting.py`, `v_q_tile.py`, `wfn_loader.py` each host 3+ categories** — the
   ranked split targets (MAP.md §4). `isdf_fitting` is an entire pipeline stage living in `common/`.
5. **~2500 L of verified-dead code** deletable in one behavior-neutral commit: `psp/archive/*`,
   `cg_posdef`, `bispinor_init`, `centroid_io`, `vcoul` (mostly), `chi_sos`, `head_wing_schur`,
   `aot_memory_model/chooser` (dead-by-clobber) + ~180 dead functions.
6. **74 suspected bugs**, several live-wrong-answer: streamed-Σc silently skips the q→0 head; negative-Ω²
   head sign flip; stale `compute_vcoul` cache key; `wfn_writer` nspinor=1 occ factor-of-2 + `run_nscf`
   broken for nk>1; 0D Coulomb volume-convention split; `slate/eigh` eigenvectors (known/unresolved).
7. **Config has three parsers and a shadow env-flag surface.** cohsex.in parser copy-pasted ×3;
   `PPMSigmaRuntimeOptions` mirrors config; SC/V_q loop knobs live only in `LORRAX_*` env vars outside
   `_DEFAULTS`. `COHSEX_INPUT.md` is substantially stale (the canonical `compute_mode` axis is undocumented).
8. **QP/eqp/Z math is duplicated ~4×** across `eqp_bgw`/`gw_output`/`gw_jax`/`sigma_dispatch`.
9. **Bispinor screened-W is unbuilt** and the SC path silently drops Σ^B — the axis is A2/A3/A5 only.
10. **Environment drift**: base `lorrax_D` modulefile FFI paths point at purged $SCRATCH (logged to
    `KNOWN_SANDBOX_ERRORS.md`); container JAX 0.5.3 vs `pyproject` jax≥0.9 causes 2 of the 5 test failures.

## Proposed attack order
0. **Gate first (non-negotiable):** green the e2e (fix/gate-off the `write_qp_wfn_h5` crash), bisect &
   re-freeze the 3.5 meV drift, add gates #1 GN-PPM + #2 3D-COHSEX + #4 IBZ-vs-full-BZ (seeds exist under
   `runs/`; GATE_AUDIT §4). A refactor is only safe once each stage has a gate a wrong-but-plausible change fails.
1. **Delete pass** — Tier F, one commit, no logic moves (~2500 L).
2. **Single-source duplicates** — config parser ×3→1; eqp math ×4→1; zeta reader old/new→1; kill `load_wfns` facade.
3. **Extract the two smeared concerns** — C6 planner → one home; B4 symmetry → one `SymMaps` table + one
   sym-action helper (memory `feedback_unified_sym_action`).
4. **Stage-by-stage move** — one branch each, gate green per commit: `isdf_fitting`→`gw/`; split `v_q_tile`;
   de-physics `wfn_loader`; `w_isdf` quadrature→minimax engine.
5. **Cross-cutting last** — head-correction seams, streamed-Σc head injection, fold `LORRAX_*` env into config.

Suspected bugs are triaged separately from the refactor (fix-or-file before moving the owning code).

## Next steps
- Decide gate-0 scope with the owner; the `write_qp_wfn_h5` fix needs a feature branch (source change).
- CONVENTIONS section for `sources/lorrax/AGENTS.md` (bundle signatures, io_callback rule, unified sym
  action, no parallel old/new paths) — the norms already exist in memory; write them down as the one-page
  standard a model loads to stay in-style.
- Docs overhaul (`COHSEX_INPUT.md` refresh) can key off `FLAGS.md` + `FEATURES.md` directly.

## Cost note
~12.8M subagent tokens (three runs; the adversarial verify layer at high effort was the bulk). The final
synthesis was done in the main loop from on-disk artifacts — no additional fan-out. Re-running is
unnecessary: all raw material is durable under this directory.
