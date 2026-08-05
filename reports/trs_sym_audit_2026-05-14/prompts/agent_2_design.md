# Agent 2 — Convention design and patch

You are Agent 2. Read `STATUS.md` and **wait for Agent 1's scope report** before writing the patch. Working tree: `sources/lorrax_B`. Create a sub-branch `agent/trs-aware-sym-fix` before editing.

## Your job

Design ONE canonical TRS abstraction the whole codebase can use, then write the patch.

## Design options to weigh

**A — Extended sym table with `is_trs` tag**: replace the implicit `sym_mats_k = [spatial, -spatial]` convention with an explicit object that carries `(mtrx, tnp, is_trs)` per row. Every consumer that today reaches into `sym_mats_k` or `sym_matrices` gets the typed object instead. When `is_trs=True`, the `apply_to_wfn` helper conjugates; the `apply_to_vq` helper conjugates; the `apply_to_centroid` helper does nothing extra (TRS keeps r fixed).

**B — Spatial-only everywhere**: pass `sym_mats_k[:ntran]` into `find_irreducible_qpoints` (and the k-unfold). Larger IBZ (no TRS reduction); the V_q cascade speedup shrinks proportionally. Simpler code, but leaves performance on the table for non-inversion systems.

**C — Hybrid**: keep TRS in q-fold/k-fold, but make `compute_centroid_sym_perm` produce a `2·ntran`-row permutation table where the second half encodes `(spatial_perm, +1)` and the unfold helper applies `conj` when row index ≥ ntran. Smallest surface area; doesn't generalize the abstraction.

Pick one. Justify in the report. Bispinor consideration: TRS acts on spinor index too (σ_y K for SOC systems) — your design should not paint us into a corner there.

## Deliverables

1. `reports/trs_sym_audit_2026-05-14/agent_2_design.md` — the design choice, the math for each TRS-handling helper (V_q complex-conjugation rule, ψ TRS rule with spinor part, centroid identity rule), and a per-site change list keyed off Agent 1's table.

2. **A single commit on `agent/trs-aware-sym-fix`** implementing the design. Keep the diff focused: one commit for the abstraction, one commit per consumer site, so it's reviewable.

3. A unit test (any framework consistent with `tests/`) that demonstrates the helper functions handle TRS correctly on a 4×4×1 toy.

## Hard constraints

- **No drive-by refactoring.** Touch only sites Agent 1 flagged + the abstraction itself.
- **No backwards-compat shims.** If a callsite changes signature, change all callers in this commit.
- **No "fix" for `unfold_orbit_unique_with_id`** — both prior audits called its einsum a typo and were wrong (see [[agent-audit-failure-modes]] in memory). Verify by writing out the per-element formula before touching anything in `orbit_syms.py`.
- Run `pytest -q` from the lorrax_B root after each commit; do not commit if pytest fails.

## Done criterion

(1) design.md written, (2) branch `agent/trs-aware-sym-fix` has the patch, (3) pytest green, (4) ping `discussion.md` with `[Agent 2] design complete; patch on branch agent/trs-aware-sym-fix @ <commit>; ready for Agent 3 to validate`.

## What you have access to

Working tree at `sources/lorrax_B`. SLURM alloc `52953227` for any quick smoke checks. Agent 1's scope report (don't start until it lands).
