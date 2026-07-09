# archive/ — superseded history of the GW refactor program

Nothing here is live. Fresh sessions start at `../HANDOFF.md`; current state lives in
`../STATUS.md` + `../NEXT_TARGETS.md`. Files below are kept as evidence/history only.

## Plans (executed or superseded)

| File | Why archived |
|---|---|
| `B4_ATTACK.md` | Executed plan — B4 unfold/sym consolidation attack; outcome folded into STATUS (step 3) and the "not worth doing" section of NEXT_TARGETS. |
| `VQ_CONSOLIDATION_AUDIT.md` | Executed audit — V_q consolidation verdict (Phase 3 CANCELLED, two live kernels); conclusion recorded in STATUS. |
| `ISDF_MOVE_PLAN.md` | Executed plan — isdf_fitting common/→gw/ move (c9fb0e2, 62ce45e); done, see STATUS step 4. |
| `ISDF_LIBRARY_PLAN.md` | Executed plan — `src/isdf/core.py` extraction (dfb6b90); done, see STATUS "New isdf/ mini-library". |
| `READER_CLEANUP_PLAN.md` | Executed plan — A1/B4 reader un-smear; step 2 done, step 4 (zeta merge) now UNBLOCKED and tracked in NEXT_TARGETS #7. |
| `OVERNIGHT_PLAN.md` | Executed plan — the 2026-07-08 overnight loop; outcome = MORNING_SUMMARY.md (kept live). |

## Audit catalogs (superseded by STATUS/NEXT_TARGETS)

| File | Why archived |
|---|---|
| `DEAD_CODE.md` | Initial-audit catalog (212 dead / 74 suspected-bug verdicts); actioned items recorded in STATUS; remainder is stale vs HEAD — re-grep before acting on any entry. |
| `FLAGS.md` | Initial-audit flag catalog (128 keys); superseded by the Σ_PPM config-seam work and the qp_solver toggle; re-grep before trusting. |
| `FEATURES.md` | Initial-audit feature catalog; orientation value superseded by MAP.md + HANDOFF.md. |
| `GATE_AUDIT.md` | Gate coverage audit + the gate-0 / MoS2-exact-agreement saga; gates since rebuilt (12 golden e2e); evidence for the ~62 meV 2D-COHSEX ceiling finding. |
| `BUGS_FOUND.md` | Session bug ledger through 2026-07-03; all entries fixed or triaged into NEXT_TARGETS. |
| `MAINTAINABILITY.md` (not present) | Lives at `reports/memplanner_cleanup_2026-07-02/MAINTAINABILITY.md`, not this dir. |
| `BUDGET_VALIDATION.md` (not present) | Lives at `reports/memplanner_cleanup_2026-07-02/BUDGET_VALIDATION.md`, not this dir. |

## Session records / research notes

| File | Why archived |
|---|---|
| `report.md` | The original 2026-07-01 kickoff report; superseded by STATUS + HANDOFF. |
| `BISPINOR_E2E_2026-07-02.md` | Evidence for the first bispinor e2e verification (pre-gate, hand-verified run); the gate itself landed 3fc93b4. |
| `BGW_INVALID_POLE_RESEARCH.md` | Research note for ppm_invalid_mode. **Partially corrected during implementation — BGW mode 3 keeps BOTH static SEX and CH terms** (the note's "SX pole → 0" was wrong); see SIGMA_PPM_MAP §2C; authoritative refs in `reports/bgw_invalid_mode_refs_2026-07-08/`. |

## Scripts, logs, raw data (evidence for the above)

| File(s) | Why archived |
|---|---|
| `bgw_aligned.py`, `bgw_crosscheck.py`, `mos2_exact_compare.py` | Ad-hoc comparison scripts for the gate-0 / MoS2-exact-agreement investigation (GATE_AUDIT evidence). |
| `verify_*.sh` (7) | One-shot gate-verification wrappers for the isdf move / IBZ / bispinor steps; superseded by the pytest regression subset. |
| `pytest_full.log`, `verify_isdf_suite.log` | Suite-run evidence for the 2026-07-01..03 commits. |
| `_raw_sorts.json`, `_raw_verdicts.json` | Raw agent-audit output backing DEAD_CODE.md. |

## Run-debris directories (evidence for specific gates/freezes)

| Dir | Why archived |
|---|---|
| `e2e_rerun_nowfnqp/` | Evidence for the gate-0 write_qp_wfn_h5 crash fix + re-freeze. |
| `e2e_gate0_verify/` | Evidence for the gate-0 verification rerun. |
| `gnppm_freeze/` | Evidence for the GN-PPM golden-reference freeze. |
| `ibz_gate/` | Evidence for the IBZ-vs-full-BZ equivalence gate (1479162). |
| `refreeze/` | Evidence for reference re-freezes during the Σ_PPM program. |
| `files/` | Misc collected outputs from the 2026-07-01 audit round. |
| `g0w0_sc_toggle_audit/` | Evidence for the qp_solver toggle audit (G0W0_SC_TOGGLE_DESIGN.md). |
| `g0w0_sc_toggle_impl/` | Implementation logs for the qp_solver toggle (RMS trajectories, single-jit check) — kept whole, logs inside. |
