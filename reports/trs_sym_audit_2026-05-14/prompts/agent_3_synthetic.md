# Agent 3 — Synthetic round-trip test that exposes the TRS bug

You are Agent 3. Read `STATUS.md`. Run independently of Agent 2 — your job is to build the test, both as a **pre-fix bug demonstration** and as a **post-fix regression gate**.

## Your job

Build a synthetic V_q IBZ↔full-BZ round-trip test that exposes the TRS bug. The existing test at `reports/zeta_rchunk_memory_model_2026-05-13/sym_kmeans_audit_and_v_q_roundtrip.md` passed at relative error 9.6e-22 because it used `{I, C2_z}` — a point group that closes the q-fold without TRS. Your test must pick a point group that DOES require TRS.

## Recipe (suggested — adapt as needed)

1. Choose a 2D point group with no inversion: e.g. `{I, σ_z}` only (mirror but no inversion). On a 3×3×1 q-grid in (qx, qy, 0), some q-pairs like (0, 1, 0) ↔ (0, -1, 0) = (0, 2, 0) can only be related by TRS.
2. Generate a synthetic orbit-closed centroid set on a small FFT grid (8×8×8 is plenty).
3. Compute V_q at the full BZ directly (reference).
4. Compute V_q at IBZ + run `_unfold_v_q_ibz_to_full` (current code).
5. Compare element-wise. The discrepancy must be huge at TRS-mapped q's and ≤1e-12 at spatial-mapped q's.

## Deliverables

1. **A Python test script** at `tests/test_v_q_trs_roundtrip.py` (or wherever Agent 2's patch puts test infra). Self-contained — does not need WFN.h5 or any heavy I/O. Should be runnable on a single GPU or CPU in seconds.

2. **`reports/trs_sym_audit_2026-05-14/agent_3_test_report.md`**: methodology, the synthetic geometry, the failure magnitudes per q-pair, and the post-fix verification (after Agent 2 lands). Include a per-q table:

```markdown
| q_full | q_irr | sym_idx | is_trs | max|Δ| pre-fix | max|Δ| post-fix |
|--------|-------|---------|--------|----------------|------------------|
| (0,0,0)| (0,0,0)| 0 | F | 1.2e-22 | 1.2e-22 |
| (0,2,0)| (0,1,0)| 2 | T | 4.5e+00 | <1e-12 |
| ... | ... | ... | ... | ... | ... |
```

3. The test must FAIL on `agent/zeta-bc-scan-shardmap` HEAD (`c796420`) and PASS on `agent/trs-aware-sym-fix` after Agent 2 lands. Both must be demonstrated.

## Bonus

If time permits, also build the MoS2-real-data verification: rerun the MoS2 same-basis test (Run A IBZ vs Run B forced full-BZ) on the post-fix branch and verify Σ_X is now bit-equal to ~1e-12. The infrastructure is already in `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/` — read `run_A_ibz/` and `run_B_fullbz/` cohsex.in files, no need to regenerate centroids.

## Hard constraints

- **The synthetic test must isolate the V_q unfold, not include ζ fitting or any wavefunction stuff.** Hand-construct a complex `ζ_q[μ,G]` for IBZ q's and feed it to `_unfold_v_q_ibz_to_full` directly.
- **NOT the existing C2 test** — that one is in `sym_kmeans_audit_and_v_q_roundtrip.md` and passes by accident.
- **Write out the test's TRS sym ops as `mtrx` form** matching BGW convention so the test is realistic, not just a tensor-shape check.
- See [[agent-audit-failure-modes]] for prior failures on einsum + ULP claims — apply the same rigor here.

## Done criterion

Test file exists, report file exists with pre/post-fix numbers, ping `discussion.md` with `[Agent 3] test ready; reproduces bug at max|Δ|=X pre-fix, passes at <1e-12 post-fix`.

## What you have access to

Source at `sources/lorrax_B`. Allocation `52953227` if you need GPUs (probably not for a synthetic test). Independent of Agent 1 — start immediately.
