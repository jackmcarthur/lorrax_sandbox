# Milestone A via the channel-blocked Dyson supermatrix

**Date:** 2026-06-16 · **Branch:** `lorrax_C agent/bispinor-supermatrix-w` (`cad7378`, off `main fc9984e`) · **Owner:** session-C

## Goal

Do milestone A (screened charge channel + bare Breit) **the supermatrix way** — build the full
channel-blocked `(δ − V·χ)` machinery now, with the transverse χ tiles set to **zero** — so that
milestone B is just "fill the χ tiles" and any failure there is isolated to χ, not the plumbing.

## What was built

- `gw/w_bispinor.py` (`24b889d`, ponytail-reviewed to 72 lines): `assemble_supermatrix` (4×4-Lorentz
  blocks → `(nq,N,N)`, `N=n_C+3·n_T`, missing blocks → zero), `solve_w_bispinor` (= reuse
  `w_isdf.solve_w` per-q LU on the supermatrix), `extract_blocks`.
- Driver wiring (`cad7378`): bispinor `do_screened` solves the supermatrix instead of the standalone
  charge `solve_w(V⁰⁰)`; `W⁰⁰ → Σ_SX/Σ_COH`, `W^{ij} → Σ^B`. `compute_sigma_x_bispinor` /
  `compute_cohsex_sigma` now take `w_ij_tiles` (the BispinorVqReader load is hoisted to `gw_jax`,
  also serving x_only). Milestone A passes `χ = {(0,0): χ₀}`.

## Validation

| Check | Result |
|---|---|
| Unit: assemble/extract round-trip + χ⁰⁰-only reduction (W⁰⁰ screened, W^{ij} bare, W^{0i}=0) | 2 passed |
| pytest (w_bispinor + sigma_x_bispinor + q_ibz_and_centroid_perm) | 16 passed |
| **E2E: bispinor screened COHSEX (MoS2 3×3, FORCE_FULL_BZ) vs analytic-A** | **sigma_diag BIT-IDENTICAL** |

Runs: `runs/MoS2/C_60Ry_bispinor_supermatrixA_2026-06-16/` (new code) vs
`runs/MoS2/C_60Ry_bispinor_cohsex_2026-06-15/` (analytic-A). The `W.exec` timing confirms the
`N=2644` supermatrix solve actually ran (3.86 s).

## Scope / limits

- **Full-BZ only** (`LORRAX_FORCE_FULL_BZ=1`). IBZ would need a per-channel unfold (charge n_C vs
  transverse n_T centroids unfold differently) — marked as the upgrade path, deferred.
- For A the wiring is numerically a no-op (W^{ij}=bare V^{ij}); its value is the validated B-foundation.
- FFI-IO is honored via `backend=config.backend.slab_io` (use_ffi_io derives from it).
- Not pushed (branch only).

## Next (Milestone B)

Un-trace χ at `w_isdf.py:181` (`'Rambn,Rambn->Rmn'` collapses the spin axes) → channel-resolved χ
tiles (charge + transverse + the charge–current cross blocks). Drop them into `chi_blocks`;
assembly / solve / extract are unchanged. The cross blocks make `W^{0i}≠0`, adding a new
charge–current term to Σ_c^B. Also pending: the 3:1 centroid ratio (640/200) keeps `N≈1240` vs the
2644 used here for the same-basis A comparison.
