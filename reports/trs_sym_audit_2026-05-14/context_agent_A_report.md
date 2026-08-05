# Context Agent A — Historical Cross-Check for ζ-Unfold Derivation

**Date**: 2026-05-14
**Role**: Read-only context provider for the math agent (`aa2a50a6be49ef63d`) deriving the ζ→V→ζ symmetry-transformation rule from first principles. Cross-checks against the existing TRS/sym audit corpus and the compare-skill conventions.

## 1. Compare-skill conventions (skills/compare/SKILL.md)

Re-read in full. Relevant for sym-vs-nosym Σ_X validation:

- **Only the total `Cor` is a valid cross-code comparison target.** `Σ⁺` vs `SX-X` and `Σ⁻` vs `CH` separately are NOT equivalent because LORRAX decomposes by band occupation while BGW decomposes via SX-X+CH (col 4 + col 10 = "Cor'") in `sigma_hp.log`. For pure Σ_X (no screening) this distinction is moot — `x_bare` in LORRAX's `sigma_freq_debug.dat` is the comparable quantity.
- BGW band indices are 1-indexed in `sigma_hp.log`; LORRAX's `sigma_freq_debug.dat` is 0-indexed. Physical band = LORRAX `n` + 1.
- `compare_bgw_gwjax.py` matches k-points **by crystal coords via WFN.h5** — this is what makes BGW symmetry-reduced grids comparable to LORRAX full-BZ. The skill says nothing about ζ-unfold conventions (out of scope); the sym handling lives in `WFNReader` / `SymMaps`, not in the parsers.
- There is no canonical entry in the compare skill for "expected µeV residual on sym vs nosym" — that gate (`≤ 1 meV`) is set by `sym_vs_nosym_pr3_validation.md` (Task #30) and is the load-bearing acceptance criterion for the current refactor.

## 2. Past TRS / sym reports — key findings

### `agent_1_scope_report.md` (2026-05-14 12:06) — enumerates 11 sym-table mismatch sites

The audit identified two distinct bug classes:
- **Class A** — consumer indexes a length-`ntran` table with a `[0, 2·ntran)` index → JAX silently clamps to last row, no exception. Hot path: `_unfold_v_q_ibz_to_full` (`v_q_tile.py:1513-1554`) reads `sym_perm` (length `ntran`) with `q_full_to_irr_sym` from `find_irreducible_qpoints` (values up to `2·ntran−1`). On MoS2 3×3 this produces the σ_h centroid perm instead of TR×identity at 4 TRS q's → 6.89 eV ΔΣ_X. **Already addressed by PR3**.
- **Class B** — table lookup succeeds but the TR-half entry is physically wrong: `U_spinor[ntran:]` is built by feeding `−S_spatial` through Markley's quaternion algorithm with a `det<0: R=−R` pre-flip, which collapses the TR half to the spatial spinor (missing the `iσ_y·conj` Wigner factor). Affects ψ-side. Site #6.

The audit also flagged the **explicit TODO at `symmetry_maps.py:548`** ("NOT SURE IF THESE SHOULD BE SYM_MATS_K OR SYM_MATS") inside `syms_crystal_to_cartesian` — author-flagged uncertainty in the very function that builds `R_cart` feeding `U_spinor`. This TODO turned out to be the root of TWO independent bugs (see §3 below).

### `algebraic_unfold_cri3.md` + `sym_vs_nosym_pr3_validation.md` — empirical evidence

- **MoS₂ 3×3 (E + σ_h)**: PR3 passes at **0.090 meV** max |ΔΣ_X|, 11× under the 1 meV gate. All 4 TRS k's behave indistinguishably from the 5 spatial k's (σ_h is involutive, `iσ_y` Wigner factor exercises but bispinor=false suppresses the Class-B propagation).
- **CrI₃ 6×6 30Ry SOC, charge channel (bispinor=false)**: 6022 meV Σ_X failure. Hand-rolled ψ-unfold (LORRAX-convention U_spinor) exactly matches LORRAX (a≡b bit-equal to 0.0), but DISAGREES with nosym ground truth at C3/S6 rows with **0.82 unitary deviation, 0.71 cross-block leakage**. Identity and −I (both involutive) pass at ULP.
- **Si 4×4×4**: 160 077 meV Σ_X failure. Same R_cart bug as CrI3 (different impact magnitude).

## 3. Two distinct bugs — both load-bearing for CrI3

Per the discussion.md timeline (16:21 → 00:25 → 00:40 entries), the corpus has converged on **two independent root causes**, both currently latent on CrI3:

1. **R_cart bug** (`symmetry_maps.py:810`): `syms_crystal_to_cartesian` passes `self.sym_mats_k` instead of `self.sym_matrices` to the cartesian similarity. For CrI3 C3, this yields a NON-orthogonal R_cart (||R Rᵀ − I||∞ = 3.46, vs the canonical 120° rotation matrix). Markley's quaternion algorithm silently accepts non-orthogonal input and outputs `U_spinor[1] = diag(exp(±i 125.5°))` instead of `diag(exp(±i 60°))`. Confirmed across Si (b1→b2 closes 1.19→3.1e-6) and CrI3.
2. **Centroid-permutation direction bug** (`orbit_syms.py:308`): `compute_centroid_sym_perm` builds `Rinv = inv(S)` (i.e., builds the INVERSE permutation) but the V_q bilinear-unfold rule needs the FORWARD direction `r_{π(μ)} = S r_μ + τ`. Invisible for involutive ops (σ_h, −I) but bites C3/C3⁻¹/S6/S6⁻¹.

Both bugs cancel for MoS₂ (σ_h is involutive AND its `mtrx` is symmetric — `R_cart` happens to be orthogonal regardless of basis convention). Both fire on CrI3.

## 4. Math agent's derivation (`zeta_unfold_derivation.md`) — cross-check verdict

I read all 418 lines. The derivation is internally consistent and reaches the correct verdict:

- §3 derives `V_full[Sq, π_s(μ), π_s(ν)] = V_ibz[q, μ, ν]` with `π_s` defined by `r_{π_s(μ)} = S · r_μ + τ` (forward).
- §4.3-4.4 correctly identifies that LORRAX builds the INVERSE direction at `orbit_syms.py:308`.
- §4.6 correctly anticipates why prior "flip the direction" attempts failed: `compute_rgrid_sym_perm:450` uses the same `Rinv = inv(S)` and must be flipped atomically with `compute_centroid_sym_perm`.
- §B correctly establishes that umklapps on G drop out of the V_q bilinear (rotation invariance of v + sum-index relabel).

**The derivation is correct and complete on its stated scope (the centroid-direction bug).** It does NOT address the R_cart / U_spinor bug — and explicitly says so at §4 line 97 ("for non-SOC ρ_nn is band-diagonal and U cancels"). This is the gap the math agent should be aware of.

## 5. FLAG FOR MATH AGENT

Appended below to `discussion.md` (not duplicated here).
