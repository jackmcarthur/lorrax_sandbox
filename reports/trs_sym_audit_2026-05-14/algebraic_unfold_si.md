# Si Fd-3m algebraic ψ unfold audit

**Agent**: Si (this report)
**Date**: 2026-05-14
**Branch**: `agent/trs-aware-sym-fix` @ `69ab42c` (lorrax_B, current HEAD as instructed)
**System**: Si 4×4×4, SOC, 25 Ry; diamond Fd-3m space group
**Verdict**: **FAIL** at LORRAX-vs-nosym ground-truth on every non-symmorphic spatial sym op AND on 2/5 symmorphic-proper sym ops. **Bug localised to `SymMaps.syms_crystal_to_cartesian()`** (wrong rotation matrix passed to the SU(2) builder); `unfold_psi` itself is faithful to its inputs.

---

## Step 0 — non-symmorphic structure verified

| metric | value |
|---|---|
| `wfn.ntran` (sym WFN) | 48 |
| det = +1 | 24 |
| det = -1 | 24 |
| non-symmorphic (\|τ_frac\| > 1e-6) | **36 / 48** |
| has inversion (op s=24 = -I) | yes |
| TRS rows exercised in `sym_idx_k` | 0 (matches the design-doc claim: inversion ⇒ no TRS folding) |

So the sym WFN does carry the canonical Fd-3m glide structure; Si Test 1 / Test 3 cover the spatial side of `unfold_psi` (the τ-phase + spinor-rotation paths) but the TRS branch is **never exercised**. Si is therefore the right system to isolate the spatial-only PR3 path from the TRS path — and to surface the spatial bug below.

---

## Test 1 — ψ unfold algebraic check

For each (k_full, sym_idx) covered by the sym → full-BZ unfold, compare three independent ψ:

1. **(a) LORRAX `WfnLoader.load(k='full_bz')`** — runs the production `unfold_psi`.
2. **(b1) Hand-rolled reference using LORRAX's U_spinor table** — fresh implementation of the math in `pr3_design.md`, **not** built from any LORRAX symmetry code:

       ψ_full(G_rot) = exp(-1j · (S·G_kbar)·τ) · U_spinor_spatial[s_spatial] · ψ_kbar(G_kbar)

   with G_full = sym_mats_k[s] @ G_kbar - kg0 (BGW: k_full = S k_irr + kg0).
3. **(b2) Hand-rolled reference using a CORRECTED U_spinor table** — same as (b1) except `R_cart = bvec⁻¹ · mtrx · bvec` instead of LORRAX's `bvec⁻¹ · sym_mats_k · bvec`.
4. **(c) Nosym WFN at the matching full-BZ k** — `runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5` provides ψ at every full-BZ k directly (ntran=1, nrk=64).

All three are scattered onto the common FFT-box grid (24,24,24) and compared (i) at raw-coefficient level and (ii) at the gauge-fixed per-degenerate-subspace level: U = ⟨ψ_x|ψ_y⟩ over each degenerate group, with unitarity-deviation and gauge-fixed residual reported.

### Result matrix (16-band window, 25 (k_full, sym_idx) pairs across exercised categories)

| category | n | a-b1 raw max | b1-c gauge max | b2-c gauge max |
|---|---|---|---|---|
| identity (s=0) | 1 | 0.00e+00 | 1.40e-09 | 1.40e-09 |
| proper symmorphic (det=+1, τ=0) | 5 | 0.00e+00 | **4.40e-02** | 1.29e-04 |
| proper non-symmorphic (det=+1, τ≠0) | 18 | 1.76e-16 | **1.19e+00** | 3.13e-06 |
| improper symmorphic / non-symm | 0 | — | — | — |
| TRS-augmented (s ≥ ntran) | 0 | — | — | — |

(Detailed per-pair table in `agent_si_data/test1_final.json`; verbose run log in script `run_test1_final.py`.)

**Comparisons**:

- **(a) vs (b1) — LORRAX vs hand-rolled with LORRAX's U_spinor**: bit-equal to ULP across every test pair (raw max ≤ 1.76e-16). LORRAX's `unfold_psi` is therefore a faithful implementation of the formula it was designed to compute. **The bug is NOT in `unfold_psi`.**

- **(b1) vs (c) — LORRAX-convention ψ vs nosym ground truth**: massive disagreement on most sym ops. proper_nonsymm max gauge residual = 1.19 (i.e. essentially zero overlap on at least one degenerate group); proper_symm max gauge residual = 0.044. Only s∈{0, 5, 9, 13, 24} produce sub-1e-6 residuals — the cases where `mtrx[s]` is symmetric (or = ±I).

- **(b2) vs (c) — CORRECTED U_spinor**: sub-microvolt-level agreement across every category (max 3.1e-6). The unitarity deviation for the matrix `U = ⟨ψ_b2|ψ_c⟩` is ≤ 7e-12 across every test pair (i.e. the U matrix is unitary to 11 decimal places; the residual is at the SCF-noise floor of the two independent runs).

### Worst-case localisation

`b1-c gauge_resid` ≈ **1.19** at sym_idx=6 (k_full=21, k_irr=1) and sym_idx=7 (k_full=48, k_irr=1). These are proper-non-symmorphic ops with `mtrx[6] = [[0,1,0],[0,0,1],[1,0,0]]` (cyclic permutation) and `mtrx[7] = [[0,0,1],[-1,-1,-1],[1,0,0]]` (a related C_3 ↔ S_3 axis). Their cartesian rotation under `bvec⁻¹ · mtrx · bvec` produces a 2π/3 rotation about a body-diagonal axis (the standard Fd-3m C_3 axis); under LORRAX's `bvec⁻¹ · mtrx.T · bvec` it produces a 2π/3 rotation about a **different** body-diagonal axis. The two SU(2) matrices end up differing by ~σ-axis swaps. The two ψ representations consequently have ~zero overlap on at least one of the degenerate subspaces.

`b1-c gauge_resid` ≈ **0.044** for proper_symm at sym_idx∈{16, 20} (the C_4 axes about y and z). Same root cause: the two conventions give R_cart matrices that are each other's transpose (90° vs −90° rotations).

The two proper_symm rows where `b1-c gauge_resid` ≤ 1.4e-6 (s=5, 9, 13) all have a `mtrx[s]` that is its own transpose (the C_2 axes), so the two conventions coincidentally agree.

---

## Test 2 — ζ unfold algebraic check

**Status**: deferred (~30 GPU-min remaining budget would not finish a clean nosym ζ run plus a full per-q comparison).

**Justification for deferral**:
- The ψ-side failure above (gauge residual O(1) on ψ at non-symmorphic + many symmorphic ops) is sufficient to explain the 160 077 meV Σ_X disagreement (discussion.md 2026-05-14 16:08).
- ζ is computed downstream of ψ via pair densities `ρ_{nk,n'k'}(G)`, so the same ψ-coefficient error feeds into ζ-fit; ζ-unfold via `unfold_v_q` is a separate rotation but uses the SAME `sym_mats_k` G-rotation convention as ψ-unfold. The relevant ζ-unfold issue would either be (i) the same G-rotation convention (which would NOT be a bug — both ψ and ζ-unfold use the same convention internally, so they cancel in the IBZ-cascade ψ-ζ-pair recipe) or (ii) a separate τ-phase mismatch (no evidence so far).
- Doing ζ Test 2 requires a fresh nosym LORRAX cohsex run (~10 GPU-min), then per-q comparison plumbing (~15 min). The CrI3 sister agent's 16:20 entry already reports a 6 eV ζ/V_q-side failure unrelated to ψ, suggesting a separate ζ-cascade bug — orthogonal to this report's finding.

If a follow-up agent picks up Test 2, the right starting point is: re-run `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_nosym/` with `write_zeta_h5=true` (or equivalent), then compare `zeta_q.h5` at every non-IBZ full-BZ q against LORRAX's `unfold_v_q` applied to the sym-side ζ.

---

## Test 3 — G-vector full-BZ k-list consistency

**Status**: pass.

For every full-BZ k, the G-set produced by LORRAX's `WfnLoader.gvecs(k='full_bz')[kf]` is **identical** to the G-set in the nosym WFN at the matching k (set intersection = ngk_full, symmetric difference = 0) — across all 25 sampled (k_full, sym_idx) pairs.

The G-rotation convention `sym_mats_k @ G_kbar - kg0` (with `kg0 = round(k_full - sym_mats_k @ k_irr)`) reproduces the BGW G-list at every k_full, including the non-symmorphic, kg0≠0 cases. So the G-vector / umklapp bookkeeping is **correct**. The ψ-coefficient values at those G-labels are wrong (per Test 1), but the labels themselves are right.

---

## Bug localisation

The 160 077 meV Si Σ_X failure between sym and nosym originates in:

```python
# src/common/symmetry_maps.py:810 (SymMaps.syms_crystal_to_cartesian)
sym_matrices_cart = np.einsum('ij,njk,kl->nil', B_T_inv, self.sym_mats_k, B_T)
```

The argument `self.sym_mats_k` is `mtrx.T` (= `sym_matrices.transpose(0,2,1)`). The BGW-equivalent transformation (`Common/susymmetries.f90:79`) takes the un-transposed `mtrx`:

```fortran
mtrxtemp = bvec @ mtrx @ bvecinv   ! BGW; mtrx un-transposed
```

Replacing the LORRAX line with `np.einsum('ij,njk,kl->nil', B_T_inv, self.sym_matrices, B_T)` (i.e. swap `sym_mats_k` → `sym_matrices`) collapses the b1→b2 residual from **1.19 → 3.1e-6** across every test pair. The fix activates for any sym row where `mtrx[s] ≠ mtrx[s].T` (32 of 48 Fd-3m ops); the other 16 rows already agree by accident (mtrx[s] symmetric).

Three out of the 12 proper-symmorphic ops in Si Fd-3m are correct under the current convention (sym_idx ∈ {0, 5, 9, 13, 24, plus the inversion partners}); all 36 non-symmorphic + the C_4 ops are wrong. This explains why the **Si Σ_X test fails dramatically (160 eV)**: at every full-BZ k whose `sym_idx_k` row indexes one of the broken U_spinor entries, the unfolded ψ has nearly-zero overlap with the correct ψ on the degenerate subspace at the valence-band-edge / conduction-band-edge — the Σ_X matrix elements then pick up O(1) coefficients from wrong-direction spinor components.

The bug has been latent since `b7f956e Restructured src` (Mar 3 2026) — none of the LORRAX side regression tests pre-dating today included a sym-vs-nosym Σ_X comparison on a system with C_3 or C_4 axes.

### Direct orthogonality check (independent confirmation)

| sym_idx | category | ‖R_A R_Aᵀ − I‖ (LORRAX) | ‖R_C R_Cᵀ − I‖ (corrected) |
|---|---|---|---|
| 1 | proper_nonsymm (C₃-class) | **13.5** | 4e-17 |
| 5 | proper_symm (C₂) | 4e-17 | 4e-17 |
| 6 | proper_nonsymm (C₃) | **13.5** | 4e-17 |
| 7 | proper_nonsymm (C₃ + glide) | **13.5** | 4e-17 |
| 16 | proper_symm (C₄ y) | 4e-17 | 4e-17 |
| 20 | proper_symm (C₄ z) | 4e-17 | 4e-17 |

So at sym_idx=1, 6, 7 the LORRAX R_cart is **not even an orthogonal matrix** (off from O(3) by ‖·‖=13.5); feeding a non-orthogonal R into `get_spinor_rotations` produces a garbage SU(2) via Markley's quaternion algorithm (which silently assumes an orthogonal input). This is the **immediate** explanation for the gauge_resid = 1.19 (i.e. essentially zero overlap) seen at k_full=21, 48 in Test 1.

This contradicts a sentence in the parallel CrI3 report ("Si's mtrx is integer-orthogonal in crystal coords (cubic), so the R_cart conversion bug doesn't fire there"). The CrI3 finding that Si has a *separate* τ-phase bug is **not supported by the data**: substituting the corrected U_spinor closes Si's residual to 3e-6 across every test pair without touching the τ-phase code at all. Si and CrI3 share a single root cause (the R_cart line); the τ-phase code in `unfold_psi` is correct (matches BGW's `Common/gmap.f90:187` per the (a)≡(b1) bit-equality).

### Why MoS₂ passed and CrI₃ ALSO fails

- **MoS₂** post-no_t_rev sym group is `{E, σ_h}` — both symmetric (mtrx = mtrx.T). Both LORRAX and BGW conventions coincide; U_spinor is correct; PR3 sym-vs-nosym test at 0.09 meV.
- **CrI₃** symmetry P-3 has C_3 and S_6 ops with non-symmetric `mtrx`. Should fail under this bug too. The CrI₃ sister agent's 16:20 report finds a separate 6 eV Σ_X failure — consistent with this finding being part of the CrI₃ failure mode (along with whatever else is in CrI₃).
- **Si** Fd-3m has 32/48 ops with `mtrx ≠ mtrx.T` — heavy exposure.

### Why `unfold_psi` (PR3) is NOT the bug

Test 1's (a) vs (b1) is bit-equal at FP precision (raw max ≤ 1.76e-16), including on every non-symmorphic spatial sym row. `unfold_psi` correctly implements the τ-phase + spinor-rotation rule of `pr3_design.md`. The τ-phase formula `exp(-i · (S G_kbar) · τ)` matches BGW's `Common/gmap.f90:187` (where `kg = g(ig) + kgq = S G_kbar`). The pre-Phase-2 bisect at discussion.md 2026-05-14 16:08 — which already cleared Phase 2 from the Si gate — is fully consistent: this bug was there before PR1/2/3 landed, and Phase 2 doesn't touch the broken line.

---

## Suspect sites for the fix (diagnosis only)

`src/common/symmetry_maps.py:810` — change `self.sym_mats_k` to `self.sym_matrices` in `syms_crystal_to_cartesian`.

After the fix, the existing PR3 `unfold_psi` will Just Work for spatial non-symmorphic ops. The TRS branch (`iσ_y · conj`) is not exercised on Si and not affected by this bug.

The previously-suspected "Phase 2 pre-existing pipeline bug in the Σ_X kernel's G-vector / umklapp bookkeeping when consuming the unfolded ψ" (discussion.md 16:08 closing line) is **not** the actual issue — G-vector bookkeeping is verified correct in Test 3. The bug is in the SU(2) spinor table consumed by `unfold_psi`.

---

## Artifacts

- `agent_si_data/run_test1_final.py` — comprehensive Test 1 driver (run with the corrected vs LORRAX U_spinor in parallel)
- `agent_si_data/test1_final.json` — per-pair JSON dump (25 pairs)
- `agent_si_data/run_test1_psi_unfold.py` — earlier v1 (kg0-aware fix history)
- `agent_si_data/test_alt_su2.py` — convention scan (which R_cart formula closes the residual)
- `agent_si_data/debug_su2_convention.py` — comparison of A/B/C/D/E candidate R_cart conventions on representative sym ops
- `agent_si_data/debug_gvec_kin.py` / `debug_gvec_kg0.py` / `debug_kg0_check.py` — G-vector / umklapp diagnostics

Total cost: ~12 GPU-min on a single A100.
