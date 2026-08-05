# Algebraic ψ + ζ unfold audit — MoS₂ 3×3 SOC

**Date**: 2026-05-14
**Agent**: 5 (MoS₂ branch)
**Source**: `lorrax_B` @ `agent/trs-aware-sym-fix` commit `69ab42c`
**Test bed**: `runs/MoS2/06_sym_vs_nosym_pr3_2026-05-14/{run_sym, run_nosym}`
**Companion reports** (when they land): `algebraic_unfold_cri3.md`, `algebraic_unfold_si.md`

## Verdicts

| Test | Verdict | Worst case |
|---|---|---|
| **Test 1 — ψ unfold (a) LORRAX vs (b) hand-rolled** | **PASS, bit-equal (0.0)** | All 9 k-points × 82 bands × 2 spinors — exact agreement |
| **Test 1 — ψ unfold (b) hand-rolled vs (c) nosym** | **PASS (at independent-SCF noise floor)** | max ‖U U† − I‖∞ = 3.1×10⁻⁷ across degen subspaces; baseline ≈ 1×10⁻⁷ from 0.1 meV eigenvalue noise |
| **Test 2 — ζ unfold (b) hand-rolled vs (c) nosym** | **PASS (at noise floor; ratio 1.00×)** | max rel‖Δζ‖ = 1.696×10⁻⁴; baseline rel‖Δζ‖ at trivial IBZ q's = 1.696×10⁻⁴ → unfold adds zero error |
| **Test 3 — G-vector full-BZ k-list consistency** | **PASS** | All 9 k-points match nosym G-set elementwise |

**Bottom line for MoS₂ 3×3 SOC**: PR3's `unfold_psi` is mathematically correct at machine precision relative to the per-element design-doc formula, AND it produces wavefunctions that match the nosym ground truth within the SCF-convergence noise floor. The ζ unfold (G-flat conj path used inside `compute_vcoul` via the V_q bilinear) is correct: the only disagreement vs nosym is the same noise floor that exists at trivial IBZ q's — the unfold operation itself contributes zero additional error. **No bug on this test bed.**

## What the test bed exercises

MoS₂ monolayer 3×3×1, SOC, **D₃ₕ point group** but with QE writing `no_t_rev=true, nosym=false` → on disk:
- `ntran = 2`: identity E and σ_h (mirror z, leaves all k in the k_z=0 plane fixed)
- `sym_mats_k.shape = (4, 3, 3)`: [E, σ_h, −E, −σ_h] (last two are TRS-augmented)
- `translations = [(0,0,0), (0,0,0)]` (symmorphic τ=0)
- 9 full-BZ k-points unfolded from 9 (sym WFN stores full BZ since σ_h is trivial on in-plane k)
- 5 IBZ q-points from 9 full-BZ q: `q_irr_full_idx = [0, 1, 3, 4, 5]`
- Non-trivial unfolds: `q_full ∈ {2, 6, 7, 8}` all use `sym_idx_q = 2`, which is the TRS-augmented identity row (sym_mats_k[2] = −E). So **every non-trivial unfold here is pure time reversal** (q → −q via TRS, centroid perm identity, τ=0 phase, conjugation).

The k-side is similar: 4 of 9 k-points use `sym_idx_k = 2` (pure TRS).

This is the **simplest non-trivial sym corner**: only TRS rows fire, no τ-phase, no spatial rotation of G, no non-trivial centroid permutation. **If TRS handling is broken anywhere, this test bed will catch it.**

## Test 1 — ψ unfold algebraic check

For each (k_full, sym_idx, kbar) determined by `SymMaps`:

**(a) LORRAX**: `WfnLoader(SYM_WFN).load(bands=(0,82), k='full_bz')` — production unfold path through `_eager_build` → `common.symmetry_maps.unfold_psi`.

**(b) Hand-rolled** (no LORRAX imports for the math): read `psi_kbar` from sym WFN at `k = sym.irr_idx_k[k_full]`. For each (sym_idx, kbar) compute by hand:
```python
S_full = sym.sym_mats_k[sym_idx]       # full op including ± sign
G_rot = (S_full @ g_kbar.T).T          # rotated G on IBZ G-axis
tau = sym.translations[sym_idx % ntran]
phase = exp(-1j * (G_rot · tau))       # = 1 for τ=0
is_trs = sym_idx >= ntran
if is_trs:
    cnk = conj(psi_kbar)
    if phase: cnk *= phase[None, None, :]
    U_eff = I_SIGMA_Y @ conj(U_spinor[sym_idx % ntran])
else:
    if phase: psi_kbar *= phase
    U_eff = U_spinor[sym_idx]
cnk_full = einsum("ij,njg->nig", U_eff, cnk)
```
where `I_SIGMA_Y = [[0, 1], [-1, 0]]`.

**(c) nosym**: `WfnLoader(NOSYM_WFN).load(bands=(0,82), k='full_bz')` — direct read from nosym WFN (ntran=1, no unfold required, IBZ ≡ full BZ).

### (a) vs (b): bit-equal at every k_full

| k_full | sym_idx | is_trs | max ‖(a) − (b)‖∞ |
|---|---|---|---|
| 0 | 0 | F | **0.0** |
| 1 | 2 | T | **0.0** |
| 2 | 0 | F | **0.0** |
| 3 | 2 | T | **0.0** |
| 4 | 2 | T | **0.0** |
| 5 | 2 | T | **0.0** |
| 6 | 0 | F | **0.0** |
| 7 | 0 | F | **0.0** |
| 8 | 0 | F | **0.0** |

`LORRAX's `unfold_psi` matches the hand-rolled per-element formula EXACTLY at every floating-point digit, on every k_full, including the four TRS-folded k-points {1, 3, 4, 5}. The PR3 implementation in `src/common/symmetry_maps.py:254-361` is correct.

### (b) vs (c): degenerate-subspace unitary check

For each degenerate band group at k_full (gap < 5 mRy ≈ 68 meV — the noise tolerance setting), build the overlap matrix `U[m, n] = ⟨ψ_handroll_m | ψ_nosym_n⟩` over (spinor, G_aligned). Check `‖U @ Uᴴ − I‖∞ < tol`.

| k_full | sym_idx | trs | max ‖U Uᴴ − I‖∞ | worst group |
|---|---|---|---|---|
| 0 | 0 | F | 8.7×10⁻⁸ | [30, 34) |
| 1 | 2 | T | 1.6×10⁻⁷ | [46, 48) |
| 2 | 0 | F | 1.6×10⁻⁷ | [46, 48) |
| 3 | 2 | T | 1.6×10⁻⁷ | [46, 48) |
| 4 | 2 | T | **3.1×10⁻⁷** | [46, 48) |
| 5 | 2 | T | 1.6×10⁻⁷ | [46, 48) |
| 6 | 0 | F | 1.6×10⁻⁷ | [46, 48) |
| 7 | 0 | F | 1.6×10⁻⁷ | [46, 48) |
| 8 | 0 | F | **3.1×10⁻⁷** | [46, 48) |

Maximum unitary deviation = 3.1×10⁻⁷. This is **at the noise floor of independent SCFs**: the sym and nosym WFNs come from independent QE SCF runs that converged to 0.1 meV eigenvalue agreement; that propagates to ~10⁻⁷ wavefunction-overlap deviation per band pair.

**Localization**: the TRS rows (sym_idx=2) and spatial rows (sym_idx=0) give the SAME magnitude of residual — they're indistinguishable from the noise floor, which is the signature of correct unfold math.

## Test 2 — ζ unfold algebraic check

`zeta_q_G[q, μ, j]` on disk stores `Σ_r exp(-2πi (q + G_j)·r) · ζ_q(r, μ)` where `G_j = gvec_components[q, :, j]`.

Symmetry rule (per `reports/zeta_ibz_2026-05-11/report.md` eq. 1 + design doc TRS rule):

**Spatial** (sym_idx < ntran):
```
ζ_{Sq, π_s(μ)}(SG) = exp(-i (Sq + SG)·τ_s) · ζ_{q, μ}(G)
```

**TRS** (sym_idx ≥ ntran, sym_mats_k = -S for s_spat = sym_idx-ntran):
```
ζ_{-q, μ}(G) = ζ*_{q, μ}(-G)
```

For MoS₂ with τ=0 and all 4 non-trivial unfolds being pure TRS (sym_idx=2 ≡ -E), the rule reduces to:
```
ζ_{-q, μ}(-G_irr) = conj(ζ_{q, μ}(G_irr))     (identity centroid perm, identity τ phase)
```

### Baseline (sym vs nosym at trivial IBZ q's, no unfold)

| i_irr_disk | q_full | max ‖Δζ‖ | refmax | rel |
|---|---|---|---|---|
| 0 | 0 | 9.43×10⁻¹ | 8.75×10³ | 1.08×10⁻⁴ |
| 1 | 1 | 5.84×10⁻¹ | 4.57×10³ | **1.28×10⁻⁴** |
| 2 | 3 | 4.56×10⁻¹ | 6.63×10³ | **6.88×10⁻⁵** |
| 3 | 4 | 5.14×10⁻¹ | 3.95×10³ | **1.30×10⁻⁴** |
| 4 | 5 | 7.61×10⁻¹ | 4.49×10³ | **1.70×10⁻⁴** |

**Baseline max rel = 1.70×10⁻⁴.** This is set by independent-SCF wavefunction differences propagating through the ISDF least-squares fit — at trivial IBZ q's (sym ≡ identity), no unfold has fired and yet sym ζ and nosym ζ disagree at this level.

### Non-trivial unfolds (hand-rolled vs nosym at same q_full)

| q_full | i_irr_disk | sym_idx | trs | max ‖Δζ‖ | rel | (μ, j)_worst | refmax |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 2 | T | 5.84×10⁻¹ | **1.28×10⁻⁴** | (5, 0) | 4.57×10³ |
| 6 | 2 | 2 | T | 4.56×10⁻¹ | **6.88×10⁻⁵** | (321, 0) | 6.63×10³ |
| 7 | 4 | 2 | T | 7.61×10⁻¹ | **1.70×10⁻⁴** | (211, 0) | 4.49×10³ |
| 8 | 3 | 2 | T | 5.14×10⁻¹ | **1.30×10⁻⁴** | (211, 0) | 3.95×10³ |

**Comparison**: Each unfolded q_full uses the IBZ-disk q at index `irr_idx_q[q_full]`. The hand-roll vs nosym discrepancy at the unfolded q is **exactly equal** (to within `1.00×` ratio) to the baseline at the same i_irr_disk:

| irr_disk | baseline rel | unfold rel | ratio |
|---|---|---|---|
| 1 | 1.28×10⁻⁴ | 1.28×10⁻⁴ | **1.00×** |
| 2 | 6.88×10⁻⁵ | 6.88×10⁻⁵ | **1.00×** |
| 4 | 1.70×10⁻⁴ | 1.70×10⁻⁴ | **1.00×** |
| 3 | 1.30×10⁻⁴ | 1.30×10⁻⁴ | **1.00×** |

This is the **strongest possible evidence the unfold is correct**: it adds zero additional error beyond what's already there at the baseline. The unfold operation `out[μ, j] = conj(ζ_irr[μ, j])` with G-axis relabeling `G_target = -G_irr` exactly reproduces the nosym ζ at the corresponding q_full, modulo the unavoidable SCF noise that's already in the IBZ slab.

## Test 3 — G-vector full-BZ k-list consistency

`WfnLoader(SYM_WFN).gvecs(k='full_bz')[k_full]` is built by `sym._get_umklapp_vector` + `S_k @ G_kbar - kg0` (per `wfn_loader.py:297-316`). For each k_full, the G-set returned must equal the nosym WFN's stored G-list at the same k_full (modulo order).

Per-k_full set equality check across all 9 k-points: **PASS** at every k_full, including the TRS-folded {1, 3, 4, 5}.

## Localization of any "failure"

There is **no failure** on MoS₂ 3×3 SOC. The PR3 unfold (Site #5 + #6 + #7 fixes in `wfn_loader.py` + `symmetry_maps.py`) is bit-equal to the per-element design-doc formula AND produces ψ + ζ that match the nosym ground truth at the SCF noise floor.

**This corroborates the prior 22:13 end-to-end Σ_X gate** (`sym_vs_nosym_pr3_validation.md`, max ‖ΔΣ_X‖ = 0.090 meV) at the level of the individual unfold operations — the ψ unfold and the ζ-side bilinear are both correct.

**Cross-system context**:
- MoS₂ exercises ONLY the TRS-fold corner (sym_idx=2 = −E). This is the simplest non-trivial sym case (τ=0, identity centroid perm, identity G rotation).
- CrI3 (`cri3_sym_vs_nosym_pr3.md` 16:20 entry) **FAILED at 6022 meV** in the end-to-end Σ_X gate. That bug fires for the `C3 + improper rotation + −I` cascade, which MoS₂ doesn't reach. The CrI3 ψ unfold test (if landed) should localize whether the bug is in `unfold_psi` for proper-improper compound ops, or in the ζ G-axis rotation for `S G_irr` where S is a non-trivial rotation.
- Si (`si_sym_vs_nosym_pr3.md` 16:08 entry) **FAILED at 160 eV** and the bisect shows this is **pre-existing**, not PR3-introduced. Suspect: non-symmorphic τ-phase bra/ket asymmetry in `compute_vcoul`/`v_q_tile` (per the existing diagnosis). The Si ψ unfold test should isolate whether the τ-phase enters ψ correctly or whether the consumer drops it.

The MoS₂ pass alone is **not enough** to clear PR3 globally; CrI3 and Si test beds expose distinct, latent sym-handling bugs that fire on richer point-group + non-symmorphic systems. But for the specific PR3-target slice (TRS handling with bispinor + iσ_y · conj), MoS₂ here is a clean PASS at the algebraic level.

## Diagnostic artifacts

- Test 1 driver: `reports/trs_sym_audit_2026-05-14/agent_5_mos2/test_psi_unfold.py`
- Test 1 log: `reports/trs_sym_audit_2026-05-14/agent_5_mos2/test_psi_unfold.log`
- Test 2 driver: `reports/trs_sym_audit_2026-05-14/agent_5_mos2/test_zeta_unfold.py`
- Test 2 log: `reports/trs_sym_audit_2026-05-14/agent_5_mos2/test_zeta_unfold.log`

Resource cost: ~2 GPU-min on a single A100 (Test 1: 1 min, Test 2: 1 min — both dominated by host I/O of ~80MB WFN slabs and ~30MB ζ slabs).

## Notes on methodology

- All hand-rolled formulas are pure numpy and do NOT import `unfold_psi`, `unfold_v_q`, `_phdf5_unfold_kernel`, or any other LORRAX unfold routine. They are derived from first principles using the equations in `pr3_design.md` § "The correct rule" and `zeta_ibz_2026-05-11/report.md` § 2.2-2.3.
- Tests use the eager backend of `WfnLoader` and `ZetaLoader` to avoid sharding/jit complications; all comparisons are on host numpy buffers at c128.
- `JAX_ENABLE_X64=1` is set (without it the eager backend silently downcast c128 → c64 on the return path, polluting the bit-equality check).
- For Test 1's (b)-vs-(c) check, the degenerate-band tolerance is 5 mRy ≈ 68 meV — wide enough to capture nearly-degenerate states that mix gauge across independent SCFs (the 0.1 meV ε noise floor between sym and nosym SCFs means single-band orbital identification is unstable for gap < ~5 meV).
- All 4 non-trivial ζ unfolds for MoS₂ are pure TRS (sym_idx=2 = sym_mats_k[2] = −E). The spatial sym σ_h (sym_idx=1) does NOT appear in `sym_idx_q` because it leaves every in-plane k unchanged — so it never fires as a unfold operator. This is a structural property of MoS₂ 3×3 in 2D; richer systems (Si non-symmorphic, CrI3 C3+inversion) exercise different unfold rows.
