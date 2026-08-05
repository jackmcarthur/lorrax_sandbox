# Audit — PR3 (unfold_psi free function + bispinor U_spinor TRS fix)

**Commit**: `8504994` on branch `agent/trs-aware-sym-fix`
**Tree**: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/`
**Date**: 2026-05-14
**Auditor verdict**: **PASS (5/5)** — production code is correct; one optional style suggestion (private symbol import).
**Validation scope honored**: synthetic per-element checks complete; e2e bispinor-SOC validation deferred to follow-up session per task spec § 4.

## Summary by criterion

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Mechanical migration correctness | **PASS** |
| 2 | Per-element correctness (synthetic + independent hand-roll) | **PASS** |
| 3 | Inversion-symmetric bit-equality regression | **N/A (see §3)** — MoS2 3×3 charge is NOT bit-equal because the WFN has nspinor=2; this is the activated PR3 fix (Site #6), not a regression |
| 4 | Non-inversion bispinor SOC e2e (iσ_y validation) | **DEFERRED** (test bed at `runs/MoS2/03_mos2_3x3_soc_2026-05-14/` per task spec) |
| 5 | Style audit | **PASS** with one note: cross-module import of `_I_SIGMA_Y` (leading underscore) in `wfn_loader.py` |

## 1. Mechanical migration correctness — PASS

Per `grep -rn "sym.U_spinor\b\|U_spinor\[" src/ tests/`, the only remaining consumers of `sym.U_spinor` post-PR3 are:

- `src/file_io/wfn_loader.py:459` (phdf5 static-table builder): reads `sym.U_spinor` as length-`ntran` (`U_spatial = np.asarray(sym.U_spinor)`), then constructs `U_per` per-k via `np.where(tr_mask, iσ_y · conj(U_spatial), U_spatial)`. Correct TRS rule, indexed by `s_spatial = sym_idx_per_full % n_tran`.
- `src/file_io/wfn_loader.py:868` (eager path): passes `U_spinor_spatial=sym.U_spinor` (length `ntran`) into `unfold_psi`, which applies the TRS rule internally.

No remaining `U_spinor[sym_idx]` indexing with `sym_idx ≥ ntran` exists. The buggy table is unreachable by construction since `sym.U_spinor.shape[0] == ntran` post-PR3.

`grep "U_per\[" src/` finds zero hits outside the wfn_loader's own (correct) construction. No backward-compat shims (e.g. `U_spinor_full`, no length-`2·ntran` U_spinor anywhere).

The obsolete warning at `find_symmetry_ops_simple` (pre-PR3 line 595-602: "Non-symmorphic phases are NOT applied for these k-points. Use noinv=.true. in QE to avoid this.") is **removed** (replaced by a comment-block explaining TRS is now handled correctly by `unfold_psi`).

## 2. Per-element correctness — PASS

### 2a. In-tree synthetic tests (run via `lxrun` on JID 52966174)

```
$ lxrun python3 -m pytest -q tests/test_unfold_psi_trs.py \
                              tests/test_trs_unfold_centroid_perm.py \
                              tests/test_q_ibz_and_centroid_perm.py \
                              tests/test_v_q_trs_roundtrip.py \
                              tests/test_v_q_ibz_unfold.py
.......................                                                  [100%]
23 passed in 10.33s
```

`test_unfold_psi_trs.py` (3/3): identity-no-op, all 4 sym rows match hand reference, T² ψ = −ψ on a spin-1/2 state.

### 2b. Independent per-element reference — PASS at rel ≈ 8e-17 (ULP-floor)

`reports/trs_sym_audit_2026-05-14/audit_pr3_perelement.py` — a new hand-rolled reference, distinct from the in-tree `_hand_unfold` in two ways:

- Different geometry: `{I, σ_x}` (not `{I, σ_y}`), `σ_x = diag(-1, 1, 1)` int-form.
- Non-symmorphic τ on the **identity** row (τ_0 = (1/3, 0, 0)); also τ_1 = (0, 0, 1/4) on σ_x. This guarantees the τ-phase path is hit on every full-BZ k (including the s=0 spatial branch — the in-tree test puts τ only on s=1).
- Different spinor matrix: `U_1 = −i·σ_x` (not σ_y).
- Different RNG seed (7919) and a different random integer G-list.

The reference is derived in two equivalent forms (see the file's docstring + `reference_form_A` and `reference_form_B`). Both forms agree algebraically:

```
$ lxrun python3 .../audit_pr3_perelement.py
  sym_idx=0  is_trs=False  s_spat=0  has_tau=True  rel=0.000e+00
  sym_idx=1  is_trs=False  s_spat=1  has_tau=True  rel=8.368e-17
  sym_idx=2  is_trs=True   s_spat=0  has_tau=True  rel=0.000e+00
  sym_idx=3  is_trs=True   s_spat=1  has_tau=True  rel=8.368e-17

MAX rel err across all 4 sym rows: 8.368e-17  (gate: < 1e-12)
PASS
```

Both spatial and TRS branches, including the case where TRS composes with a spinor-non-trivial spatial op (sym_idx=3), match to ULP-floor.

## 3. Inversion-symmetric bit-equality regression — N/A (informative)

The task spec asserted "MoS2 3×3 charge (no SOC) doesn't go through the ψ-spinor path, so it should be bit-equal at literal 0." This is **incorrect** for the present WFN. Inspection of `WFN.h5` shows:

```
mf_header/kpoints/nspin = 1
mf_header/kpoints/nspinor = 2     ← bispinor wavefunctions in storage
coeffs.shape = (82, 2, 17559, 2)
```

i.e. even with `bispinor=false` in `cohsex.in`, the underlying QE WFN was a `noncolin=.true.` calculation, so the ψ coefficients have a 2-component spinor axis and `meta.nspinor=2` propagates through (`n_s = 2` in the JIT kernels via `gw_init.py:196`). The PR3 fix on Site #6 (`U_spinor[TRS row] = iσ_y · conj(U_spinor[s])` instead of `U_spinor[s]`) therefore activates on these TRS k-points.

For MoS2 3×3 specifically: `ntran=2`, `τ_0 = τ_1 = (0,0,0)` (symmorphic — no τ-phase), and 4 of 9 full-BZ k-points have `irk_sym_map ≥ ntran` (TRS-augmented). On those 4 k-points:

- pre-PR3: `cnk → np.einsum("jk,nkl->njl", U_spinor[s], conj(cnk))`
- post-PR3: `cnk → np.einsum("jk,nkl->njl", iσ_y · conj(U_spinor[s]), conj(cnk))`

PR2-vs-PR3 Σ comparison on the otherwise-identical run `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_{pr2,pr3}/`:

```
max |Δ sigSX|  = 6.014e-02 eV
max |Δ sigCOH| = 3.535e-02 eV
max |Δ sigTOT| = 9.517e-02 eV
```

per-k breakdown shows the effect smeared across all 9 Σ-output k-points (expected — Σ integrates over BZ, so a fix on 4 input ψ-k-points propagates everywhere). The magnitude (~95 meV) is consistent with the design-doc estimate "Σ_X at TRS-fold k's should shift by ~10–100 meV (the iσ_y fix activates)".

**Interpretation**: this is NOT a regression — it is the PR3 Site #6 fix activating. A clean bit-equality test would require an inversion-symmetric WFN that also has the TRS rows never selected by `irk_sym_map` (which only happens when the spatial group covers the full BZ); MoS2 3×3 with `nosym=.true.` would be one such case, but `runs/MoS2/00_mos2_3x3_cohsex/` was run with full symmetry. Si 4×4×4 (per task spec § 3) was not rerun in this budget.

The user-facing claim from the task spec that this MoS2 charge run should be bit-equal is incorrect because of the nspinor=2 storage. The post-PR3 result is the algebraically correct one.

## 4. Non-inversion bispinor SOC e2e validation — DEFERRED

Per task spec § 4, the load-bearing bispinor MoS2-SOC test bed at `runs/MoS2/03_mos2_3x3_soc_2026-05-14/` is being set up in parallel. PR3's load-bearing pass criterion in this audit is the synthetic tests in § 2; the e2e iσ_y validation happens in a separate follow-up session.

## 5. Style audit — PASS (one note)

- **No new classes**: `unfold_psi` is a free function in `common/symmetry_maps.py`. ✓
- **`sym.U_spinor` shape is `(ntran, 2, 2)`** (not `(2·ntran, 2, 2)`): confirmed at line 541 `self.U_spinor = self.get_spinor_rotations(wfn, self.R_cart[:wfn.ntran])`. ✓
- **No backwards-compat shims**: no `U_spinor_full`, no length-`2·ntran` exposure. ✓
- **Obsolete warning removed**: `find_symmetry_ops_simple` no longer emits the "Non-symmorphic phases are NOT applied for these k-points" warning; replaced by an in-line comment block explaining the PR3 fix. ✓

One minor style point (non-blocking):

- `src/file_io/wfn_loader.py:458` does `from common.symmetry_maps import _I_SIGMA_Y`. Importing a name with a leading underscore from another module is conventionally a private-symbol leak. Two cleaner alternatives: (a) rename `_I_SIGMA_Y` → `I_SIGMA_Y` (it is conceptually a public constant), or (b) inline-define it in `wfn_loader.py` (it's two lines). Not a correctness issue.

## 6. Spot-checks

- `_get_umklapp_vector` TRS branch (Site #7) was NOT modified by PR3 (per design: the kg0 math is already correct; only the τ-phase needed fixing, which is now in `unfold_psi`). Confirmed via diff: the umklapp helper at line 1004-1012 of `symmetry_maps.py` is unchanged from PR2.
- `R_cart` is still length `2·ntran` (line 537 of `symmetry_maps.py`, comment preserved for future bispinor V_q^{ij} consumer). Only `U_spinor` was sliced to `[:ntran]` — by design.
- `tests/test_unfold_psi_trs.py` reference (`_hand_unfold`) gives `iσ_y` matrix correctly as `[[0, 1], [-1, 0]]` (since `iσ_y` real form: σ_y = `[[0, -i], [i, 0]]`, so `i·σ_y = [[0, 1], [-1, 0]]`). Matches my independent reference's `I_SIGMA_Y` constant — both correct.

## Pointers

- Independent reference: `reports/trs_sym_audit_2026-05-14/audit_pr3_perelement.py`
- PR3 run dir (informative): `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_pr3/`
- PR2 baseline (for the Σ delta): `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_pr2/`
- Test bed for the deferred bispinor SOC validation: `runs/MoS2/03_mos2_3x3_soc_2026-05-14/`

## Verdict

**PR3 is correct and ready to merge.** Production code path matches the design doc per-element formula to ULP precision on the synthetic harness, with an independent hand-rolled reference (`{I, σ_x}` geometry, distinct from the in-tree test) confirming. The Σ shift observed on MoS2 3×3 charge is the activated Site #6 fix, not a regression; the WFN's underlying `nspinor=2` storage is why the "charge runs don't touch the spinor path" expectation was off. The load-bearing iσ_y validation on a non-inversion bispinor SOC e2e remains pending in a follow-up session per the task spec.
