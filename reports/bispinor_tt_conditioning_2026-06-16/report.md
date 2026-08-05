# Bispinor transverse V_q is numerically ill-conditioned (CrI3 C3 gate fallout)

**Date:** 2026-06-16 · **Branch:** lorrax_C `agent/bispinor-ibz-lorentz-unfold` (no source edits — diagnosis only)
**TL;DR:** The CrI3 C3 IBZ-vs-full-BZ gate "failed" on the transverse (Σ^B) block. Root-causing shows
this is **NOT an IBZ-unfold bug**. The bare transverse (Breit) **V_q itself is numerically
ill-conditioned** — its magnitude diverges with transverse-centroid count and its in-plane (x,y)
Lorentz block loses C3-covariance. Charge and the z-channel are exact. The IBZ unfold (`c546c74`)
is exonerated.

## How we got here
The CrI3 C3 gate (`runs/CrI3/C_cri3_{ibz_active,full_bz_ref}_2026-06-16`, 6×6 hex, 300 charge / 102
transverse cent, x_only bare Σ^B) reported: **CC charge unfold correct (sigma_diag 1 ULP); TT
transverse Σ^B differs ~2 eV in-plane, z-z exact** (gw_xonly.out lines ~634/836). MoS2 never caught
this — `milestone_a` makes sigma_diag CC-only, so TT was never compared there.

## Localization (scripts in this dir, run via CPU `lxrun`)
1. **`_tt_unfold_diag.py`** — `SymMaps.R_proper` IS orthogonal for the C3 ops (|RRᵀ−I|∞=1.1e-7,
   = rounded-avec round-off); the `unfold_v_q_bispinor_lorentz` einsum (Rᵀ M R) is a valid
   orthogonal congruence (preserves the Lorentz trace). **The unfold helper math is sound.**
2. **`_tt_convention_probe.py` / `_tt_findR.py`** — using the gauge-clean centroid-trace
   M[q]_{ij}=Σ_μ V^{ij}(q)[μ,μ] aligned to sym order by integer q-vector: the **full-BZ-direct**
   transverse in-plane data is itself NOT C3-covariant (a rotation can't change a trace/eigenvalues,
   yet they vary within an orbit). z-z and charge are covariant to 1e-8.
3. **`_tt_covar_control.py` / `_tt_verify_scaling.py`** — the discriminator: vary transverse
   centroid count, measure in-plane covariance + magnitude.

## The verdict (two independent scripts + raw-tile check agree)
Full-BZ-direct, **charge fixed at 300**, transverse varied:

| transverse cent | charge-tr spread | z-z spread | in-plane eig-spread | max\|in-plane trace\| |
|---|---|---|---|---|
| 102 | 2.7e-8 | 1.4e-8 | 0.186 | 2.4e6 |
| 206 | 2.7e-8 | 9.5e-8 | 0.255 | 3.5e8 |
| 308 | 2.7e-8 | 1.8e-7 | 0.076 | 4.3e9 |
| 410 | 2.7e-8 | 1.3e-7 | **2.49** | **9.5e13** |

Raw max\|tile\| (102 → 410): **TT_22 (y-y) 4.3e4 → 4.1e12 (~1e8×)**, TT_11 (x-x) ~1000×, TT_33 (z-z)
~1000×, TT_12 ~1e5×; CC unchanged (same 300 cent). The in-plane non-covariance does **not** converge
and the magnitude **diverges**, worst and **asymmetrically** in y-y. This is the full-BZ-direct data,
so the IBZ unfold is not implicated.

## Interpretation
The transverse (Breit) ζ̃/V_q construction is numerically ill-conditioned, worsening with basis
size — consistent with the **indefinite transverse CCT** (`project_bispinor_isdf`: "μ_L=i CCT
indefinite, LU not Cholesky"). The ζ̃ solve amplifies spurious in-plane components. The **x/y
asymmetry** (y-y ≫ x-x) breaks the hexagonal x↔y equivalence → a *sharper localized* defect, not
uniform conditioning; the γ̃²=σ_y channel (purely imaginary) is the prime suspect for a
conjugation/phase issue. Production runs (102–200 cent) sit in a "least-bad/accidentally-small"
regime (transverse V_q ~100× < charge at 102 → Σ^B≈−0.15 eV looks plausible) but are **not
converged** and degrade fast with basis size.

## Recommended next steps
1. **Debug the transverse ζ̃ fit conditioning** — inspect the indefinite-CCT solve (regularization,
   pivoting, condition number vs centroid count); pin why **y-y** diverges 1e5× more than x-x
   (start at the γ̃²=σ_y pair-density / conjugation path).
2. Decide whether bare Σ^B at production counts is trustworthy (its accidental smallness, not
   convergence, is what made Milestone-A look fine).
3. Only after the transverse reference is stable+covariant does the TT IBZ-unfold validation become
   meaningful (the unfold itself is correct; a separate Rᵀ M R vs R M R^T transpose question is moot
   until then).

## Side issue (non-blocking)
All forced-full-BZ x_only runs crash AFTER v_q write at `gw_output.py:288`
(`e_dft_ev_full[irr_idx]` IndexError 36 vs 36) → no sigma_diag under FORCE_FULL_BZ. One-line clamp/skip.

## Artifacts
- Runs: `runs/CrI3/C_cri3_full_bz_ref_2026-06-16/` (102) + `C_cri3_covar_t{204,306,408}_2026-06-16/`
  (206/308/410) — full-BZ-direct, tmp/v_q_bispinor.h5 with all 6 TT tiles + CC.
- Scripts: this dir (`_tt_*.py`). Memory: `project_bispinor_tt_noncovariance`.

---

## CLEAN RE-SWEEP (2026-06-16, identical methodology) — supersedes the muddied table above

The prior sweep above was muddied: the 410 leg required a per-run memory tweak
(`memory_per_device_gb`: 36 → 26) because its peak exceeded budget, so it was NOT run with
identical settings. **This clean re-sweep holds EVERYTHING identical** across all counts:

- **kmeans:** `LORRAX_NGPU=1 ... centroid.kmeans_cli <N> --orbit --density-mode current --seed 42`,
  oversample 1.5, same 1 GPU, from inside each run dir — for every count.
- **gw:** full-BZ-direct (`LORRAX_FORCE_FULL_BZ=1`), x_only, **4×A100-40GB**, **budget 36 GB**,
  charge fixed at the same 300 set. Verified identical: every run reports
  `Devices: 4 · Mesh 2×2 · full BZ (36 q-points) · Memory estimate peak 34.92 GB (budget 36.00)` —
  peak is bottleneck=fft, set by the fixed 300 charge centroids, so it is identical regardless of
  transverse N. No per-run tweak. Charge-tr & z-z spread = ~1e-8 in every run (runs are valid).

| transverse cent (target) | in-plane MAX eig-spread | median\|TT_22\| | max\|TT_22\| | charge-tr / z-z spread |
|---|---|---|---|---|
| 102 (baseline ref) | **0.186** | 8.9e2 | 4.3e4 | 1.1e-8 / 1.4e-8 |
| 152 (150) | **0.072** | 9.6e3 | 1.7e6 | 1.1e-8 / 1.7e-8 |
| 212 (210) | **0.332** | 5.2e4 | 1.2e7 | 1.1e-8 / 6.3e-8 |
| 266 (264) | **0.255** | 1.4e5 | 7.7e7 | 1.1e-8 / 4.6e-8 |
| 320 (318) | **0.086** | 2.65e5 | 5.6e7 | 1.1e-8 / 1.1e-7 |

### Per-orbit in-plane eigenvalue-spread (the covariance metric), per count
(orbit id | n_q in orbit | spread; orbit 7 is the q=Γ-type 2-member orbit, always ~0)

- **102:** o1 0.186 · o2 0.102 · o3 0.084 · o4 0.113 · o5 0.073 · o6 0.072 · o7 2e-10
- **152:** o1 0.072 · o2 0.032 · o3 0.027 · o4 0.035 · o5 0.021 · o6 0.018 · o7 4e-9
- **212:** o1 0.332 · o2 0.225 · o3 0.210 · o4 0.174 · o5 0.141 · o6 0.122 · o7 4e-9
- **266:** o1 0.202 · o2 0.190 · o3 0.255 · o4 0.212 · o5 0.226 · o6 0.222 · o7 1.5e-8
- **320:** o1 0.055 · o2 0.021 · o3 0.039 · o4 0.086 · o5 0.057 · o6 0.044 · o7 3e-8

### VERDICT
The in-plane covariance violation **does NOT decrease monotonically toward 0**. Under fully
identical methodology it is **erratic**, bouncing in a **0.07–0.33 band** with no convergence trend:
102→0.186, 152→0.072, 212→0.332, 266→0.255, 320→0.086. The lowest values (152→0.072, 320→0.086)
are no better than the 102 baseline's 0.186 in any consistent way, and intermediate counts are
*worse*. There is no undersampling signature (which would be a clean monotone decay). The
domain-expert's "well-conditioned, shrinks with centroids" expectation is **not borne out**.

Simultaneously, the raw in-plane magnitude **inflates monotonically and steeply** with centroid
count: median|TT_22| 8.9e2 → 9.6e3 → 5.2e4 → 1.4e5 → 2.65e5 (≈**300× over 102→320**), max|TT_22|
4.3e4 → 5.6e7 (≈1300×). The charge (CC, fixed 300 set) is untouched. So the clean methodology
**reproduces the prior sweep's magnitude inflation and erratic in-plane non-covariance** — it was
not a methodology artifact. This is a **real residual** in the transverse (Breit) ζ̃/V_q
construction (consistent with the indefinite transverse CCT, `project_bispinor_isdf`), not basis
undersampling. The recommended next step (debug the transverse ζ̃-fit conditioning; chase the
y-y / γ̃²=σ_y asymmetry) stands.

- Clean runs: `runs/CrI3/C_cri3_sweep_t{150,210,264,318}_2026-06-16/` (full-BZ-direct,
  tmp/v_q_bispinor.h5, all 6 TT tiles + CC). Magnitude scan: `_tt22_magnitude.py`.

---

## CORRECTION (2026-06-16) — measure the physical Σ^B, not V_q tiles. Both verdicts above are WRONG.

Everything above measures **V_q ISDF tiles**, which are **gauge/normalization-dependent**. The
gauge-invariant physical quantity is the band-traced **Σ^B** (printed in every run's gw_xonly.out as
`Σ^B tile … tr Σ`, before the gw_output.py:288 crash). Reading THAT flips both conclusions:

**(1) The transverse self-energy is WELL-CONDITIONED and CONVERGENT.** Full-BZ-direct physical Σ^B
(charge fixed 300; transverse swept):

| transverse cent | xx | yy | zz | xy | xx−yy |
|---|---|---|---|---|---|
| 102 | −8.54 | −8.75 | −8.69 | −0.62 | 0.20 |
| 206 | −8.58 | −8.58 | −8.69 | −0.64 | 0.001 |
| 308 | −8.60 | −8.59 | −8.72 | −0.63 | 0.015 |
| 320 | −8.60 | −8.58 | −8.71 | −0.63 | 0.024 |

Stable, convergent, C3-covariant (xx≈yy), **more isotropic with more centroids**. The tile-level
"magnitude inflation (TT_22 4e4→4e12)" and "erratic in-plane eigenvalue spread (0.07–0.33)" are a
**gauge/normalization artifact that CANCELS in Σ^B**. The domain expert's prior was correct.

**(2) The real, narrower bug: the transverse in-plane IBZ-UNFOLD is wrong.** Same fixed 102 basis,
IBZ-unfold vs full-BZ-direct physical Σ^B:

| | xx | yy | zz | xy |
|---|---|---|---|---|
| IBZ-unfold | −10.60 | −10.61 | −8.69 | **+1.35** |
| full-BZ (truth) | −8.54 | −8.75 | −8.69 | **−0.62** |

z-z exact; in-plane wrong (trace inflated 23% −21.2 vs −17.3 + xy sign flip). So `c546c74`'s
`unfold_v_q_bispinor_lorentz` IS buggy for the in-plane transverse channels (the earlier
"exonerated" was based on the misleading tile metrics). R_proper is orthogonal and the einsum is a
valid congruence, so it's the **Lorentz-mix ↔ transverse current-ISDF-basis-gauge interaction** —
naive R⊗R on the Lorentz index is not the full symmetry action on ζ̃^i.

**Path forward:** (A) fix the transverse in-plane unfold (route ζ̃^i through one sym-action helper),
or (B) **pragmatic**: keep IBZ for charge (the expensive ~640-cent channel) and use full-BZ-direct
for the cheap ~200-cent transverse — sidesteps the unfold bug at small cost. **Lesson: validate
bispinor symmetry on the physical Σ^B, never on raw ISDF tiles.**

---

## FINAL: the unfold is correct; bug + FIX are at the indefinite-CCT solve (lorrax_C 9128728)

The CORRECTION above (full-BZ-direct as "truth") was itself wrong in-plane. The decisive metric is
the **gauge-invariant** `S(q)=Σ_ij‖V^{ij}_TT(q)‖²_F` — invariant under both the per-q ζ̃ basis unitary
AND the channel rotation R (RRᵀ=I ⇒ Σ‖RVRᵀ‖²=Σ‖V‖²), so it MUST be orbit-constant for any correct
tile set. Findings (`_vq_gaugeinv.py`, `_step3b_Mi_covar.py`, `_spinor_consistency.py`):
- **R⊗R unfold is correct**: IBZ-unfold tiles are S-orbit-constant to 1e-7.
- Loaded ψ carries U_spinor (n(child)=U n(parent)U† to 3e-18); transverse current/γ̃ Gram covariant
  (‖g_child−Rᵀg_parent R‖=1.1e-7 vs lab-frame 0.22); R_proper=SO(3) image of U_spinor (1e-7). No σ_y
  sign bug.
- **Real bug = the per-q indefinite-CCT solve** `_ridge_indef_solve`. Full-BZ-fit tiles have in-plane
  S-orbit-spread 6–18% (so even brute-force full-BZ is non-covariant in-plane). The transverse CCT is
  Hermitian-INDEFINITE; the TRS-paired in-plane near-null modes (|λ|~1e-7·σ_max, cond≈3e7) sit above
  the old fixed ridge (1e-12), so LU amplifies their sub-floor noise as 1/λ. A +ridge can't fix an
  indefinite matrix — at ε=1e-4 a small −λ crosses zero and **Σ^B_yy → −912 eV**.

**FIX:** relative-|λ| truncated-eigendecomposition pseudoinverse in `_ridge_indef_solve`
(drop `|λ|<RCOND_INDEF·|λ|max`, default 1e-5). Covariance-preserving; transverse `auto` routes here.

**Validation** (`runs/CrI3/C_cri3_{fix_validate,ibz_fix}_2026-06-16`, rcond=1e-5):

| metric | bug (1e-12 ridge) | FIX (rcond=1e-5) |
|---|---|---|
| IBZ Σ^B (xx,yy,zz) | (−10.60,−10.61,−8.69) | (−5.14,−5.13,−5.62) |
| full-BZ Σ^B (xx,yy,zz) | (−8.54,−8.75,−8.69) | (−5.32,−5.22,−5.62) |
| IBZ-vs-fullBZ in-plane gap | ~23% | **~3%** |
| xy term | sign-flipped (+1.35 vs −0.62) | consistent (−0.20 vs −0.07) |
| gauge-inv S(q) full-BZ in-plane | 6–18% | 0.3–1.5% |

The 23% anisotropy + xy sign-flip are gone; both isotropic; z exact. IBZ-unfold is exactly covariant
by construction (the trustworthy one). Residual ~3% = the CCT's ~1e-4 covariance floor (transverse
signal overlaps the near-null modes; `rcond=1e-4` over-truncates → 0). Knob `LORRAX_RCOND_INDEF`.
TODO: port the cuSolverMp getrf+getrs transverse branch to the PSD `(LᴴL+δ²I)` Cholesky-Tikhonov form
for huge transverse blocks. Scripts: `_vq_gaugeinv.py`, `_vq_gi_fix.py`, `_spinor_consistency.py`,
`_step3b_Mi_covar.py`, `_hand_unfold.py`.
