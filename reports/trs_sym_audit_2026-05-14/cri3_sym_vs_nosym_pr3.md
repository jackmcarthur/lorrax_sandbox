# CrI3 sym vs nosym PR3 Σ_X validation — CrI3 6×6×1 30 Ry SOC

**Date**: 2026-05-14
**Task**: #30 mirror — belt-and-suspenders check for PR3 sym handling on a second system with inversion (D3d-like point group)
**Source**: `lorrax_B` @ `agent/trs-aware-sym-fix` (PR3 commit `8504994` + `a45f039` + cleanup `69ab42c`)
**Run dir**: `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/`
**Companions**:
- MoS₂ test (PASS, 0.090 meV) — `reports/trs_sym_audit_2026-05-14/sym_vs_nosym_pr3_validation.md`
- Si test (FAIL, 160 eV — τ-phase bug) — `reports/trs_sym_audit_2026-05-14/si_sym_vs_nosym_pr3.md`

## Verdict

**FAIL.** max |ΔΣ_X(k, n)| = **6022.5 meV ≈ 6.02 eV** across 36 k-points × 84 bands = 3024 (k, n) pairs. Pass gate was ≤ 1 meV; observed residual is **6000× over threshold**.

This is **NOT a PR3 firing** (CrI3 has spatial inversion, so no TRS-fold rows). The residual exposes a **separate sym-handling bug in the IBZ→full-BZ V_q (or ζ) cascade for CrI3's symmorphic 6-op point group** (P-3 = E, 2 C3, I, 2 S6, all with τ ≡ 0).

The MoS₂ pass is therefore not enough on its own — PR1+PR2+PR3 hold for the simplest sym (E + σ_h) but a different cascade-related bug breaks on the C3 + inversion combination.

## What the test bed exercises

CrI3 monolayer, space group **P-3** (#147):
```
ntran = 6 (in sym WFN, with no_t_rev=true)
sym ops: {E, C3, C3², −I, S6, S6⁵}    (3 proper + 3 improper)
has_inversion = TRUE                   (mtrx op 3 = −I, det=−1, trace=−3)
τ (fractional translations) = 0 for ALL 6 ops (symmorphic)
```

Because `−I ∈ mtrx`, the spatial group already maps every q to −q. TRS does **not** need to fire in any V_q or ψ unfold path. Static-analysis prediction: PR3 (the iσ_y · conj patch and the τ-phase fix) is a strict no-op for CrI3. **This test confirms PR3 doesn't fire** — the bug exposed here is upstream of PR3, in the basic IBZ→full-BZ cascade machinery, and only fires for groups richer than E + σ_h.

## Test design

Two LORRAX cohsex runs on the SAME Phase-2 code, using the SAME orbit-closed 300-centroid file, differing only in the symmetry of the input WFN:

| run | WFN source | ntran | nrk | unfold path |
|-----|------------|-------|-----|-------------|
| `run_sym/`   | `M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5` | 6  | 48 | IBZ→full V_q cascade (8 IBZ q × 6 sym → 36 full) |
| `run_nosym/` | `qe_nosym/nscf/WFN.h5` (regenerated) | 1 | 36 | trivial: IBZ ≡ full BZ |

**cohsex.in** is identical between runs except for the WFN file. `x_only=true`, `do_screened=false`, `bispinor=false`, `bare_coulomb_cutoff=30.0` Ry, `nval=70`, `ncond=14`, `nband=84`.

### Cascade activation verified via gw.out diagnostics

```
SYM   gw.out:  Zeta output: shape (n_q_disk=8 of 6·6·1=36 full-BZ, n_rtot=243000, n_rmu=300)
                V_q g-flat [CC]: n_q_ibz=8, ... unfold=IBZ→full
NOSYM gw.out:  Zeta output: shape (n_q_disk=36 of 6·6·1=36 full-BZ, n_rtot=243000, n_rmu=300)
                V_q g-flat [CC]: n_q_ibz=36, ... unfold=IBZ→full (trivially identity)
```

ζ at q=0 is **bit-equal** between the two runs (`max|ζ|=1.478e+03, L2=3.310e+05` in both) — confirming the ISDF basis itself is consistent. The disagreement appears once we expand from IBZ → full.

## Test bed sanity checks

- **Same physical k-points**: SYM `qe/nscf/WFN.h5` and NOSYM `qe_nosym/nscf/WFN.h5` both list the same 36 unfolded k-points (the sym WFN has 12 additional "extra" k-pts with weight 1/72 from QE's `no_t_rev=true` BZ folding; LORRAX processes the first 36 = full-BZ standard set).
- **DFT eigenvalues bit-equal**: `max |ΔE_dft| = 0.001 meV` across all 36 k × 86 bands. The two WFNs differ only by the symmetry-folding metadata; the underlying ψ_k(G) sets describe the same DFT solution.
- **Same centroid set**: both runs use `centroids_frac_300.txt` — 300 orbit-closed real-space points (built from the sym WFN's 6 sym ops, 50 reps × 6 unfold = 300).

## Per-(k, n) Σ_X table

Comparison: `x_bare` column of `sigma_freq_debug.dat` between `run_sym/` and `run_nosym/`. Values in meV. Full table in `compare_sigma_x.log`.

```
TRS-fold k-points in sym WFN: {} (empty — inversion in mtrx ⇒ no TRS rows fire)

Summary (3024 pairs total, 36 k × 84 bands):
   max |Δx_bare|       = 3011.26 meV    (sym k=18 bands 60–61, valence top)
   max |Δsex_0|        = 3011.26 meV
   max |Δcoh_0|        = 0.00 meV       (do_screened=false branch)
   max |ΔΣ_X (total)|  = 6022.51 meV    (x_bare and sex_0 both equal x_bare here)
   mean ΔΣ_X (total)   = −2045.6 meV    (sym is SYSTEMATICALLY more negative)
   stddev              = 1430.0 meV

|ΔΣ_X| histogram:
   0.00 – 0.01 meV :   0
   0.01 – 0.10 meV :   2
   0.10 – 1.00 meV :   0
   1.00 – 10.0 meV :  16
  10.0  – 100  meV : 260
  100   – 1000 meV : 540
 1000   – 6000 meV : 2206
```

### Per-k breakdown (max|ΔΣ_X|, every k-point fails)

All 36 k-points show ~5 eV residuals. No "clean" k. Sample rows:

```
   ik   trs?    max|ΔΣ_X|     mean|ΔΣ_X|   max|Δx_bare|   N_bands
    0      -    5556.82       2135.78       2778.41        84
    1      -    5135.33       2086.25       2567.67        84
   18      -    6022.51       2090.89       3011.26        84
   30      -    5089.66       2088.41       2544.83        84
   35      -    5023.65       2077.14       2511.82        84
```

The residual is uniformly large across the BZ — not a single-k artifact. Spread by k is 4611 → 6022 meV, all roughly the same order.

### Top-15 worst rows

```
   ik   n   E_dft(eV)     x_bare_sym    x_bare_nosym    Δx_bare       ΔΣ_X(total)
   18   61    -6.7433     -16.289519    -13.278262    -3011.26       -6022.51
   18   60    -6.7433     -16.289519    -13.278262    -3011.26       -6022.51
   12   60    -6.7011     -15.945128    -13.123956    -2821.17       -5642.34
   12   61    -6.7011     -15.945128    -13.123956    -2821.17       -5642.34
   23   61    -6.7279     -16.392921    -13.581677    -2811.24       -5622.49
   23   60    -6.7279     -16.392921    -13.581677    -2811.24       -5622.49
   ...
```

Worst rows cluster at **valence-top bands 60–61 / 56–57 / 64–65** — i.e. the partially-filled d-bands near the gap that dominate exchange. Σ_X for semicore bands (b=0–4) is closer (~7–140 meV diff at k=0), but still much larger than the MoS₂ residuals.

`sex_0` and `x_bare` are identical per row (do_screened=false), so `ΔΣ_X(total) = 2 × Δx_bare`.

## Diagnostics — what this is NOT

| Hypothesis                                  | Evidence                                          | Status     |
|---------------------------------------------|---------------------------------------------------|------------|
| PR3 ψ-side TRS rotation bug firing          | CrI3 mtrx contains −I ⇒ 0 TRS-fold k-pts          | **Ruled out** |
| ISDF basis mismatch between runs            | ζ(q=0) max&L2 bit-equal between sym/nosym         | **Ruled out** |
| Centroid file incompatibility               | Same real-space points used in both runs          | **Ruled out** |
| WFN gauge difference at degenerate manifolds | DFT eigenvalues bit-equal to ULP; kin_ion bit-equal at k=0 | **Ruled out** |
| QE k-list difference (48 vs 36)             | LORRAX processes first 36 of sym WFN (matches nosym 36) | **Ruled out** |
| **IBZ→full V_q (or ζ) cascade unfold bug for C3 + improper sym ops** | sym uses cascade (n_q_disk=8 → 36 unfold), nosym is direct (n_q_disk=36); residual is uniform across BZ | **NEW BUG** |

## Comparison across systems

| System          | Sym ops in WFN | Inversion? | τ phases? | max \|ΔΣ_X\| | Verdict |
|-----------------|----------------|------------|-----------|----------------|---------|
| MoS₂ 3×3        | 2 (E + σ_h)    | no         | no        | **0.090 meV**  | PASS    |
| CrI3 6×6 30 Ry  | 6 (E, 2C3, −I, 2S6) | **yes** | no        | **6022 meV**   | **FAIL** |
| Si 4×4×4        | 48 (Fd-3m)     | yes        | **yes**   | 160,077 meV    | FAIL    |

Three independent systems, three different sym-content profiles. MoS₂ passes; CrI3 and Si fail with different root causes:
- **CrI3**: PR3 doesn't fire (no TRS rows) but the cascade UNFOLD for C3 + improper-ops is broken.
- **Si**: PR3 doesn't fire (inversion) but the τ-phase code path in `unfold_psi` is broken.

CrI3 is **symmorphic** (τ = 0 for all ops), so the Si τ-phase bug cannot explain the CrI3 failure. The CrI3 failure is a **third, distinct sym handling bug**, exposed only when the cascade unfolds via the C3 / S6 (det=−1) rotations.

## Pipeline timings

| step                                      | NGPU | wallclock |
|-------------------------------------------|------|-----------|
| nosym NSCF (4 GPUs)                       | 4    | 17 s      |
| nosym pw2bgw + wfn2hdf                    | 1    | ~75 s + 8 s |
| kmeans (reused from existing CrI3 testbed) | -    | -        |
| kin_ion.h5 (sym, nosym)                   | 1 each | ~7 s each |
| `run_sym/` cohsex                         | 2    | ~80 s     |
| `run_nosym/` cohsex                       | 2    | ~70 s     |
| compare (Python)                          | 0    | <1 s     |

Total GPU-min: ~8.

## Known transient quirks during run setup

1. **Pre-existing `qp_wfn` writer crash (`write_qp_wfn_h5: U shape (36, 84, 84) inconsistent with (nk=48, nb_active=84)`)**: Triggered by sym WFN's nrk=48 vs LORRAX's 36-point unfolded U_kmn array. Worked around by setting `write_wfn_h5 = false` in cohsex.in. **Should be fixed separately**: `qp_wfn.write_qp_wfn_h5` shape check needs to use `meta.nkpts_unfolded` (36), not `wfn.nkpts` (48).
2. **Pre-existing `write_results` IndexError (`index 36 out of bounds for axis 0 with size 36`)** at `gw_output.py:288, e_dft_ev_irr = e_dft_ev_full[irr_idx]`: Same root cause — sym WFN nrk=48 but irr_idx targets the 36-point unfolded set. Fires AFTER `sigma_freq_debug.dat` is written, so this validation is unaffected. **Should be fixed separately.**
3. **Stale `__pycache__` caused `ImportError: cannot import name 'unfold_v_q' from 'common.symmetry_maps'`** on a fresh run while the Si testbed agent was actively re-running on the same node. `rm -rf src/**/__pycache__` resolved it. (Documenting because multi-agent contention with `.pyc` files is a real pattern when several lorrax_B agents share `/global/u2/j/jackm/software/lorrax_B`.)

## Recommendation

CrI3 monolayer is now established as a load-bearing test bed for the **C3 + inversion cascade path**, complementary to MoS₂ (E + σ_h, sufficient gate) and Si (Fd-3m non-symmorphic τ phases). Any sym refactor must pass all three.

Priority order for the implied fixes:
1. Triage the IBZ→full V_q (or ζ) cascade for det = −1 improper sym ops. The systematic ~3 eV per-band valence offset (sym more negative than nosym) suggests a SIGN or CONJUGATE flip missing somewhere in the unfold path — possibly a missing `*` on ζ(−q+G) when the sym op carries a det = −1 factor.
2. Fix the qp_wfn writer (`U_kmn.shape` check) and `write_results` (`e_dft_ev_irr`) to use the unfolded k-count, not `wfn.nkpts` (which can be > unfolded count for BGW WFNs of inversion-containing systems).

## Files

- `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/sigma_freq_debug.dat` (sym, 36 k × 84 bands)
- `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_nosym/sigma_freq_debug.dat` (nosym, 36 k × 84 bands)
- `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/compare_sigma_x.py` — comparison script
- `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/compare_sigma_x.log` — full table
- `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/manifest.yaml`
