# LORRAX `ppm_invalid_mode` (zero vs 2ry) — validation against the BGW invalid-mode references

**Date**: 2026-07-08. **Scope**: runs + parsing + analysis only — **no LORRAX source was
modified** (shared checkout `lorrax_D`, other agents live on it). Companion to
[`report.md`](report.md) (the BGW three-mode reference set) — this executes its
"How to validate LORRAX" §, items 1–4, for the two modes wired today
(`zero` ↔ BGW `invalid_gpp_mode 0`, `2ry` ↔ BGW mode 2; `static_limit` still
NotImplemented). This file is also the scaffold the `static_limit` implementation
extends to a three-way comparison (see "Extending to static_limit" below).

## Runs

Two new LORRAX variants under `runs/Si/00_si_4x4x4_60band/`, reusing the run's
QE/WFN artifacts (WFN.h5, kih.dat symlinked from `qe/nscf`; centroids_frac_480.txt,
dipole.h5, kin_ion.h5 symlinked from `00_lorrax_cohsex`), per build_inputs variant
rules. Base input = `01_lorrax_gn_ppm/cohsex.in` (GN-PPM, 480 centroids, nband 60):

| dir | `ppm_invalid_mode` | BGW analogue |
|---|---|---|
| `03_lorrax_gnppm_invalidmode_zero` | `zero` (LORRAX default) | `01b_bgw_gn_mode0` |
| `03_lorrax_gnppm_invalidmode_2ry` | `2ry` | `01_bgw_gn_ppm` (mode 2) |

**The two cohsex.in differ in the single key `ppm_invalid_mode`** (verified by diff).
Deliberate deviations from the parent input (identical in both runs):

- `ncond = 8` (was 52): sigma window = bands 1–16 = exactly the BGW sigma window
  (`band_index 1..16`, 8 valence + 8 conduction). Band **sums** are unchanged
  (right/sum range is `(b1, b4)` = all 60 bands = BGW `number_bands`).
- `sigma_omega_min_ev = -13.0` (was −10): the deep valence bands 1–2 sit at
  Edft−Ef = −12.33 eV; the parent grid gave NaN `sig_c(Edft)` there, and bands 1–2
  are BGW's largest movers (+23 meV) — they must be covered.
- `sigma_regularization_ev = 0.35` (was 0.25): keeps the crossing-window minimax
  fit in the fast regime, see "operational note" below.

Executed on Perlmutter, 1×A100 each (`LORRAX_NGPU=1 lxrun`, jobs 55674298 →
killed when that allocation expired mid-fit → rerun to completion on 55674933).
LORRAX evaluates Σ on the full 4×4×4 BZ (64 k); BGW on the 8 irreducible k.
LORRAX k were grouped into the 8 BGW stars by DFT-eigenvalue fingerprint
(E_dft vs sigma_hp Eo over the 16 window bands; every one of the 64 k matched a
unique star, star sizes `[1, 8, 4, 6, 24, 12, 3, 6]` sum to 64).

**Operational note (why −15 eV failed)**: the first attempt used
`sigma_omega_min_ev = -15`, which puts the ω<E_F valence crossing window at
A_core = 2(|ω|max/Ry + 1.5ξ)/ξ ≈ 124. The shipped crossing tables stop at A=60
and the exact Remez fall-back, fast at A≈83 (the parent's regime), did not finish
in 7+ min of CPU spin at A≈124. Grid −13..+10 eV with ξ=0.35 eV gives A_core ≈ 77
(neg half) / 61 (pos half); the fit then completes (~4 min once; the minimax disk
cache makes the second run's fits instant — total 40 s). Kept as a practical
data-point for grid planning; killed-attempt log:
`03_lorrax_gnppm_invalidmode_zero/gw_attempt1_killed_alloc_expired.out`.

## Invalid-pole population (task item 2)

```
LORRAX (both runs):  GN invalid modes: 167092/14745600 (1.13%)   [64 q × 480² ISDF centroid pairs]
BGW GN reference:    186608/2162680 considered (G,G') pairs (8.63%)  [8 irr q, exact offline count]
```

- **Not vacuous**: 167k invalid ISDF poles, same order of magnitude as BGW's
  fraction (7.6× lower). Fractions are over *different pair spaces* (ISDF centroid
  pairs vs plane-wave pairs), so exact agreement was never expected.
- The count is **identical between the two runs** (the population is a property of
  the fit, not of the treatment), and bit-identical statics confirm the toggle is
  the only difference: max|Δ| of `x_bare`, `kin_ion`, `V_H` between the runs =
  **0.0 eV** exactly.
- Symmetry consistency: per-star spread of the delta ≤ **0.38 meV** over all
  (star, band).

## Delta table (task item 3): LORRAX Δ(2ry − zero) vs BGW Δ(mode2 − mode0), meV

LORRAX = star-mean of Δ`sig_c(Edft).Re` (compare-skill `parse_sigma_freq_debug_v2`);
BGW = ΔEqp0 ≡ ΔCor′ (from `mode_table_gn.dat`). Full per-star data:
`lorrax_mode_table.dat`; analysis script: `lorrax_mode_diff.py`.

| band | BGW mean | BGW max\|Δ\| | LORRAX mean | LORRAX max\|Δ\| | sign match |
|---|---|---|---|---|---|
| 1–2 (deep val) | **+22.74** | 39.10 | **+1.13** | 1.16 | ✓ |
| 3–4 (VBM) | +0.86 | 9.26 | +0.83 | 1.11 | ✓ |
| 5–8 (VBM manifold) | −2.5…−3.0 | 4.91 | +0.40…+0.47 | 0.65 | ✗ |
| 9–10 (CBM) | +6.29 | 15.06 | −0.45 | 0.66 | ✗ |
| 11–12 | +9.91 | 15.06 | −0.38 | 0.56 | ✗ |
| 13–14 | +3.91 | 13.15 | −0.35 | 0.61 | ✗ |
| 15–16 (window top) | **−3.33** | 39.58 | **−0.48** | 0.97 | ✓ |

Summary stats:

| | max\|Δ\| | mean\|Δ\| |
|---|---|---|
| BGW (m2−m0, 8 irr k × 16 b) | 39.58 meV | 8.47 meV |
| LORRAX (2ry−zero, 8 stars × 16 b) | **1.16 meV** | **0.56 meV** |

- Per-band-mean sign agreement 6/16; Pearson r (band means) = 0.32.
- Γ direct-gap (b9−b3) delta shift: BGW **+7.90 meV**, LORRAX **−1.05 meV**.
- LORRAX ΔEqp0 ≡ ΔΣc to 0.001 meV (statics cancel — mirrors BGW's ΔEqp0 == ΔCor′).

## Verdict (task item 4)

**Wiring: PASS. Quantitative BGW anchoring: FAIL (with an identified, physical
reason — not papered over).**

1. **The modes are wired correctly.** Both run, the toggle changes *only* the
   invalid-pole treatment (statics bit-identical, identical n_invalid), and the
   delta is none of the ref report's bug signatures ("zero, opposite sign at the
   principal movers, or ~eV-scale"): it is a finite, meV-scale, k-symmetric effect
   with the right sign at the three strongest BGW features — deep valence **up**
   (+1.1), window-top conduction **down** (−0.5), VBM ≈ unmoved (+0.8, and both
   codes agree VBM moves least among the big movers).
2. **Magnitude is ~15–34× below BGW** (mean 0.56 vs 8.5 meV; max 1.16 vs 39.6).
   The gross scale is roughly consistent with the smaller population
   (8.5 meV × 1.13/8.63 ≈ 1.1 meV ≈ LORRAX's largest band-mean), i.e. per-pole
   efficacy is comparable; there are simply ~7.6× fewer invalid poles, in
   different places.
3. **Band pattern disagrees in the mid-window**: BGW's CBM-manifold moves
   +4…+10 meV; LORRAX's moves −0.4 meV. Sign agreement is confined to the window
   edges (bands 1–4, 15–16). So the ISDF invalid-pole population does not sit on
   the same (pair, q) structure as BGW's plane-wave one — expected in kind
   (ref report §3 warned the populations differ by construction), but the
   mid-window sign flip means the BGW mode-delta tables **cannot serve as a
   quantitative acceptance band for LORRAX invalid-mode physics**; they remain a
   sign/order sanity anchor only at the window edges.

## One-level-deeper dig (task item 4b)

Script + output: `invalid_pole_weight_dig.py`, `invalid_pole_weight_dig.txt`.

- **BGW side (exact, q→0 of `01_bgw_gn_ppm/eps0mat.h5`)**: 8.84% of pairs invalid
  (reproduces the reference count); they carry **8.2% of Σ|W_c|** (mean |W_c| of an
  invalid pair ≈ 0.92× a valid one — the population is *not* weight-suppressed);
  **0/537 diagonal** pairs invalid; 166/559 wing (G=0 row/col) pairs invalid;
  only 11% of invalid pairs have both G-indices in the lowest-|G| quartile →
  BGW's invalid poles live on **off-diagonal, mid/high-G pairs at every q**,
  which is how they reach every band (deep valence and window-top hardest) with
  tens of meV.
- **LORRAX side**: per-q localization of the 167k invalid ISDF poles **could not
  be established offline** from existing artifacts. The only saved W snapshot
  (`01_lorrax_gn_ppm/w_copies_debug.h5`, April, q→0 only) stores **full W, not
  W_c** — the current pipeline subtracts bare V in-flight
  (`ppm_sigma.py: Wc0_q = W0_q − V_q` before `fit_gn_ppm_from_wc_pair`), and V_qμν
  is not on disk — so an offline refit on that file is convention-confounded
  (it produces a spurious ~92%-invalid figure at q0; do not quote it). Noted for
  the next source session: a one-line per-q breakdown of `n_invalid` in
  `build_ppm_from_w_pair` (or a Wc dump flag) would settle where LORRAX's invalid
  poles sit; the `static_limit` implementation needs per-pair W_c(0) = −2B/Ω
  anyway.
- **Physics reading**: in the ISDF product basis each pole is fitted on a
  centroid-pair matrix element that *averages many G-pairs*; averaging pulls
  marginally-negative ω̃² pairs back positive, shrinking the invalid population
  and relocating it relative to the plane-wave picture. The mid-window sign flip
  is consistent with LORRAX's surviving invalid poles coupling to bands with a
  different (μ,ν)-structure than BGW's mid-G pairs.

## Extending to static_limit (scaffold)

- Third run dir: `03_lorrax_gnppm_invalidmode_staticlimit` (same cohsex.in,
  `ppm_invalid_mode = static_limit`) once implemented; BGW anchor
  `01c_bgw_gn_mode3` (= BGW default, `01d`).
- `lorrax_mode_diff.py` extends directly: parse the third `sigma_freq_debug.dat`,
  add Δ(static_limit − zero) and Δ(static_limit − 2ry) columns against BGW's
  (3−0) and (3−2) tables (both in `mode_diff_summary.txt`).
- Acceptance realism (given this report): expect sign/edge-band agreement and
  ~population-scaled magnitudes, not the BGW meV values; the sharper *internal*
  acceptance is Δ(static_limit − 2ry) ≪ Δ(static_limit − zero) (BGW: 3.1 vs
  8.6 meV mean) and the analytic −½W_c0 term reproducing `2ry`'s deltas within
  ~×2 while never moving statics.

## Artifacts

| file | what |
|---|---|
| `lorrax_mode_diff.py` | delta-vs-delta analysis (parsers per compare skill §2c v2 + §2a) |
| `lorrax_mode_table.dat` | per-band + per-(star, band) delta table |
| `invalid_pole_weight_dig.py` / `.txt` | invalid-pole population/weight dig, both codes |
| `w_convention_check.py` | proof the April w_copies file stores full W (not W_c) |
| `runs/Si/00_si_4x4x4_60band/03_lorrax_gnppm_invalidmode_{zero,2ry}/` | the runs (gw.out, sigma_freq_debug.dat, eqp0/1.dat, sigma_mnk.h5) |

## Status

- [x] LORRAX `zero` vs `2ry` runs on the BGW-matched Si 4×4×4 window (item 5 of ref report closed)
- [x] Invalid fraction recorded: 1.13% (ISDF pairs) vs BGW 8.63% (plane-wave pairs) — non-vacuous
- [x] Delta-vs-delta table + verdict (wiring PASS; quantitative anchoring FAIL — population location differs)
- [ ] Per-q localization of LORRAX invalid poles (needs a 1-line per-q count print or Wc dump — source session)
- [ ] `static_limit` implementation + three-way comparison (scaffold above)
