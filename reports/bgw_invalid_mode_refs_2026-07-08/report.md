# BGW invalid-GPP-pole mode references (Si 4×4×4) — validation targets for LORRAX `ppm_invalid_mode`

**Date**: 2026-07-08. **Purpose**: produce BerkeleyGW reference GW runs that exercise the three
invalid-PPM-pole handling modes (`invalid_gpp_mode` 0 / 2 / 3), so LORRAX's `ppm_invalid_mode`
feature (`zero` / `2ry` / `static_limit`) has a BGW-anchored physical target. BGW-reference
production + analysis only — **no LORRAX source was modified**.

Background: `reports/gw_refactor_map_2026-07-01/BGW_INVALID_POLE_RESEARCH.md` (semantics,
`mtxel_cor.f90` lines) and `SIGMA_PPM_MAP.md` §2C (LORRAX wiring).

## Summary

Two complete three-mode reference lines now exist on the Si 4×4×4 60-band run
(`runs/Si/00_si_4x4x4_60band/`), reusing its existing WFN + eps artifacts:

- **GN-GPP line** (`frequency_dependence 3` — the flavor LORRAX's GN-PPM mirrors):
  modes 0/2/3 + a default-keyword run. **8.63% of all considered (G,G′) pairs have invalid
  poles** (Re ω̃² < 0; exact offline count, BGW's own formula). Mode choice moves Eqp0 by up to
  **51 meV** (mean |Δ| ≈ 8.5 meV).
- **HL-GPP line** (`frequency_dependence 1`): modes 0/2/3. Mode choice moves Eqp0 by up to
  **104 meV** (mean |Δ| ≈ 32–37 meV).
- **Empirically verified: omitting `invalid_gpp_mode` (default −1) is bit-identical to mode 3**
  (`01d_bgw_gn_default/sigma_hp.log` == `01c_bgw_gn_mode3/sigma_hp.log`).
- The mutated legacy dir `02_bgw_hl_ppm` (sigma.inp lost) is bit-identical to the fresh
  `02d_bgw_hl_mode2` → it was a mode-2 run.

## Mode semantics (from `sources/BerkeleyGW/Sigma/mtxel_cor.f90:778-844`)

A pole is invalid when Re ω̃² < 0. HL: ω̃² = Ω²/I_ε (I_ε = δ_GG′ − ε⁻¹_GG′); GN:
ω̃² = |dFreqBrd(2)|²·I_ε(iω₂)/(I_ε(0) − I_ε(iω₂)).

| BGW `invalid_gpp_mode` | ω̃ set to | Σc effect | LORRAX `ppm_invalid_mode` |
|---|---|---|---|
| 0 | 0 | pole dropped (SX and CH terms → 0) | `zero`/`skip` (current LORRAX default) |
| 2 | 2 Ry | finite real pole, residue rebuilt | `2ry` |
| 3 = **default** (−1) | 1/TOL_ZERO ≈ ∞ | static-COHSEX treatment of that mode | `static_limit` (**NotImplemented** as of 2026-07-04 wiring) |
| 1 | i·√\|ω̃²\| | damped imaginary pole | unsupported |

## Run directories

All under `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/`. Within each line
**only `invalid_gpp_mode` differs** — same eps, WFN, cutoffs, k/band ranges (bands 1–16 = 8
valence + 8 conduction, 8 irreducible k, `screened_coulomb_cutoff 25.0`, `number_bands 60`).
Each sigma run: ~29 s on 4×A100 (job 55674176, 2026-07-08).

| dir | freq_dep | invalid_gpp_mode | provenance |
|---|---|---|---|
| `01b_bgw_gn_mode0` | 3 | 0 | **new** — eps reused from `01_bgw_gn_ppm` |
| `01_bgw_gn_ppm` | 3 | 2 | pre-existing (2026-04-05), the GN reference |
| `01c_bgw_gn_mode3` | 3 | 3 | **new** |
| `01d_bgw_gn_default` | 3 | *(omitted)* | **new** — proves default == 3 (bit-identical to 01c) |
| `02b_bgw_hl_mode0` | 1 | 0 | pre-existing (2026-04-28) — static eps from `00_bgw_cohsex` + RHO |
| `02d_bgw_hl_mode2` | 1 | 2 | **new** — reproduces mutated `02_bgw_hl_ppm` bit-identically |
| `02c_bgw_hl_mode3` | 1 | 3 | pre-existing (2026-04-28) |

## Invalid-pole counts (GN line, exact)

`count_invalid_gn_poles.py` (in the run dir) replicates `mtxel_cor.f90:805-844` on the stored
ε⁻¹(0), ε⁻¹(iω₂) matrices: I₁ = δ−ε⁻¹(0), I₂ = δ−ε⁻¹(iω₂); skip if both |I| < TOL_Small=1e-6;
invalid iff Re[I₂/(I₁−I₂)] < 0. The |dFreqBrd(2)|² prefactor is positive and the symmetry-unfold
phases cancel in the ratio, so per-irreducible-q counts are exactly BGW's counts for every
symmetry image. Full table: `invalid_pole_counts_gn.txt`.

| q (irr.) | nmtx | considered pairs | invalid | frac |
|---|---|---|---|---|
| q→0 (0.00025,0,0) | 537 | 251,108 | 22,188 | 8.84% |
| (0,0,¼) | 570 | 274,473 | 24,218 | 8.82% |
| (0,0,½) | 568 | 272,228 | 23,732 | 8.72% |
| (0,¼,¼) | 576 | 276,504 | 23,942 | 8.66% |
| (0,¼,½) | 562 | 267,913 | 23,062 | 8.61% |
| (0,¼,¾) | 574 | 274,942 | 23,270 | 8.46% |
| (0,½,½) | 588 | 281,296 | 23,908 | 8.50% |
| (¼,½,¾) | 560 | 264,216 | 22,288 | 8.44% |
| **TOTAL** | | **2,162,680** | **186,608** | **8.63%** |

The population is large and uniform across q — the comparison is decidedly **not vacuous**.
(HL-line counts were not computed offline — the HL Ω² needs the RHO sum-rule machinery
(`wpeff.f90`); presence there is established by the 30–104 meV mode-to-mode output deltas and the
identical branch code.)

## Mode-diff tables

Parsed with the compare-skill `parse_sigma_hp` (extended for the 11-column `freq_dep=1` layout —
see KNOWN_SANDBOX_ERRORS 2026-07-08). Full per-k/per-band data: `mode_table_gn.dat`,
`mode_table_hl.dat`; full stats: `mode_diff_summary.txt`. Since eps and everything else is fixed
within a line, ΔEqp0 == ΔCor′ == ΔΣc exactly — the deltas below are pure invalid-mode physics.

### GN-GPP line (freq_dep=3) — ΔEqp0 in meV over all 8 k × 16 bands

| mode pair | max \|Δ\| | mean \|Δ\| |
|---|---|---|
| 2 − 0 (`2ry` vs `zero`) | 39.6 | 8.5 |
| 3 − 0 (`static_limit` vs `zero`) | 50.8 | 8.6 |
| 3 − 2 (`static_limit` vs `2ry`) | 11.2 | 3.1 |

Per-band structure (mean Δ over k, meV): the movers are the **deep valence bands 1–2**
(2−0: +22.7; 3−0: +17.2) and the **top of the conduction window, bands 15–16**
(2−0: −3.3 mean / 39.6 max; 3−0: −5.2 mean / 50.8 max); the VBM manifold (bands 3–8) barely
moves (|mean| ≤ 3 meV), CBM manifold (9–14) moves +4…+12 meV. Γ-point direct-gap
(band 9 − band 3) shifts: **+7.9 meV (2−0), +6.0 meV (3−0), −1.9 meV (3−2)**.

### HL-GPP line (freq_dep=1) — ΔEqp0 in meV over all 8 k × 16 bands

| mode pair | max \|Δ\| | mean \|Δ\| |
|---|---|---|
| 2 − 0 | 94.4 | 32.5 |
| 3 − 0 | 104.0 | 36.9 |
| 3 − 2 | 45.8 | 17.7 |

HL invalid poles hit ~3–4× harder than GN and shift every band by tens of meV
(bands 1–2: +51 (2−0); bands 13–16: −40…−62 mean). Γ direct-gap shifts: +30.6 (2−0),
+48.6 (3−0), +18.0 (3−2) meV.

### Γ-point per-band detail (GN line, Eqp0 in eV)

| n | mode 0 | mode 2 | mode 3 |
|---|---|---|---|
| 1–2 (deep val) | −5.274943 | −5.235846 | −5.245193 |
| 3–4 (VBM) | 6.356822 | 6.354943 | 6.358712 |
| 5–8 (VBM manifold) | 6.404355 | 6.402496 | 6.406273 |
| 9–10 (CBM) | 9.657211 | 9.663229 | 9.665121 |
| 11–14 | 9.687598 | 9.693561 | 9.695422 |
| 15–16 | 11.162235 | 11.122652 | 11.111405 |

(HL Γ detail in `mode_diff_summary.txt`.)

## How to validate LORRAX against these references

1. **Mode mapping**: LORRAX `zero`/`skip` ↔ `01b_bgw_gn_mode0`; `2ry` ↔ `01_bgw_gn_ppm`;
   `static_limit` (once implemented) ↔ `01c_bgw_gn_mode3` (= BGW default, `01d`). The GN line is
   the primary target — LORRAX's GN-PPM is the same pole flavor. The HL line is a robustness
   cross-check of the trend only.
2. **Compare mode-to-mode DELTAS, not absolute eqp/Σc.** LORRAX GN-PPM fits poles in the ISDF
   basis from a (W(0), W(iω_p)) pair per centroid pair, while BGW fits per plane-wave (G,G′) pair;
   absolute Σc differs (and BGW-HL vs LORRAX-GN differ further). The invariant physical content is
   how Σc/Eqp0 move when the invalid-pole treatment changes. Concretely: run the same LORRAX
   calculation (Si 4×4×4, 60 bands, same `cohsex.in`) three times varying only `ppm_invalid_mode`,
   extract `sig_c(Edft)` per band via `parse_sigma_freq_debug`, and compare LORRAX's
   Δ(mode_a − mode_b) per band against BGW's GN-line Δ table above.
3. **Expected agreement**: sign and band-structure of the deltas (deep-valence up, window-top
   conduction down, VBM manifold ≈ unmoved), magnitudes of the same order (means ~10 meV, max
   ~tens of meV). Exact meV agreement is not expected: LORRAX's invalid-pole *population* is
   defined on ISDF-fitted poles (Ω² ≤ 0 in `fit_gn_ppm_from_wc_pair`), not on 8.6% of plane-wave
   (G,G′) pairs. A LORRAX zero↔2ry delta that is zero, of opposite sign, or ~eV-scale would flag a
   wiring bug.
4. **Testable today**: `zero` vs `2ry` (both wired as of 2026-07-04). `static_limit` raises
   NotImplementedError — the `01c`/`01d` references are the acceptance target for implementing it
   (analytic −½·W_c0 term, see BGW_INVALID_POLE_RESEARCH.md). Note LORRAX's current default
   (`zero`) corresponds to the **non-default** BGW mode 0; BGW's default is mode 3.

## Artifacts

- `mode_diff_tables.py` — analysis (compare-skill parser + 11-col HL extension), writes the tables.
- `mode_table_gn.dat`, `mode_table_hl.dat` — per-k/per-band Eqp0 + Cor′ for all modes.
- `mode_diff_summary.txt` — full per-band delta stats + Γ detail, both lines.
- `invalid_pole_counts_gn.txt` — per-q GN invalid-pole counts.
- `runs/Si/00_si_4x4x4_60band/count_invalid_gn_poles.py` — the counter.

## Status / open items

- [x] Three-mode GN reference line (+ default==3 proof)
- [x] Three-mode HL reference line
- [x] Exact GN invalid-pole counts (8.63%)
- [x] Compare-skill parser extended for freq_dep=1 logs (KNOWN_SANDBOX_ERRORS entry filed)
- [x] LORRAX side: run `ppm_invalid_mode = zero` vs `2ry` on this Si setup and compare deltas —
  **done 2026-07-08**, see `lorrax_zero_2ry_validation.md` (wiring PASS; invalid fraction 1.13%
  of ISDF pairs, non-vacuous; deltas 15–34× below BGW with edge-band sign agreement only —
  the BGW tables anchor sign/order at the window edges, not meV values)
- [x] LORRAX side: `static_limit` **implemented + validated + made DEFAULT 2026-07-08**
  (lorrax_D agent/memplanner-cleanup, commit fdf89c2 + default-flip commit). Three-way
  same-code rerun triple `03b_..._zero` / `03b_..._2ry` / `03_..._static` (the original 03
  pair predates the qp_solver/pad refactors); table: `lorrax_mode_table3.dat`, script
  `lorrax_mode_diff3.py`. **Verdict: PASS** per the edge-band protocol —
  Δ(static−zero): deep valence **+1.52** meV (BGW m3−m0 **+17.2**, ✓ up), window top
  **−0.49** (BGW **−5.19**, ✓ down), VBM manifold smallest mover; mean|Δ| 0.71 vs BGW
  8.56 meV (~12× smaller, consistent with the 7.6× smaller invalid population);
  internal hierarchy Δ(static−2ry)=0.21 ≪ Δ(static−zero)=0.71 meV (BGW 3.05 ≪ 8.56);
  statics bit-identical (max|Δ x_bare|=0). Si-window Σ_static: max|diag| 1.7 meV.
  **Implementation note**: BGW mode 3 keeps BOTH the static SEX (occupied) and CH terms
  (`mtxel_cor.f90` ω̃→∞ gives `ssx→−I_ε`, `sch→−½·I_ε`; the earlier research-note line
  "SX pole → 0" is wrong) — LORRAX adds
  `Σ_static = sigma_sx(G_occ, Wc0·mask) + sigma_coh(Wc0·mask)` (cohsex kernels reused;
  `Wc0_q` retained on `PPMBuildResult`), the exact Ω→∞ limit of the two-branch pole sum
  (occ → −½Wc0 = B/Ω, unocc → +½Wc0). A CH-only term would flip the occupied channel —
  ruled out by static−zero tracking 2ry−zero band-by-band.
- [x] Per-q localization of LORRAX invalid poles (source item from
  `lorrax_zero_2ry_validation.md`): per-q `n_invalid` print added; Si shows strong
  q-clustering, counts 166–11134 per q (uniform 1.13% total) — unlike BGW's
  q-uniform ~8.8%.
- [ ] (optional) offline HL invalid count — needs a RHO reader + `wpeff` sum-rule replica
