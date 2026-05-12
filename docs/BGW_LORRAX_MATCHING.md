# BGW ↔ LORRAX matching conventions

This is the single source of truth for the input-side flags and source-side
overrides required to make BerkeleyGW and LORRAX (GWJAX) produce
**numerically matched** self-energies for the same QE wavefunction.

It is *not* a tutorial on BGW or LORRAX inputs. It assumes you have working
runs of both codes and want them to agree. The receipts column points at the
8 keeper reports under `reports/` that established each convention.

Scope: static COHSEX and GN-PPM. Hybertsen-Louie PPM has additional caveats
not captured here (see `reports/log_anomalies_2026-05-07/` and CHANGELOG entry
2026-05-06 "HL-PPM mismatch traced…").

## Two modes per material

`examples/<material>/` ships two BGW configurations and two LORRAX
configurations per Σ-scheme:

| Mode | BGW config | LORRAX config |
|---|---|---|
| **Defaults** | BGW library defaults (`exact_static_ch 1`, native MC-vcoul, native head) | LORRAX library defaults (`apply_head_diagonal` as-default, native v(q+G), native wcoul0 from ISDF) |
| **Matched** | flags below set to mirror LORRAX's static-CH and head conventions | flags below set to consume BGW's vcoul + head |

The "defaults" pair is what you run when you want to know what each code says
"out of the box." The "matched" pair is what you run when you want their
output columns to agree numerically — typically within tens of meV for
COHSEX, ~100 meV for GN-PPM at converged centroid budgets.

## The conventions

| # | Convention | BGW side | LORRAX side | Receipt |
|---|---|---|---|---|
| 1 | Coulomb cutoff for screening | `screened_coulomb_cutoff = ecutwfc` | `bare_coulomb_cutoff = ecutwfc`  ⚠ LORRAX default is 4·ecutwfc | `mos2_3x3_nosym_cohsex_barecut_headnk_2026-04-10` |
| 2 | 2D slab truncation | `cell_slab_truncation` in epsilon.inp **and** sigma.inp | `sys_dim = 2` in cohsex.in | `head_fix_2026-04-04` |
| 3 | 0D truncation | `cell_box_truncation` (epsilon + sigma) | `sys_dim = 0` | (CO regression in CHANGELOG 2026-04-04) |
| 4 | Static-CH formula | `exact_static_ch 0` in sigma.inp | (implicit — LORRAX implements scheme 0) | `cohsex_head_investigation_2026-04-04` |
| 5 | Body v(q+G) overlay | (BGW MC-averages internally; needs `write_vcoul`) | `use_bgw_vcoul = true` + `bgw_vcoul_file = <bgw_dir>/vcoul` | `bse_bgw_vcoul_2026-04-23` |
| 6a | W head at q→0 (auto) | (BGW reads eps0mat.h5 epshead) | `wcoul0_source = epshead` (reads BGW eps0mat.h5) | `cohsex_head_investigation_2026-04-04` |
| 6b | W head at q→0 (explicit) | (BGW prints v(q=0) and W(q=0) in epsilon.log under `Wcoul head (MiniBZ)`) | `vhead = <BGW v(q=0) Ry>` + `whead_0freq = <BGW W(q=0,ω=0) Ry>` numerical overrides (Si production pattern) | `cohsex_si_444_gamma_agreement_2026-05-02` |
| 13 | BGW finite-q wing rescaling | `cell_average_cutoff 1.0d-12` in sigma.inp **and** epsilon.inp — disables BGW's `fixwings(q0flag=false)` per-q rescaling | (no LORRAX analog; matching requires BGW-side disable) | `cohsex_si_444_gamma_agreement_2026-05-02` |
| 14 | nosym path | `no_symmetries_q_grid` in BGW + `nosym=.true.` in QE NSCF | reads the full-BZ WFN as-is | `non_symmorphic_phase_2026-04-06` + `cohsex_si_444_gamma_agreement_2026-05-02` |
| 7 | GN-PPM head diagonal | (BGW does not add a separate head diagonal; head enters via plane-wave W) | `apply_head_diagonal = false`  ⚠ removes the rank-1 head injection that gives ~2.5 eV spurious shift on 2D | `head_fix_2026-04-04` |
| 8 | Non-symmorphic phase | (intrinsic) | applies `exp[-i (S G_source)·τ]` in full-zone reconstruction; τ in radians in WFN | `non_symmorphic_phase_2026-04-06` |
| 9 | Σ output column mapping | LORRAX `sigSX − sigX + sigCOH` ↔ BGW `(SX − X) + CH'` (primed col 11), **not** unprimed col 6 | `si_444_nosym_cohsex_mae_2026-04-09` |
| 10 | Pseudoband centroid pair-mode | — | `pair_mode = val_cond` (NOT `isdf_asym`) at centroid-selection time; `isdf_asym` is fine inside the GW fit | `pseudoband_diagnostics_2026-04-18` |
| 11 | Pseudoband scissor in-grid mask | — | scissor extrapolation tests against `E_DFT`, not `eigvalsh(H_qp)` | `scissor_shift_2026-04-18` |
| 12 | Centroid generation | — | `kmeans_cli` with N ≈ 1.5·N_μ, then pivoted-Cholesky prune to N_μ ≈ 8·n_band; ≤5 meV converged | `centroid_sweep_2026-04-27` |

## Why convention 13 (`cell_average_cutoff`) matters

**This is the single largest convention-mismatch contributor on validated systems.**

BGW's `epsilon` runs a per-q "fixwings" pass that rescales the
(G=0, G'≠0) and (G≠0, G'=0) wings of ε(q) by a factor based on
the mini-BZ-averaged 1/|q+G| (see `BerkeleyGW/Common/fixwings.f90`).
This pass fires at every q for which `|q|² < cell_average_cutoff`.
The default `cell_average_cutoff = 1e12` (a huge number) means it
fires everywhere. LORRAX has no analog of this rescaling.

Setting `cell_average_cutoff 1.0d-12` restricts the fixwings pass
to the literal q=0 point only (head + wing zeroing), which is the
convention LORRAX implicitly uses. On Si 4×4×4 nosym this changes
the matched-pair MAE from **16.43 meV → 0.12 meV** (per
`cohsex_si_444_gamma_agreement_2026-05-02/report.md`).

Add `cell_average_cutoff 1.0d-12` to BGW `sigma.inp` for any matched
benchmark. (It is also accepted in `epsilon.inp` but the sigma-side
setting is what actually affects ε(q) consumption in Σ.)

## Why convention 5 ("BGW vcoul overlay") matters

BGW's `epsilon` Monte-Carlo-averages v(q+G) inside each mini-BZ for the
screened-Coulomb body and writes the per-q,per-G table to `vcoul`. LORRAX
computes v(q+G) analytically. The two differ by ~percent at small G
(more at q→0) and the disagreement propagates into W. For tight agreement,
LORRAX has to consume BGW's `vcoul` table verbatim. The flag pair is:

```ini
use_bgw_vcoul = true
bgw_vcoul_file = /abs/path/to/<bgw_run>/vcoul
```

Independently of this, the W head at q=0,G=0 must come from BGW's
`eps0mat.h5` `epshead` (convention 6). LORRAX's `wcoul0_source = epshead`
reads that file directly. Without both, COHSEX agreement plateaus at
~150-200 meV; with both, it tightens to ~50 meV on 2D MoS2 1×1 / 3×3.

## Why convention 7 ("apply_head_diagonal = false") matters

For GN-PPM in 2D, LORRAX historically applied a rank-1 head correction
`(wcoul0/Ω)|ζ(0)⟩⟨ζ(0)|` to the diagonal of Σ_c. BGW does **not** do this —
the head enters BGW's correlation via the plane-wave W and contributes
~0.5 meV (because `v_coul(G=0)` at the shifted q₀ is ~0.45 a.u., not the
mini-BZ average ~315 a.u.). Setting `apply_head_diagonal = false` removes
the spurious ~2.5 eV LORRAX-side shift. The 2026-04-04 head-fix landed
this as the default; the flag remains as an explicit opt-in.

For COHSEX, `apply_head_correction()` adds the head to both V and W in
the ISDF basis; COH = ½(W − V) cancels most of it, leaving a residual
that depends on `wcoul0` accuracy — hence the `wcoul0_source = epshead`
override.

## Quick checklist for a matched run

When you want LORRAX to match a BGW run on the same QE wavefunction:

- [ ] In LORRAX's input set `bare_coulomb_cutoff` equal to BGW's `screened_coulomb_cutoff` (= `ecutwfc`).
- [ ] If 2D, `sys_dim = 2` + BGW `cell_slab_truncation` in both inp files.
- [ ] If COHSEX, BGW `exact_static_ch 0`. LORRAX is implicitly scheme 0.
- [ ] **BGW `cell_average_cutoff 1.0d-12` in sigma.inp** (convention 13 — disables `fixwings`; 16 meV → 0.12 meV on Si).
- [ ] If running nosym on the LORRAX side: BGW `no_symmetries_q_grid` + QE `nosym=.true.` (convention 14).
- [ ] Run BGW's `epsilon` with `write_vcoul` enabled, then set LORRAX `use_bgw_vcoul = true` + `bgw_vcoul_file = <bgw>/vcoul`.
- [ ] Choose head treatment (convention 6):
  - **automatic**: `wcoul0_source = epshead` (needs `<bgw>/eps0mat.h5` reachable). Convenient; LORRAX computes the q→0 average internally.
  - **explicit** (Si production pattern): set `vhead = <Ry>` + `whead_0freq = <Ry>` from BGW's `epsilon.log` "Wcoul head (MiniBZ)" values. More reproducible.
- [ ] If GN-PPM, set LORRAX `apply_head_diagonal = false`.
- [ ] Compare LORRAX `sigSX − sigX + sigCOH` against BGW **primed CH'** (col 11), not unprimed col 6.
- [ ] Centroid file: `kmeans_cli` with N ≥ 1.5·N_μ + pivoted-Cholesky prune to N_μ ≈ 8·n_band. Use `pair_mode = val_cond` if pseudobands are in play.

## What is *not* yet a stable convention

- **HL-PPM matching**: known mismatch traced to LORRAX's finite-band
  two-point surrogate vs BGW's f-sum-rule density (CHANGELOG 2026-05-06).
  No flag-level fix lands today.
- **Bispinor (SOC) ζ-fit μ_L = i**: must use LU not Cholesky
  (`project_bispinor_isdf` memory). Active development; conventions may
  shift.
- **CrI3 6×6 80 Ry NSCF with `nosym=.true.`** crashes on GPU
  (`KNOWN_SANDBOX_ERRORS.md` 2026-05-07). Workaround: drop `nosym`,
  keep `no_t_rev=.true. noinv=.true.` only.

## See also

- `reports/<keeper>/report.md` for the receipts cited above.
- `docs/docs_gwjax/COHSEX_INPUT.md` for the exhaustive `cohsex.in` flag reference.
- `docs/docs_bgw/` for BGW input file specs.
- `examples/<material>/` for ready-to-run pairs.
