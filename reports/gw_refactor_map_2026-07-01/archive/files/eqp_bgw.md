# src/gw/eqp_bgw.py — deep-read notes (gw refactor map, 2026-07-01)

467 LOC. Pure-numpy + h5py; no JAX, no device arrays, no sharding. Everything here is
host-side post-processing of diagonal Σ pieces.

## Purpose

Writer for BerkeleyGW-format `eqp0.dat` / `eqp1.dat` from LORRAX gw_jax outputs,
byte-compatible (modulo a prepended `#` provenance line) with what BGW's
`Sigma/sigma_main.f90:3174-3193` emits, so BSE/Inteqp interface files match a reference
BGW Sigma run. Implements the standard QP-diagonal math without ever forming V_xc:

    Δ(k,n) = kin_ion + V_H + Σ_x + Σ_c(E_DFT) − E_DFT     (all diagonal, real part)
    eqp0   = E_DFT + Δ
    eqp1   = E_DFT + Z·Δ,   Z = 1 / (1 − dRe[Σ_c]/dω |_{ω=E_DFT})

Static modes (COHSEX): Z=1 ⇒ eqp1 == eqp0 (matches BGW). Dynamic modes (GN-PPM,
HL-PPM): Z from central difference of Σ_c(ω) interpolated on the run's ω-grid.

Category: **I/O: BGW-format QP-energy writer (postprocessing / output stage)**.

## Entry points and callers (grep evidence)

Grepped `src tests tools scripts` for `eqp_bgw`, and for each public symbol name.

| symbol | callers |
|---|---|
| `write_eqp_bgw_in_memory` | `src/gw/gw_output.py:235,324` (`write_results`, the standard live gw_jax write path) |
| `compute_eqp_diag` | `src/gw/gw_jax.py:808,896` (sigma_freq_debug table, rank-0, gated on `config.debug.sigma_freq_debug_output`); internal: `write_eqp_bgw_in_memory`, `make_eqp_bgw` |
| `compute_z_factor_from_omega_grid` | `src/gw/gw_jax.py:808,848` (same debug block); internal: both entry points |
| `write_bgw_eqp` | `tests/active/test_eqp_bgw.py:14,55,68,82,93,95`; internal: `write_eqp_bgw_pair` |
| `write_eqp_bgw_pair` | internal only (both entry points) |
| `make_eqp_bgw` | only `main()` in this file (CLI `python -m gw.eqp_bgw <run_dir>`); zero callers in src/tests/tools/scripts; zero hits in sandbox `skills/` and `runs/` |
| `main` / `_build_parser` | `__main__` guard only |

Related helpers outside the file: `gw.qsgw_utils.interp_along_omega` (imported lazily at
line 162); `bse.bse_io.read_bgw_eqp` (bse_io.py:537) is the LORRAX-side *reader* of these
files and transparently skips the `#` provenance line (downstream BGW binaries do not —
docstring says `tail -n +2`).

## Function-by-function

### `write_bgw_eqp(path, kpoints_irr_frac, e_dft_ev, e_qp_ev, *, band_offset, nspin=1) -> str` — lines 77–132
Pure text formatter matching BGW `sigma_main.f90` byte-for-byte after the provenance
line. Per IBZ k-point: header `(3f13.9, i8)` = kx ky kz (nspin·nb); body rows
`(2i8, 2f15.9)` = ispin, 1-based absolute band (`band_offset + ib + 1`), E_DFT_eV,
E_QP_eV. Validates shapes `(nk,3)` vs `(nk,nb)`; makedirs parent. Writes
`common.provenance.provenance_header()` first. Arrays: host numpy float64, `(nk_irr, nb_window)`.
No flags consumed. Consumers: `write_eqp_bgw_pair`, tests.

### `compute_z_factor_from_omega_grid(*, sigma_c_omega_diag_ev (n_omega,nk,nb), omega_rel_ev (n_omega,), e_dft_rel_ev (nk,nb), dE_ev=0.5) -> (sigma_c_at_dft (nk,nb) complex, z_factor (nk,nb) real)` — lines 139–173
Physics: linear interpolation of Σ_c(ω) at ω = E_DFT − E_F, then

    Z[k,n] = 1 / (1 − (Re Σ_c(E+dE) − Re Σ_c(E−dE)) / (2·dE))

`dE_ev=0.5` = BGW `finite_difference_spacing` default; comment notes gw_jax ω-grid
spacing 0.25 eV so ±0.5 eV falls on grid points. Delegates interpolation to
`qsgw_utils.interp_along_omega` (lazy relative import at line 162). ω axis and E_DFT
are both **relative to the DFT mid-gap Fermi level**, in eV. Consumers:
`write_eqp_bgw_in_memory`, `make_eqp_bgw`, `gw_jax.py:848` (debug table — deliberately
reuses this exact recipe so the `sig_c(Edft)` debug column is bit-consistent with
eqp{0,1}.dat; gw_jax comment notes the PPM pipeline's own interp differs by ~10 meV due
to a different vectorisation path).

### `compute_eqp_diag(*, kin_ion_diag_ev, hartree_diag_ev, sigma_x_diag_ev, sigma_c_at_dft_diag_ev, e_dft_ev, z_factor=None) -> (eqp0, eqp1)` — lines 176–211
Physics (all `(nk,nb)` eV):

    Δ = Re(kin_ion + V_H + Σ_x + Σ_c(E_DFT) − E_DFT)
    eqp0 = E_DFT + Δ;   eqp1 = E_DFT + Z·Δ  (Z=None ⇒ eqp1=eqp0)

`sigma_c_at_dft_diag_ev` is complex; only `.real` of the sum enters. In static modes the
caller passes Σ_SX+Σ_COH diagonal in the Σ_c slot with `z_factor=None`. Consumers:
both entry points; `gw_jax.py:896` (debug eqp0/eqp1 columns).

### `write_eqp_bgw_pair(*, eqp0_path, eqp1_path, kpoints_irr_frac, e_dft_ev, eqp0_ev, eqp1_ev, band_offset, nspin=1)` — lines 214–235
Thin wrapper: two `write_bgw_eqp` calls sharing header/k-list/E_DFT column; only E_QP
differs. Internal consumers only.

### `write_eqp_bgw_in_memory(*, eqp0_path, eqp1_path, kpoints_irr_frac, band_offset, e_dft_ev, kin_ion_diag_ev, hartree_diag_ev, sigma_x_diag_ev, sigma_c_at_dft_diag_ev, sigma_c_omega_diag_ev=None, omega_rel_ev=None, e_dft_rel_ev=None, dE_ev=0.5, nspin=1)` — lines 243–301
Live-run entry point used by `gw_output.write_results` (avoids re-reading sigma_mnk.h5).
If `sigma_c_omega_diag_ev is None` ⇒ static ⇒ Z=1. Else requires `omega_rel_ev` and
`e_dft_rel_ev`, and **re-derives** Σ_c(E_DFT) from the ω-grid (discarding the passed
`sigma_c_at_dft_diag_ev`? — no: it re-assigns the local name, so the caller-passed
on-shell value is ignored in dynamic mode, by design "for self-consistency with the
Z-factor central difference", line 279-280). Then `compute_eqp_diag` → `write_eqp_bgw_pair`.
Caller (`gw_output.py:280-338`) subsets full-BZ arrays to the IBZ wedge with
`kirr_to_kfull` and uses the canonical `results.efermi_ev` from
`ppm_pipeline._eval_sigma_c_at_dft_energies` (comment there records a past bug where the
writer recomputed E_F as top-of-window).

### `make_eqp_bgw(run_dir, *, wfn_path=None, kin_ion_path=None, sigma_mnk_path=None, qp_rotations_path=None, eqp0_out="eqp0.dat", eqp1_out="eqp1.dat", finite_difference_spacing_ev=0.5)` — lines 308–422
Post-hoc disk orchestrator (CLI path). Reads:
- `WFN.h5`: `mf_header/kpoints/rk` (IBZ kpts frac), `nspin`, `el` (nspin,nk,nb_total) Ry, `ifmax` (nspin,nk) 1-based. Raises `NotImplementedError` if nspin != 1 (line 342-343).
- `qp_wfn_rotations.h5`: `band_range` (2,) int64 half-open [start,stop), `kirr_to_kfull` (nk_irr,) int64.
- `kin_ion.h5`: `kin_ion` (nk_full, nb, nb) Ry; subset `kin_full[kirr_to_kfull, band_start:band_stop, band_start:band_stop]`, diag, ×RYD_TO_EV.
- `sigma_mnk.h5`: `omega_ev` (n_omega,), `sigma_sx_kij_ev` (nk_full,nb,nb), `hartree_kij_ev`, `sigma_c_kij_ev` (n_omega,nk_full,nb,nb) — all already eV, ω relative to E_F.

Recomputes the mid-gap Fermi level **locally** from WFN `ifmax` + window energies:
`n_occ = max(ifmax[0])`; `occ_idx_local = n_occ - 1 - band_start`;
`vbm = max(e_dft[:, :occ+1])`, `cbm = min(e_dft[:, occ+1:])`, `E_F = (vbm+cbm)/2`
(lines 367-378, with a bracket check raising ValueError). Index gymnastics for Σ_c:
`sigma_c_full[:, kirr_to_kfull][:, :, band_start:band_stop, band_start:band_stop]`
then `np.diagonal(..., axis1=2, axis2=3)` → (n_omega, nk_irr, nb). Then always calls
`compute_z_factor_from_omega_grid` (assumes a dynamic run — a static/COHSEX sigma_mnk.h5
without `omega_ev`/`sigma_c_kij_ev` datasets would KeyError; only the PPM output layout
is supported by the CLI path).

### `_build_parser` (429–446) / `main` (449–463)
argparse CLI, flags `--wfn --kin-ion --sigma-mnk --qp-rotations --eqp0 --eqp1
--finite-difference-spacing`. `python -m gw.eqp_bgw <run_dir>`.

## I/O summary

Reads (make_eqp_bgw only): `WFN.h5` (mf_header/kpoints/{rk,nspin,el,ifmax}),
`qp_wfn_rotations.h5` (band_range, kirr_to_kfull), `kin_ion.h5` (kin_ion),
`sigma_mnk.h5` (omega_ev, sigma_sx_kij_ev, hartree_kij_ev, sigma_c_kij_ev).
Writes: `eqp0.dat`, `eqp1.dat` — BGW text format with `# Generated by LORRAX <version>
at <UTC>` provenance line prepended.

## Flags consumed

None of LorraxConfig / cohsex.in directly. Indirect gates at call sites:
`config.debug.sigma_freq_debug_output` + `config.debug.sigma_freq_debug_file`
(gw_jax.py:805-908 debug consumer of the two compute helpers). CLI flag
`--finite-difference-spacing` mirrors BGW `finite_difference_spacing` (default 0.5 eV).

## Suspects

### Dead
- `make_eqp_bgw` + `main` + `_build_parser`: no callers in src/tests/tools/scripts other
  than the module's own `main()`; no `python -m gw.eqp_bgw` hits in sandbox `skills/` or
  `runs/` (grepped `gw.eqp_bgw` across both trees). The live path (`write_eqp_bgw_in_memory`
  via `gw_output.write_results`) always writes eqp{0,1}.dat during a run, so the post-hoc
  CLI is a recovery/debug tool that appears unused in practice. Not provably dead (it IS
  the module's documented CLI), but a refactor could question keeping ~160 LOC of disk
  orchestration.

### Redundancy
- `make_eqp_bgw` (disk) vs `write_eqp_bgw_in_memory` (in-memory) are acknowledged
  parallel entry points (docstring line 264: "Parallel of make_eqp_bgw"). Core math IS
  shared (compute_eqp_diag / compute_z_factor / write_eqp_bgw_pair), but the surrounding
  assembly (IBZ subsetting, diag extraction, unit conversion) is duplicated a THIRD time
  in `gw_output.write_results` (gw_output.py:285-322) and a FOURTH (partial) time in the
  gw_jax.py:805-908 debug block. Given the "no parallel old/new paths" project rule,
  candidates for consolidation.
- Fermi-level derivation exists twice: canonical `results.efermi_ev` (from
  ppm_pipeline, used by the live path — gw_output.py:306-311 explicitly documents that a
  local recompute was a past silent bug) vs `make_eqp_bgw`'s own ifmax-based mid-gap
  recompute (lines 367-378). The two can disagree (ifmax/window vs nelec-based); the CLI
  path has no access to the canonical value since sigma_mnk.h5 doesn't store E_F.

### Weird code
- Lines 125-127: comment that BGW's Fortran format string advertises `3f15.9` but
  sigma_main.f90 only passes 4 args (ispin, iband, E_DFT, E_QP) — deliberate
  convention-matching, load-bearing for byte compatibility.
- Line 342: `nspin != 1 → NotImplementedError` in the CLI path; hard nspin=1 assumption
  also baked into `write_eqp_bgw_in_memory(nspin=1)` call in gw_output.py:337. Relevant
  to the active bispinor work (bispinor runs — how do they map onto nspin here?).
- Lines 271-286: in dynamic mode `write_eqp_bgw_in_memory` silently *overwrites* the
  caller-supplied `sigma_c_at_dft_diag_ev` with a re-interpolated value. Intentional
  (bit-consistency with Z) per comment, but the parameter being required-yet-ignored in
  that branch is a trap.
- Line 397: chained fancy-index `sigma_c_full[:, kirr_to_kfull][:, :, bs:be, bs:be]`
  materialises an intermediate copy of the full (n_omega, nk_irr, nb_total, nb_total)
  array — harmless at these sizes, just stylistically different from the two-step
  subsetting used for Σ_x/V_H at lines 395-396.
- `make_eqp_bgw` unconditionally assumes dynamic-mode sigma_mnk.h5 (reads `omega_ev`,
  `sigma_c_kij_ev`); a COHSEX run's output would crash the CLI with a KeyError rather
  than falling back to Z=1.
- Tabs for indentation in this file (matches gw_jax.py house style) while gw_output.py
  uses spaces — cosmetic, but notable for refactor diffs.
