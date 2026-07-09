# src/gw/gw_output.py — deep-read notes (2026-07-01)

LOC: 381. Module docstring: "GW driver output: banner, summary, and result serialization. Analogous to QE's `punch()` / `pw_restart_new`."

## Purpose

Pure-host (numpy-only, no JAX arrays except a diagnostic `jax.devices()[0].memory_stats()` probe in the banner) output layer for the GW driver. Holds the `GWResults` dataclass (the driver→writer contract) and `write_results`, the single "punch('all')" gateway that emits sigma_diag.dat, BGW-format eqp0/eqp1.dat, optional eqp_g0w0.dat, and qp_wfn_rotations.h5. Also three trivial console-formatting helpers (banner, section divider, system summary).

Category: **I/O: GW result serialization / console banner** (writer stage, no physics kernels; only unit conversion and IBZ-wedge subsetting).

## Imports / cross-module deps

- `common.units.RYD_TO_EV` (top-level).
- Lazy inside `write_results`: `file_io.write_sigma_to_file`, `file_io.write_eqp_g0w0`, `file_io.write_qp_rotations_h5` (defined in `src/file_io/sigma_output.py:9`, `:82`, and `src/file_io/qp_wfn.py:16`).
- Lazy: `gw.eqp_bgw.write_eqp_bgw_in_memory` (src/gw/eqp_bgw.py; its comment at :239 says "In-memory entry point — used by the standard ``gw_output.write_results``").
- Optional `import jax` inside `print_banner` (try/except, diagnostics only).

## Entry points and callers (grep over src, tests, tools, scripts)

| Symbol | Callers |
|---|---|
| `GWResults` | gw_jax.py:129 (import), :912 (construction). No other constructors found. |
| `print_banner` | gw_jax.py:130 |
| `print_section` | gw_jax.py:129 (import), ppm_pipeline.py:327-328 |
| `print_system_summary` | gw_jax.py:142 |
| `write_results` | gw_jax.py:936 (rank-0 only, `if meta.rank == 0`) |

No references from tests/ (grep "gw_output\|GWResults" in tests → zero hits), tools/, or scripts/ (scripts/ contains only checks/ and profiling/; no hits).

## Function-by-function

### `GWResults` (dataclass, lines 20–97)
Contract between `gw_jax.main` and the writer. All Σ arrays in **Rydberg**; writer converts to eV.

Fields (all host numpy):
- `sig_sx (nk,nb,nb)` — static screened-exchange Σ_SX (Ry); in PPM mode still the static COHSEX value ("retained for diagnostics/restart").
- `sig_coh (nk,nb,nb)` — static Coulomb-hole Σ_COH (Ry).
- `sig_h (nk,nb,nb)` — Hartree self-energy (Ry).
- `sig_x (nk,nb,nb)` — bare exchange Σ_X (Ry); "sigX" column in PPM mode, quality-of-fit check in COHSEX.
- `E_qp_ry (nk,nb)`, `U_qp (nk,nb,nb)` with U[k,m,n]=⟨m_DFT|n_QP⟩, `E_dft_ry (nk,nb)`, `kin_ion_ry (nk,nb,nb)` = H₀ = T+V_ion.
- `band_start, band_stop` — 0-based [b0,b3) window.
- `use_ppm` (bool) — switches SX/COH labels → X/C and pulls correlated diag from `sigma_c_diag_at_dft_ry`.
- `self_consistent` (bool).
- `sigma_c_diag_at_dft_ry (nk,nb)|None` — Σ_c(E_DFT) diagonal, PPM non-SC only.
- `sigma_xc_at_dft_ev (nk,nb)|None` — Σ_xc(E_DFT) diagonal in eV, G₀W₀-PPM non-SC only (drives eqp_g0w0.dat).
- `sigma_c_omega_diag_ev (n_omega,nk,nb_sigma)|None`, `omega_rel_ev|None` — full ω-grid Σ_c diagonal (eV, ω relative to DFT mid-gap E_F); drives BGW Z-factor in eqp1.dat; None in static modes ⇒ Z=1.
- `efermi_ev|None` — mid-gap E_F computed once in ppm_pipeline from `meta.nelec` (comment stresses NOT from the sigma-window band count; a previous bug recomputed it from band_stop−band_start).
- `sigma_omega_h5_path|None`, `tensors_filename|None` — status-line strings only.

nk convention: driver-internal arrays are on the unfolded **full BZ** (nk_full); the writer subsets to the IBZ wedge via `kirr_to_kfull`.

### `print_banner(backend, n_devices, grid_x, grid_y, n_procs, device_kind, print_fn=print)` (104–146)
Prints header: mesh, backend, `XLA_PYTHON_CLIENT_PREALLOCATE` / `XLA_PYTHON_CLIENT_MEM_FRACTION` env vars, and (best-effort) XLA pool stats via `jax.devices()[0].memory_stats()` wrapped in bare `try/except Exception: pass`. Caller: gw_jax.py:130. No physics.

### `print_section(title, print_fn=print)` (149–154)
72-dash divider. Callers: gw_jax (imported), ppm_pipeline.py:328.

### `print_system_summary(n_rmu, fft_grid, cell_volume, print_fn=print)` (157–169)
Prints ISDF centroid count, FFT grid, cell volume. Caller: gw_jax.py:142.

### `write_results(results, sigma_diag_file, eqp0_file, eqp1_file, input_dir, kpoints_crys, kgrid, kpoints_irr_frac, kpoints_reduced=None, kirr_to_kfull=None, print_fn=print, *, eqp_dE_ev=0.5)` (176–381)
The unified writer. Steps:

1. **sigma_diag.dat** (239–269): mode switch —
   - PPM: `sx_arr = sig_x`; `corr_arr` = zeros_like(sig_coh) with `corr_arr[:, idx, idx] = sigma_c_diag_at_dft_ry` (diag Ry values expanded into a band-diagonal matrix). Labels sigX/sigC/sigXC.
   - COHSEX: `sx_arr = sig_sx`, `corr_arr = sig_coh`. Labels sigSX/sigCOH/sigTOT.
   - Convert to eV (`r2e = RYD_TO_EV`), call `file_io.write_sigma_to_file(sx_out, sigma_diag_file, sigma_coh_kij_eV=corr_out, hartree_kij_eV=sig_h_out, sx_label=..., corr_label=..., total_label=...)`. Note VH column always printed.
2. **eqp0/eqp1.dat** (271–338): requires `kirr_to_kfull` (raises ValueError if None — despite the signature giving it default None; effectively mandatory). Subsets full-BZ arrays with `irr_idx`:
   - `e_dft_ev_irr = E_dft_ry[irr_idx]*r2e`
   - `kin_ion_diag_ev = Re diag(kin_ion_ry[irr_idx]) * r2e` (np.diagonal axis1=1, axis2=2)
   - `hartree_diag_ev`, `sigma_x_diag_ev` similarly (diag of already-eV arrays).
   - `sigma_c_at_dft_diag_ev`: PPM → `sigma_c_diag_at_dft_ry[irr_idx]*r2e` as complex128; static → `np.diagonal(corr_out[irr_idx])` (static Σ_COH diag; note NOT wrapped in np.real — kept complex).
   - If ω-grid present: subset `sigma_c_omega_diag_ev[:, irr_idx, :]`; require `results.efermi_ev` (raise if None); `e_dft_rel_ev_irr = e_dft_ev_irr − efermi_ev`.
   - Delegate all linearization math (central-difference Z, Newton update; equation eqp1 = eqp0 + Z·(Σ_c(E) evaluated via central difference dE=eqp_dE_ev)) to `eqp_bgw.write_eqp_bgw_in_memory(..., band_offset=results.band_start, ..., dE_ev=eqp_dE_ev, nspin=1)`. Static modes hand `sigma_c_omega_diag_ev=None` ⇒ Z=1 ⇒ eqp1==eqp0 (matches BGW static behavior).
3. **eqp_g0w0.dat** (340–357): only if `not self_consistent and sigma_xc_at_dft_ev is not None`. `h0_diag = Re diag(kin_ion_ry + sig_h)*r2e`; writes `E_dft_ev` and `h0_diag + sigma_xc_at_dft_ev` (complex Re/Im) via `write_eqp_g0w0(input_dir/eqp_g0w0.dat, ...)`. Note: uses **full-BZ** arrays, no irr_idx subsetting (unlike eqp0/eqp1) — intentional? It is a "hand-debugging" dump.
4. **qp_wfn_rotations.h5** (359–371): `write_qp_rotations_h5(input_dir/qp_wfn_rotations.h5, U_mnk=U_qp, E_qp_nk=E_qp_ry/2.0  # Ry → Hartree, band_start, band_stop, kpoints_crys, nkx,nky,nkz, kpoints_reduced, kirr_to_kfull)`. Full-BZ eigenvectors, energies in **Hartree** (third unit in one file: sigma_diag eV, eqp eV, rotations Ha).
5. Status-line prints (paths, optional σ(ω) h5 path and restart tensors filename).

Docstring notes degenerate-set averaging is applied **upstream** at the H-build seam in gw_jax; the writer does no re-averaging (confirmed by gw_jax.py:934 comment).

No einsums in this file.

## LorraxConfig flags / cohsex.in keys consumed

None directly. Path plumbing comes from the caller: `config.paths.sigma_diag_file`, `config.paths.eqp0_file`, `config.paths.eqp1_file` (gw_jax.py:938-941). Mode switches come via GWResults fields (`use_ppm` ← `mode.is_dynamic`, `self_consistent` ← `config.self_consistent`). Env vars read (banner, print-only): `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION`.

## I/O

All writes; no reads. Actual serialization lives in file_io / eqp_bgw; this module orchestrates:

| File | Format | Written by | Content |
|---|---|---|---|
| `sigma_diag.dat` | text table | file_io.sigma_output.write_sigma_to_file | per-(k,n) diag Σ decomposition, eV; columns sigSX/sigCOH/sigTOT or sigX/sigC/sigXC + VH |
| `eqp0.dat`, `eqp1.dat` | BGW text format, IBZ wedge only | gw.eqp_bgw.write_eqp_bgw_in_memory | E_DFT, E_qp0, E_qp1 with Z-factor (PPM) or Z=1 (static) |
| `eqp_g0w0.dat` | text | file_io.sigma_output.write_eqp_g0w0 | Re/Im of H₀+Σ_xc(E_DFT), full BZ, PPM non-SC only |
| `qp_wfn_rotations.h5` | HDF5 | file_io.qp_wfn.write_qp_rotations_h5 | U_qp eigenvectors, E_qp in Hartree, band window, k-mesh metadata |

## Suspects

### dead_suspects
None. All five public symbols (GWResults, print_banner, print_section, print_system_summary, write_results) have live callers in gw_jax.py and/or ppm_pipeline.py (grep `print_banner|print_system_summary|write_results|GWResults|print_section` over src/tests/tools/scripts). But note: **zero test coverage** — grep of tests/ for "gw_output|GWResults" returns nothing.

### redundancy_suspects
- Signature bloat / vestigial default: `kirr_to_kfull=None` and `kpoints_reduced=None` are "optional" in the signature, but `kirr_to_kfull=None` immediately raises ValueError at line 280-284 — it is mandatory. The Optional default is dead-path scaffolding.
- gw_jax.py:936-948 passes `kpoints_reduced=np.array(wfn.kpoints)` and `kpoints_irr_frac=np.array(wfn.kpoints)` — the same array under two parameter names; the two-name distinction may be a legacy of an older reduced-vs-irr split.
- `eqp_dE_ev=0.5` keyword-only param is never overridden by any caller (grep `eqp_dE_ev` → only gw_output.py itself), so the plumbing is currently decorative.
- In PPM mode the diag-vector `sigma_c_diag_at_dft_ry` is expanded to a full (nk,nb,nb) band-diagonal matrix (lines 247-251) purely so `write_sigma_to_file` can take a matrix, then only its diagonal is printed — round-trip vector→matrix→diagonal.

### weird_code
- **Line 189 / 336**: magic constant `eqp_dE_ev=0.5` eV central-difference spacing for the Z-factor, never configurable from cohsex.in; BGW's own default finite-difference spacing should be checked when comparing Z-factors.
- **Line 337**: `nspin=1` hardcoded in the `write_eqp_bgw_in_memory` call. For bispinor/spin-polarized runs the eqp files will always claim nspin=1. Hypothesis: predates bispinor work; fine while LORRAX treats bispinor bands as one joint index, but a refactor landmine.
- **Line 364**: `E_qp_ry / 2.0  # Ry → Hartree` — qp_wfn_rotations.h5 is in Hartree while every text output is eV and GWResults is Ry; three unit systems in one writer. Deliberate (downstream band-interp consumer expects Ha) but easy to trip over.
- **Lines 340-357**: eqp_g0w0.dat is written on the **full BZ** (no `irr_idx` subsetting) while eqp0/eqp1 are IBZ-only; inconsistent k-set between sibling outputs. Hypothesis: intentional debug dump, but undocumented asymmetry.
- **Line 302**: static-mode `sigma_c_at_dft_diag_ev = np.diagonal(corr_out[...])` is NOT wrapped in `np.real` (unlike kin_ion/hartree/sigma_x diagonals at 289-293) — complex is passed through to the BGW writer. Presumably deliberate (Σ_c can be complex; eqp writer handles Re/Im), but the real() treatment is inconsistent across columns.
- **Lines 132-143**: bare `try: import jax ... except Exception: pass` in print_banner — silent swallow of any error, GPU-touching call (`memory_stats()`) inside a nominally host-only output module.
- **Lines 304-310**: long comment documenting a *fixed* bug (writer used to recompute efermi from `band_stop − band_start`, silently treating every sigma-window band as occupied). Historical note kept as guard rationale for the line-317 ValueError.
