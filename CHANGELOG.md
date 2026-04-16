# Changelog

## 2026-04-16: JAX profiling stack — skill, helpers, k-parallel run_nscf

New sandbox-level `skills/profiling_stack/` and `scripts/profiling/` that
turn an unfamiliar LORRAX module into a ranked punch-list of bottlenecks
in one command. Four categories covered: memory, compute time, sharding,
compilation.

### Deliverables
- `scripts/profiling/pf.py` — helper library (`setup_env`, `trace_profile`,
  `region`, `annotate`, `snapshot_memory`, `aot_report`, `attach_compile_log`).
  Handles jax.distributed bootstrap, JAX_ENABLE_X64 latching, and the
  per-rank perfetto-trace race that broke multi-process runs.
- `scripts/profiling/run_profiled.py` — one-shot launcher wrapping
  `python -m <module>` with the whole env (XLA_FLAGS dump, JAX_LOG_COMPILES,
  IR dump, xprof trace, pprof snapshot).
- `scripts/profiling/analyze_hlo_dump.py` — XLA dump → ranked
  `hlo_summary.{md,json}` (Memory, Compute + custom calls, Sharding
  collectives, Rematerialization warnings, Retrace groups).
- `scripts/profiling/analyze_compile_log.py` — JAX compile log → ranked
  `compile_summary.{md,json}` (wall-clock totals, cache misses by source
  location, persistent-cache misses).
- `skills/profiling_stack/` — SKILL.md (entry point) + four category docs
  (memory / compute_time / sharding / compilation) + aot_reports.md +
  cookbook.md. All docs lead with "read the ranked summaries first, drill
  into source second" — per-function inspection is the secondary tool.

### LORRAX code change — branch `agent/run-nscf-kpar` (`4617f6e`)
- `src/psp/run_nscf.py`: module-level `_maybe_init_jax_distributed()`
  (same pattern as `gw.gw_jax`); Davidson k-loop strides over
  `jax.process_index()`; `process_allgather` of evals + packed coeffs;
  only rank 0 writes WFN.h5.

### Validation — Si 2×2×2 / 60 Ry / 12 bands
- `runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/00_davidson_only/`:
  1 GPU, Davidson 7.91 s (1 rank), evals[0]=-0.418717 Ry.
- `runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/02_davidson_4gpu/`:
  4 GPU k-parallel, Davidson 6.99 s (4 ranks). **WFN.h5 bit-identical to
  1-GPU** (eigenvalue maxabs diff 0.0, coefs maxabs diff 0.0).
- Analyzer on 4-GPU run surfaces 4 collectives (all-gather-start on
  f64[1,8,12] evals + c128[1,8,12,2,2120] coeffs, 31 MiB each) — the
  expected multihost_utils payloads.
- `uv run python -m pytest -q` → 14 passed when login-node GPU not saturated.

### Report
`reports/profiling_stack_2026-04-16/report.md` — deliverables, validation,
top-3 bottlenecks found from the very first profile (memory in
`jit__apply_H_sparse`, 33 % of wallclock spent in XLA compile, 163 cache
misses localised to `solvers/davidson.py` + `psp/vnl_ops.py`).

### Next steps
- A communication-heavy smoke test (multi-GPU `gw.gw_jax`) would exercise
  the Sharding + Rematerialization view at scale — `run_nscf` is
  embarrassingly k-parallel so only holds single-digit MiB collectives.
  Waiting on direction for the next target module.
- Collapse the `jit_multiply` x58 / `jit_broadcast_in_dim` x45 retrace
  groups by wrapping the Davidson k-loop body in one outer jit (or
  `lax.scan`).

## 2026-04-16: Symmetric Si 2x2x2 failure traced to SymMaps index conflation

- Reproduced the current symmetry-path failure directly from
  `runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/nscf/WFN.h5`:
  `SymMaps(WFNReader(...))` raises
  `IndexError: index 8 is out of bounds for axis 0 with size 8`.
- Root cause is in `sources/lorrax/src/common/symmetry_maps.py`:
  `create_kpoint_symmetry_map()` stores **symmetry-operation indices** in
  `kpoint_map`, but `kpoint_map_irrbz_ids()` later treats those values as
  **full-/irreducible-k indices** and indexes `full_kpts[idx]`.
- For the Si `2x2x2` WFN this is fatal because `nk_full=8` but the stored
  symmetry ids include `8` and `12`; the symmetric `4x4x4` path only
  appears to survive because its mistaken symmetry ids remain `< 64`.
- Compared against BerkeleyGW `Sigma/genwf_mpi.f90` and
  `Common/find_kpt_match.f90`, which keep irreducible-k index and symmetry
  index as separate state. This is the active bug; time reversal is only a
  secondary latent concern for future TR+nonsymmorphic cases.
- Fixed on source branch `agent/symmetry-maps-fix`:
  `create_kpoint_symmetry_map()` now stores irreducible-k ids rather than
  symmetry ids, and `kpoint_map_irrbz_ids()` now validates that direct map
  instead of reinterpreting it as a full-grid index.
- Added `src/common/symmetry_test.py`, a debug checker that validates both
  atomic-position invariance under the stored spatial symmetries and full-grid
  k-point unfolding from the irreducible wedge.
- Validation:
  `uv run python -m pytest -q` → `14 passed, 1 warning`;
  `uv run python -m common.symmetry_test .../Si_pseudobands/.../WFN.h5`
  → `48/48` symmetries and `8/8` k-points valid;
  `uv run python -m common.symmetry_test .../Si/05_si_4x4x4_sym/.../WFN.h5`
  → `48/48` symmetries and `64/64` k-points valid.

## 2026-04-15: Bare Σ_X invariance analysis — ISDF quality confirmed OK

### Bare exchange is nearly invariant (17 meV shift, BGW: 0 meV)
- Added bare Σ_X diagnostic print to gw_jax.py
- Ran 4 COHSEX calculations with the diagnostic: baseline (400c, 2000c), V1 PB, V2 PB
- Result: bare X shifts only 17-20 meV with pseudobands
- Centroids don't affect bare X (400c vs 2000c identical)
- ISDF quality for exchange is acceptable

### Decomposed comparison vs BGW (using CH' = exact static, per BGW sigma_hp.log)
- LORRAX absolute X differs from BGW by 5.5 eV — nk convention (8 vs 4 k-points)
- PB screening shifts: LORRAX ΔCH ≈ -1.4 to -1.7 eV, BGW ΔCH' ≈ -1.1 to -1.8 eV — within 20%
- Baseline CH offset (LORRAX -6.77 vs BGW -8.46) is k-grid dependent: 1.7 eV at 2×2×2, 0.6 eV at 4×4×4
- No evidence of COHSEX implementation regression from recent refactors

## 2026-04-15: Pseudobands v2 (Gauss-quadrature energies) — implemented, tested, V1 still wins

Branch `agent/nscf-clean-scaffold` (+6 commits).

### New module: `solvers/pseudobands_v2.py`
- **Shifted CJ boundaries** (δ = π/2M) for quadratic POU: Σw_j² ≈ 1 ± 0.04
- **Gauss quadrature** from windowed DOS moments (Stieltjes/Jacobi algorithm)
  gives per-band energies and weights. Numerically fragile for large n_eff;
  falls back to Ritz eigenvalues + uniform weight.
- **Davidson windows**: no-matvec Galerkin from stored eigenvalues
- **n_min = k floor** prevents pathologically narrow windows
- **Window placement** with automatic n_min enforcement
- Wired into `run_nscf.py` via `pb_version = 2` in nscf.in

### COHSEX comparison (Si 2×2×2, VBM)

| Method | sigTOT (eV) | Δ from 40-band |
|:--|:--:|:--:|
| Baseline 40-band | -12.824 | — |
| **V1 hybrid PB** | **-14.145** | **-1.32** |
| V2 Gauss PB | -14.428 | -1.60 |
| V2 Ritz energies | -14.419 | -1.60 |
| BGW reference | — | -1.18 |

**V1 remains the better scheme** (-1.32 vs -1.60 excess). The v2 shifted
boundaries and different window placement create 0.3 eV more over-screening.
Energy assignment (Gauss vs Ritz) has negligible effect (< 10 meV).

### Key findings
- Dominant error: ISDF quality degradation with pseudobands (89 meV sigSX shift)
- Energy assignment is NOT the bottleneck — Gauss vs Ritz ≈ same result
- The v2 infrastructure is complete and working, but the shifted boundaries
  need further investigation to understand why they increase over-screening
- `dos_cjwindows.py` diagnostic plots CJ window indicators on the full spectrum

### Test directories (runs/Si_pseudobands/00_si_2x2x2_60Ry/)
```
11_lorrax_pb_v2_k4_40win/    — v2 k=4, 41 windows (192 bands)
12_lorrax_pb_v2_k6_60win/    — v2 k=6, 59 windows (382 bands)
13_lorrax_cohsex_v2/          — COHSEX with v2 Gauss energies
14_lorrax_pb_v2_ritz_energies/ — v2 with Ritz energies
15_lorrax_cohsex_v2_ritz/     — COHSEX with v2 Ritz energies
```

## 2026-04-15: Hybrid stochastic/CJ-Ritz pseudobands — cross-window fix

Branch `agent/nscf-clean-scaffold` (+1 commit on top of prior work).

### Architecture change
- **Hybrid pseudobands**: three construction modes per window:
  - **Stochastic**: random-phase sums of exact eigenstates (for windows
    where CJ filter can't resolve — near conduction edge).
  - **CJ-Ritz**: Chebyshev-filtered Galerkin-Ritz (high-energy windows).
  - **CJ-0**: zero-weight placeholder (spectral gaps, CJ produces garbage).
- Det bands split into "protected" (below window start, included as-is)
  and "available" (consumed by stochastic construction). Extends Davidson
  deeper (nbnd=60) to provide exact eigenstates for transition zone.

### Bug fixes
- **Window start below det max**: E_cross was 1.31 Ry but det bands
  went to 2.23 Ry. First 3-4 windows were in the det manifold — after
  deflation, CJ produced noise. Now: stochastic for those windows.
- **Zero-norm NaN**: WFNReader clamped zero norms to 1e-30, ISDF divided
  by it → 10^30 → NaN in all zeta. Fixed: clamp to 1.0 (no-op division).
- **n_protected consistency**: fixed band count across k-points by passing
  n_protected from k=0 to subsequent k-points.

### Results (Si 2×2×2, 60 Ry)
- COHSEX pseudobands shift: **-1.32 eV** (was -1.77 eV broken, BGW ref -1.18 eV)
- Excess over BGW: **0.14 eV** (was 0.59 eV — 76% reduction)
- No more NaN output, no cross-window leakage

### Next
- Investigate remaining 0.14 eV excess (ISDF quality with pseudobands)
- Test with more centroids (5000+) to separate ISDF error from PB error
- Consider global QR for CJ windows to further reduce cross-window overlap

## 2026-04-14: NSCF refactor — clean scaffold, 2D Coulomb fix, module reorganization

Branch `agent/nscf-clean-scaffold` (14 commits).

### Bug fix
- **MoS2 2D Coulomb truncation**: `compute_V_H_and_V_xc` hardcoded `truncation_2d=False`
  for V_H Poisson solve. QE's `assume_isolated='2D'` now auto-detected from XML and
  applied to both V_loc and V_H. MoS2: 594 mRy → 0.013 mRy offset. Si unchanged.

### Module reorganization
- **`src/solvers/davidson.py`**: generic eigensolver (BSE-ready). `nspinor→n_channels`,
  `n_tgt→n_eig`, `nG→dim`. `psp/davidson.py` → shim.
- **`psp/pseudos.py`**: `load_pseudopotentials`, `symbol_to_Z`, `AtomPP` extracted from
  `get_DFT_mtxels.py` (-300 lines from the kitchen sink).
- **`psp/gvec_utils.py`**: `build_master_gvec_list`, `select_gvecs_for_k`, `compute_ngkmax`,
  `reorder_to_qe` consolidated.
- **`psp/radial/`**: `radial_jax.py`, `solid_harmonics.py`, `build_projectors_qe.py`.
- **`psp/upf/`**: `load_upf.py`, `normalize.py`, `upf_model_2_0_1/`.
- **`file_io/`**: `qe_save_reader.py` + `wfn_writer.py` joined `WFNReader` et al.
- **`dft_operators.py`**: now owns `poisson_potential_from_rhoG`, `generate_gvectors_k`,
  `build_G_cart` (moved from `get_DFT_mtxels` and `charge_density`).
- **Deleted**: `kpar.py`, `get_dipole_mtxels_chunked.py`, debug functions (~750 lines).
- **Archived**: `charge_density.py` (85% dead SCF code).
- **`get_DFT_mtxels.py`**: 1281 → 974 lines.

All three entry points (`run_nscf`, `get_DFT_mtxels`, `get_dipole_mtxels`) and GW drivers
now import shared routines from canonical locations. Validated: Si 0.001 mRy, MoS2 0.013 mRy.

## 2026-04-14: NSCF driver, WFN.h5 writer, k-parallel, MoS2 validation

### New modules
- **`psp/run_nscf.py`**: Full NSCF driver (QE .save → Davidson → WFN.h5)
- **`psp/kpar.py`**: K-point parallel diag via 2D mesh ('k', 'g')  
- **`compare_wfn.py`** (sandbox): Permanent WFN.h5 comparison tool

### WFN.h5 accuracy
- **Si 4×4×4**: 33/37 fields EXACT, eigenvalues 0.0009 mRy MAE, timing competitive with QE
- **MoS2 3×3×1**: 36/37 fields EXACT (all structural, G-vectors byte-identical after QE convention matching). Eigenvalues: 2.7 mRy MAE at Gamma, 1.0 mRy at other k-points.

### Bug fixes
- **bvec.T transpose bugs**: bdot, adot, atom_crys, G_cart — all hidden by cubic Si, exposed by hexagonal MoS2. Fixed in qe_save_reader.py, wfn_writer.py, ionic_gspace.py, charge_density.py.
- **QE G-vector ordering**: Matched exactly via `(round(|G|²×1e8), g1, g2, g3)` lexicographic sort
- **nosym symmetry convention**: ntran=1, identity only, zero-padded to 48
- **scipy_erf**: Replaces jax.scipy.erf in table construction (avoids Shifter PTX crash)

### MoS2 NSCF eigenvalue discrepancy — FIXED
**Root cause**: `compute_V_H_and_V_xc` hardcoded `truncation_2d=False` for V_H Poisson solve.
QE's MoS2 input uses `assume_isolated='2D'`, applying 2D Coulomb truncation to both V_H and
V_loc. LORRAX applied it to V_loc but not V_H, causing 594 mRy offset.

**Fix** (branch `agent/nscf-2d-truncation`): Added `truncation_2d` kwarg to
`compute_V_H_and_V_xc` and threaded from `run_nscf.py`. After fix: **0.013 mRy offset,
0.002 mRy MAE-no-offset** across all 9 k-points. Si unchanged at 0.001 mRy.

## 2026-04-13: Unified ionic G-space pipeline — 195s → 31s (setup: 177s → 5s)

Three changes on branch `agent/rho-core-table-interpolate`:

1. **Unified `build_ionic_and_core`** (new `psp/ionic_gspace.py`):
   - V_loc(r) and ρ_core(r) built in one pass via shared `lax.scan` primitives
   - `species_structure_factors` + `accumulate_species_on_G` — jittable, scannable
   - Cold: 2.37s. Warm: 0.01s. Previously V_loc=1.5s + rho_core=155s.

2. **SciPy CPU table construction** (`radial_jax._spherical_hankel_table_np`):
   - Replaced JIT-compiled `spherical_hankel_table_jax` for one-time setup
   - l=1 table build: 20.27s → 0.24s (84× faster, no JIT overhead)
   - JAX version kept for gradient computations

3. **VNL table reduction** (`vnl_ops.build_vnl_setup` n_q: 50000 → 4000):
   - Linear interpolation accurate to <1e-6 Ry at dq~0.001
   - vnl_setup: 21.5s → 2.6s

Full pipeline Si 4×4×4 nosym 64 k-points: **195s → 31s** total (26s is per-k JIT).
Setup (V_loc+NLCC+VNL): **177s → 5.0s**. Eigenvalues ≤0.0001 mRy.
Branch: `agent/rho-core-table-interpolate`, commits `8e50cbc`..`3c95c63`.
- **Next**: wire `build_ionic_and_core` into `test_dft_hamiltonian.py` callers,
  consider further per-k JIT reduction, merge to main.

## 2026-04-13: Active PSP callers migrated onto unified JAX VNL path

- Switched the remaining active preprocessing callers off the old
  `projector_pipeline` execution backend:
  `psp.get_dipole_mtxels`, `psp.get_dipole_mtxels_chunked`,
  `psp.get_DFT_mtxels`, and `gw.kin_ion_io_chunked` now build one
  `vnl_ops.build_vnl_setup(...)` and use per-k
  `build_vnl_kdata_from_kvec(...)` plus dense JAX contractions for `V_NL`.
- Added canonical sparse-G helpers to `psp.dft_operators` so the active caller
  scripts share one gather / `V_NL` matrix-element path rather than
  reimplementing host-side extraction logic.
- Preserved the custom JAX radial/spline/Bessel handling in one place:
  the migration still flows through `psp.radial_jax` and `psp.vnl_ops` for
  uniform-table interpolation, derivative tables, and stable spherical-Bessel
  behaviour.
- Archived the old CPU-heavy compatibility modules under `src/psp/archive/`:
  `build_projectors.py` and `projector_pipeline.py`.
- Validation:
  `uv run python -m pytest -q` → `13 passed, 1 warning in 19.27s`
  and real sandbox smokes both completed on local GPU:
  `gw.kin_ion_io_chunked` wrote `/tmp/kin_ion_migrated_smoke.h5`
  with shape `(64, 8, 8)` in `38.769 s`, and
  `psp.get_dipole_mtxels_chunked` wrote `dipole.h5`
  with shape `(3, 64, 60, 60)` from a temp staging directory.
- Revalidated both migrated preprocessors in the documented Perlmutter
  interactive-node Shifter environment on job `51487668` so profiling stays
  comparable to earlier sandbox runs:
  `gw.kin_ion_io_chunked` completed with `Total recorded: 17.793 s`
  and `real 30.31`, while
  `psp.get_dipole_mtxels_chunked --vnl-mode analytic` completed with
  `real 49.57`.

## 2026-04-12: Unified JAX radial backend for PSP setup path

- Added a shared source backend for radial transforms:
  [src/psp/radial_jax.py](/global/u2/j/jackm/software/lorrax/src/psp/radial_jax.py:1).
  This now owns the common spherical-Bessel kernels, uniform radial tables,
  interpolation, and radial integration weights used to form `V_NL`, `V_loc`,
  and NLCC/core charge.
- Switched the active production builders away from the old SciPy spline path:
  `vnl_ops.build_vnl_setup(...)`,
  `build_projectors_qe.build_local_ionic_potential_on_G_total(...)`, and
  `charge_density.build_core_density(...)` now all use the shared JAX/table
  backend.
- Simplified the autodiff `V_NL` channel extraction path in
  `dft_operators.py` so it consumes the same uniform tables rather than SciPy
  spline internals.
- Removed a duplicate spherical-Bessel implementation from
  `projector_pipeline.py` by importing the shared backend instead.
- Validation:
  `uv run python -m pytest -q` → `13 passed, 1 warning in 15.24s`
  and the canonical Si DFT-H reproducer still passes with
  `Max MAE: 0.0001 mRy = 0.00 meV`.
- Measured canonical launcher wall time after the refactor:
  `/usr/bin/time -p ./launch_test_dft_hamiltonian.sh` →
  `real 25.67`, `user 0.05`, `sys 0.04`.
- Followed up with a terminology cleanup in the active path so plan/bundle
  fields now prefer `radial_tables` over `splines`, reducing conceptual drift
  after the backend swap.
- Added report:
  [reports/jax_unified_psp_radial_2026-04-12/report.md](/global/homes/j/jackm/scratchperl/lorrax_sandbox/reports/jax_unified_psp_radial_2026-04-12/report.md)

## 2026-04-12: Standalone psp DFT-H validation now documented and runnable

- Fast-forwarded `sources/lorrax` again from `f7bc2e2` to `273a7d8`, picking up
  the new upstream reproducer `src/psp/tests/test_dft_hamiltonian.py` and the
  expanded `src/psp/dev_status.md`.
- Logged a new sandbox mismatch in `KNOWN_SANDBOX_ERRORS.md`: the local
  `runs/Si/04_si_4x4x4_davidson/00_davidson/` helper scripts still pointed at
  deleted `psp` setup helpers, so they were no longer a valid entrypoint.
- Added a sandbox-side canonical entrypoint for the standalone DFT path:
  [runs/Si/04_si_4x4x4_davidson/00_davidson/README.md](/global/homes/j/jackm/scratchperl/lorrax_sandbox/runs/Si/04_si_4x4x4_davidson/00_davidson/README.md)
  and
  [runs/Si/04_si_4x4x4_davidson/00_davidson/launch_test_dft_hamiltonian.sh](/global/homes/j/jackm/scratchperl/lorrax_sandbox/runs/Si/04_si_4x4x4_davidson/00_davidson/launch_test_dft_hamiltonian.sh),
  both using this sandbox's real paths and the Shifter environment that
  includes `$SANDBOX/sources` for `jax_xc_local`.
- First launcher run exposed a real upstream test bug: `test_dft_hamiltonian.py`
  passed `CrystalData` into `vnl_ops.build_vnl_setup(...)`, but the current
  implementation needs the `WFNReader` for its k-dependent G-vector scan.
  Patched locally on source branch `agent/test-dft-hamiltonian-fix`.
- Re-ran the canonical test on interactive job `51470500` and obtained:
  `Max MAE: 0.0000 mRy = 0.00 meV`
  and
  `PASS: all k-points match QE to < 0.01 mRy`
  for all 8 Si `4x4x4` IBZ k-points.
- Added report:
  [reports/dft_hamiltonian_validation_2026-04-12/report.md](/global/homes/j/jackm/scratchperl/lorrax_sandbox/reports/dft_hamiltonian_validation_2026-04-12/report.md)

## 2026-04-12: Si 4x4x4 no-sym COHSEX output-format rerun

- Created `runs/Si/02_si_4x4x4_nosym/16_lorrax_cohsex_rerun_4gpu_repeat/` as a fresh clone of variant `15` and reran GWJAX on interactive job `51470500` (1 node / 4 GPUs) so the updated logging/output-writing behavior would land in a new `gw.out` without overwriting prior outputs.
- Run completed end to end in `26.661 s`; artifacts written successfully: `gw.out`, `eqp0.dat`, `qp_wfn_rotations.h5`, and `tmp/isdf_tensors_480.h5`.
- The new `gw.out` differs materially from variant `15`: no initial `srun` step line, denser chunked-ISDF setup summary, progress-bar style zeta/V_q status lines, a new `STATIC HEAD TERMS` block, and inline XLA rematerialization warnings captured in the file.
- `eqp0.dat` from variant `16` is not byte-identical to variant `15`, so this should be treated as more than a cosmetic logging-only rerun.

## 2026-04-12: Housekeeping sync

- Fast-forwarded `sources/lorrax` on local `main` from `b0b02f9` to `f7bc2e2` to match `origin/main`.
- Logged a sandbox inconsistency in `KNOWN_SANDBOX_ERRORS.md`: the newest report directory (`reports/mos2_kgrid_gnppm_head_convergence_2026-4-10/`) does not contain the documented `report.md`.
- Added sandbox-local `jax_xc_local` wiring for the standalone `psp` DFT path:
  `sources/jax_xc_local -> /global/u2/j/jackm/software/jax_xc_local_lorrax_sandbox`
  and `sources/jax_xc -> /global/u2/j/jackm/software/jax_xc`.
  Verified `jax_xc_local.pbe` and `psp.dft_operators.compute_V_H_and_V_xc` import and execute under the documented Shifter flow when `PYTHONPATH` includes `$SANDBOX/sources`.
- Pulled the current Si Davidson/NSCF test drivers from `../lorrax_sandbox_fresh` into
  `runs/Si/04_si_4x4x4_davidson/00_davidson/` and updated `run_direct_diag_v2.py`
  to the current `origin/main` `psp` API (`setup_H_k`, `build_matrix_k`, `vnl_ops.build_vnl_setup`).
- First live Perlmutter/Shifter validation now works end-to-end for the direct-diag rung.
  `run_direct_diag_v2.py` reaches all 8 IBZ k-points and reports:
  diagonalized occupied-band MAE `94.890 mRy`, offset `-94.890 mRy`, MAE-no-offset
  `19.943 mRy`, max `153.478 mRy`. Nontrivial k-points show `H` non-Hermitian warnings
  (`~1e-4` to `4.5e-4`) and pathological Rayleigh quotients, which is the clearest
  current testing signal before Davidson wall-time work.

## 2026-04-12: Major code clarity refactor

Session focused on making gw_jax.py main() read like a physics outline.

**Screening pipeline surfaced at top level:**
- `compute_chi0(wfns, quad, meta, mesh_xy)` and `solve_w(V_q, chi0_q, meta, mesh_xy)` now visible in main() for both COHSEX and PPM paths
- `build_static_quadrature` / `build_imag_quadrature` are clean one-liners for quadrature setup
- `fit_gn_ppm(W_q, Wiwp_q, V_q, omega_p, mesh_xy)` extracted from monolithic PPM builder

**ppm_sigma.py (-347 lines):**
- PPM arrays stored as flat-q (nq,μ,μ) — eliminated transpose round-trip
- Fixed _mu_nu_sharding (was 5D for dead k-last layout)
- Fixed _build_single_sigma_window missing mask_B args (would crash on kernel_sign=-1)
- Stripped all profiling boilerplate; replaced verbose prints with per-window summary
- _convolve_sigma_branch_kij takes wfns bundle (28→22 params)

**gw_jax.py (-267 lines from ISDF move, +gw_output.py):**
- ISDF pipeline moved to gw_init.py (fixes circular import), split: fit_zeta + compute_V_q
- Output formatting extracted to gw_output.py (GWResults dataclass + write_results)
- V_q/W_q naming used consistently everywhere (no more bare V/W aliases)
- solve_w_from_chi_q_jax → solve_w; print0= → print_fn= standardized

**w_isdf.py:**
- Fixed chi0 accumulator sharding for non-divisible k-grids: P(None,'x','y')
- Fixed Dyson solve padding order (pad before reshard)
- Both verified on 4×A100 with MoS2 3×3 (nk=9)

All changes GPU-regression-tested (MoS2 3×3 COHSEX, 4×A100-40GB, bit-identical).
COHSEX chi0_W timing dropped from 2.7s→1.7s (old path computed unnecessary PPM head terms).

## 2026-04-09: GWJAX pipeline refactor status

Primary initiative: remove non-jitted stages, eliminate incorrect host/replicated
materializations, and make the active no-symmetry GWJAX pipeline safe on multi-GPU
Si `4x4x4` and `10x10x10`.


## Current status

What is now in good shape:
- head corrections for sigma_{X,static SX-X/CH, GN-PPM cor}
- active multi-GPU minimax screening path
- active GN-PPM fit path
- active dynamic sigma path
- post-PPM tail safety on `10x10x10`
- one process per GPU execution

What still looks worth improving:
- `compute_sigma_c_ppm_omega_grid` dominates runtime on large grids
- post-PPM fixed-point / QSGW work is safer now, but not yet distributed over
  band tiles on the `XY` mesh. This is a significant issue.
- likely next architectural step is a band-sharded `sigma_mnk.h5` / post-PPM
  path over `(omega, k, m_X, n_Y)`

## Known environment notes

- For multi-GPU GWJAX on Perlmutter, use Shifter, not `uv run`.
- *Keep one MPI rank per GPU. Do not ever run one mpi rank per node with 4 GPUs or so forth.*
