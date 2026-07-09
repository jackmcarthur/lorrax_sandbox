# src/gw/gw_config.py (1088 LOC)

Deep-read notes for the GW refactor map, 2026-07-01. Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.

## Purpose

Unified configuration layer for LORRAX GW. Parses the `[cohsex]` section of `cohsex.in` (INI via `configparser`, with a QE-style `K_POINTS crystal_b` block stripped and parsed separately) into a flat dict, then builds the frozen `LorraxConfig` dataclass tree (`paths` / `head` / `screening` / `ppm` / `memory` / `backend` / `debug` / `bse` sub-dataclasses) that is threaded through the entire driver. Also holds the four mode enums (`ComputeMode`, `SlabIOBackend`, `GspaceIO`, `ScreeningSolver`), the single `_DEFAULTS` table (single source of truth for every input key + inline docs), runtime resolution (GPU memory auto-detect, env-var chunk knobs, CPU-backend FFI auto-rerouting), and back-compat aliases for renamed flags.

Category: **configuration / input parsing (driver plumbing, no physics math)**.

## Structure — function/class table

| Symbol | Lines | Role |
|---|---|---|
| `ComputeMode` (enum) | 47–80 | Self-energy axis: `X_ONLY` / `COHSEX` / `GN_PPM` / `HL_PPM`. Properties: `needs_screening` (≠X_ONLY), `is_dynamic` (PPM modes), `ppm_model` (→ `"gn"`/`"hl"`/None). Consumers: `gw_jax.py:26`, `sigma_dispatch.py:32`, `screening.py:38`, `ppm_pipeline.py:29`. |
| `SlabIOBackend` (enum) | 83–101 | HDF5-writer axis: `PHDF5_FFI` (GPU, cudaMemcpyAsync D2H + collective MPI-IO, ~5× faster w/ Lustre striping), `PHDF5_HOST` (mpi4py + h5py-parallel), `H5PY_ALLGATHER` (rank-0 serial fallback). Consumers: `file_io/slab_io.py`, `file_io/kin_ion.py`, `file_io/sigma_output.py`, `file_io/_slab_io_mpi_host.py`, `common/isdf_fitting.py:1920`, `gw/gw_driver_helpers.py:13`. |
| `GspaceIO` (enum) | 104–119 | ψ(G) lifecycle: `HOST_CACHE` (read once, resident host RAM) vs `FILE_REREAD` (phdf5 collective re-read per r-chunk, zero persistent host residency). Both pull one band-chunk into jit via io_callback. Consumer: `common/isdf_fitting.py`. |
| `ScreeningSolver` (enum) | 122–132 | Dyson solver for W = (1 − V·χ₀)⁻¹·V: `JAX_NATIVE` (q-parallel reshard + `lu_factor`/`lu_solve` per q) vs `CUBLASMP_FFI` (fused symmetric Cholesky W = X·H⁻¹·X† via cuBLASMp/cuSOLVERMp; for nq·n_rmu² > VRAM). Consumer: `w_isdf.py` (`_normalize_screening_solver`, `_resolve_w_solve_fn`), driven from `gw_jax.py:262,265` and `screening.py:164` as `config.backend.screening_solver`. |
| `_LEGACY_ISDF_MEMORY_MODE` / `_SCREENING_SOLVER_TO_LEGACY` | 137–145 | legacy `isdf_memory_mode` string ↔ `ScreeningSolver` maps (`auto`/`high_mem`→JAX_NATIVE, `low_mem`→CUBLASMP_FFI). **Private dict imported cross-module** by `w_isdf.py:320` (`_normalize_screening_solver`). |
| `_DEFAULTS` (dict) | 152–350 | ~90 keys, the single defaults table + de-facto flag documentation. Type of default drives parse coercion (bool/int/float/None-nullable-float/str). |
| `_NORMALIZE_STR` (set) | 353–360 | keys lowercased+stripped after parse. |
| `read_lorrax_input(filename) -> dict` | 367–492 | INI parser. Locates `[cohsex]` (falls back to first `[...]` section, or pure defaults if none), strips the `K_POINTS` block from the INI text, rejects legacy `use_shipped_minimax_tables` (hard error), deprecation-warns and ignores `output_file`/`eqp_output_file`. Builds params from `_DEFAULTS` with type-directed coercion. Then parses the K_POINTS block: `nseg` lines of `kx ky kz [n] [#|!|; label]` → `params["kpoints_crystal_b"] = {"segments": [{"k", "n", "label"}, ...]}`. |
| `read_cohsex_input` | 496 | back-compat alias `= read_lorrax_input`. Re-exported through `gw/__init__.py:9` and `gw/gw_init.py:516`; called by `bandstructure/htransform.py:880`, `gw/kin_ion_io.py` (NB: that one imports the *psp copy*, see redundancy), `scripts/checks/sigma_direct_check.py:372`, `scripts/checks/w_from_eps0_0d_check.py:430`, `tests/archive/*`. |
| `FilePaths` | 507–520 | wfn/centroids(+optional `centroids_file_current` for bispinor Gordon-current centroids, None = charge-only CC-tile path)/kin_ion/sigma_diag/eqp0/eqp1/sigma_omega_h5/sigma_kij_h5. Consumers via `config.paths.*`: `gw_jax.py:136,138,468,938–940`, `gw_init.py:757–769,962`, `ppm_pipeline.py:254,395`, `gw_driver_helpers.py:281`, `sigma_x_bispinor.py:135`. |
| `HeadConfig` | 523–543 | q→0 Coulomb head: `wcoul0_source` ("s_tensor"\|"epshead"), `wcoul0_eta`, explicit overrides `vhead`/`whead_0freq`/`whead_imfreq`, `mc_average_vcoul_body`, `bare_coulomb_cutoff`, `zeta_cutoff`, BGW-vcoul diagnostic override (`use_bgw_vcoul`, `bgw_vcoul_file`, `bgw_vcoul_sym_wfn` — aux WFN for 48-op sym group when main WFN is nosym). Consumed by `gw.head_correction.HeadResolver` per docstring. |
| `ScreeningConfig` | 546–553 | `method` ("minimax" only), minimax target error / max nodes / regenerate flag / energy reference ("midgap"\|"vbm"). |
| `PPMConfig` | 556–590 | PPM ansatz (`model` "gn"\|"hl", `omega_p` probe — imaginary for GN ≈2 Ry, real for HL default 200 Ry, `fallback_omega`, `head_omega_h_ry` override Ω_h = √(ω_p²/(1−ε_head⁻¹)) for BGW comparison), σ-quadrature minimax (`sigma_target_error`, `sigma_max_nodes`), Σ_c(ω) ω-grid (min/max/step eV, regularization, window edge factor, batch size, `omega_accumulation` "auto"\|"kij"\|"kij_stream"), on-shell knobs (`sigma_scale`, `sigma_flip_neg`, `invalid_mode` "static_limit", `fermi_reference`, `sigma_at_dft_extrapolate`, `sigma_at_dft_energies`). |
| `MemoryConfig` | 593–612 | `per_device_gb` (0 = auto-detect), `chunk_target_utilization` (env `ISDF_CHUNK_TARGET_UTILIZATION`, default 0.97, clamped [0.85,1.0]), `chunk_size` (legacy, −1 = none), `band_chunk_size` (default 16), `r_chunk_override`, `zct_stage_cap_gb` (env `ISDF_ZCT_STAGE_CAP_GB`/`_FRAC`), `use_aot_chunk_chooser` + `chunk_chooser_mode` ("heuristic"\|"analytic", needs artifacts at `src/gw/aot_memory_model/artifacts/fit_one_rchunk__current__*.json`), `gflat_chunk_size` (0 = one-shot, cohsex.in > 0 **overrides the gflat planner**), `vq_g_chunk_size` (0 = auto `_pick_g_chunk(ngkmax)` → largest divisor ≤ 4096). Referenced by `gflat_memory_model.py:690` and `gw_init.py:617` (comments). |
| `BackendConfig` | 615–641 | `slab_io` / `gspace_io` / `screening_solver` enums + `cusolvermp_charge`/`cusolvermp_lu` ("auto"\|"on"\|"off"; auto = cuSolverMp iff true 2D mesh p_x≥2 AND p_y≥2) + `gamma_contract_mode` ("take"\|"einsum"\|"scan" — HLO-variant of `common.gamma_matrices.gamma_double_contract`, math-identical). `summary()` (632–641) one-line run banner. Consumers: `gw_init.py:547` (`set_gamma_contract_mode`), `gw_init.py:719–722,841–844,1006–1300` (slab_io + cusolvermp), `gw_jax.py:262,265,311,366,469`, `screening.py:164`, `ppm_pipeline.py:267`. |
| `DebugConfig` | 644–657 | debug flags + aux filenames; `write_wfn_h5` (default True → end-of-run `WFN_qp.h5`, BGW format, ψ rotated by final U, E→E_QP) lives here despite not being debug-only. |
| `BSEConfig` | 660–670 | `get_centroids_fi` master gate + `wfn_fi_min/max` band sub-window + `kgrid_fi` string; feeds `bandstructure.bse_setup.compute_wfns_fi` (htransform-driven fine-k wfn recovery). |
| `LorraxConfig` | 673–1088 | Frozen top-level: geometry (`nval`/`ncond`/`nband`/`sys_dim`), mode flags (`restart`, `compute_mode_raw`, `do_screened`, `bispinor`, `do_G0`, `self_consistent`, `use_ppm_sigma` legacy mirror, `no_degen_averaging`, `degen_avg_tol_ry` = BGW `TOL_Degeneracy` 1e-6 Ry), 8 sub-groups, `kpoints_crystal_b`, `input_dir`. |
| `LorraxConfig.compute_mode` (property) | 733–762 | Resolves enum: explicit `compute_mode` wins; `"auto"` infers from legacy `use_ppm_sigma` (+`ppm.model` gn/hl; raises if `do_screened=False`) else `do_screened` → COHSEX/X_ONLY. |
| `.minimax_config` (property) | 764–773 | on-demand `gw.minimax_config.MinimaxConfig` from screening group. Callers: `gw_jax.py:237`, `screening.py:153,157`. |
| `.sigma_quadrature_config` (property) | 775–785 | on-demand `SigmaQuadratureConfig`; hardcodes `crossing_max_nodes=max(500, sigma_max_nodes)`, `crossing_eps_q=1.0e-3`. Caller: `ppm_pipeline.py:355`. |
| `.omega_grid_ry` / `.omega_grid_ev` (properties) | 787–803 | `np.arange(min, max + 0.5·step, step)` in Ry / eV. NOTE: most `omega_grid_*` grep hits are `PPMSigmaRuntimeOptions.omega_grid_*` in `gw_driver_helpers.py:255–265` which **recomputes the same grid independently** rather than calling these properties (see redundancy). |
| `.use_ffi_io` (property) | 812–821 | legacy bool view: True iff slab_io ∈ {PHDF5_FFI, PHDF5_HOST}. **No caller found** (see dead suspects). |
| `.gspace_mode` (property) | 823–826 | legacy string view. Live: `gw_init.py:720,842`. |
| `.isdf_memory_mode` (property) | 828–831 | legacy string view via `_SCREENING_SOLVER_TO_LEGACY`. **No caller found.** |
| `LorraxConfig.from_input_file(filename, *, print_fn)` (classmethod) | 837–1088 | The factory. Steps: (1) `read_lorrax_input`; (2) `file_io.resolve_input_paths(params, input_dir)` — mutates params to absolute paths; (3) validate legacy `isdf_memory_mode`; (4) memory auto-detect via `common.gpu_utils.get_device_memory_gb()` when ≤0; (5) env knobs (chunk utilization, ZCT stage cap — cap-frac path queries `get_device_memory_info()` only on GPU backend); (6) build the 8 sub-dataclasses; (7) CPU-backend auto-routing: `use_ffi_io=true` → PHDF5_HOST if h5py has MPI, else H5PY_ALLGATHER with warning; forces `cusolvermp_charge`/`cusolvermp_lu` to "off" on CPU. Callers: `gw_jax.py:113` (main driver, the only production call); referenced in docs of `file_io/slab_io.py` and `file_io/_slab_io_allgather.py`. |

## Entry points (grep evidence, across src/ tests/ tools/ scripts/)

- `LorraxConfig.from_input_file` <- `gw/gw_jax.py:113` (main driver).
- `LorraxConfig` (type refs) <- `gw/gw_jax.py`, `gw/ppm_pipeline.py`, `gw/gw_driver_helpers.py` (`setup_runtime`, `build_ppm_runtime_options`), `file_io/slab_io.py`, `file_io/_slab_io_allgather.py` (docs).
- `read_lorrax_input` <- `gw/__init__.py`, `gw/gw_init.py:516` (re-export only, `# noqa: F401`).
- `read_cohsex_input` (alias) <- `gw/__init__.py`, `gw/gw_init.py` re-exports; `bandstructure/htransform.py:880`, `scripts/checks/sigma_direct_check.py`, `scripts/checks/w_from_eps0_0d_check.py`, `tests/archive/test_chunked_wfn_loading.py`, `tests/archive/test_input_file.py` (that one imports from long-gone `gw_isdf.cohsex_isdf`).
- `ComputeMode` <- `gw_jax.py`, `sigma_dispatch.py`, `screening.py`, `ppm_pipeline.py`.
- `SlabIOBackend` <- `file_io/{slab_io, kin_ion, sigma_output, _slab_io_mpi_host}.py`, `common/isdf_fitting.py:1920`, `gw/gw_driver_helpers.py`.
- `GspaceIO` <- `common/isdf_fitting.py`.
- `ScreeningSolver`, `_LEGACY_ISDF_MEMORY_MODE` <- `gw/w_isdf.py:320,339,667`.

## Flags consumed

- Input file: every key in `_DEFAULTS` (~90 keys of `cohsex.in [cohsex]`), plus the `K_POINTS crystal_b` block.
- Env vars: `ISDF_CHUNK_TARGET_UTILIZATION`, `ISDF_ZCT_STAGE_CAP_GB`, `ISDF_ZCT_STAGE_CAP_FRAC`.
- Runtime probes: `jax.default_backend()`, `common.gpu_utils.get_device_memory_gb()` / `get_device_memory_info()`, `h5py.get_config().mpi`, `import mpi4py`.

## I/O

- Reads: `cohsex.in` (INI text with embedded QE-style K_POINTS block). No other file I/O; it *names* the run's files (WFN.h5, centroids_frac.txt, kin_ion.h5, sigma_diag.dat, eqp0.dat, eqp1.dat, sigma_mnk.h5, optional sigma_kij h5, WFN_qp.h5, debug .dat files) but reads/writes none of them itself.

## Cross-module deps

`common.units.RYD_TO_EV`, `common.gpu_utils` (memory detect), `file_io.resolve_input_paths` (path resolution), `gw.minimax_config` (derived config objects, lazy import), `jax` (backend probe only).

## Dead suspects

- `LorraxConfig.use_ffi_io` property (lines 812–821): grepped `\.use_ffi_io\b` across src/tests/tools/scripts — every hit is a `use_ffi_io` *kwarg* on file_io/gw writer functions or the `self.use_ffi_io` attribute of the SlabIO class in `file_io/slab_io.py:275`; zero reads of the LorraxConfig property.
- `LorraxConfig.isdf_memory_mode` property (828–831): grepped `\.isdf_memory_mode\b` — no attribute-read hits outside gw_config.py (w_isdf takes the enum / raw string, not this property).
- (Contrast: `.gspace_mode` property IS live at `gw_init.py:720,842`.)

## Redundancy suspects

- **Full copy-paste parser**: `src/psp/get_DFT_mtxels.py:93` `read_cohsex_input` re-implements this file's parser line-for-line (docstring admits "mirrors the robust parser used in gw.gw_init.read_cohsex_input"), with its own defaults table. Consumed by `psp/get_dipole_mtxels.py:30`, `psp/run_sternheimer.py:1488`, and — confusingly — `gw/kin_ion_io.py:34` imports the *psp copy*, not the gw one. Classic fetch_X_dyn-next-to-fetch_X cruft.
- Dual naming `read_lorrax_input` / `read_cohsex_input` alias + double re-export via both `gw/__init__.py` and `gw/gw_init.py:516`.
- Legacy triples: `compute_mode` vs (`do_screened`, `use_ppm_sigma`, `ppm_model`); `isdf_memory_mode` string vs `ScreeningSolver`; `use_ffi_io` bool vs `SlabIOBackend` — each kept parsed+mirrored for back-compat, with translation logic split between here and `file_io/slab_io.py:_normalize_slab_backend` / `w_isdf.py:_normalize_screening_solver`.
- `omega_grid_ry`/`omega_grid_ev`: same `np.arange(min, max+0.5·step, step)` grid built independently in `gw_driver_helpers.py:255–258` for `PPMSigmaRuntimeOptions` instead of using the LorraxConfig properties (properties ARE used elsewhere? — the grep hits at gw_jax/ppm_pipeline are on `ppm_options.omega_grid_*`, not `config.omega_grid_*`, so the config properties themselves may be near-dead too).

## Weird code

- `w_isdf.py:320` imports the **private** `_LEGACY_ISDF_MEMORY_MODE` dict across module boundary — the "legacy strings cross over here only" contract lives in the consumer, not the owner.
- Magic constants in `sigma_quadrature_config` (781–784): `crossing_max_nodes = max(500, ...)`, `crossing_eps_q = 1.0e-3` — hardcoded, not input keys, undocumented in COHSEX_INPUT terms.
- Chunk utilization clamp `max(0.85, min(1.0, x))` (line 875) and ZCT frac clamp `max(0.10, min(0.95, frac))` (line 894) — silent clamping of env-var values, `except Exception: pass` swallows bad values (lines 886–887, 896–897).
- `from_input_file` imports `jax` at line 878 unconditionally for the ZCT-cap branch, then again defensively in a try/except at 987–991 for backend detect — inconsistent import discipline in the same function.
- `_DEFAULTS["sys_dim"] = 2` — 2D default; a 3D user who omits `sys_dim` silently gets slab-truncated Coulomb assumptions.
- `_DEFAULTS["nval"]=5, "ncond"=5, "nband"=100` — arbitrary physics defaults that will run "successfully" on the wrong band window if omitted.
- `bare_coulomb_cutoff` default None: per project memory this resolves downstream to 4·ecutwfc, mismatching BGW's ecutwfc — the mismatch is documented only in sandbox memory, not here (only the `zeta_cutoff` comment at 270–276 hints at the ecutwfc default).
- ω-grid `np.arange` with `+0.5·step` endpoint fudge (791–803) — float-step arange, endpoint inclusion depends on rounding.
- Parser fallback (383–386): if no `[cohsex]` section, silently uses the *first* section of any name; if no section at all, silently returns pure defaults (457) — a typo'd section header yields a default-everything run with no warning.
- K_POINTS parsing exists in duplicate here (459–491) and in the psp copy; the parser also tolerates malformed segment counts via bare `except Exception: seg_count = 0`.
