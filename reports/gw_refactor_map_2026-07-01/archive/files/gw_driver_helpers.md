# src/gw/gw_driver_helpers.py (285 LOC)

Deep-read notes, 2026-07-01. Repo: /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D

## Purpose

"Small driver-side helpers to keep `gw_jax.py` focused on orchestration" (module
docstring). Four independent utilities: a frozen options dataclass for the PPM sigma
stage, a best-effort profiling context manager, runtime pre-init (NCCL/MPI/compile
cache), and the BGW-vcoul override closure builder. No physics equations are
implemented here — it is glue between `LorraxConfig` and the physics stages.

Category: **driver glue / config plumbing** (with one runtime-init and one profiling
sub-role).

## Entry points (grep across src/, tests/, tools/, scripts/)

Greps run:
```
grep -rn "gw_driver_helpers|profile_section|setup_runtime|build_bgw_v_grid_fn|build_ppm_sigma_runtime_options|PPMSigmaRuntimeOptions" src tests tools scripts
```

| Symbol | Callers |
|---|---|
| `PPMSigmaRuntimeOptions` | src/gw/ppm_pipeline.py:31,53,111,169,239 (type annotations on pipeline stage fns/dataclass field); referenced in comment src/gw/gw_config.py:25 |
| `build_ppm_sigma_runtime_options` | src/gw/ppm_pipeline.py:325; src/gw/sc_iteration.py:665-668 (local import inside function) |
| `profile_section` | src/gw/ppm_pipeline.py:352 (`with timing.section("sigma.exec"), profile_section("sigma_ppm", ...)`) |
| `setup_runtime` | src/gw/gw_jax.py:127 (main driver) |
| `build_bgw_v_grid_fn` | src/gw/gw_jax.py:164 (main driver) |
| `_resolve_input_path` | internal only (3 call sites in this file) |

No test/tools/scripts callers found.

## Function-by-function

### `PPMSigmaRuntimeOptions` (dataclass, L16-42)
Frozen flat struct of 23 fields: resolved-once PPM sigma options. Mixture of physics
knobs (`omega_p_ry`, `ppm_fallback`, `omega_grid_ev/ry`, `sigma_regularization_ry`,
`sigma_edge_factor`, `fermi_reference`, `sigma_at_dft_extrapolate/energies`),
execution knobs (`sigma_omega_batch_size`, `sigma_omega_accumulation`), fudge/debug
knobs (`ppm_sigma_scale`, `ppm_sigma_flip_neg`, `ppm_invalid_mode`,
`ppm_sigma_debug_static_norm`, `ppm_static_cohsex_check`, `sigma_debug_quadrature*`,
`sigma_freq_debug_*`, `write_w_copies_debug`, `w_copies_debug_file`), and one path
(`sigma_kij_h5_path`). Only numpy arrays crossing: `omega_grid_ev`/`omega_grid_ry`
(shape `(n_omega,)` float64, host-resident). Consumed by the PPM sigma pipeline
stages in `ppm_pipeline.py` and via `sc_iteration.py`.
gw_config.py:25 explicitly notes this struct as a legacy still-alive parallel of the
typed config ("`PPMSigmaRuntimeOptions` from `gw_driver_helpers.py` ... are still").

### `_resolve_input_path(input_dir, path)` (L45-48)
Join relative path onto input dir; empty string passes through unchanged. Private.
Duplicates the inner `_resolve_path` of `file_io/paths.py:resolve_input_paths`
(which handles a fixed dict-key list instead). Two parallel path-resolution idioms.

### `class profile_section` (L51-125)
Context manager wrapping optional `pf` profiling hooks (memory snapshot pre/post +
xprof region). All failures swallowed; degrades to one `print_fn` line.
- `_try_import_pf` (L87-95): **hardcodes**
  `sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")` —
  machine/sandbox-absolute path baked into LORRAX source; docstring (L57-58) claims
  the relative `scripts/profiling/pf.py`.
- `_snapshot` (L97-104): writes `{artifacts_dir}/memprof/{name}_{pre|post}.prof`;
  `artifacts_dir` defaults to env `PF_ARTIFACTS_DIR` else `"profile"`.
- `__enter__`/`__exit__` (L106-125): standard, every pf call wrapped in bare
  `except Exception: pass`.
Single caller: ppm_pipeline.py:352 (name "sigma_ppm").

### `setup_runtime(config, mesh_xy, *, print_fn)` (L128-173)
Best-effort startup: (1) `runtime.nccl_warmup(mesh_xy)` inside
`common.timing.section("nccl_warmup")`; (2) if
`config.backend.slab_io is SlabIOBackend.PHDF5_FFI` → `ffi.common.ffi_loader.
phdf5_init_mpi()` (eager `MPI_Init_thread` MPI_THREAD_MULTIPLE, ~400 ms off critical
path before first collective `H5Fcreate` in `zeta_fit_chunked`); if `PHDF5_HOST` →
`import mpi4py.MPI` for the same amortization; (3)
`common.jax_compile_cache.ensure_jax_compile_cache()` (opt-out via env
`ISDF_JAX_CACHE_DIR=""`). Items 2-3 swallow exceptions with a print. Called once
from `gw_jax.main` (gw_jax.py:127). Docstring: "All three were inlined in
`main()`; this helper unifies them."

### `build_bgw_v_grid_fn(config, *, wfn, sym, input_dir, print_fn)` (L176-235)
Returns `None` unless `config.head.use_bgw_vcoul`; else builds a closure
`q_frac_tuple -> dense v(q+G) grid (np.ndarray on FFT grid)` substituting BGW's
MC-averaged Coulomb for LORRAX's internal head-only mini-BZ average (bit-reproducible
BGW comparisons). Raises if `head.bgw_vcoul_file` unset.
- Loads `read_bgw_vcoul(bgw_path)` (text vcoul table: `q_fracs`, `G_miller_per_q`)
  from `file_io/read_bgw_vcoul.py`.
- Captures `wfn.cell_volume` (float), `wfn.fft_grid` (3-tuple).
- Symmetry source fork (L219-227): if `head.bgw_vcoul_sym_wfn` set, reads
  `mf_header/symmetry/mtrx` (int32, shape `(nsym,3,3)`) from that aux WFN h5 and
  applies `sym_mats_k = sym_real.transpose(0, 2, 1).copy()` — real-space→k-space
  matrix transpose convention; else uses `sym.sym_mats_k` directly (no transpose;
  the SymMaps object presumably already stores the k-space form). Rationale comment:
  nosym WFN stores only identity, so unfolding BGW's IBZ-only q list needs the full
  48-op group from an aux sym-reduced WFN.
- Closure delegates to `fill_v_grid_for_q(bgw_table, q_frac_tuple, fft_grid,
  cell_volume, sym_mats_k=sym_mats_k)`.
Called by gw_jax.py:164; result threaded into `compute_V_q` per docstring.

### `build_ppm_sigma_runtime_options(config, *, input_dir)` (L238-285)
Pure config→struct translation with validation:
- Validates `ppm.omega_step_ev > 0`, `omega_max_ev >= omega_min_ev`,
  `fermi_reference in ("vbm","midgap")`.
- `n_omega = floor((omega_max - omega_min)/step + 0.5) + 1` (round-half-up trick,
  L252-254); builds uniform `omega_grid_ev` and `omega_grid_ry = ev / RYD_TO_EV`.
- `sigma_regularization_ry = ppm.regularization_ev / RYD_TO_EV`.
- Resolves 3 paths against `input_dir`: `config.paths.sigma_kij_h5_file`,
  `debug.w_copies_debug_file`, `debug.sigma_freq_debug_file`.
Callers: ppm_pipeline.py:325, sc_iteration.py:668.

## LorraxConfig flags consumed (config attribute paths)

- `config.backend.slab_io` (enum SlabIOBackend: PHDF5_FFI / PHDF5_HOST branches)
- `config.head.use_bgw_vcoul`, `config.head.bgw_vcoul_file`,
  `config.head.bgw_vcoul_sym_wfn`
- `config.ppm.{omega_p, fallback_omega, omega_min_ev, omega_max_ev, omega_step_ev,
  regularization_ev, window_edge_factor, omega_batch_size, omega_accumulation,
  sigma_scale, sigma_flip_neg, invalid_mode, fermi_reference,
  sigma_at_dft_extrapolate, sigma_at_dft_energies}`
- `config.debug.{sigma_freq_debug_output, ppm_sigma_debug_static_norm,
  ppm_static_cohsex_check, sigma_debug_quadrature, sigma_debug_quadrature_samples,
  write_w_copies_debug, w_copies_debug_file, sigma_freq_debug_file}`
- `config.paths.sigma_kij_h5_file`
- Env vars: `PF_ARTIFACTS_DIR` (profile_section default), `ISDF_JAX_CACHE_DIR`
  (documented opt-out, actually read inside `ensure_jax_compile_cache`).

## I/O

- **Reads** BGW vcoul table file (path from `head.bgw_vcoul_file`) via
  `file_io.read_bgw_vcoul` (BGW `vcoul` text format: q fracs + G Miller indices +
  v values).
- **Reads** aux WFN HDF5 (`head.bgw_vcoul_sym_wfn`): dataset
  `mf_header/symmetry/mtrx` only.
- **Writes** (via pf, best-effort) memory snapshots
  `{artifacts_dir}/memprof/{name}_{pre,post}.prof`.

## Suspects

### dead_suspects
None. Every public symbol has ≥1 caller (grep evidence above). `_resolve_input_path`
is private with 3 internal call sites.

### redundancy_suspects
1. `_resolve_input_path` (L45) vs `file_io/paths.py:resolve_input_paths` inner
   `_resolve_path` (L17-18) — identical logic, two homes.
2. `PPMSigmaRuntimeOptions` is a 23-field flat mirror of `config.ppm` +
   `config.debug` + one `config.paths` field; gw_config.py:25 comment acknowledges
   it as a surviving legacy layer. A refactor could pass the typed config sections
   directly (with the 3 derived quantities: omega grids, regularization_ry,
   resolved paths).
3. `setup_runtime`'s PHDF5_FFI vs PHDF5_HOST branches are two spellings of "init
   MPI early" (minor).

### weird_code
1. L90: hardcoded absolute sandbox path
   `"/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling"` inserted into sys.path
   — non-portable, contradicts docstring's relative path claim; hypothesis: quick
   hack to make pf importable under srun/shifter without PYTHONPATH plumbing.
2. L225: `sym_mats_k = sym_real.transpose(0, 2, 1).copy()` — silent real-space↔
   k-space convention transpose applied only in the aux-WFN branch; the
   `sym.sym_mats_k` branch takes matrices as-is. Correct if SymMaps already stores
   k-space matrices, but the asymmetry is a convention-bug magnet.
3. L270-272 / L28-30: `ppm_sigma_scale`, `ppm_sigma_flip_neg`, `ppm_invalid_mode` —
   a multiplicative fudge factor and a sign-flip flag for sigma baked into
   production options; hypothesis: leftover debug knobs from PPM bring-up that
   should be quarantined or removed.
4. L252-254: `floor(x + 0.5) + 1` grid-count rounding — intentional round-half-up
   against float noise, but a magic idiom worth a named helper.
5. Pervasive `except Exception: pass/print` in `profile_section` and
   `setup_runtime` — deliberate best-effort design, but hides real MPI init
   failures until the first collective.
