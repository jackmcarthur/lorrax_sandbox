# GW Refactor Map — group: common utilities

Files: `src/common/{meta,units,provenance,progress,timing,jax_compile_cache,jax_profile,gpu_utils,__init__}.py`
Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`. Caller greps ran across `src/`, `tests/`, `tools/`, `scripts/` (tools/ and scripts/ produced no hits for any symbol in this group).

---

## src/common/meta.py (160 loc)

**Purpose.** Defines the `Meta` dataclass — the run-wide shape/bookkeeping record threaded through the entire GW pipeline: band-window edges `b_id_0..b_id_4` (frozen core / valence / conduction / sigma / padded-total), FFT grid, k-grid, spin/spinor counts, centroid counts, and the padded-for-sharding variants of each extent. `Meta.from_system(wfn, sym, ...)` is the canonical constructor; it computes `b_id_4 = round_up(nband, world_size)` so band axes divide the device mesh.

**Category.** core data structure: run metadata / shape bookkeeping.

**Functions.**
| symbol | role |
|---|---|
| `_round_up(x, n)` | ceil-round helper |
| `Meta` (dataclass) | 24 fields + `__post_init__` derived attrs (`nelec`, `nb_sigma`, `kgrid`, `kgrid_np/jax`, `fft_grid_np/jax`, `band_edges`, `band_ranges` SimpleNamespace) |
| `Meta.band_range(name)` | named lookup into `band_ranges` |
| `Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor)` | canonical constructor; pads band + μ axes to `world_size` |

**Entry points / callers.**
- `Meta.from_system` <- `gw/gw_jax.py:147`, `gw/kin_ion_io.py:110`, `bandstructure/htransform.py:594`, `centroid/pivoted_cholesky.py:905`, `psp/run_sternheimer.py:1112`
- `Meta` (type import) <- `common/isdf_fitting.py`, `common/load_wfns.py`, `gw/vcoul.py`, `gw/w_isdf.py`, `gw/coulomb/{base,bulk_3d,slab_2d,box_0d}.py`, `psp/get_DFT_mtxels.py`, `common/__init__.py`
- Direct `Meta(...)` construction (bypasses `from_system`) <- `gw/aot_memory_model/kernels/fit_one_rchunk.py:435`, `tests/archive/test_frequency_integration_toy.py:38`, `tests/archive/test_freqint_stage23.py:32`
- `dataclasses.replace(meta, n_rmu=..., n_rmu_jax=..., n_rmu_padded=...)` <- `gw/gw_init.py:783-789` (bispinor transverse-centroid refresh)

**I/O.** None (pure in-memory).

**Dead suspects.**
- `Meta.band_range()` and the whole `band_ranges` SimpleNamespace: grep `\.band_range\(|band_ranges\.` across src/tests/tools/scripts → zero hits outside meta.py. Callers use raw `b_id_*` / `band_edges` instead (only 2 `band_edges` users: `gw_jax.py:154` via `BandSlices.from_band_edges`, `gw_init.py:192`).
- Derived attrs `kgrid_jax`, `fft_grid_jax`, `kgrid_np`, `fft_grid_np`: grep `\.kgrid_jax` etc. → zero readers each.
- Fields `nbnd_jax`, `n_rtot_jax`: grep `\.nbnd_jax|\.n_rtot_jax` → zero attribute readers (only set at construction; kw appears in the aot kernel + archive-test constructors).
- Local `rank_topo` in `from_system` (line 94): computed, never used.

**Weird code.**
- `meta.py:94` — `rank_topo = np.where(np.asarray(jax.devices()) == rank)` compares Device objects against an int process index; always-empty result, then unused. Hypothesis: leftover from an abandoned rank→device-topology mapping.
- `from_system` passes literal `1` for `nfreq` (line 147) — frequency count hardwired at construction; real freq handling lives elsewhere.
- `n_rmu_jax` explicitly documented as legacy/wrong-divisor (host-count instead of world-size) but still kept and still refreshed in `gw_init.py:786`; parallel to `n_rmu_padded`. Same duplication pattern for `nbnd_jax` (n_proc-rounded) vs `b_id_4` (world-size-rounded).
- `sys_dim` is monkey-patched onto `meta` by `gw_jax.main` (no dataclass field), so `dataclasses.replace` drops it — `gw_init.py:795` manually re-copies it with an apologetic comment. Refactor target: make it a field.

**Redundancy suspects.** `n_rmu_jax` vs `n_rmu_padded`; `nbnd_jax` vs `b_id_4` — two generations of padding conventions coexisting.

---

## src/common/units.py (9 loc)

**Purpose.** Single-source Rydberg↔eV constants: `RYD_TO_EV = 13.6056980659`, `EV_TO_RYD`.

**Category.** core constants.

**Callers.** `RYD_TO_EV` <- `gw/{gw_output,ppm_pipeline,sc_iteration,sigma_dispatch,gw_config,head_correction,eqp_bgw,ppm_sigma,gw_driver_helpers,gw_jax}.py`, `file_io/sigma_output.py`, `common/__init__.py`.

**Dead suspects.** `EV_TO_RYD`: grep across src/tests/tools/scripts → zero uses outside units.py/__init__.py.

**Redundancy suspects.** The docstring claims the constant was "previously inlined at ~25 sites"; the cleanup only reached `gw/` + `file_io/`. `grep 13.6056` still finds ~15 inlined copies, all in `src/bse/` (`davidson_absorption.py:192`, `eigvals_to_eps2.py:85`, `test_bse.py` x3, `write_eigenvectors.py:151`, `absorption_common.py:23`, `pseudopoles_sweep.py:23`, `bse_io.py:39,700`, `bse_kpm.py:34`, `test_davidson_bse.py:46`, `bse_jax.py:191,329,369`) under four different local names (`ryd2ev`, `RYD2EV`, `RY_TO_EV`, `ry_to_ev`). Refactor: sweep bse/ onto `common.units`.

---

## src/common/provenance.py (29 loc)

**Purpose.** One-line provenance stamps ("Generated by LORRAX <ver> at <UTC>") for human-readable output files.

**Category.** I/O support: output-file provenance.

**Functions.** `lorrax_version()` (importlib.metadata, "unknown" fallback); `provenance_header(comment="#")`.

**Callers.** `provenance_header` <- `gw/eqp_bgw.py:117`, `file_io/sigma_output.py:40,111,185`. `lorrax_version` — internal only. Tested by `tests/active/test_eqp_bgw.py::test_reader_skips_provenance_header`.

**I/O.** Writes header lines into `eqp0.dat`/`eqp1.dat`/`sigma_diag.dat`-style text outputs (via callers).

**Dead/weird/redundancy.** None. Clean.

---

## src/common/progress.py (230 loc)

**Purpose.** BGW-style progress bars for long loops. Two variants: `LoopProgress` (host Python loops, milestone-masked printing with ETA) and `scan_progress` (decorator for `jax.lax.scan` bodies using `io_callback`, one int32 crossing the boundary per milestone).

**Category.** diagnostics: progress reporting.

**Functions.**
| symbol | role |
|---|---|
| `_fmt_time`, `_milestone_mask`, `_format_progress` | formatting helpers |
| `LoopProgress` (`step`, `finish`, ctx-manager) | host-loop progress; rank-0-only by default |
| `_ScanLogState` | mutable host-side start-time holder for the scan variant |
| `scan_progress(num_steps, print_fn, ...)` | decorator; requires `xs = (idx, ...)` with `idx = arange` |

**Callers.**
- `LoopProgress` <- `gw/v_q_tile.py:1230` (V_q tile loop), `gw/ppm_sigma.py:1344` (sigma convolution), `common/isdf_fitting.py:2487` (r-chunk fit loop).
- `scan_progress` <- **nobody** (grep `scan_progress` across src/tests/tools/scripts → only progress.py itself).

**I/O.** stdout via injected `print_fn` only.

**Dead suspects.**
- `scan_progress` + supporting `_ScanLogState` — zero callers. ~90 loc (lines 140-230) of io_callback machinery with no consumer. Either a planned hook for scan-ified loops (cf. the scan-inside-shard_map refactor direction) or delete.
- Typing aliases `CarryT`, `ScanBody` (lines 35-36) unused even within the file.

**Weird code.** None beyond the dead half.

**Redundancy suspects.** Two parallel implementations (LoopProgress vs scan_progress) of the same output format — acceptable by design, but only one is live.

---

## src/common/timing.py (208 loc)

**Purpose.** Hierarchical wall-clock timing: a global thread-local-stacked `TimingCollector` with nested named sections (inclusive/exclusive/count), a `timed` decorator, and a formatted `report()`. `TimingSection.watch(x)` registers `block_until_ready` callables so device work is flushed before the section closes.

**Category.** diagnostics: timing instrumentation.

**Functions.**
| symbol | role |
|---|---|
| `TimingNode` | tree node (count, inclusive, exclusive, children) |
| `TimingSection` | ctx manager; `.watch()` collects `block_until_ready` from pytrees |
| `TimingCollector` | `.reset/.section/.timed/.report/.format`, private `_rows` |
| module-level `get_collector/reset/section/timed/report` | thin wrappers over `_GLOBAL_COLLECTOR` singleton |

**Callers.** Heavily used: `timing.section` ~140 call sites across `gw/` (`gw_jax`, `gw_init`, `ppm_pipeline`, `compute_vcoul`, `kin_ion_io`, `v_q_tile`, `gw_driver_helpers`), `bse/` (all drivers), `psp/` (`get_DFT_mtxels`, `dft_operators`, `kpm_dos`), `centroid/kmeans_isdf`, `common/{isdf_fitting,load_wfns,psi_G_store}`. `timing.timed` <- only `psp/get_DFT_mtxels.py:421,603`. `timing.report`/`reset` <- driver mains (~10 sites each). `get_collector` <- `tests/archive/test_chunked_wfn_loading.py:971` only. `.watch` <- `bse/test_bse.py`, `psp/get_DFT_mtxels.py`.

**I/O.** stdout via `print_fn`.

**Dead suspects.**
- `TimingCollector.format()` (line 177): grep `timing\.format|\.format(**` → zero callers.
- `get_collector()` module function: only an archived test uses it.

**Weird code.**
- `_rows` `min_percent` filter (lines 146-147): a below-threshold node is skipped from output but its children are still recursed and printed at the child's own depth — indentation then implies a parent that isn't shown. Probably intentional ("keep deep hot spots visible") but surprising.
- File uses tab indentation while most of common/ uses spaces (cosmetic; shared with gw_init.py etc.).
- `gw/gw_jax.py:234` comment warns "Do NOT use `_chi_sec.watch(...)` here" — the watch mechanism has a known footgun (forces sync/holds buffer) documented only at the call site.

**Redundancy suspects.** None internal. Overlaps conceptually with `jax_profile.py` annotations (two instrumentation systems), but they serve different outputs (text table vs XLA trace).

---

## src/common/jax_compile_cache.py (120 loc)

**Purpose.** One-shot idempotent activation of JAX's persistent compilation cache, partitioned per `{base}/np{n_proc}/rank{r}` to avoid cross-rank device-assignment cache-key mismatches. Env knobs: `ISDF_JAX_CACHE_DIR` (override; `""` = opt out); default `~/.cache/isdf_jax_compilation`.

**Category.** resource mgmt: JAX compile-cache setup.

**Functions.** `ensure_jax_compile_cache()` (public, idempotent via `_COMPILATION_CACHE_READY`); `_ensure_compilation_cache` back-compat alias.

**Callers.** `ensure_jax_compile_cache` <- `centroid/kmeans_cli.py:192`, `bse/bse_feast.py:1159`, `gw/w_isdf.py:39` (inside a local wrapper), `gw/gw_driver_helpers.py:170`, `psp/run_nscf.py:504`, `psp/run_sternheimer.py:1505`.

**I/O.** Writes/reads the on-disk XLA compile cache dir (~267 entries / 1.9 MB at MoS2 3x3 scale per docstring).

**Dead suspects.** Module-level alias `_ensure_compilation_cache` (line 120): grep → zero importers. The name that IS called (`w_isdf._ensure_compilation_cache`, used by `ppm_sigma.py:520,573,622` and `w_isdf.py:584,626,668`) is a **separately defined local wrapper** in `gw/w_isdf.py:38`, not this alias.

**Redundancy suspects.** `gw/w_isdf.py:38` re-wraps `ensure_jax_compile_cache` under the legacy name and `gw/ppm_sigma.py` reaches through `w_isdf._ensure_compilation_cache` — a parallel old/new access path (the classic `fetch_X_dyn` pattern). Refactor: call `common.jax_compile_cache.ensure_jax_compile_cache` directly everywhere, delete both aliases.

**Weird code.**
- Env var kept as `ISDF_JAX_CACHE_DIR` for shell-alias back-compat though it now caches everything (documented, deliberate).
- Global `warnings.filterwarnings` on "Error reading persistent compilation cache entry" (lines 76-80) — message-targeted but process-global side effect.
- Four nested bare `except Exception: pass/return` ladders — cache setup can silently no-op (by design, but invisible when it fails).

---

## src/common/jax_profile.py (81 loc)

**Purpose.** Thin optional wrappers around `jax.profiler`: `trace_section` starts a trace when `ISDF_JAX_PROFILE_DIR` is set (per-section, timestamped, per-process suffixed dirs); `step_annotation` / `annotation` label host regions inside traces. Everything degrades to no-op when the profiler or env var is absent.

**Category.** diagnostics: XLA profiler integration.

**Functions.** `_trace_path`, `_log_once`, `trace_section(section)`, `step_annotation(name, step_num, detail)`, `annotation(name)`.

**Callers.** `trace_section` <- `gw/gw_jax.py` (chi0_W), `gw/gw_init.py` (zeta_fit, V_q_compute, bispinor variants), `bse/test_bse.py`; `step_annotation` <- `gw/ppm_sigma.py`, `common/isdf_fitting.py`, `bse/test_bse.py`; `annotation` <- `gw/w_isdf.py` (W_solve), `gw/ppm_sigma.py` (sigma_branch).

**I/O.** Writes JAX profiler trace dirs under `$ISDF_JAX_PROFILE_DIR/<section>-<ts>-p<rank>/` (TensorBoard/perfetto format).

**Dead/weird/redundancy.** None significant. `_log_once` flag is misnamed-global-ish (`_warned_trace_failure` gates ALL future messages, not just trace failures) — harmless since only one message exists. Env-var shares the legacy `ISDF_` prefix.

---

## src/common/gpu_utils.py (187 loc)

**Purpose.** Device-memory autodetection for sizing chunk parameters: `get_device_memory_gb()` returns a per-device budget (0.9 x `bytes_limit` from `jax.memory_stats()`, nvidia-smi fallback, `/proc/meminfo`+psutil for CPU backend), `get_device_memory_info()` returns a detail dict.

**Category.** resource mgmt: memory budget detection.

**Functions.**
| symbol | role |
|---|---|
| `_query_nvidia_smi_memory(field)` | subprocess nvidia-smi query, MiB→GB |
| `get_gpu_memory_nvidia_smi()` | free GPU mem via nvidia-smi |
| `_get_jax_gpu_memory_bytes()` | (limit, in_use, available) from `jax.devices()[0].memory_stats()`; forces backend init with `jnp.zeros(1)` |
| `get_cpu_memory_total()` | psutil → /proc/meminfo fallback |
| `get_device_memory_gb(n_devices)` | budget = 0.9·bytes_limit (GPU) or 0.9·total/n_dev (CPU); 4.0 GB last resort |
| `get_device_memory_info()` | dict {backend, total_gb, available_gb, budget_gb, source, n_devices} |

**Callers.** `get_device_memory_gb` <- `centroid/pivoted_cholesky.py:922`, `gw/gw_config.py:863`; `get_device_memory_info` <- `gw/gw_init.py:900`, `gw/gw_config.py:890`; plus `tests/archive/test_chunked_wfn_loading.py:46`.

**I/O.** Reads `/proc/meminfo`; shells out to `nvidia-smi`. No files written (one WARNING print to stderr on fallback).

**Dead suspects.** `get_gpu_memory_nvidia_smi` and `get_cpu_memory_total` are in `__all__` but have zero external callers (grep across src/tests/tools/scripts, excluding gpu_utils.py) — internal helpers masquerading as API.

**Weird code.**
- Budget policy inconsistency: `get_device_memory_gb` = 0.90 x `bytes_limit` (pool size, rank-constant — the documented policy), but `get_device_memory_info`'s `budget_gb` = 0.90 x `bytes_available` (line 149) — a different number that varies with in-use memory. Callers of `get_device_memory_info` (gw_init/gw_config) may see a budget inconsistent with `get_device_memory_gb`.
- Magic constants: 0.90 safety factor (x4 sites), 4.0 GB universal fallback, 8.0 GB default total.
- `_get_jax_gpu_memory_bytes` allocates `jnp.zeros(1).block_until_ready()` purely to force backend init — a hidden side effect (triggers full GPU pool preallocation if PREALLOCATE=true).
- `tests/archive/test_kmeans.py:5` does `from common.gpu_utils import cp` — a cupy symbol that no longer exists; evidence of an excised cupy code path (archived test now un-importable).

**Redundancy suspects.** `get_device_memory_gb` vs `get_device_memory_info` duplicate the whole detection ladder with divergent policies; could be one function returning the dict.

---

## src/common/__init__.py (12 loc)

**Purpose.** Package facade: re-exports `Meta`, `get_enk_bandrange`, `RYD_TO_EV`/`EV_TO_RYD`, and five `cholesky_2d` functions.

**Category.** package plumbing.

**Callers of the re-exports (via `from common import X`).** `Meta` (many, see meta.py) and `RYD_TO_EV` (`gw/gw_jax.py:60`) are genuinely consumed through the facade; also bare `from common import timing / jax_profile` module imports go through the package.

**Dead suspects.**
- `get_enk_bandrange` re-export: all callers (`gw/gw_jax.py:25`, `gw/sc_iteration.py:152`, `gw/gw_init.py:1156`, `bandstructure/htransform.py:25`) import from `common.load_wfns` directly — facade path unused.
- All five `cholesky_2d` re-exports (`cholesky_2d_single`, `cholesky_2d_batched`, `cholesky_solve_2d`, `dense_to_tiles`, `tiles_to_dense`): grep `from common import .*cholesky` → zero hits; consumers import `common.cholesky_2d` directly.
- `EV_TO_RYD` (see units.py).

**Weird code.** Importing `common` eagerly imports `load_wfns` (h5py, big module) and `cholesky_2d` (JAX kernels) even for callers that only want `Meta` or `RYD_TO_EV` — import-time weight for the whole package. Refactor: trim `__init__` to `Meta` + units, or drop the facade entirely (most of the codebase already imports submodules directly).

---

## Cross-cutting observations for the refactor

1. **Two padding generations in `Meta`** (`*_jax` host-count-rounded vs `*_padded`/`b_id_4` world-size-rounded); the legacy fields have near-zero readers — collapse to the world-size convention.
2. **Legacy `ISDF_` env-var namespace** (`ISDF_JAX_CACHE_DIR`, `ISDF_JAX_PROFILE_DIR`) predates the LORRAX rename; documented in `src/ffi/phdf5/ARCHITECTURE.md` and `gw/gw_driver_helpers.py` — a rename would touch run scripts.
3. **`w_isdf._ensure_compilation_cache`** is the one live legacy-alias path; delete both it and the module-level alias in jax_compile_cache.py.
4. **bse/ never adopted `common.units`** — 15 inlined constant copies remain.
5. **`scan_progress` (half of progress.py) is dead** but aligns with the planned scan-inside-shard_map loop refactor; decide keep-and-wire vs delete.
