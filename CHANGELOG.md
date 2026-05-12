# Changelog

## 2026-05-11: Bispinor V_q orchestrator on G-flat — end-to-end MoS2 3×3 [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

The 7-tile bispinor V_q^{μ_L, ν_L} hot loop is now end-to-end on
the G-flat ζ disk format.  All seven unique tiles (CC + 3 TT
diagonal + 3 TT off-diagonal) run through the new per-q +
G-chunked kernel.

Run directory: `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_bispinor_2026-05-11/`
Report:        `reports/gflat_e2e_bispinor_mos2_3x3_2026-05-11/report.md`

### Code changes

- `gw/v_q_g_flat.py`: factored a private `_compute_V_q_g_flat_one_tile`
  helper (~250 LOC) that drives one tile end-to-end.  Charge wrapper
  `compute_all_V_q_g_flat` reduced to a ~30-line bare-Coulomb
  v_per_G builder + helper call.  Kernel parametrized over
  `(n_rmu_L, n_rmu_R)` with separate L/R buffers; the same_zeta
  path still aliases L=R inside the jit.
- `gw/v_q_bispinor.py`: added `compute_V_q_bispinor_g_flat_to_h5`
  (~120 LOC).  Loops over `UNIQUE_TILES`, builds per-tile
  `v(q+G)` via new `_make_per_q_v_builder_for_tile` (CC = bare
  Coulomb; TT = bare · `(δ_ij − K̂_i K̂_j)`), calls the shared
  helper, streams each tile to HDF5.  Reuses the existing
  `tile_dataset_name`, `UNIQUE_TILES`, `HERMITIAN_PAIRS`,
  `BispinorVqReader` (output format is unchanged).
- `gw/gw_init.py`: bispinor dispatch reads the charge ζ's
  `isdf_header.zeta_layout` and routes to the new orchestrator
  on G-flat (opens 4 `ZetaReader` handles).  Legacy r-space path
  preserved as fallback.  Also: copy `sys_dim` onto `meta_curr`
  (dataclasses.replace strips dynamic attrs — caught by the
  bispinor shakedown).

### Numbers (vs the legacy bispinor smoke A_bispinor_smoke_2026-05-08)

ζ disk-shrink (per file, all in `tmp/`):

| File              | Legacy r-space | G-flat new | Ratio |
|-------------------|----------------|------------|-------|
| `zeta_q.h5`       | 4.0 GB         | 177 MB     | 23×   |
| `zeta_q_mu1.h5`   | 2.6 GB         | 181 MB     | 14×   |
| `zeta_q_mu2.h5`   | 4.2 GB         | 181 MB     | 23×   |
| `zeta_q_mu3.h5`   | 4.2 GB         | 181 MB     | 23×   |
| **Total ζ**       | **15.0 GB**    | **720 MB** | **~21×** |
| `v_q_bispinor.h5` | 446 MB         | 424 MB     | 1.05× |

`v_q_bispinor.h5` size is unchanged by design — V_q has
(μ × μ) axes, no G-axis.

V_q wall: 4.2 s for all 7 tiles on 4× A100 (extrapolated ~6×
faster than the legacy μ × ν tile driver from the charge-only
shakedown; total bispinor pipeline 47.3 s).

### Numerics

Bare Σ_X print at k=Γ matches the legacy r-space baseline to
**0.01 eV** band-by-band (-40.0326 new vs -40.0277 legacy at
band 1; matching delta of ~5 meV across all sampled bands).  The
residual is per-q sphere ⊂ shared sphere drop-out of cutoff-edge
G's, as designed.

Bispinor unit tests in `tests/test_compute_V_q_bispinor_g_flat.py`
(committed in `ac735cc`):
* 7 tiles agree with a per-q einsum reference V^{μ_L, ν_L}[μ, ν]
  = Σ_G conj(ζ_L) · v_q^{μ_L, ν_L} · ζ_R to 1e-10;
* CC tile from the bispinor orchestrator is bit-identical to the
  charge-only orchestrator on the same ζ_C file (confirms genuine
  code path sharing, not just structural duplication).

### Notes

- Each new kernel compile emits ~8 `Involuntary full
  rematerialization` SPMD warnings (disk read at `P(None, ('x','y'),
  None)` reshards to `P(('x','y'), None)` inside the jit; XLA does
  full rematerialization instead of an all-to-all).  Non-fatal;
  ~20 MB per q per rank lost to the copy, dwarfed by the kernel
  time.  Followup: have the loader expose a directly-sharded read.

- `eqp0_bisp.dat` (full bispinor Σ^B reading the TT tiles) was
  not emitted by either the baseline or this run for this config —
  appears to need an output flag we haven't enabled.  Separate
  followup.

## 2026-05-11: G-flat ζ + new V_q orchestrator — end-to-end MoS2 3×3 [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

End-to-end shakedown on Perlmutter (1 node × 4× A100, 4-rank).
Writer ran in G-flat mode (`LORRAX_WRITE_G_FLAT_ZETA=1`), V_q via
the new `gw.v_q_g_flat.compute_all_V_q_g_flat` orchestrator, Σ
through the existing path.  No code changes to the kernel since
yesterday's swap commit — only orchestration patches caught in
shakedown.

Run directory: `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_xonly_2026-05-11/`
Report:        `reports/gflat_e2e_mos2_3x3_2026-05-11/report.md`

### Numbers (vs r-space baseline 02_lorrax_xonly)

| Quantity                 | r-space   | G-flat     | Ratio |
|--------------------------|-----------|------------|-------|
| `zeta_q.h5` size         | 2.3 GB    | **101 MB** | **23×** |
| Total wallclock          | 17.2 s    | **11.4 s** | 1.5× |
| `zeta_fit.close_io`      | 3.8 s     | 0.1 s      | **~38×** |
| `V_q_compute`            | 4.4 s     | **0.7 s**  | **6.3×** |
| Σ stage                  | 3.1 s     | 2.9 s      | 1.07× |

sigma_diag agreement: **5 decimals vs r-space baseline** at every
k, band sampled (per-q sphere is a strict subset of the legacy
shared sphere; the few cutoff-edge G's drop out by design since
`v(q+G) → 0` past `zeta_cutoff_ry`).

### Shakedown fixes (commit 6ebfc3e)

- `compute_all_V_q` dispatcher: async prefetch default OFF (env
  `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH=0`).  The worker-thread G-flat
  read deadlocks against the PHDF5 FFI collective in production
  (NCCL kernel collectives interleave with the MPI read collective
  via the GIL in ways that hang).  Sync loop is already 6.3× faster
  than the legacy driver — async is a future opt-in.
- `v_q_g_flat.compute_all_V_q_g_flat`: caller in `gw_init.py`
  passes `ZetaReader`, not `ZetaLoader` (the unit tests use the
  loader).  Orchestrator now detects which API is on hand and
  dispatches accordingly.
- Per-q progress print on the sync path (`read=…s, kernel=…s`):
  one line per q so a stuck JIT compile or NCCL hang is visible
  in `tail -F run.log`.
- `gw_init.py` ζ-peek diagnostic: reads `'zeta_q_G'` on G-flat
  files (was hard-coded to `'zeta_q'`).
- `compare_bgw_gwjax.py` (sandbox top-level): replaced the stale
  `common.wfnreader.WFNReader` import with raw h5py k-list read.

### Followups (unchanged)

- BGW agreement: 0.5 eV gap at band 19 for MoS2 3×3 x-only is
  **pre-existing in the r-space baseline** — not introduced by
  this rewrite.  Worth a separate dig.
- Async prefetch re-enable (NCCL ↔ MPI interleave).
- Bispinor 7-tile orchestrator
  (`gw.v_q_bispinor.compute_V_q_bispinor_to_h5`).

## 2026-05-11: V_q driver swap — new G-flat orchestrator [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

The G-flat V_q hot loop is now end-to-end: ζ̃ read off disk in WFN.h5
per-q sphere layout → v(q+G) built at the per-q Miller components →
G-chunked contract → dynamic_update_slice into (V_acc, g0_acc) →
IBZ-to-full unfold.

- `gw/v_q_g_flat.py` (NEW, ~280 LOC) — `compute_all_V_q_g_flat`.
  Replaces the legacy ``compute_V_q_tile`` / ``_choose_v_q_chunks``
  pipeline for the G-flat-on-disk case.  μ × ν tiling, the chooser,
  the in-V_q FFT, and the shared-sphere conversion all collapse:
  per q, one read + one ``compute_v_q_per_q_g_chunked`` call.  Async
  prefetch (single-step) overlaps the next q's read with the current
  q's contract (borrowed from `v_q_tile`).
- `compute_vcoul.compute_all_V_q` now dispatches on
  ``zeta_io.zeta_layout``: G-flat → new orchestrator; r-space →
  legacy `v_q_tile` path (kept as fallback).
- Tests in ``tests/test_compute_all_V_q_g_flat.py``: synthesised
  G-flat ζ file → orchestrator output bit-matches a one-shot
  einsum reference; async vs sync identical; r-space loader is
  rejected with a clear error.

### Followups

- Larger profile (Si 4×4×4 / MoS2 3×3×1) with the new path enabled
  to validate the disk-shrinkage + I/O-overlap wins claimed for
  the writer.
- Bispinor V^{μ_L, ν_L} 7-tile driver
  (`gw/v_q_bispinor.compute_V_q_bispinor_to_h5`) still uses the
  legacy r-space path; swap follows the same pattern (1 q at a time,
  G-chunked, per-tile signed v).

## 2026-05-11: per-q G-chunked V_q kernel + ζ-cutoff separate from V_q cutoff [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

### Two cutoffs, separate plumbing

`cfg.head.zeta_cutoff` is now an independent knob from
`cfg.head.bare_coulomb_cutoff`.  Both default to `ecutwfc` and cap at
`ecutrho`; `bare_coulomb > zeta` is a hard error (V_q would need ζ̃
values the writer never stored).  The on-disk per-q sphere is built
at `zeta_cutoff`; V_q's `sqrt_v(q+G)` mask uses
`bare_coulomb_cutoff`.

- `gw_config.HeadConfig.zeta_cutoff` (new field).
- `gw_init.fit_zeta`: shared `_resolve_cutoff` helper validates ≤
  ecutrho, raises on `bare > zeta`.
- `isdf_fitting.fit_zeta_to_h5(zeta_cutoff_ry=)` builds the per-q
  sphere at that cutoff and writes it to
  `isdf_header/zeta_cutoff_ry` (renamed from
  `bare_coulomb_cutoff_ry`).
- `ZetaReader` / `ZetaLoader.zeta_cutoff_ry` surfaces it.

### Per-q, G-chunked V_q kernel

New `compute_vcoul.compute_v_q_per_q_g_chunked(zeta_q_L, zeta_q_R,
v_q, g_chunk=...)` evaluates

    V_q[μ,ν] = Σ_G  conj(ζ̃_μ(G)) · v(q+G) · ζ̃_ν(G)

at a single q with the G-axis chunked into `g_chunk` slices.  Each
chunk is a GEMM-shape einsum `'mG,nG->mn'` on
`(n_rmu, g_chunk)` blocks — contiguous G access, no FFT, no
shared-sphere conversion.  Accumulator is donated so repeated calls
(e.g. one per q) sum in place under jit.

The companion `compute_v_q_per_G(q_irr_frac, gvec_components, ...)`
builds `v(q+G)` at the writer's per-q Miller list (matches the
legacy kernel's full-FFT-grid `get_sqrt_v_and_phase` output at the
sphere positions for both 2D slab and 3D bulk — tested).

This is the kernel the rewritten V_q driver will call once swapped
over; the existing `compute_V_q_tile` driver in `v_q_tile.py`
remains in place for the current production hot path.

### Tests

- `tests/test_v_q_per_q_g_chunked.py` (NEW, 9 tests):
  - kernel matches one-shot einsum (3 g_chunk sizes);
  - bispinor off-diagonal (L ≠ R, signed/complex v);
  - pad-slot invariance (ζ̃ = 0 at j ≥ ngk[q] ⇒ zero contribution
    regardless of v(G) there);
  - accumulator donation across multiple kernel calls;
  - alignment-error path (ngkmax not divisible by g_chunk);
  - `compute_v_q_per_G` ≡ legacy `get_sqrt_v_and_phase` at the
    sphere positions, for `sys_dim ∈ {2, 3}`.

### Followups

- Swap `compute_V_q_tile` / `_choose_v_q_chunks` over to the new
  per-q kernel.  The chooser shrinks to G-chunk + memory model
  (q-batching gone in this scope; comment marks the seam for a
  future opt-in).
- Sigma readers that consume V_q[μν] are unchanged — V_q's μ × ν
  output shape is identical to the legacy kernel's.

## 2026-05-11: G-flat ζ on-disk with WFN.h5-style per-q sphere padding [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

Writer now produces ``zeta_q_G(n_q, n_rmu, ngkmax)`` instead of
``zeta_q(n_q, n_rtot, n_rmu)`` when ``LORRAX_WRITE_G_FLAT_ZETA=1`` is
set, with per-q WFN.h5-style sphere components stored alongside.

- **`sources/lorrax_D/src/common/coulomb_sphere.py` (NEW)**
  - `compute_bare_coulomb_sphere_idx(...)` — shared single sphere
    used by V_q kernel.  Extracted from inline code in
    `compute_vcoul.py:246-263` so the writer and consumer share one
    source of truth.
  - `compute_per_q_bare_coulomb_components(...)` — per-q sphere
    `{G : |q+G|² ≤ cutoff}` for every IBZ q, padded uniformly to
    `ngkmax = max_q ngk[q]` with sentinel Miller index
    `(-nx/2, -ny/2, -nz/2)`.  Returns `sphere_idx_padded`,
    `gvec_components_padded`, `ngk_per_q`, `ngkmax`.
  - Fixed `_q_max_cart` bug: enumerates the actual BGW-wrapped
    q-list instead of using the `±0.5/kgrid` half-BZ corners
    (under-bound for the real q-list — even kgrid leaves q=K/2 at
    `q_frac = 1/2` outside the Wigner-Seitz cell).
- **`sources/lorrax_D/src/gw/compute_vcoul.py`**
  - Inline sphere construction at lines 246-263 replaced by a call
    to `compute_bare_coulomb_sphere_idx`.
- **`sources/lorrax_D/src/common/wfn_transforms.py`**
  - `accumulate_rchunk_to_gflat` accepts a 2-D per-q
    `sphere_idx (n_q, ngkmax)` in addition to the legacy 1-D shared
    sphere.  Uses `jnp.take_along_axis(mode='promise_in_bounds')`
    to dodge the XLA x64+shard_map verifier bug.
- **`sources/lorrax_D/src/file_io/isdf_header.py`**
  - New fields: `gvec_components (n_q, 3, ngkmax)`, `ngk_per_q (n_q,)`,
    `bare_coulomb_cutoff_ry`.  Required by `IsdfHeader.build` when
    `zeta_layout == 'G_flat'`; legacy r-space files read with these
    fields set to `None`.
- **`sources/lorrax_D/src/common/isdf_fitting.py`**
  - `fit_zeta_to_h5(..., vcoul_cutoff_ry=...)` accepts the bare
    cutoff; builds the per-q sphere, allocates
    `gflat_acc(n_q, n_rmu_padded, ngkmax)`, gathers per-q after each
    chunk's FFT, masks pad slots to zero post-loop, and persists
    components + ngk + cutoff in the isdf_header.
- **`sources/lorrax_D/src/gw/gw_init.py`**
  - Plumbs `vcoul_cutoff_ry` into both `fit_zeta_to_h5` call sites
    (scalar charge + bispinor transverse μ_L=1,2,3).
- **`sources/lorrax_D/src/file_io/zeta_loader.py` / `zeta_reader.py`**
  - Expose `gvec_components`, `ngk_per_q`, `bare_coulomb_cutoff_ry`,
    `ngkmax_zeta`.
  - Loader: G-flat-on-disk reads `zeta_q_G` directly via the new
    `_read_g_flat_disk` helper.  `layout='r_space'` raises
    `NotImplementedError` against a G-flat file (would need IFFT).
  - Reader: G-flat path raises `NotImplementedError` for the
    "narrow to shared sphere" sub-case (per-q → shared scatter not
    yet wired into the V_q hot loop); raw slab returns work.
- **Disk-size win** (`n_G_sph / n_rtot`, smaller is better):
  - MoS2 3×3×1, cutoff=30 Ry: **11.5%** of r-space (~8.7× shrinkage).
  - Si 4×4×4, cutoff=30 Ry: **16.9%** (~5.9× shrinkage).
  - Si 4×4×4, cutoff=120 Ry (=ecutrho): 94.4% (near full FFT box at
    the rho cutoff — expected).
- **Tests**
  - `tests/test_per_q_sphere.py` (NEW, 6 tests): helper correctness
    vs direct `(q+G)` enumeration, shared-sphere ⊇ per-q-sphere
    invariant, per-q accumulate matches reference FFT+gather,
    header round-trip + validation errors.
  - `tests/test_zeta_loader.py`: bumped 1 test's `IsdfHeader.build`
    call to supply the new required G-flat fields.
- **Validation**
  - 33/33 new + existing G-flat tests pass.
  - Full non-GPU pytest sweep: 181 passed, 20 skipped.  Pre-existing
    `test_kmeans_sharded` failures unchanged (independent of this
    branch).  GPU regression needs a CUDA job allocation (login-node
    cuSolver init fails — same as before).
- **Followups**
  - Wire the per-q → shared-sphere scatter into the V_q wrapper so
    the kernel can consume the new G-flat on-disk format.  Until
    then, the kernel keeps using r-space ζ files.

## 2026-05-11: chunk-capable local FFT helpers + slab-only phase helpers [agent]

Branch `agent/fft-batch-chunks` on `lorrax_A`, rebased onto `origin/main`
`92cbd83`.

- **`sources/lorrax_A/src/common/fft_helpers.py`**
  - added `apply_local_fft(...)`, a reusable device-local FFT helper with
    optional `fft_batch_chunks=` batching over all non-transform axes
  - threaded `fft_batch_chunks=` through
    `make_sharded_{f,if}ftn_3d`, `make_flat_k_fft`, and
    `query_fft_peak_bytes`
  - default remains `fft_batch_chunks=1`, so current production callers keep
    today’s one-shot FFT behavior unless a future refactor opts in
- **`sources/lorrax_A/src/common/wfn_transforms.py`**
  - added generic flat-r helpers:
    `extract_flat_rchunk`, `embed_flat_rchunk`,
    `apply_bloch_phase_flat_points`, `apply_bloch_phase_flat_rchunk`
  - `to_rmu(..., kvecs_frac=...)` now phases only the gathered centroid
    points instead of the whole FFT box
  - `to_rchunk(..., kvecs_frac=...)` now slices the flat-r slab first and
    applies the Bloch phase only on that retained slab
  - `to_rbox` / `to_rmu` / `to_rchunk` now also accept `fft_batch_chunks=`
    for future opt-in use
- **`sources/lorrax_A/src/file_io/zeta_reader.py`** and
  **`sources/lorrax_A/src/file_io/zeta_loader.py`**
  - threaded `fft_batch_chunks=` into the `G_flat` zeta read path so the
    upcoming `rchunk <-> G_flat` zeta/V refactor can reuse the same helper
    without reopening reader internals
- **Tests**
  - `tests/test_fft_helpers.py`: new chunked-helper coverage and
    chunk-aware `query_fft_peak_bytes` coverage
  - `tests/test_wfn_transforms.py`: new phased `to_rmu`,
    phased/chunked `to_rchunk`, and flat-r helper coverage
- **Validation**
  - `uv run python -m pytest -q tests/test_fft_helpers.py` → `5 passed`
  - `uv run python -m pytest -q tests/test_wfn_transforms.py` → `16 passed`
  - `uv run python -m pytest -q` → `182 passed, 20 skipped, 4 failed`
  - remaining failures are unchanged from `main`:
    - `tests/test_gw_jax_regression.py::test_gw_jax_matches_reference`
      (`write_qp_wfn_h5` shape mismatch)
    - `tests/test_kmeans_sharded.py::test_refactored_matches_naive[fcc-avec1]`
    - `tests/test_kmeans_sharded.py::test_refactored_matches_naive[skew-avec2]`
    - `tests/test_kmeans_sharded.py::test_pbc_distance_scan_matches_naive_fcc`
- **Report**
  - `reports/fft_helper_unification_2026-05-11/report.md`

## 2026-04-21: analytic chunk chooser + γ-calibrated AOT memory model [agent C]

Branch `agent-C/aot-memory-model` on `lorrax_C`.  Closes Phase 6 of the
AOT memory-model initiative — the chooser now predicts runtime peak
bytes to ≤1% error after γ calibration.

- **`src/gw/aot_memory_model/chooser.py`**: new
  `choose_chunks_analytic(sys, mesh, budget)` — regroups the 7 memory
  primitives into 4 scaling classes via a `PRIMITIVE_CLASSES` dict on
  the kernel (`const`, `cr`, `bc`, `crbc`).  The feasibility bound
  ``peak ≤ M`` is linear in chunk_r at fixed bc, so
  ``chunk_r_max(bc) = (M − α₀ − α_bc·bc) / (α_cr + α_crbc·bc)`` is a
  closed-form inversion.  Chooser 1-D searches over bc candidates, no
  2-D grid.  Adds an optional `fft_launch_overhead_flops` knob for
  calibrating the "small-bc performance hit" post-hoc.
- **`src/common/isdf_fitting.py`**: replaced the
  `jax.devices()[0].memory_stats()` peak tracker (returns `None` on
  this CUDA PJRT) with a single `nvidia-smi` sample at the end of the
  r-chunk loop.  Per-chunk sampling inside the Shifter container was
  observed to hang on some Perlmutter node types.
- **`src/gw/gw_init.py`**: prints `γ = runtime_peak / aot_predicted`
  at the end of `fit_zeta` whenever both numbers are available.  Also
  corrected the `aot_sys.n_b` passed to the predictors: use the union
  range (`nb_full`) not `nb_L + nb_R`; the cost primitive's factor of
  2 handles the L+R sum.
- **`fit_one_rchunk__current__fit.json`**: records **γ=0.510**
  calibrated at MoS2 3×3 nosym (runtime nvidia-smi = 3.06 GB vs AOT
  worst-case = 6.00 GB).  `Fit.gamma` is applied by both
  `predict_peak` and the analytic chooser's `_group_alpha`, so
  chooser-predicted peaks now match runtime to within measurement
  noise.
- **Validation** — with `memory_per_device_gb=4` and
  `use_aot_chunk_chooser=true` at MoS2 3×3:

  ```
  AOT chooser: chunk_r=46080 band_chunk=80 (1×1 jits,
      peak=3.06 GB / 3.88 GB = 79%, total=7.3 GF)
  ```

  Chooser-predicted 3.06 GB matches the earlier runtime nvidia-smi
  measurement (3.06 GB).  Budget is genuinely hit.  Bit-identical
  eqp0 vs baseline `c8fc139fb22d2653d585874fe19c72a7`.
- **Follow-ups**: (1) widen the bc DoE axis — current 11-sample fit
  collapses bc-sensitivity into zero β.  (2) γ calibration at Si 4×4×4
  60Ry — may need mesh-scaling γ rather than a global scalar.
- **Report**: [reports/aot_memory_model_poc_2026-04-20/report.md](
  reports/aot_memory_model_poc_2026-04-20/report.md) — Phase 6 section
  with α decomposition, γ measurement, budget-hit validation table.

## 2026-04-21: phdf5 on-demand G-space during ISDF fit [agent C]

Branch `agent-C/aot-memory-model` on `lorrax_C`.  Follow-on to same-day
"jit the r-chunk body" work.

- **`src/common/isdf_fitting.py`**: new `use_phdf5_gspace: bool`
  parameter to `fit_zeta_chunked_to_h5`.  When True, the driver skips
  the device-resident G-space cache (`load_gspace_for_bands`) and
  instead calls `PhdfWfnReader.coeffs_gspace(band_range)` fresh per
  r-chunk per band-chunk.  The tuple is `del`'d right after the
  `fit_one_rchunk` jit returns — nothing persists between r-chunks.
- **`src/gw/gw_config.py` + `gw_init.py`**: `use_phdf5_gspace` surfaces
  as a `cohsex.in` flag and threads into `fit_zeta`.
- **Duck-type**: `PhdfWfnReader.coeffs_gspace` already returns
  `(n_k, nb_pad, n_s, nx, ny, nz)` with
  `P(None, ('x','y'), None, None, None, None)`, matching the cached
  path's shape/sharding contract exactly.  No FFI-reader signature
  changes were needed; the driver-side factory is four lines.
- **Validation** (MoS2 3×3, `use_phdf5_gspace=true`):
  - single r-chunk + `use_ffi_io=true`:
    md5 `c8fc139fb22d2653d585874fe19c72a7` ✓
  - multi-chunk (5×10000 + remainder 6080) + `use_ffi_io=false`:
    same md5 ✓
  - Multi-chunk + `use_ffi_io=true` fails with concurrent HDF5 MPI-IO
    errors in the async zeta_q writer.  Pre-existing interaction —
    PhdfWfnReader + SlabIO-FFI race on MPI-IO state on the same ranks.
    When both flags are needed, use `use_ffi_io=false`.
- **Memory win**: zero persistent GPU footprint for the per-band-chunk
  G-space cache between r-chunks.  ~265 MB per rank saved at MoS2 3×3
  (small); multi-GB at Si 10×10×10 1000+ bands, where it pushes the
  pre-rchunk CCT/cholesky stages back under budget.
- **Timing**: +0.2 s total at MoS2 3×3 multi-chunk (4.3 s vs 4.1 s).
  Negligible under phdf5; would be slow under legacy h5py (keep flag
  opt-in).
- **AOT model**: per-r-chunk peak unchanged — `psi_bc_G_tuple` is
  still a jit input, so `argument_size_in_bytes` is identical.  Phase
  1b benefits show up in the *between-rchunk* GPU residency, which the
  AOT kernel doesn't currently measure.
- **Report**: [reports/aot_memory_model_poc_2026-04-20/report.md](
  reports/aot_memory_model_poc_2026-04-20/report.md) — new Phase 1b
  section with validation matrix and the FFI-writer conflict note.

## 2026-04-21: jit the r-chunk body + AOT-model fit_one_rchunk [agent C]

Branch `agent-C/aot-memory-model` on `lorrax_C`.

- **`src/common/isdf_fitting.py`**: new
  `_make_fit_one_rchunk_kernel` factory + `fit_one_rchunk` entry point.
  The full per-r-chunk body (FFT+reshard per band-chunk, streamed
  spin-traced pair-density accumulate, ZCT, Z→col reshard, Cholesky
  solve) is now one jitted kernel.  `fit_zeta_chunked_to_h5` calls it
  once per r-chunk.  Two compile variants per run (full + remainder).
- **`src/common/load_wfns.py`**: `get_sharded_wfns_rchunk_slice`
  signature refactored `(r_start, r_end)` → `(r_start, r_chunk_size)`
  so `r_start` can be a tracer inside an outer jit.  Callers in
  `iter_psi_rchunk_bandwise` updated.
- **`src/gw/aot_memory_model/kernels/fit_one_rchunk.py`**: new
  composite AOT kernel mirroring the production factory.  Captures the
  driver-level memory peak including coexisting buffers that per-stage
  kernels can't see.  Primitives: `Pacc`, `PrBc`, `psiBc`, `psiBcY`,
  `psi_cent`, `L_q`, `psiG_total`.
- **`src/gw/aot_memory_model/core.py`**: `SysDims` gains an optional
  `fft_grid` field + `fft_shape` property for kernels that need both
  k-grid and real-space FFT box.
- **`src/gw/aot_memory_model/presets.py`**: `points_fit_one_rchunk`
  for `mos2_3x3` and `si444_60Ry`.
- **`src/gw/gw_init.py`**: logs the AOT-predicted driver peak
  alongside the existing per-stage heuristic — sanity-check-only, does
  not override `chunk_r` yet.
- **Validation**: MoS2 3×3 COHSEX single-chunk (46080 pts) and
  multi-chunk (r_chunk_size=10000, 5 chunks + remainder 6080) both
  produce `md5sum eqp0.dat == c8fc139fb22d2653d585874fe19c72a7` matching
  the reshard-fix baseline.
- **NNLS fit** (11 DoE points, residual RMS 0.23 GB on ~3 GB peaks):
  β[PrBc]=1.03, β[L_q]=5.02, β[psiG_total]=1.65.  Saved at
  `src/gw/aot_memory_model/artifacts/fit_one_rchunk__current__{fit,samples}.json`.
- **Report**: [reports/aot_memory_model_poc_2026-04-20/report.md](
  reports/aot_memory_model_poc_2026-04-20/report.md) — new "Update
  2026-04-21" section with primitives table, fit coefficients, next
  steps.
- **Next**: (1) host-resident `cached_gspace` via phdf5-like duck type
  — attacks the `psiG_total=1.65` primitive directly; (2) switch
  `compute_optimal_chunks` to use AOT prediction for the
  pair+zct+reshard+solve sub-loop; (3) γ-calibrate at Si 4×4×4 60 Ry.

## 2026-04-20: phdf5 FFI — independent writes by default, Cray MPICH now viable [agent A]

Branch `agent-A/independent-writes-default` on `lorrax_A`.

- **`src/ffi/phdf5/cpp/ctx.h`**: split `use_collective` into
  `use_collective_read` (default `true`) and `use_collective_write`
  (default `false`).  `coll_metadata` now defaults to `false`.
- **`src/ffi/phdf5/cpp/context.cc`**: new env-var surface.
  `LORRAX_PHDF5_INDEPENDENT=1` still forces reads independent too (power
  user override).  New `LORRAX_PHDF5_COLLECTIVE_WRITES=1` to opt writes
  back into collective (do NOT set on Cray).  New `LORRAX_PHDF5_COLL_META=1`
  to re-enable collective metadata.
- **`src/ffi/phdf5/cpp/write_ffi.cc` + `read_ffi.cc`**: dxpl selection
  now uses the per-direction flag.
- **Why**: the Cray MPICH collective write driver
  (`ad_cray_write_coll.c:669`) OOMs at ≥ 1 GB/rank regardless of
  `cb_*`, `stripe_*`, `alloc_time`, or `cray_cb_write_lock_mode` knobs.
  The fix that prior investigation missed was the combination of
  `H5FD_MPIO_INDEPENDENT` writes AND non-collective metadata ops —
  both are needed to fully bypass the buggy driver.  Independent
  writes are neutral on OpenMPI at our measured sizes; collective
  reads are preserved (ROMIO two-phase is optimal on both stacks).
- **Regression data** (1 node / 4 GPUs, post-fix defaults):

  | workload | OpenMPI | Cray MPICH |
  |---|---|---|
  | MoS2 3×3 `phdf5_multi_offset_test` | PASS | PASS |
  | MoS2 3×3 `phdf5_profile` (45 MB WFN) | 18.1 ms (was 18.3) | **17.9 ms** |
  | MoS2 3×3 `phdf5_profile --centroids` (gw_jax load) | 26.2 ms | 26.4 ms (parity) |
  | n=16384 C128 `phdf5_read_bench` (4.29 GB) | 3.04 GB/s | **3.79 GB/s** (was CRASH) |

  Cray now works at all scales and beats OpenMPI at large scale; at
  MoS2 3×3 scale the two stacks are within noise.  Unification around
  Cray for cross-cluster portability is viable.
- **Docs**: `src/ffi/phdf5/ARCHITECTURE.md` env-var table and
  `src/ffi/PORTING.md` Option B write-up refreshed.

## 2026-04-20: flat-k FFT helper — one wrapper for kx/ky/kz across the GW pipeline [agent C]

Branch merged to `main` as commit `c9bd801`.

- **`src/common/fft_helpers.py`** — new `make_flat_k_fft` /
  `make_flat_k_ifftn` / `make_flat_k_fftn`.  Callers hand it flat-k
  `(nk, *trail)` arrays, the helper does
  `reshape → with_sharding_constraint → custom-partitioned 3-D FFT →
  reshape back`.  `kgrid` and the 3-D PartitionSpec are closure state;
  the 3-D form never appears in caller code.
- **Call sites wired through**:
  - `gw/w_isdf.py` chi0 minimax — three FFT closures collapsed to
    helper calls (`Gv_ifftn`, `Gc_fftn`, `chi_fftn_local`).
  - `gw/ppm_sigma.py` `_sigma_kij_kernel` — `_fft_flat_G` /
    `_fft_flat_V` closures replaced.
  - `gw/gw_jax.py` — `_make_fft_pair` factory removed in favor of
    direct helper calls for G and V.
  - `common/isdf_fitting.py` — `CCT_LR`, `CCT_LR_spin_matrix`,
    `ZCT_LR`, `ZCT_LR_spin_matrix` all refactored to take flat-k
    input end-to-end.  The pre-ZCT `reshape → with_sharding_constraint`
    in the r-chunk loop was deleted (ZCT now takes flat-k directly).
    `donate_argnums` re-enabled on CCT_LR (0, 1), ZCT_LR `_left_ifft_conj`
    (0) and `_right_ifft_mul_fft` (0, 1), and on the spin-matrix
    variants — a handful of one-shot trace-time XLA aliasing warnings
    remain (rank-3 → rank-5 intermediate), but there are no per-call
    donation failures.
- **Validation** (`runs/Si/C_flatk_si10/`, Si 10×10×10 mem12, 4 GPUs):
  - `eqp0.dat` **byte-identical** to `C_stream_si10_transposed`
    baseline (0-byte diff).
  - Total runtime 304.7 s → **275.9 s** (9.5 % faster).  Savings
    concentrated in `zeta_fit.chunk_loop` (30 s → 25.5 s) from
    pair-density donation; `close_io` unchanged (65.9 s → 63.2 s);
    Σ computation unchanged.
- **Why it matters**: single point where the 3-D FFT happens, so the
  NUFFT substitution the user has in mind is a one-file change with
  no call-site churn.

## 2026-04-18 (midday): scissor-shift for out-of-grid bands + Si 4×4×4 pseudobands end-to-end [agent A]

Branch `agent-A/scissor-shift-sc-gw` on `lorrax_A`
(commits `dfc880c`, `9b0e666`).  Full write-up in
`reports/scissor_shift_2026-04-18/report.md`.

- **`src/gw/scissor.py`** — `ScissorFit` dataclass, `fit_scissor` (numpy
  OLS, separate valence / conduction lines), `extrapolate_delta_e`, and
  `add_diag_to_H_kmn` (shard_map-based diagonal add onto a
  `P(None,'x','y')`-sharded Hamiltonian, ready for the future SC loop).
  Smoke-tested 4-GPU: fit recovers synthetic slopes to <6e-6, sharded
  diagonal add bit-identical to numpy (maxabs 0.0), P(None,'x','y')
  output sharding preserved, divisibility check raises cleanly.
- **`src/gw/gw_jax.py`** — G0W0 PPM post-processing now honors the
  `sigma_at_dft_extrapolate` config knob: out-of-grid bands get the
  fitted affine QP correction instead of the static-COHSEX fallback.
  Fixed two adjacent bugs while wiring:
  - `E_qp_ev * ryd2ev` unit double-count in the original `in_grid`
    mask — every state looked out-of-grid, so the diagonal Sigma
    fixed-point was silently discarded for all bands.
  - in-grid test must use `E_DFT`, not `eigvalsh(H_qp)`, because
    pseudobands' non-unit norms scale `<n|H|n>` by the pseudoband
    weight and produce garbage eigenvalues for compressed states.
- **First end-to-end test** — `runs/Si/A_06_si_4x4x4_scissor/`.
  Si 4×4×4 nosym, BGW-convention pseudobands via
  `psp.run_nscf --pseudobands` (50 windows × 2 pseudobands = 8 prot +
  98 pseudo = 106 bands).  Σ(ω) grid ±5 eV so all pseudobands
  (onset ~+10 eV) are out-of-grid.  GN-PPM G0W0 + scissor on 4×A100:
  run wall 30 s, 668/6784 in-grid, valence fit
  α=-0.44, β=-6.24 eV (RMSE 1.07 eV), conduction fit
  α=-0.64, β=-0.61 eV (RMSE 2.37 eV).  E_QP vs E_DFT is a single
  smooth line across the full 0→330 eV bandrange with no jump at the
  in-grid / out-of-grid boundary — continuity goal met.  Magnitudes
  over-correct at the high-E tail (E_QP ≈ 0.36·E_DFT → highest
  pseudoband lands at ~120 eV vs DFT 330 eV); expected for a line fit
  over 10 eV extrapolated to 300 eV, worth revisiting with a softer
  A + B/E tail or a damped law later.
- **Known issue documented**: `psp.get_dipole_mtxels` crashes on a
  pseudobands WFN (`vnl_velocity_matrix` hits a `None` `dZ`).  Worked
  around in this run by routing the q→0 head through the BGW
  `eps0mat.h5` from `runs/Si/02_si_4x4x4_nosym/01_bgw_gnppm/` with
  `wcoul0_source = epshead` in `cohsex.in`.

Validation: `uv run python -m pytest -q` on `lorrax_A` → 13 passed,
1 pre-existing reshard failure (unchanged from prior state).

## 2026-04-18 (overnight): sigma_ppm cleanup + compile-cache trims + zeta_fit probe [agent C]

Branch `agent/C-sigma-ppm-cleanup` on `lorrax_C`. Full write-up in
`reports/session_2026-04-18_async_probe/report.md`.

MoS2 3×3 / 4-GPU run_module wall: **47.3 s → 34.7 s (−27 %)**, eqp0.dat
bit-identical at every commit (16 substantive + 1 TEMP profiling).

- **Reduce-scatter in `_sigma_kij_kernel`**: `projection_kernel.project_ri`
  tail replaced by a shard_map'd local einsum + `psum_scatter × 2` (m on x,
  n on y).  σ^τ now emerges `(m_X, n_Y)`-sharded; every downstream ω-kernel
  multiply + accumulate is rank-local.  HLO diff shows 4× `all-reduce
  c128[9,2,2,320,80]` flipping to `reduce-scatter c128[2,9,40,2,320]` per
  τ step, same byte volume but output is now sharded.
- **σ^τ as a (re, im) tuple** from the shard_map — removes the
  `sigma_ri[0]/[1]` indexing pjits and the `is_fully_addressable` assert
  that a multi-process tuple-unpack of a sharded (2, …) stacked array
  would trigger.
- **New `_ReduceScatterGpuAccumulator`** is the default buffered path;
  Σ_c(ω, k, m, n) is held sharded on GPU so it's n_b²/p² per rank instead
  of replicated.  `_BufferedGpuAccumulator` deleted (was redundant).
- **lax.scan τ-loop infrastructure** landed as `_get_sigma_tau_scan_kernel`
  + `_ReduceScatterGpuAccumulator.run_window_scan`, **off by default** —
  regresses at MoS2 3×3 scale (fewer overlap opps with big fused module +
  per-window compile).  Reconsider at padded-τ or larger mesh.
- **Physics visibility pass**: module docstring states the quadrature
  formula directly; `_iter_branches` NamedTuple with comments deriving
  kernel_sign / scale flips; `_run_sigma_branch` reads like a physics
  outline; `_combine_coeff_with_sigma_tau` documents the re/im split and
  drops the dead "real" branch; `_convolve_sigma_branch_kij` →
  `_run_sigma_branch`; 'channel' scrubbed from factory names.
- **Dead-param / dead-class purge**: `omega_sign_flip` (always +1), the
  unused `_BufferedGpuAccumulator`, the one-line wrapper
  `_accumulate_tau_into_window`.  −203 / +102 lines in one commit.
- **Compile-cache trims — numpy for tiny host-side helpers**:
  `get_enk_bandrange`, `fft_integer_axes`, `exp_ikr_fftbox`, `_build_Gij`,
  `_build_occ`.  Each had emitted ~8–16 standalone pjits at trace time
  for pure host bookkeeping.  TRACING CACHE MISS 313 → 269 (−44).
  `wavefunction_setup` section **1.79 s → 0.18 s** (the old `jnp.zeros_like`
  + `.at[].set` on sharded input had a non-trivial runtime tied to
  cross-device scatter).
- **zeta_fit chunk loop**: dropped the per-chunk `sync_global_devices`
  (the allgather is itself a collective; one rendezvous at the end is
  enough).  Investigated async-allgather paths and confirmed JAX has no
  async `process_allgather`-to-host API; the 1.95 s first-collective
  NCCL setup is the floor without pre-warming or the phdf5 FFI path.

Future work documented in-tree (heavy comments at each extension point):
  τ batching, m-chunking, `_CollectiveFlushSlabIoAccumulator` (FFI SlabIO
  collective-write variant for multi-process streamed output).  `zeta_fit`
  remains the dominant cost bucket (47.6 % of total) and is the natural
  next target.

## 2026-04-17 (pm): k-means ISDF — parallelism refactor + 4-GPU sharding prototype [agent B]

Branch `agent/kmeans-sharded` on `lorrax_B`. Full write-up in
`reports/kmeans_sharded_2026-04-17/report.md`.

- **Refactored `centroid/kmeans_isdf.kmeans_update_step`** to eliminate the
  double (P, K, 3) tensor materialization: segment-sum over labels replaces
  the one-hot-mask weighted mean; a `lax.scan` over K-chunks replaces the full
  pairwise distance tensor (peak (P, `k_block`, 3) instead of (P, K, 3)).
  PBC minimal image and metric tensor behavior are byte-compatible with the
  old implementation (new regression test covers orthorhombic / FCC / skew
  cells and the cross-boundary minimum-image case).
- **Added `make_sharded_kmeans_update`** — `shard_map`-based parallel Lloyd
  step. P sharded on mesh axis `'x'`, centroids replicated, one `lax.psum`
  per iteration on the (K, 3) / (K,) accumulators. Verified bit-identical
  single-GPU vs 4-GPU on Si 4×4×4 (matching md5 on `centroids_frac_128.txt`),
  same 71-step trajectory.
- **Fixed latent `alat`-vs-`Å` mislabel in `main()`.** BGW WFN.h5 stores
  `avec` in alat units and `alat` in Bohr; the old code treated `|avec row|`
  as Å, which silently inverted a ~2× grid upsample into a ~0.6× downsample.
  `main()` now converts to Å once via `wfn.alat * BOHR_TO_ANG`; the kmeans
  function docstring states distances inherit the caller's avec units.
- **Multi-process bootstrap.** Added the standard `_maybe_init_jax_distributed`
  to the module so `srun -n N>1` works (matches `psp/run_nscf.py`, `gw/gw_jax.py`).
  Prototype uses the simpler single-process-4-GPU path.
- **New tests** (`tests/test_kmeans_sharded.py`): 5 cases, all pass. Full
  suite: 18 pass, 1 pre-existing failure in `test_reshard_all_to_all.py`
  unrelated to this branch.
- **Sandbox doc hardening.** `skills/execute_workflow/SKILL.md` now says
  explicitly: never export a `SLURM_JOBID` you did not allocate yourself; the
  interactive-allocation section documents the background `salloc` +
  `-J lorrax_X_agent` naming pattern. Matching memory pointer at
  `memory/feedback_never_share_allocation.md`.
- **Run**: `runs/Si_B/00_si_4x4x4/` — fresh 4×4×4 Si sym-reduced QE (8 IBZ
  k-pts, 24³ FFT, 16 bands) → 48³ kmeans grid. Three sub-dirs hold the
  baseline, refactored-single-GPU, and sharded-4-GPU centroid outputs for
  the equivalence check.

## 2026-04-17: Three parallel LORRAX checkouts (A/B/C) for concurrent agents

Consolidated the previous per-sandbox LORRAX clones into three sibling
checkouts at `$HOME/software/lorrax_{A,B,C}`, symlinked into the sandbox
as `sources/lorrax_{A,B,C}`. Each agent session claims one letter and
touches only its own checkout. Shared Shifter stage trees remain at
`/pscratch/sd/j/jackm/lorrax_{nvhpc,phdf5_openmpi}` (read-only in the
container), so the three variants share bind-mounted deps but build
their own `src/ffi/common/cpp/build/liblorrax_ffi.so`.

- `config/perlmutter/install.sh`, modulefile template: new
  `LORRAX_MODULE_NAME` variable lets each checkout install its own
  modulefile (`lorrax_A`, `lorrax_B`, `lorrax_C`). `family("lorrax")`
  makes variants mutually exclusive in a single shell; across shells
  they are fully independent. Landed on `main` (LORRAX feature branch
  `agent/multi-checkout`, fast-forwarded).
- Sandbox `AGENTS.md`: new "Which agent are you?" section at the top,
  revised source-code table, non-negotiable rule #7 ("only edit your
  assigned checkout"). `execute_workflow`, `checkpoint`,
  `profiling_stack` skills updated to say `sources/lorrax_X` /
  `module load lorrax_X`.
- Deleted stale sandboxes: `lorrax_sandbox_fresh`,
  `lorrax_sandbox_profiling`, and their backing clones
  `$HOME/software/lorrax_{bse,profile_ppm}`.
- `pyproject.toml`: dropped the sandbox-level `lorrax` editable
  dependency; the path no longer resolves to a single variant. Host
  Python that imports LORRAX should run inside Shifter via `lxrun`, or
  `uv run` from inside a specific `sources/lorrax_X`.

## 2026-04-17: WFNReader full-zone symmetry wrappers

- Audited the raw wavefunction reader usage after the nonsymmorphic-phase work.
  In active `src/`, raw `get_cnk()` / `get_cnk_batch()` are only consumed by
  `SymMaps.get_cnk_fullzone*`; there was no active path pairing unfolded
  `get_gvecs_kfull()` output with raw irreducible-zone coefficients.
- Clarified the API in both `src/common/wfnreader.py` and
  `src/file_io/wfnreader.py`: raw `get_cnk*` / `get_gvec_nk` remain explicit
  irreducible-zone readers, and new `get_gvecs_kfull`,
  `get_cnk_fullzone`, and `get_cnk_fullzone_batch` wrappers now route full-BZ
  access through `SymMaps` so the non-symmorphic `tau` phase is applied in the
  safe path by construction.
- Switched active consumers in `src/common/load_wfns.py`,
  `src/bandstructure/htransform.py`, and
  `src/centroid/get_charge_density.py` to the new WFNReader full-zone
  wrappers.
- Verified on Si `4x4x4` symmetry-vs-nosym WFNs that for all `44` k-points
  unfolded with nonzero `tau`, `get_gvecs_kfull()` matches the nosym G-list as
  a set, confirming `tau` does not act on the integer G-list itself.
- Added wrapper regression coverage in
  `tests/test_symmetry_maps_nonsymmorphic.py`.
- Validation: `uv run python -m pytest -q tests/test_symmetry_maps_nonsymmorphic.py`
  passed (`4 passed`), and full `uv run python -m pytest -q` passed
  (`15 passed, 1 warning`).

## 2026-04-16 (pm): cuSOLVERMp FFI unblocked — NCCL-backed cal_comm_create works

Follow-up to the earlier "cuSOLVERMp WIP" entry.  The SIGFPE in
`cusolverMpSyevd_bufferSize` was a communicator-plumbing bug:
NVIDIA's sample passes `ncclComm_t` directly to a `cal_comm_t`-typed
API, which works under `MPI_Init` but *not* in a JAX-only C++ process
(C implicit pointer-conversion quietly becomes a bug in C++).  The
documented non-MPI CAL path — `cal_comm_create` with user
allgather/req_test/req_free callbacks — routes through NCCL cleanly.

### Result (job 51659364, nid001033, 1 node × 4×A100)

| Path | n   | type        | max \|evals − ref\| |
|------|-----|-------------|---------------------|
| cuSOLVERMp (multi-proc, NCCL)   | 128 | F64  sym    | 9.1e-13 |
| cuSOLVERMp (multi-proc, NCCL)   | 128 | C128 Herm   | 5.7e-13 |

Both on a 2×2 process grid with `NamedSharding(P('x','y'))`.

### Source changes (branch `agent/ffi-cusolvermp`, commits 22ed74a, 7c716a7)

- `src/ffi/cusolvermp/cpp/ctx.h`: add `CalNcclShim` (NCCL comm + stream
  + persistent device scratch buffer), add `cal_comm_t` field on Ctx.
- `src/ffi/cusolvermp/cpp/context.cc`: three static callbacks
  (`cal_nccl_allgather` = H2D→`ncclAllGather`→D2H→stream-sync,
  `cal_nccl_req_test/free` trivial since we're synchronous).  Replace
  `reinterpret_cast<cal_comm_t>(ncclComm)` with a real
  `cal_comm_create(params, &ctx->cal_comm)`.  Teardown extended.
- `src/common/cusolvermp_eigh_test.py`: `gather_to_numpy` now uses
  `multihost_utils.process_allgather(x, tiled=True)` so each rank
  can verify the full logical array in multi-process mode.
- `src/ffi/AGENTS.md`: mark cuSOLVERMp status as working, document the
  three required env vars and why.

### Required runtime env

```
CUSOLVERMP_FORCE_NCCL=1              # route libcal's runtime collectives via NCCL
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5   # leave headroom for cuSOLVERMp workspace
XLA_PYTHON_CLIENT_PREALLOCATE=false  # allocate on demand, not up front
```

Without the first flag, libcal's internal reduce goes through UCC and
trips `Failed to parse ib device list` in the container.  Without the
memory settings, `cudaMalloc(scratch)` inside UCC fails because JAX's
modulefile default reserves 95% of VRAM up front.

### Why this is the real scaffold for ELPA

The multi-process NCCL bootstrap is the hard part; everything else
(`XLA_FFI_DEFINE_HANDLER_SYMBOL` in C++, `jax.ffi.ffi_call` wrapped in
`shard_map` in Python, ctypes-loaded .so, bind-mounted host libs, JAX
KV-store unique-id broadcast) transfers to ELPA 1-for-1.  ELPA takes
an MPI communicator instead of NCCL; that swaps `ncclGetUniqueId` /
`ncclCommInitRank` for their MPI equivalents but does not change the
control flow.

### Report

`reports/ffi_cusolvermp_nccl_2026-04-16/report.md`.

## 2026-04-16: JAX FFI scaffolding — cuSOLVERMg eigh working on 4 GPUs; cuSOLVERMp WIP

New directory `sources/lorrax/src/ffi/` with pluggable scaffolding for
calling compiled parallel-LA libraries from JAX via the XLA FFI.  No
pybind/nanobind; the `.so` is plain C ABI loaded with `ctypes.CDLL` and
its XLA handlers wrapped via `jax.ffi.pycapsule` — the pattern from
NVIDIA's JAX FFI tutorial.

### Working — `ffi.cusolvermg` (single-process, multi-GPU)

- `src/ffi/cusolvermg/cpp/eigh_mg_ffi.cc`: XLA FFI handler that owns a
  lazy `cusolverMgHandle_t` + pairwise peer access, scatters the
  device-0 input into cuSOLVERMg's column-tile layout via
  `cudaMemcpyPeerAsync`, runs `cusolverMgSyevd` across all visible GPUs,
  and gathers `Q` back to device 0.
- `src/ffi/cusolvermg/eigh.py`: `eigh_mg(A, tile_size=32, max_gpus=0)`.
- `src/common/cusolvermg_eigh_test.py`: 4-GPU Python test.

Validation on 1 node × 4×A100 (job 51656242, nid001164):

| n    | tile | max \|evals − ref\| | wall (post-warmup) |
|------|------|---------------------|--------------------|
| 128  | 32   | 9.1e-13             | 57 ms              |
| 2048 | 256  | 2.2e-11             | 509 ms             |

Eigenvector residuals `‖A q_i − λ_i q_i‖∞` ≈ 7e-14 (F64).

### WIP — `ffi.cusolvermp` (multi-process, multi-GPU/multi-node)

Everything builds, links, and runs up to the solve.  NCCL bootstrap
works via `jax.distributed.global_state.client` KV-store broadcast of
a 128-byte `ncclUniqueId` (note: `multihost_utils.broadcast_one_to_all`
silently promotes `uint8 → uint64` under `jax_enable_x64=True` and
scrambles it — workaround is documented in the code).  `cusolverMpCreate`,
`CreateDeviceGrid`, `CreateMatrixDesc` all succeed.  `cusolverMpSyevd_bufferSize`
then SIGFPEs (integer divide-by-zero) at a constant offset inside
`libcusolverMp.so`.

Most likely cause: NVIDIA's `mp_syevd.c` sample passes `ncclComm_t`
directly to an API typed `cal_comm_t` — the C implicit pointer
conversion plus an MPI-initialised libcal recognises the wrap; our
JAX-only process never calls `MPI_Init` so libcal's NCCL-detection path
never arms, `cal_comm_get_size` returns 0, and bufferSize divides.
Preserved as branch `agent/ffi-cusolvermp`; follow-ups
documented in `src/ffi/AGENTS.md` and the report.

### Build + runtime environment

- Container: `nvcr.io/nvidia/jax:25.04-py3` (CUDA 12.9, JAX 0.5.3.dev,
  `libcusolver*`, `libcusolverMg*` in-container).
- NVHPC (for cuSOLVERMp ONLY): `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/`
  staged to `/pscratch/sd/j/jackm/lorrax_nvhpc` and bind-mounted into
  Shifter at `/lorrax_nvhpc` — the Mg path needs nothing outside the
  container.
- Build: `src/ffi/common/cpp/build.sh` via
  `src/ffi/common/cpp/run_shifter.sh`.
- `LORRAX_NTASKS` env added to `run_shifter.sh` so single-process
  multi-GPU runs (1 task × N GPUs) are as easy as multi-process
  (N tasks × 1 GPU each).

### Report

`reports/ffi_cusolvermg_2026-04-16/report.md`.

### Regression

`uv run python -m pytest -q` → 12 passed, 1 OOM failure (GPU contention
with the interactive 4-GPU alloc; not a regression).

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
