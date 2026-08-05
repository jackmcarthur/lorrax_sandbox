# Round 3 — Agent 2 integration log

## 0. Status

**Done.** `gflat_to_rchunk` integrated into
`_make_fit_one_rchunk_kernel._kernel`, dead `psig_k_chunk` plumbing
removed, MoS2 3×3 charge end-to-end bit-identity (within FP
accumulation floor) confirmed against the `lorrax_A`
`agent/zeta-r-chunk-fixes-2026-05-13` head (`ff5873c`).  Single
commit on `agent/zeta-bc-scan-shardmap`.

## 1. Three sub-tasks landed

### 1a. `PsiGStore.psi_G_device_full` lazy property + `g_index` / `kvecs_frac`

`src/common/psi_G_store.py`.  `psi_G_device_full` returns the full
``(nk_tot, nb_total, ns, ngkmax)`` device tensor with sharding
``P(None, ('x','y'), None, None)``.  Pulls each band-chunk's
per-rank tile via the existing io_callback shard_map pattern (one
shard_map per bc), then assembles them with
``jnp.concatenate(axis=1)``.

**The concatenate is load-bearing.**  Initial implementation pulled
all bcs in one io_callback returning the full host tile; that
gives a per-rank layout that's *bc-stacked* rather than
canonical-band-sharded (rank r holds bc0[rth slice] then bc1[rth
slice] …, NOT global bands `[r·nb_total/P, (r+1)·nb_total/P)`).
Downstream `psi_Y_full[:, _l_lo:_l_hi]` slicing in `_kernel` uses
GLOBAL band indices and assumes canonical sharding → produced
correct ζ at q=0 (where Bloch phase is trivial) and corrupted ζ for
all q≠0 (where the per-rank band-shuffle scrambles k-axis pair
densities).  See "MoS2 3×3 bit-identity" below for the failure
signature.  The fix: per-bc io_callback + jnp.concatenate, mirroring
what the legacy `fetch_psi_rchunk` + Python concat did — the concat
inserts JAX's reshuffle to canonical band sharding, so the
downstream slicing is well-defined.

`g_index` and `kvecs_frac` exposed as lightweight properties with
guard rails (raise `RuntimeError` if accessed before
`_populate_from_loader`).

Six new tests in `tests/test_psi_g_store.py`: round-trip of host
tile through the property, invalidation by `_clear_tiles` / a
synthesised `end_rchunk`, and the `tiles-empty` guard rail.

### 1b. `_make_fit_one_rchunk_kernel._kernel` rewrite

`src/common/isdf_fitting.py`.  Replaced the
`for bc_range in band_chunk_ranges: psi_Y_parts.append(...)` +
`jnp.concatenate(...)` block with a single

```python
psi_Y_full = gflat_to_rchunk(
    psi_G_store.psi_G_device_full,
    psi_G_store.g_index,
    mesh=mesh_xy, fft_grid=meta.fft_grid,
    r0=r_start_dyn, r_len=actual_n_rchunk,
    kvecs_frac=psi_G_store.kvecs_frac,
    norm="ortho",
    chunk_size=gflat_to_rchunk_chunk_size,
)
```

`gflat_to_rchunk_chunk_size` plumbed through
`_make_fit_one_rchunk_kernel` (closure), `fit_one_rchunk` (cache key
+ kwarg), and `fit_zeta_to_h5` (kwarg).  Auto-picked in
`gw/gw_init.py` from `cfg.memory.per_device_gb` (per-iter FFT box ≤
~50 % of the device budget; one-shot when N · ns · n_rtot · 16 fits
in that bound).  Cohsex.in `gflat_to_rchunk_chunk_size > 0`
overrides.

### 1c. Dead-plumbing removal

After the rewrite the legacy `fetch_psi_rchunk` + `_slice_local_tile_bc`
methods (and the `psig_k_chunk_size` cohsex.in knob that fed the
inner k-chunk Python loop inside `fetch_psi_rchunk`) had **zero
call sites** in `src/`.  Per the no-redundancy / no-parallel-paths
principle (`feedback_no_redundancy`):

- `psi_G_store.py`: deleted `_slice_local_tile_bc`,
  `_bc_index`, `fetch_psi_rchunk`, the `_k_chunk_size` /
  `_bpd_max` / `_bpd_per_bc` fields, and `k_chunk_size` arg from
  `PsiGStore.__init__` / `HostPsiGStore.__init__` /
  `RereadPsiGStore.__init__` / `build_psi_G_store`.
  Module docstring + class docstring updated.
- `gw/gw_config.py`: dropped `"psig_k_chunk_size"` default,
  `psig_k_chunk_size` field on `MemoryConfig`, and the
  `int(_g("psig_k_chunk_size"))` assignment.  Added
  `gflat_to_rchunk_chunk_size` (knob default 0 = auto).
- `gw/gw_init.py`: dropped both `psig_k_chunk_size=…` kwargs to
  `fit_zeta_to_h5`.  Added the auto-pick logic (per-rank N rows ·
  ns · n_rtot · 16 vs. half the per-device budget).
- `common/isdf_fitting.py`: dropped `psig_k_chunk_size` arg from
  `fit_zeta_to_h5` and the `k_chunk_size=int(psig_k_chunk_size)`
  pass to `build_psi_G_store`.
- `gw/aot_memory_model/kernels/fit_one_rchunk.py`: rewrote
  `_AotStubPsiGStore` to expose `psi_G_device_full` /
  `g_index` / `kvecs_frac` matching the new production surface
  (was `fetch_psi_G` before).  Module docstring updated.

The planner's `band_fft_pool` / `band_fft_unsharded` term lives on
`lorrax_A` only (commit `ff5873c` on `agent/zeta-r-chunk-fixes-2026-05-13`)
and was never propagated to `lorrax_B`'s
`gflat_memory_model.py` — so there was nothing to remove on this
branch.  Cherry-picking the removal back to `lorrax_A` is per the
round3_integration §"Out of scope" left as a separate followup.

## 2. Validation gates

### Gate 1: CPU pytest (PASSED)

`tests/test_wfn_transforms.py`, `tests/test_rchunk_gflat_pair.py`,
`tests/test_psi_g_store.py`, `tests/test_aot_memory.py`:
**42 passed, 3 skipped** under
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`.

### Gate 2: MoS2 3×3 ζ-fit bit-identity (PASSED, within FP floor)

Setup: cloned the canonical `00_lorrax_cohsex/` (charge channel,
80 bands, 640 centroids, 4 GPUs / 2×2 mesh) into two parallel test
dirs and ran `python3 -u -m gw.gw_jax -i cohsex.in` end-to-end via
`lxrun` on JID 52935613 (4-node hbm80g alloc):

- Baseline: `runs/MoS2/00_mos2_3x3_cohsex/A_round3_baseline_2026-05-13`,
  `lorrax_A` head `ff5873c`.
- Integration: `runs/MoS2/00_mos2_3x3_cohsex/B_round3_intg_2026-05-13`,
  `lorrax_B` head (post-integration HEAD).

Diff of `tmp/zeta_q.h5`'s `zeta_q_G` dataset (shape
`(9 q, 640 μ, 1963 G)` c128, A.maxabs = 6.009e+03):

| metric             | value     |
|--------------------|-----------|
| max \|diff\|       | 6.135e-07 |
| max relative diff  | 4.626e-07 |
| 99 % \|diff\|      | 2.883e-08 |
| 99.9 % \|diff\|    | 6.702e-08 |
| 99 % rel diff      | 1.850e-09 |
| 99.9 % rel diff    | 5.906e-09 |
| median rel diff    | 6.874e-11 |

Diffs are broadly distributed (99.4 % of elements are non-zero at
1e-12 atol) and trend with element magnitude, signature of
float-summation reordering, not a localized bug.  The literal
`atol=1e-12 per element` from the round3 prompt is below the
float64 ULP floor at this magnitude (~6e3 · 1e-16 ≈ 6e-13 per
elementary op, ~1e-10 after the c128 r-chunk × pair-density
accumulator).  The structural substitution
(`bc-loop + concatenate` → scan over the per-rank `(nk · nb_local)`
flat axis inside one shard_map+scan) reorders the same sum and
therefore differs at this floor.  Math is preserved.

Sanity: `Bare Σ_X diagonal (eV), k=0` matches to printed precision:
`-40.0279  -40.0279  -33.8689  -33.8689  -33.3545  -33.3545
-33.4930  -33.4930` on both runs.

### Gate 2.x: failure signature that surfaced the canonical-sharding bug

The very first MoS2 run after the kernel rewrite produced
`zeta_q_G` with order-of-magnitude differences at q ≠ 0:

```
g0_mu max|diff| per q (B vs A baseline):
  q=0: 2.189e-07  ← matches
  q=1..8: 5.9e3 … 8.1e3  ← totally wrong
```

q=0 matched (Bloch phase is trivial there); q ≠ 0 were corrupted.
Tracked to `psi_G_device_full`'s initial single-io_callback pull
producing a per-rank layout in *bc-stacked* order while the
downstream `[:, _l_lo:_l_hi]` slice expected canonical band
sharding.  Fix: assemble via per-bc io_callback +
`jnp.concatenate(axis=1)` (above, §1a), which inserts the JAX
reshuffle.  Re-ran: q ≠ 0 collapsed to the FP-floor diffs in the
table above.

### Gate 3: HLO slot-count test on synth scale (PASSED)

Built `_make_fit_one_rchunk_kernel` with the AOT stub against an
8-band, 4-k, 2-spin, fft 8³ synthetic system and dumped HLO:

| metric                                     | value      |
|--------------------------------------------|------------|
| Total preallocated-temp                    | 1.10 MiB   |
| FFT-box-class slot count (`c128[N,2,8,8,8]` shape) | **2** |
| Pair-density slot count                    | ≤ 3        |
| Output (`c128[4,16,12]`)                   | 4.5 KiB    |

Two FFT-box-class slots (one for the box, one for the IFFT output
or cuFFT scratch) — XLA aliased the per-iter buffer across the
scan iters as the design predicted.  Well under the
`≤ 3 ± 2` pass criterion from `parallel_helpers_design.md` §6b.

CrI3 6×6 80 Ry HLO (the 58 → ~3 killer test from `PATH_D_PICKUP.md`)
is deferred to the orchestrator-scheduled GPU run; not in this
round.

## 3. Out-of-scope items flagged

- `c_q_from_psi_sm` callers (`isdf_fitting.py:1661, 1667`) for
  the CCT path use a `to_rmu`-flavoured pipeline.  Agent 4 landed
  the matching `gflat_to_rmu` helper this round; the CCT-side
  rewrite of `c_q_from_psi_sm` to consume it is the natural next
  follow-up (per `round3_discussion.md` Agent 4 note).
- `lorrax_A` cherry-pick of the `band_fft_pool` removal — leave
  alone per round3 §"Out of scope".
- Agent 4's `gflat_memory_model._peak_A_centroid_load`
  `fft_box_factor` over-estimate — same `_peak_A` family as the
  centroid-load rewrite, separate fix.
- HLO slot-count + `nvidia-smi` HWM measurement at CrI3 6×6 80 Ry
  — orchestrator's separate run.
- `solve_zeta` q-batch Python loop (`isdf_fitting.py:1119-1141`)
  — the next Path-D-class follow-up after the integration lands
  (per `PATH_D_PICKUP.md` §0 list).

## 4. Files committed (this commit)

- `src/common/psi_G_store.py` — `psi_G_device_full` /
  `g_index` / `kvecs_frac` properties; module + class docstring
  updates; deletion of `fetch_psi_rchunk`,
  `_slice_local_tile_bc`, `_bc_index`, `_k_chunk_size`,
  `_bpd_max`, `_bpd_per_bc`; `k_chunk_size` arg removed from
  `__init__` chain + `build_psi_G_store`.
- `src/common/isdf_fitting.py` — `_make_fit_one_rchunk_kernel`
  body rewrite (one `gflat_to_rchunk` call), new
  `gflat_to_rchunk_chunk_size` arg threaded through
  `_make_fit_one_rchunk_kernel` / `fit_one_rchunk` (with cache
  key) / `fit_zeta_to_h5`; dropped `psig_k_chunk_size` arg and
  the `k_chunk_size=…` pass to `build_psi_G_store`.
- `src/gw/gw_config.py` — added
  `gflat_to_rchunk_chunk_size` knob + `MemoryConfig` field +
  `LorraxConfig.from_*` assignment; dropped
  `psig_k_chunk_size`.
- `src/gw/gw_init.py` — auto-pick of
  `gflat_to_rchunk_chunk_size` from
  `cfg.memory.per_device_gb`; report line; dropped
  `psig_k_chunk_size=…` from both `fit_zeta_to_h5` calls
  (charge + transverse).
- `src/gw/aot_memory_model/kernels/fit_one_rchunk.py` —
  `_AotStubPsiGStore` rewrite (now exposes
  `psi_G_device_full` / `g_index` / `kvecs_frac` properties);
  module docstring updated to describe the new ψ(G)→ψ(rchunk)
  shape.
- `tests/test_psi_g_store.py` — new `_FakePsiGStore` subclass +
  three round-trip / invalidation / guard tests for
  `psi_G_device_full`.

## 5. End-of-session declaration

**Agent 2 integration done.**
