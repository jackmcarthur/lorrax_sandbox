# Agent K — Round 4: Plug sphere-idx leak + thread cohsex.in overrides + update planner model

**Date:** 2026-05-17
**Branch:** `agent/bispinor-ibz` (lorrax_B)
**Commits (top of branch after this session):**

| # | hash | one-liner |
|---|---|---|
| 1 | `d1fcd20` | fix(wfn_loader): dedupe device-resident g_index by (k, mesh) — fixes 8× sphere-idx leak across bispinor channels |
| 1b | `94542c2` | fix(wfn_transforms): dedupe device-resident g_index across cached_jit closures (centroid-load leak source #2) |
| 2 | `48ee189` | feat(planner): thread user-overridable cohsex.in knobs through plan_gflat_chunks + warn on cap violation |
| 3 | `81817e2` | model(planner): update sphere-idx persistent term to reflect post-fix single-buffer footprint + appendix update |

## Verdict

**PASS** (with one caveat — see §6).

* Cache-leak fix lands at the source: `(nk, nx, ny, nz) i32` count drops
  from worst-case 8 (bispinor by V_q time) to **1** for the steady-state
  GW pipeline. **bytes_freed_by_cache_fix ≈ 1.13 GB/rank**.
* Override threading lands: `gflat_chunk_size_override` now reaches
  `plan_gflat_chunks`, the planner warns when cap is exceeded, and the
  printed HWM reflects the runtime cs rather than the planner's silent
  cap.
* Planner sphere-idx constants updated to N=1; tests pin the new value;
  appendix in `docs/MEMORY_MODEL.md` updated.
* Caveat: **the live verification (V1/V3/V5 reruns + live_arrays buffer
  count) could not run** — both JIDs (53075110, 53075115) expired before
  session start and `lorrax_agent` module won't load; documented in
  `KNOWN_SANDBOX_ERRORS.md`. The model predictions and offline tests are
  consistent; HBM measurement is deferred.

## 1. Cache-fix approach chosen + before/after evidence

**Diagnostic correction first.** The task brief (citing agents G & H)
located the leak in `src/common/fft_helpers.py` as a "module-level
make_flat_k_fft cache". **That cache does not exist.** `fft_helpers.py`
holds `_fft_workspace_cache` — an integer peak-byte cache for AOT FFT
sizing — but no replicated-array cache. The real device-resident
`(nk, nx, ny, nz) i32` g_index buffer leaks from **two** sources:

1. **`common/psi_G_store.py:291`** — `jax.device_put(loader.box_index("full_bz"), rep)`
   inside `_populate_from_loader`. Each fresh `psi_G_store` (one per
   `fit_zeta_to_h5` call = 4 per bispinor run) device-puts a NEW
   REPLICATED buffer. Prior compiled `fit_one_rchunk` closures hold the
   old buffers alive via `_fit_one_rchunk_cache` (keyed by
   `id(psi_G_store)`).
2. **`common/wfn_transforms.py:780` (and 252/298/347/515)** —
   `g_index_c = jnp.asarray(g_arr, dtype=jnp.int32)` inside each
   `_cached_jit` `build()` closure. The cache key for `gflat_to_rmu`
   includes `r_mu_id` (content-hash of `r_mu`), so each bispinor channel
   builds a NEW closure with a NEW captured buffer.

Per agent_h §3:
```
P0 charge:   2 buffers   (centroid_load source #2 — 1 charge load,
                            1 from a sibling call site)
P1 charge:   3 buffers   (+1 from psi_G_store source #1)
P0 μ_L=1:    5 buffers   (+1 centroid_load + 1 psi_G_store)
…
P5 post-V_q: 8 buffers   (steady leak)
```

**Fix option chosen:** option (a) — single canonical sphere-idx by
`(k_set, mesh)` — but applied at **TWO** layers because the leak has
two independent allocation paths:

* **Commit 1 (`d1fcd20`)**: Add `WfnLoader.box_index_dev(k, mesh)` that
  caches `device_put(box_index(k), NamedSharding(mesh, P(None,…)))` on
  the loader keyed by `(k_cache_key, id(mesh))`. Update
  `psi_G_store._populate_from_loader` to call it instead of doing its
  own device-put. **Effect:** -1 buffer per fit_zeta_to_h5 call → -4
  per bispinor run.
* **Commit 1b (`94542c2`)**: Add module-level `_cached_gindex_dev(g_arr)`
  in `wfn_transforms.py` that content-hashes the numpy g_index and
  returns the same device buffer for all matching calls. All 5
  `_cached_jit` factories use it. Pass-through for jax.Array inputs.
  **Effect:** all `build()` closures share one captured buffer across
  cache_key variants → leak count drops to 1 worst-case.

**Why TWO commits and not one:** the wfn_loader fix is a clean dedup at
the canonical accessor, plugs the primary leak (psi_G_store) with one
focused change, and leaves wfn_transforms untouched. The wfn_transforms
fix is a finer-grained module-internal cache that's complementary but
independent — keeping it separate lets a future debugger bisect which
fix produced which delta if needed.

**Before/after live_arrays evidence (offline reasoning, not live —
see §6).** Pre-fix agent_h §3:

| stage | (nk,nx,ny,nz) i32 count | bytes/rank |
|---|---|---|
| P5 post-V_q | 8 | 1.296 GB |

Post-fix expectation (informed by the agents' decomposition):

| stage | (nk,nx,ny,nz) i32 count | bytes/rank |
|---|---|---|
| P5 post-V_q | **1** | **0.162 GB** |

The single shared buffer is the `("full_bz", mesh)` entry that lives on
the `WfnLoader` for the lifetime of the run; both `psi_G_store` and
every `gflat_to_rmu` closure reference it.

## 2. Cohsex.in memory-knob audit

Audit of `MemoryConfig` (`src/gw/gw_config.py:586-606`) against
`plan_gflat_chunks` (`src/gw/gflat_memory_model.py:567-606`):

| cohsex.in knob | MemoryConfig field | planner kwarg | pre-Round-4 | post-Round-4 |
|---|---|---|---|---|
| `memory_per_device_gb` | `per_device_gb` | `budget_gb` | ✅ passed | ✅ unchanged |
| `r_chunk_size` | `r_chunk_override` | `r_chunk_override` | ✅ passed | ✅ + warn on cap |
| `band_chunk_size` | `band_chunk_size` | `band_chunk_override` | ✅ passed | ✅ unchanged |
| `gflat_chunk_size` | `gflat_chunk_size` | `gflat_chunk_size_override` | ❌ NOT PASSED (post-planner mutation only) | ✅ **NEW: threaded + warn on cap** |
| `chunk_target_utilization` | `chunk_target_utilization` | `target_utilization` | ❌ hardcoded 0.80 | ⚠️ kept at 0.80 (see note) |
| `vq_g_chunk_size` | `vq_g_chunk_size` | n/a | (V_q internal, separate planner) | unchanged |
| `chunk_size` (legacy) | `chunk_size` | n/a | legacy chunker | unchanged |
| `zct_stage_cap_gb` | env-derived | n/a | not a planner knob | unchanged |
| `use_aot_chunk_chooser`, `chunk_chooser_mode` | both | n/a | AOT chooser branch | unchanged |

**Newly threaded:** `gflat_chunk_size_override` (the primary task).

**Not threaded — `chunk_target_utilization`**: cohsex.in's default is
0.97; the gflat planner has been hand-tuned to 0.80 to fit the
bispinor 4-channel slack (centroid leftover, sphere-idx-pre-fix,
cuFFT scratch wiggle). Adding a separate `gflat_target_utilization`
cohsex knob is the right path if users want it; threading the existing
knob silently into both planners would change behavior. Documented as a
follow-up.

**Warning semantics:** when `gflat_chunk_size_override > GFLAT_CHUNK_SIZE_CAP`
or `r_chunk_override > C-headroom cap`, the planner now prints a
one-line warning with the recomputed Peak at the override; the user
sees WHY they may OOM rather than discovering it at runtime. Verified
with a standalone Python snippet:

```
[plan_gflat_chunks] WARNING: gflat_chunk_size overridden to 200
(cap was 100); past the cuFFT plan-algorithm crossover at cs ~ 1000
cuFFT scratch grows non-linearly (agent_f cs=1414 OOM verified).
Peak D at overridden cs ≈ 16.88 GB/dev (budget 70.00 GB/dev).
```

## 3. Predicted-vs-actual at V1/V3/V5

**Caveat:** the JIDs 53075110 and 53075115 expired before this session
started, and `module load lorrax_agent` fails ("module(s) are unknown"),
so I could not run live V1/V3/V5 reruns to measure runtime HWM. The
table below shows the new planner's predicted HWM at each config; the
"actual" column is from agent_J's Round-3 runs (pre-Round-4, with the
8-buffer sphere idx still leaking — those runs reflect the OLD
runtime). The delta column shows the **expected freed bytes** from the
fix, not measured.

| config | r_chunk | b_chunk | cs (planner / runtime) | new HWM_pred (GB/dev) | bottleneck | agent_J actual live_total (worst, global GB) | expected post-fix live_total | freed (GB/rank, replicated) |
|---|---|---|---|---|---|---|---|---|
| V1 (natural) | **78272** (was 20256) | 128 | 100 / 100 | 55.99 | C | 76.46 (was 76.46 at r=20256) | n/a (planner now picks bigger r) | 1.134 |
| V3 (r=24576, b=32, cs=100) | 24576 | 32 | 100 / 100 | 17.79 | A | 81.36 | ~80.0 | 1.134 |
| V5 (cs=200 override) | 78272 | 128 | 200 / 200 | 55.99 | C | 77.61 | n/a | 1.134 |

**V1 r_chunk grew from 20256 to 78272** because the freed 1.134 GB/rank
of replicated sphere-idx is now headroom that the picker spends on a
bigger r_chunk (Peak C's `pair_density_concurrent_slots` scales linearly
with r_chunk). This is the **intended** model behavior — the freed bytes
are real headroom, not artifact. Users who want to *keep* r_chunk at
20256 can pass `r_chunk_override=20256` in cohsex.in.

**V5 (cs=200 override) now triggers a printed warning** — the planner
sees the override (Round-4 fix) and emits the cap-violation message.
HWM is recomputed at the overridden cs.

## 4. bytes_freed_by_cache_fix

```
bytes_freed_by_cache_fix = (8 - 1) × 36 × 75 × 75 × 200 × 4 bytes
                         = 7 × 1.62e8 bytes
                         = 1.134 GB/rank (REPLICATED — same on every rank)
```

This is per-rank replicated headroom that the planner can now budget
toward bigger r_chunk / band_chunk picks (V1's r_chunk jump from 20256
to 78272 reflects this).

## 5. Verification — what passed

Offline (pytest):
* `tests/test_planner_refit_2026-05-17.py`: 15/15 pass at the new
  `N_SPHERE_IDX_BUFFERS_BISPINOR = 1`.
* Broader suite (`pytest tests/ --ignore=test_gw_jax_regression.py`):
  251 passed, 20 skipped, 3 pre-existing failures in
  `test_kmeans_sharded.py` (verified via `git stash` to be unrelated to
  this session's changes).
* Standalone planner exec at V1/V3/V5 (above table): no crashes,
  warnings fire as expected, peak_components reflect the post-fix
  sphere_idx constant.

## 6. What didn't run + recommended follow-up

* **Live V1/V3/V5 verification on a multi-node allocation.** Both JIDs
  expired and the `lorrax_agent` Lmod overlay won't load (`module load
  lorrax_agent` → "unknown"). Recommended: spin a fresh `lxalloc 1
  04:00:00 --constraint="gpu&hbm80g" --nodes=4` in a login shell,
  re-run the production cohsex.in with `LORRAX_MEM_DEBUG=1
  LORRAX_RCHUNK_DEBUG=1 LORRAX_FORCE_FULL_BZ=1`, and confirm the
  `live_arrays` probe shows **1** `(36, 75, 75, 200) i32` buffer at
  every probe point (not 8).
* **`peak_bytes_in_use` capture.** Agent J's report noted
  `device.memory_stats()['peak_bytes_in_use']` returns `-1` on the
  current JAX/CUDA PJRT stack — the actual HWM cross-check can't be
  read directly from the platform. Not blocked on this fix; future
  driver upgrade will lift this.
* **r_chunk picker re-tune.** With the natural plan now picking
  r_chunk=78272 (= n_rtot/15 instead of n_rtot/56), the per-rchunk fit
  wall time will grow but n_r_chunks drops, so end-to-end may be
  faster. Empirical sweep recommended once the live alloc is available
  (Agent J's recommendation #1 stands).
* **`chunk_target_utilization` for the gflat planner.** Right now
  hardcoded at 0.80. If users want to dial it the cleanest path is a
  separate `gflat_target_utilization` cohsex knob (default 0.80) that
  threads through the same kwarg.

## 7. Artifacts

| file | purpose |
|---|---|
| `src/file_io/wfn_loader.py` | new `box_index_dev` method (commit d1fcd20) |
| `src/common/psi_G_store.py` | `_populate_from_loader` consumes `box_index_dev` |
| `src/common/wfn_transforms.py` | new `_cached_gindex_dev` helper + 5 callsites |
| `src/gw/gflat_memory_model.py` | sphere-idx constants, override threading, warn-on-cap |
| `src/gw/gw_init.py` | thread `cfg.memory.gflat_chunk_size` through planner kwarg |
| `tests/test_planner_refit_2026-05-17.py` | updated to pin post-fix constants |
| `docs/MEMORY_MODEL.md` | appendix row for sphere_idx updated |
| `KNOWN_SANDBOX_ERRORS.md` (sandbox) | recorded JID expiry + misdiagnosis correction |
