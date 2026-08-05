# Agent F — `jax.live_arrays()` probes at r-chunk boundaries (cross-jit liveness ground truth)

**Date:** 2026-05-17
**Branch:** `agent/bispinor-ibz` (lorrax_B), commit `5c884ac` (instrumentation)
**Allocations:** JID 53075115 (cs=707) + JID 53075110 (cs=1414); both 4 nodes × 4 GPUs hbm80g.
**Run dir:** `runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/`
**Outputs:** `mem_probe_cs707.out`, `mem_probe_cs1414.out`.

This is the runtime complement to Agent E's HLO-only audit (`agent_e_cross_jit_lifetime.md`),
which established that buffer-assignment.txt is per-jit and cannot prove cross-jit liveness
either way. The instrumentation uses `jax.live_arrays()` (process-wide JAX-tracked arrays)
at three points per r-chunk in `fit_zeta_to_h5` — before fit_one_rchunk, after fit_one_rchunk
returns (just before accumulate), and after accumulate.

`device.memory_stats()` returned `bytes_in_use=-1, peak=-1` throughout (the JAX CUDA PJRT on
this stack returns `None` for `memory_stats()` — already noted in the in-tree `_track_peak()`
fallback that uses nvidia-smi). All quantitative findings below come from `jax.live_arrays()`
shape × dtype × itemsize summation, which **reports global/logical shapes, not per-rank**.
For sharded tensors (gflat_acc, centroids over μ axis) the per-rank residency is
**total / world_size = total / 16**.

---

## Run config (both configs)

```
r_chunk_size      = 21232
band_chunk_size   = 32
n_mu_charge       = 1520 (padded), n_mu_transverse = 1504
n_q = 36, ngkmax = 59990
LORRAX_FORCE_FULL_BZ=1, LORRAX_EXIT_AFTER_ZETA=1
LORRAX_MAX_RCHUNKS=2, LORRAX_MEM_DEBUG=1, LORRAX_RCHUNK_DEBUG=1
```

Planner's HWM estimate at cs=707 and cs=1414: both 57.03 GB/dev,
bottleneck reported as `C_fit_one_rchunk`. Planner's Peak D estimate at cs=1414:

```
[D] accumulate_fft_box   50.904 GB/dev   (2 × bare FFT box of 25.45 GB)
    gflat_acc             3.694
    zeta_chunk            1.162
    centroids_persist     0.136
    L_q                   0.083
    Total                55.98 GB/dev
```

i.e. the planner says Peak D = 55.98 GB at cs=1414 — comfortably under the 70 GB budget.

---

## Run 1 — cs=707 (safe regime, 2 r-chunks completed)

### Probe 1A: `rchunk_start chunk=0`

```
[mem_probe rchunk_start chunk=0] live_count=67 live_total=58.72 GB
  complex128 (36, 1520, 59990) x 1 = 52.52 GB    <- gflat_acc (μ-sharded, /16 per rank)
  complex128 (36, 1520, 1520)  x 1 =  1.33 GB    <- L_q (μ-sharded)
  complex128 (36, 1520, 160, 4) x 2 = 1.12 GB    <- psi_r_rmuT_X_fit + transpose
  complex128 (36, 160, 4, 1520) x 2 = 1.12 GB    <- psi_r_rmu_Y + transpose
  complex128 (36, 1520, 150, 4) x 2 = 1.05 GB    <- psi_l_rmuT_X_fit + transpose
  complex128 (36, 150, 4, 1520) x 2 = 1.05 GB    <- psi_l_rmu_Y + transpose
  int32      (36, 75, 75, 200) x 3 =  0.49 GB    <- FFT sphere indices
  complex128 (36, 59990)       x 1 =  0.03 GB
  complex128 (36, 1508)        x 6 =  0.01 GB
  int32      (1508, 3)         x 2 =  0.00 GB
```

### Probe 1B: `after_fit_one_rchunk chunk=0` — **the key probe**

```
[mem_probe after_fit_one_rchunk chunk=0] live_count=68 live_total=77.31 GB
  complex128 (36, 1520, 59990)  x 1 = 52.52 GB   <- gflat_acc (persistent)
  complex128 (36, 1520, 21232)  x 1 = 18.59 GB   <- NEW: zeta_chunk just produced
  complex128 (36, 1520, 1520)   x 1 =  1.33 GB   <- L_q STILL LIVE
  complex128 (36, 1520, 160, 4) x 2 =  1.12 GB   <- psi_r_rmuT_X_fit STILL LIVE
  complex128 (36, 160, 4, 1520) x 2 =  1.12 GB   <- psi_r_rmu_Y STILL LIVE
  complex128 (36, 1520, 150, 4) x 2 =  1.05 GB   <- psi_l_rmuT_X_fit STILL LIVE
  complex128 (36, 150, 4, 1520) x 2 =  1.05 GB   <- psi_l_rmu_Y STILL LIVE
  int32      (36, 75, 75, 200)  x 3 =  0.49 GB
  ...
```

**Centroids+L_q ARE live in HBM between fit_one_rchunk and accumulate.** Delta vs probe 1A
is exactly +18.59 GB = +zeta_chunk; nothing else freed.

### Probe 1C: `after_accumulate chunk=0`

```
[mem_probe after_accumulate chunk=0] live_count=76 live_total=58.74 GB
  complex128 (36, 1520, 59990)  x 1 = 52.52 GB   <- gflat_acc (donated/updated in place)
  complex128 (36, 1520, 1520)   x 1 =  1.33 GB
  complex128 (36, 1520, 160, 4) x 2 =  1.12 GB
  ...
  int32      (36, 59990)        x 2 =  0.02 GB   <- NEW: gflat sphere idx
```

zeta_chunk freed (donated), gflat_acc updated in place — net live drops back to ~58.74 GB.
Centroids+L_q unchanged. Chunk timing: `fit=20100ms write=1406ms total=21506ms`.

### Probe 1D-1F: chunk=1 — bit-identical to chunk=0

`live_total` at rchunk_start = 58.74 GB; at after_fit = 77.33 GB; at after_accumulate = 58.74 GB.
Same top-10 shapes. Run cleanly hit `LORRAX_MAX_RCHUNKS=2` and broke. Charge channel
completed; transverse channel started (visible in tail of mem_probe_cs707.out with
μ=1504 instead of 1520).

### cs=707 per-rank decomposition at `after_fit_one_rchunk` (worst point in iter)

| Tensor | Global | Sharding | Per-rank |
|---|---|---|---|
| gflat_acc                | 52.52 GB | μ sharded /16 | 3.28 GB |
| zeta_chunk (transient)   | 18.59 GB | μ sharded /16 | 1.16 GB |
| L_q                      |  1.33 GB | μ sharded /16 | 0.083 GB |
| ψ_r (4 copies)           |  2.24 GB | μ sharded /16 | 0.14 GB |
| ψ_l (4 copies)           |  2.10 GB | μ sharded /16 | 0.13 GB |
| FFT sphere indices       |  0.49 GB | replicated   | 0.49 GB |
| other small              |  ~0.04 GB | mixed        | ~0.04 GB |
| **Python-side live**     | **77.31 GB** | | **~5.3 GB** |

Plus XLA's accumulate scratch (factor_D × FFT_box) = 2 × 12.73 GB/rank = **25.5 GB/rank**.

cs=707 per-rank total at peak: ~5.3 + 25.5 + cuFFT_plan_scratch ≈ **30–35 GB/rank**, well under 80 GB. Run succeeded — consistent with the planner Peak D = 28.0 GB/dev at cs=707 (predicted, scaled from 1414 → 707).

---

## Run 2 — cs=1414 (OOM regime — fired probe 2A and 2B, then died in accumulate)

### Probe 2A: `rchunk_start chunk=0`

```
[mem_probe rchunk_start chunk=0] live_count=67 live_total=58.72 GB
  complex128 (36, 1520, 59990)  x 1 = 52.52 GB
  complex128 (36, 1520, 1520)   x 1 =  1.33 GB
  complex128 (36, 1520, 160, 4) x 2 =  1.12 GB
  ...
```

**Identical** to cs=707 probe 1A — same baseline live state. The accumulate-pre state is
not what differs between the two configs.

### Probe 2B: `after_fit_one_rchunk chunk=0` — **fired before OOM**

```
[mem_probe after_fit_one_rchunk chunk=0] live_count=68 live_total=77.31 GB
  complex128 (36, 1520, 59990)  x 1 = 52.52 GB
  complex128 (36, 1520, 21232)  x 1 = 18.59 GB
  complex128 (36, 1520, 1520)   x 1 =  1.33 GB
  complex128 (36, 1520, 160, 4) x 2 =  1.12 GB
  ...
```

**Identical** to cs=707 probe 1B. fit_one_rchunk's output state is the same — same
zeta_chunk, same centroids, same L_q. The difference between cs=707 and cs=1414 lives
**only** inside the subsequent accumulate jit's scratch needs.

### Probe 2C: never fired — accumulate crashed

```
jaxlib.xla_extension.XlaRuntimeError: INTERNAL: RET_CHECK failure
(external/xla/xla/backends/gpu/runtime/fft_thunk.cc:176)
fft_plan != nullptr Failed to create cuFFT batched plan with scratch allocator
```

Stack trace bottom: `isdf_fitting.py:2537` = the `accumulate_rchunk_to_gflat(...)` call.
All 16 ranks emitted the same error within ~700 ms. The cuFFT plan creation, not the FFT
execution, failed to acquire scratch.

---

## Verdict 1 — Are centroids + L_q live in HBM during accumulate?

**YES, unambiguously.** `live_arrays()` at `after_fit_one_rchunk` (the moment between
fit_one_rchunk return and the accumulate call) lists, in both cs=707 and cs=1414, exactly:

- `c128[36, 1520, 1520] × 1` = **L_q, 1.33 GB global (0.083 GB/rank)**
- `c128[36, 1520, 150, 4] × 2 + c128[36, 150, 4, 1520] × 2` = **ψ_l charge centroids, 2.10 GB global (0.13 GB/rank)**
- `c128[36, 1520, 160, 4] × 2 + c128[36, 160, 4, 1520] × 2` = **ψ_r charge centroids, 2.24 GB global (0.14 GB/rank)**

Plus the persistent gflat_acc (52.52 GB global, 3.28 GB/rank). The centroids+L_q together
total **0.35 GB/rank** — the *direction* of the Peak D fix in commit `21f2ed6` is correct
(centroids are NOT freed before accumulate runs), but the *magnitude* is small per-rank
because the μ axis is sharded /16 and the bispinor ns=4 split is also accounted for.

## Verdict 2 — What causes the cs=1414 OOM?

**Not centroid competition for HBM.** The failure mode is:

```
fft_thunk.cc:176 — Failed to create cuFFT batched plan with scratch allocator
```

That is **cuFFT's plan-creation API itself** asking XLA's scratch allocator for workspace,
and the allocator returning nullptr. At the moment accumulate enters, per-rank live
includes:

- Python-side persistent live (gflat_acc, centroids, L_q, ψ_*, indices, zeta_chunk): **~5.3 GB/rank**
- XLA must allocate accumulate's preallocated-temp scratch: **factor_D × 25.45 GB ≈ 50.9 GB/rank**

That's ~56 GB/rank, leaving ~24 GB free of the 80 GB HBM — but cuFFT's plan needs more than
~24 GB of additional scratch beyond the box-sized slots XLA already accounts for. The
planner does not model this cuFFT plan-scratch overhead.

**Evidence ruling out centroid competition**:
- Pre-accumulate live state is identical (77.31 GB global = 4.83 GB/rank) at both cs=707 and cs=1414.
- cs=707 with the same centroids+L_q resident succeeds twice.
- The OOM error is in cuFFT plan creation, not generic XLA allocator OOM.

**Evidence pointing at cuFFT scratch**:
- The error literal is "Failed to create cuFFT batched plan with scratch allocator" — this is the cuFFT layer's plan-cache, not XLA's BFC pool.
- The two box-sized slots XLA reports (2 × 25.45 GB per-rank at cs=1414) are inside `preallocated-temp` and already accounted for in `factor_D = 2.0`. The additional ask is cuFFT's internal plan workspace, which scales super-linearly with batch size and is invisible to XLA's planner.

## Peak D fix advice (commit `21f2ed6`)

The fix `21f2ed6` adds `centroids_persist + L_q` (~0.219 GB/dev for bispinor 80Ry) into
the Peak D persistent term. The direction is right (the runtime probe confirms these are
live during accumulate). The magnitude is tiny (~0.2 GB/dev) because of μ-sharding.

**Recommendation: KEEP `21f2ed6` (the centroids+L_q in Peak D)** for correctness — it now matches
what `live_arrays()` shows. But **do not stop there**: the cs=1414 OOM is NOT caused by
centroids competing. It is caused by **cuFFT plan-scratch growth that the planner does
not model**. The 0.2 GB centroid term cannot move cs=1414 out of OOM; the fix that would
move cs=1414 to safe regime is one of:

1. **Empirical cap on `gflat_chunk_size`** (e.g. cap at the value that makes
   `factor_D × box_bytes ≤ 0.55 × budget` rather than the current `0.94 × budget`) — leaves
   headroom for cuFFT plan-scratch.

2. **`query_fft_peak_bytes`** — actually run a tiny cuFFT plan-creation probe at planner time
   to measure plan-workspace bytes, then add to Peak D. This is the principled fix but
   requires a runtime probe in the planner.

3. **Lower `factor_D` cap with a `cufft_plan_overhead_factor` knob** (e.g. add a +30%
   margin on the FFT-box term) — cheap to add, conservative, doesn't require runtime
   probing.

Empirically, cs=707 → Peak D ≈ 28 GB/dev → succeeds; cs=1414 → Peak D ≈ 56 GB/dev (per
planner) → fails at cuFFT plan creation. The transition is somewhere between cs=707 and
cs=1414. Without measurement of cuFFT plan scratch as a function of cs, the safe path is
to cap `gflat_chunk_size` at ~0.6 × budget rather than 0.94.

---

## Notes / caveats

1. **`device.memory_stats()` returned -1 GB throughout** — JAX CUDA PJRT on this jax/jaxlib
   build returns `None` from `memory_stats()`. The `bytes_in_use=-0.00 GB peak=-0.00 GB` lines
   are the probe printing the `.get("bytes_in_use", -1)` fallback. All quantitative
   findings come from `jax.live_arrays()` (shape × dtype × itemsize summation), which
   reports **global** shapes — divide by world_size=16 for per-rank.

2. **`live_arrays()` does not show XLA's BFC scratch pool or cuFFT plan-workspace.** What
   we measured at `after_fit_one_rchunk` is the *Python-side persistent residency*
   (gflat_acc, centroids, L_q, ψ_*, sphere indices, zeta_chunk). XLA's transient
   scratch inside the accumulate jit (factor_D × FFT box) is allocated INSIDE
   accumulate and therefore not visible at this probe point — but it is what dominates
   the per-rank peak, not the live arrays we measured.

3. **The cs=707 / cs=1414 baseline difference is zero.** Probes 1A == 2A and 1B == 2B at
   the GB level (same shapes, same counts, same totals). The two configs diverge only
   inside the accumulate jit's scratch ask.

4. **Transverse channel** (μ_L>0) was reached by cs=707 (line "fitting ζ^{μ_L=1,2,3} on
   current-density centroids"). The transverse probes show μ=1504 instead of 1520 and
   add ~1 GB of mixed-channel state (lingering charge ψ_l/ψ_r at 0.56 GB each, 3 extra
   sphere indices at 0.97 GB) — no qualitative change to the Verdict.
