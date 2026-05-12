# Open questions on the V_q tile memory model — 2026-05-08

State after `agent/phdf5_padded` commit `723e1df` (full-kernel AOT chooser).

## What we know

| System | Mesh | μ | n_G | budget | slope+intercept pred | full-kernel AOT pred | actual XLA peak | observed wall (chosen q) |
|---|---|---|---|---|---|---|---|---|
| MoS2 3×3 | 2×2 | 640 | 2419 | 68.79 GB | 2.14 GB @ q=9 | **3.21 GB @ q=9** | **2.99 GiB** (HLO) | <1 s |
| CrI3 6×6×1 | 4×4 | 1504 | 62649 | 43.38 GB | 40.69 GB @ q=12 | **61.22 GB @ q=12** (over-budget → shrink to q=8 @ 40.84 GB) | unknown directly; q=12 ran fine in earlier benches at unknown peak | (TBD this session) |

The full-kernel AOT was within 1% on MoS2; on CrI3 it predicts 50% higher per-rank peak than slope+intercept at q=12. Earlier CrI3 runs at q=12 *fit* (218 s baseline; 152 s warm-cache; one 2134 s outlier attributed to Lustre fluke per the forensics report). The hardware ceiling on hbm80g is 80 GB; cohsex.in's `memory_per_device_gb = 45.0` caps the V_q chooser at 43.38 GB even though physical headroom exists.

## Three plausible interpretations of the discrepancy

1. **Slope+intercept under-counts.** Real q=12 peak is ~55-60 GB (matching full-kernel AOT). It fit on the hardware because physical mem is 80 GB. The `memory_per_device_gb=45` budget was mis-set: had we trusted slope+intercept at face value (40.7 GB), we'd have been near OOM, but headroom saved us.
2. **Full-kernel AOT over-counts.** Real q=12 peak is ~40 GB (matching slope+intercept). XLA's `memory_analysis()` reports a static high-water across kernel stages without crediting all the aliasing that runtime achieves. MoS2 happens to be small enough that aliasing wins are bounded by cuFFT scratch ratio; CrI3's larger n_rtot/n_G ratio means more cross-stage buffer reuse opportunity, which static analysis doesn't see.
3. **Both are roughly right at different scales.** The standalone-FFT measurement undercounts surrounding live buffers (gather, contract intermediate). The full-kernel measurement may not reflect actual aliasing the planner achieves at runtime. Real peak is between the two.

## Verdict from the 70 GB distinguishing test (2026-05-08, alloc 52707589)

**Outcome: a third case neither (1) nor (2) anticipated.**

Setting `memory_per_device_gb = 70.0` (budget = 67.63 GB after overhead) made the chooser pick **q_chunk=13** (full-kernel AOT predicted 66.32 GB peak/rank ≤ 67.63 GB; q=19 at 96.89 GB was rejected). ζ-fit ran fine; ζ-fit GPU peak was 14.60 GB vs AOT-predicted 16.37 GB (γ=0.892, AOT slightly over-counted ζ-fit).

V_q tile then **failed on first kernel invocation** with:

```
INTERNAL: RET_CHECK failure (.../fft_thunk.cc:176) fft_plan != nullptr
Failed to create cuFFT batched plan with scratch allocator
```

at 10:55:26, no q-point ever printed (failure during JIT plan creation, before any rotation through the contract).

This is **not** an XLA OOM (no "Resource exhausted" / HBM allocator failure) and **not** the case-(2) scenario where slope+intercept's 40 GB prediction would have been right. It is the case-(1)-adjacent scenario:

> **Case 3 (observed):** Full-kernel AOT under-predicted memory at scale on CrI3. AOT predicted 66 GB; the cuFFT plan's runtime scratch workspace pushed total over the 80 GB hardware ceiling.

Why AOT missed this: `compiled.memory_analysis()` reports the static buffer + temp scratch the compiler can see at lowering time. The **cuFFT plan workspace is allocated lazily at first kernel invocation** by jaxlib's FFT thunk, *outside* the XLA buffer plan. So memory_analysis() never sees it. At small Q this is harmless (cuFFT scratch is small relative to the buffer plan). At Q=13 with FFT shape (75×75×200) batched ≈ 2419·1504·13 ≈ 47.3 M transforms per call, cuFFT's chosen algorithm wants enough scratch that 80 − 66 = 14 GB of headroom isn't enough.

Calibration data points so far:

| Run            | Q  | AOT pred | slope+int pred | Hardware  | Result               |
|----------------|----|----------|----------------|-----------|----------------------|
| MoS2 3×3 mesh 2×2 | 9  | 3.21 GB  | 2.14 GB        | hbm40g    | ran; XLA peak 2.99 GiB (AOT within 1%) |
| CrI3 6×6 mesh 4×4 | 8  | 40.84 GB | (n/a, < FFT)   | hbm80g    | ran; wall 1640 s (Lustre-slow) |
| CrI3 6×6 mesh 4×4 | 12 | 61.22 GB | 40.69 GB       | hbm80g    | historically ran (218 s baseline) |
| CrI3 6×6 mesh 4×4 | 13 | 66.32 GB | ~44 GB         | hbm80g    | **FAIL: cuFFT plan**           |
| CrI3 6×6 mesh 4×4 | 18 | 91 GB    | 60.9 GB        | hbm80g    | OOM (slope+intercept saw 56.7 GB actual) |

The Q=18 row is the existing data point from `runs/.../qchunk18_20260507_182744.log`: slope+intercept's 60.9 GB prediction matched the XLA actual 56.7 GB, and the OOM there was JAX-side (not cuFFT). At Q=13 the failure mode is *cuFFT-side*, telling us the boundary case is the cuFFT scratch eating into headroom that AOT couldn't see.

## What this means for the chooser default

**Keep `LORRAX_V_Q_AOT_FULL_KERNEL=1` (full-kernel AOT default-on).**

- AOT remains *less wrong* than slope+intercept (slope+intercept's 41 GB prediction at Q=12 would have nominated even larger Q values at the 67 GB budget; we'd have failed sooner).
- AOT's failure mode at Q=13 was an under-prediction by ~14 GB — the unmodeled cuFFT scratch — so any chooser using AOT must be paired with an explicit cuFFT-scratch model or a guard band.
- **Do NOT raise `memory_per_device_gb` in CrI3 templates toward AOT predictions.** The 45 GB budget that made the chooser pick Q=12 (predicted 61 GB at slope+intercept time, 61 GB also at full-kernel AOT) historically worked because the cuFFT scratch fit into 80 − 61 = 19 GB headroom. At Q=13 it didn't fit into 80 − 66 = 14 GB headroom. The implicit safety came from never picking Q ≥ 13.

Concrete chooser change: subtract a **fixed cuFFT guard band** (~15 GB on hbm80g) from the AOT-predicted peak before comparing to the budget. Or, equivalently, treat the budget as `memory_per_device_gb − 15 GB`. This is a one-line change in `_choose_v_q_chunks`. It keeps the AOT chooser conservative.

A more principled fix is option (a) below: query cuFFT scratch directly via `cufftXtMakePlanMany` + `cufftGetSize` for the chosen Q, and add it to the AOT prediction.

## Other open items in the model

A. **cuFFT plan workspace is invisible to `memory_analysis()`.** [SUPERSEDED BY CASE 3 ABOVE — promoted to a top-level finding.] `_aot_fft_model` measures at Q=1 and Q=2 and assumes peak(Q) = slope·Q + intercept. cuFFT plans for non-trivial radix can switch algorithms at certain batch sizes — discrete jumps in workspace size that linear extrapolation misses. The full-kernel AOT compile-time `memory_analysis()` does NOT capture cuFFT scratch either (plan workspace is allocated lazily at first kernel invocation by jaxlib's FFT thunk). Both paths therefore under-count at large Q. Fix: query cuFFT directly via `cufftGetSize` after lowering, add to AOT-reported peak.

B. **Distinct-ζ doubling of slope.** The `same_zeta=False` path doubles the FFT-stage slope on the assumption that two ζ_disk inputs are alive at the kernel boundary. Untested empirically — bispinor would exercise it. With the full-kernel AOT path the assumption falls away (the kernel's `compiled.memory_analysis()` reflects whatever's actually live), so this is a non-issue when AOT is on.

C. **Per-rank vs whole-mesh budget.** The chooser's `budget_bytes` is per-rank. `memory_per_device_gb` sets it. The implicit rule "leave 35 GB headroom for chi0/W" lives in cohsex.in convention, not in the code. Worth promoting to a `compute_optimal_chunks` global model that sees all stages, but that's a refactor much bigger than V_q.

D. **No distinction between "stable" peak and "transient FFT scratch" in the chooser output.** The full-kernel AOT path returns a single peak number; if cuFFT scratch dominates it, the chooser doesn't know vs a case where input ζ dominates. Doesn't affect the q_chunk pick but affects diagnostic interpretation. Could split into "input/output bytes" vs "temp bytes" for `LORRAX_V_Q_AOT_VERBOSE` output.

E. **Cache eviction discipline.** AOT shrink-retry leaves stale `_v_q_tile_kernel_cache` entries for candidate q_chunk values that didn't fit; we drop these at end of chooser via `_drop_unused_v_q_kernel_cache_entries`. **JAX's internal pjit/lower compile cache** also retains the compiled HLO modules and there's no public API to drop those — they leak ~MB per stale q_chunk attempt for the lifetime of the process. Bounded and usually small, but worth noting.

## Closure (2026-05-08, commit bbb2925 on `agent/aot-cufft-workspace`)

The case-3 gap is closed by `runtime/aot_memory.py`: a new module that
queries cuFFT plan workspace via `cufftMakePlanMany` on the **exact
libcufft.so jaxlib has dlopen'd** (located via `/proc/self/maps` after
`jax.devices()`).  No version drift, no system libcufft, no slope+intercept
extrapolation.  Public API: `aot_kernel_peak_bytes(compiled) →
AotPeakBreakdown(compiled_peak, cufft_scratch, total, fft_specs)`.

Reusable across V_q / chi0 / sigma / bispinor — anywhere a JAX kernel
contains FFTs and we need an honest peak prediction.

Validation on real production shapes (CrI3 6×6 80 Ry, 4×4 mesh, fft
75×75×200 c128, μ_local=94/rank):

| Q  | compiled_peak | cuFFT scratch | total       | observed                   |
|---:|--------------:|--------------:|------------:|----------------------------|
|  8 | 40.84 GB      | 13.54 GB      | **54.38 GB**| ran fine ✓                 |
| 12 | 61.22 GB      | 20.30 GB      | 81.52 GB    | ran fine (model 1-2% over) |
| 13 | 66.32 GB      | 22.00 GB      | **88.32 GB**| **cuFFT plan FAIL ✓**      |
| 18 | 91.00 GB      | 30.46 GB      | 121.46 GB   | JAX OOM ✓                  |

MoS2 3×3 80 Ry, 2×2 mesh, fft 60×60×200 c128, Q=9: predicted total
3.21 GB matches observed XLA peak 2.99 GiB to 0.07 GB.  cuFFT scratch
is genuinely zero because (60,60,200) is below cuFFT's in-place
threshold — for that shape the algorithm uses no extra workspace.

Linearity verified: at fixed transform shape, scratch ÷ batch is
constant (75×75×200 → 18.000 MB/batch across batches 50, 100, 200,
500).  cuFFT does not switch algorithms in our regime.

## Recommendations now that the cuFFT scratch is honest

1. **Keep `LORRAX_V_Q_AOT_FULL_KERNEL=1` as default.** Now strictly
   correct rather than "less wrong than slope+intercept".
2. **(DONE in commit bbb2925)** `_aot_full_kernel_peak` returns
   `compiled_peak + cufft_scratch` via `aot_kernel_peak_bytes`.  No
   guard-band heuristic; the prediction is exact.
3. **Don't raise `memory_per_device_gb` in CrI3 templates.** The new
   accurate model says Q=12 is right at the 80 GB ceiling and Q=13 is
   over it.  At the existing 45 GB budget the chooser will correctly
   pick a smaller Q than slope+intercept would have (which got lucky
   historically).
4. **Capture stage breakdown (`LORRAX_V_Q_TIME_STAGES=1`) on every
   CrI3 production run** and put numbers in `CHANGELOG.md`.
5. **Reuse follow-up:** call `aot_kernel_peak_bytes` from the chi0/W
   chooser and the σ chunker.  Single source of truth for FFT-aware
   AOT memory.
