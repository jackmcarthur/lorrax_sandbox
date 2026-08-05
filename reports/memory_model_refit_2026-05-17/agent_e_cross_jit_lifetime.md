# Agent E — Cross-jit lifetime audit: are centroids+L_q in HBM during accumulate?

**Date:** 2026-05-17. **Method:** read-only HLO inspection. Pure file:line citations, no code edits.

## TL;DR

The HLO is unambiguous: the `accumulate_rchunk_to_gflat` jit has **two parameters only** — `zeta_chunk` slab + `gflat_acc` — and NO centroid- or L_q-shaped parameters. Buffer-assignment.txt is per-jit (XLA does NOT report process-wide allocator state), so it cannot DIRECTLY confirm or deny centroid persistence in HBM outside the jit. But Q1+Q2 prove centroids+L_q are not used **inside** the accumulate jit, while Q4+Q5 establish the OOM cannot be explained by the accumulate jit's own scratch.

---

## Q1: centroid-sized buffers in accumulate jit's HLO? — **No.**

Bispinor centroid shapes (from M1 in `agent_d_hlo_calibration.md`): `c128[36, 380, 150, 4]` ψ_L (125.24 MiB/rank) and `c128[36, 380, 160, 4]` ψ_R (133.59 MiB/rank). At cs=360, `module_0474.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt` lines 7–14 list every allocation:

```
allocation 7: 16.61 GiB preallocated-temp
allocation 0: 3.06 GiB parameter 1   = c128[36,95,59990] (gflat_acc, donated/output)
allocation 1: 1.25 GiB parameter 0   = c128[36,95,24576] (zeta_chunk slab)
allocation 2-6: <8.3 MiB total       (constants + s32[] scalar)
```

No allocation matches the centroid shapes. Same in cs=1 (`module_0363` doesn't exist for accumulate at gflat=1; the matching gflat=1 accumulate jit is given in agent_d M3 — 4.35 GiB total, again only zeta+gflat_acc params).

`module_0474.*-buffer-assignment.txt:1-17` enumerates allocations 0–7 by parameter index — `parameter 0 = c128[36,95,24576]`, `parameter 1 = c128[36,95,59990]`, `parameter 2 = s32[]`. **There is no parameter 3+, no centroid input.**

## Q2: buffer-assignment.txt liveness — confirms per-jit view only

`module_0474.*-buffer-assignment.txt` lists 30+ `value:` entries inside `allocation 7` (the preallocated-temp scratch). Every offset is one of: `c128[360,1125000]` (FFT-box-flat, offset 6480000384), `c128[360,75,75,200]` (FFT-box-spatial, offset 384), `c128[3600,59990]` (gflat scan-carry, offset 12960000384), `c128[3600,24576]` (zeta slab pad, offset 16415424384), and small index/scalar tuples. **No 1.03 GiB ψ_L slab, no 1.10 GiB ψ_R slab, no L_q slot.** The scratch contains only the FFT-pipeline + scan-carry working set the accumulate body uses.

## Q3: Process-level evidence — **XLA's HLO does not provide this.**

This is the cite-able limitation. `*-buffer-assignment.txt` describes only the buffer plan for one HLO module (one jit). It has no "Total allocated memory" or "live ranges" table that crosses jit boundaries. JAX/XLA tracks process-wide live arrays separately (via `jax.live_arrays()` / `memprof`). So the HLO alone cannot DIRECTLY tell us whether the Python-scope centroid arrays are still resident in BFC pool when accumulate is invoked.

**However** — `module_0474.jit__kernel.sm_8.0_gpu_after_optimizations.txt` (the optimized HLO) declares its full parameter list. The fact that centroids are not parameters means **XLA's allocator is free to evict them between fit_one_rchunk's outer call and accumulate's outer call**: there is no XLA-side liveness requirement keeping them resident.

## Q4: cross-jit lifetime via shared offsets — **disjoint by construction**

`module_0343.jit_fn.*-buffer-assignment.txt:1-15` (fit_one_rchunk, cs=21232 case, in the failed-verify run): centroids appear as `allocation 2: parameter 1, c128[36,380,160,4]` and `allocation 3: parameter 0, c128[36,380,150,4]`. Each fused kernel allocates its scratch independently — `module_0377` (accumulate, cs=1414) and `module_0343` (fit_one_rchunk) have **separate `allocation` numbering** and **independent offset spaces**. Comparing offsets across jits is not meaningful; XLA's BFC allocator reassigns offsets per-call. The relevant question — does Python still hold a ref to the centroid `Array` — has the answer "yes" from a glance at `sources/lorrax_B/src/common/isdf_fitting.py` (the centroids are created once and reused across r-chunks/g-flat chunks), but **HLO cannot prove this**.

## Q5: cs=1414 case in totality

`module_0377.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt:1-15`:

```
Total bytes used: 60870263156 (56.69 GiB)
allocation 7:  52.54 GiB preallocated-temp
allocation 0:   3.06 GiB parameter 1 (c128[36,95,59990] gflat_acc, donated/output)
allocation 1:   1.08 GiB parameter 0 (c128[36,95,21232] zeta slab)
```

Inside the 52.54 GiB scratch:

```
23.70 GiB  2×c128[1414, 1125000]       (FFT-box flat, in+out)
23.70 GiB    c128[1414, 75, 75, 200]   (FFT-box spatial)
 3.79 GiB  5×c128[4242, 59990]         (gflat scan-carry, padded N=4242)
 1.34 GiB    c128[4242, 21232]         (zeta slab pad)
```

That's `factor_D = 2` (× FFT box of 22.2 GiB at cs=1414) + persistent scan-carry + slab — **56.69 GiB total, not 120 GiB**. The accumulate jit's own footprint at cs=1414 is **comfortably below the 70 GiB cohsex.in budget**. If the run OOM'd at runtime, the missing ~63 GiB must come from outside this jit — either (a) Python-held centroids+L_q + other zeta-fit allocators that **HLO cannot account for**, or (b) cuFFT plan-scratch handed out by cuDNN below XLA's accounting (the "out-of-place 3D FFT cs=1414" plan can be ~10 GB per device).

---

## Verdict

The proposed `_peak_D_accumulate` fix **cannot be justified by HLO citation alone**. Two facts:

1. **Centroids+L_q are NOT parameters of the accumulate jit**, and they do NOT appear as offsets in either the preallocated-temp scratch or the value-list of `module_0474` (gflat=360) / `module_0377` (gflat=1414). The accumulate jit's HLO-reported footprint at cs=1414 is 56.69 GiB — fits the budget by itself.

2. **Whether they are resident in HBM during accumulate is a Python/JAX-runtime question, not an HLO question.** XLA's buffer-assignment.txt is per-module. If centroids are still referenced from Python scope (which `isdf_fitting.py` strongly implies — same `L`, `R` arrays passed to each r-chunk), JAX's BFC pool keeps them resident, but **no HLO citation can prove this**.

**Recommendation:** The Peak D model should add centroids+L_q **only with a separate, non-HLO citation** — a `jax.live_arrays()` snapshot inside the accumulate loop, or a `memprof/*.prof` taken between fit_one_rchunk and accumulate calls in the same run. Without that, the proposed fix is plausible but unproven, and the cs=1414 OOM has at least one untested alternative cause: **cuFFT plan-scratch growing super-linearly with cs**, which the planner also doesn't model. The user's "no planner parameter without smoking-gun HLO" rule says: **do not commit the fix yet**.

If forced to choose between adding centroids (~2 GiB) vs L_q (~3 GiB) based on HLO alone: **neither has HLO evidence.** Both omissions are equally justified or unjustified by the dumps. Pick the cause with non-HLO evidence (`memprof` or `live_arrays`).
