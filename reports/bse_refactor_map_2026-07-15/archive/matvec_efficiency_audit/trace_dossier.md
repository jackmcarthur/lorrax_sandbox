# BSE ring/stack matvec — trace dossier (TRACER)

Checkout `sources/lorrax_A` @ `agent/bse-phase2` HEAD `6ca714b`. All numbers are
measured on Perlmutter A100 (job 56010372, 1 node, 4 GPU), module-free
srun+shifter (`nvcr.io/nvidia/jax:25.04-py3`), `XLA_PYTHON_CLIENT_ALLOCATOR=platform`,
JAX x64. Synthetic in-device fixtures (no GW run) mirroring
`bse_ring_comm.ring_matvec_smoke_test`; shapes/shardings identical to production,
values random (physics-irrelevant for a structural/efficiency trace).

Harness + raw artifacts under `raw_traces/`:
`matvec_profile.py` (builder+timer+HLO+mem), `runlx.sh` (runner), `hlo_analyze.py`
/ `trace_analyze.py` / `trace_timeline.py` (login-node parsers), `hlo/*.hlo`
(compiled `as_text()`), `logs/*.log`, `traces/*` (xprof).

Two regimes:
| regime | nc | nv | ns | nk (grid) | μ_pad | note |
|--------|----|----|----|-----------|-------|------|
| **fixture** | 2 | 2 | 2 | 9 (3×3×1) | 400 | MoS2 gnppm gate size — latency regime |
| **inflated** | 48 | 48 | 2 | 16 (4×4×1) | 800 | compute/bandwidth regime |

Three matvecs: **stack** = `build_bse_stack_matvec` (production TDA, scan-in-shard_map);
**ring** = `build_bse_ring_matvec` (legacy TDA, ppermute rings); **full** =
`build_bse_ring_matvec_full` (non-TDA S=[[A,B],[-B,-A]], the W(0)/finite-q resolvent
SOLVE operator via `screening=True`).

---

## 0. Headline findings

1. **Both matvecs are bandwidth/latency-bound, never compute-bound.** Stack
   inflated 1×1 nt1: `bytes_accessed=8.96 GB` vs `flops=349 MFLOP` → arithmetic
   intensity **0.039 FLOP/byte**. Warm 15.1 ms ≈ **2.5×** the ~6 ms HBM-traffic
   floor (8.96 GB ÷ 1.5 TB/s). At fixture size everything is collective-latency
   bound (µs of compute under ms of launches).

2. **The stack W-term round-trips the full T-tensor ~6–8× through HBM.** Two
   **materialized** T-scale transposes (655 MB each at inflated 1×1) bracket the
   FFT — the k-batch↔k-minor reorder GSPMD cannot avoid because the encode/decode
   are batched-GEMMs (k outermost) and cuFFT needs k minor. These two transposes
   are ~25–30% of W-term HBM traffic = the single largest *avoidable*-bandwidth
   lever (§2, §4).

3. **Stack vs ring is a memory-vs-time trade with a clean crossover at n_trials.**
   The ring **batches** trials (b on the T axis) → collective count + FFT/GEMM
   passes are FIXED in n_trials; memory is LINEAR. The stack **scans** trials →
   memory FLAT (one T alive) but collectives + compute ×n_trials, SERIALIZED
   across the while loop. Measured: at **nt1** the ring is as-fast-or-faster AND
   lower-memory than the stack in every cell. The stack's only win is memory at
   nt>1 (§1).

4. **Collective inventory / matvec (2×2):** stack TDA = 2 all-reduce (V) + 4
   async/trial (W: 2 all-gather + 2 reduce-scatter) → ~6 @nt1, ~34 @nt8. Ring TDA
   = ~11 (8 ppermute + 1 all-reduce + 2 reduce-scatter), FIXED in nt. Full non-TDA
   = ~36 (16 ppermute ops ×~2 runtime + 3 RS + 1 AR) — the "~20 ms SOLVE" (§2).

5. **Two live donation defects** ("Some donated buffers were not usable"): (a)
   ring `apply_W_from_T` `donate_argnums=(0,)` on **T** fails at BOTH regimes
   (`c128[1,400,400,2,2,9]` fixture, `c128[1,800,800,2,2,16]` inflated) → copy
   fallback; (b) the known `c128[200,200,3,3,1]` = **W_q** at 2×2 fixture, emitter
   `bse_lanczos.solve_bse_sharded` `donate_argnums=(6,)` (§3).

---

## 1. Kernel-level timing — both regimes, 1 GPU + 2×2 (warm min-of-15, ms)

| matvec | regime | nt | 1×1 min | 1×1 med | 2×2 min | 2×2 med | temp/rank | peak/rank |
|--------|--------|----|--------:|--------:|--------:|--------:|----------:|----------:|
| stack | fixture | 1 | **2.46** | 2.53 | 15.35 | 16.50 | 69–185 MB | 76–211 MB |
| stack | fixture | 8 | 15.40 | 15.42 | 15.18 | 20.68 | flat | flat |
| stack | inflated | 1 | 15.13 | 15.62 | 18.80 | 27.94 | 1350 / 819 MB | 1604 / 902 |
| stack | inflated | 8 | 90.93 | 94.45 | 33.93 | 47.90 | 1359 / 578 MB | 1622 / 663 |
| ring | fixture | 1 | 2.42 | 7.61 | 9.23 | 20.00 | 185 / 69 MB | 211 / 76 |
| ring | inflated | 1 | 14.09 | 17.58 | 12.35 | 24.09 | 1350 / 347 MB | 1604 / 430 |
| full | fixture | 1 | — | — | 16.13 | 20.13 | 185 MB | 192 MB |

(temp/peak shown as 1×1 / 2×2 per rank where both measured.)

Reading it:

* **Fixture = pure collective latency.** stack nt1 1×1 = 2.46 ms → 2×2 = 15.35 ms
  (**6.2×** for ~6 collectives; wall α ≈ 2 ms/collective, device α from §"timeline").
  Ring nt1 fixture 2×2 = 9.23 ms < stack 15.35 ms: ppermute of tiny messages is
  cheaper than the stack's all-gathers of larger (band) tensors.

* **Scan serialization (stack, in n_trials).** 1×1 fixture nt1→nt8: 2.46→15.40 ms
  (**6.3×**). 1×1 inflated nt1→nt8: 15.13→90.93 ms (**6.0×**). Memory is FLAT
  across nt (temp 1350→1359 MB) — the design's promise holds — but time is ~linear
  in nt because the while loop runs each trial's GEMM/FFT/collectives in series.

* **Regime crossover (where 2×2 pays).** inflated nt8: 2×2 33.93 ms **beats** 1×1
  90.93 ms (2.68×) — compute large enough that 4-way parallelism wins. inflated
  nt1: 2×2 18.80 ≈ 1×1 15.13 (collective overhead cancels the parallelism).
  So 2×2 helps only once per-device work ≫ collective latency (large μ AND many
  trials).

* **nt1 stack ≥ ring in time, ≥ in memory, everywhere.** The single-vector
  consumers (`estimate_spectral_bounds_sharded`; the W(0)/finite-q resolvent
  SOLVE, which uses `full`) get no benefit from the stack — the stack exists for
  block solvers (nt>1). At inflated 2×2 nt1 the stack costs 819 MB / 18.8 ms vs
  the ring's 347 MB / 12.35 ms.

### Device-kernel category split (xprof, per-call over 10, % of summed device time)

`traces/*` via `trace_analyze.py` (sum over the 4 GPUs' compute+NCCL streams;
collective % includes NCCL wait — the OOM-/latency-relevant quantity).

| config | collective | gemm | fft | copy/transpose | elem | max concurrent (overlap) |
|--------|-----------:|-----:|----:|---------------:|-----:|--------------------------|
| stack fixture 1×1 nt1 | — | 23.7% | 30.0% | **36.7%** | 9.7% | 2 (memcpy/compute) |
| stack inflated 1×1 nt1 | — | 35.3% | 28.4% | **17.0%** | 19.3% | 2 |
| stack inflated 2×2 nt1 | **72.8%** | 8.8% | 5.3% | 5.4% | 7.7% | 9 |
| stack inflated 2×2 nt8 | **32.1%** | 24.3% | 22.0% | 14.4% | 6.9% | 8 |
| stack fixture 2×2 nt8 | **81.5%** | 4.4% | 6.3% | 5.1% | 2.4% | 8 |
| ring inflated 2×2 nt1 | **76.4%** | 9.0% | 5.6% | 3.4% | 5.2% | 8 |
| ring fixture 2×2 nt1 | **96.0%** | 0.4% | 1.0% | 0.6% | 1.7% | 7 |

**Flag A/B (COMMS #4) — the latency-hiding scheduler is already on by default.**
stack inflated 2×2 nt8, warm min: default **33.93 ms**; forced
`--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_all_gather=true`
= 38.70 ms (no gain — slightly worse, the pipelined-AG perturbs); forced
`=false` = 43.90 ms (**+29%**). So the default already schedules the compute↔collective
overlap seen below; the flag is a no-op, and *disabling* it costs 29%. The residual
cost is structural (collective wait + while-loop trial barrier), not a flag away.

**Overlap IS present (decisive).** `max concurrent = 7–9` at every 2×2 config →
XLA schedules compute concurrently with the async collectives: collectives run on
dedicated per-GPU NCCL streams (`Stream #139/#146/#151/#156`), GEMM/FFT/transpose
on compute streams (`#46/#64/#82/#100`). At 1×1 (no collectives) concurrency = 2
(async memcpy overlaps compute). So "enable overlap" is NOT a lever — it is on.

**Yet collectives still dominate (32–96%)** because at these small per-device sizes
the NCCL kernels are **wait-dominated**: the same all-gather shows GPU:0 = 621 µs vs
GPU:3 = 3081 µs, and all-reduce ≈ 2–3 ms — the "duration" is mostly spin-wait for a
straggler, not bandwidth. There isn't enough compute to fill the collective windows.
The straggler is **ROTATING, not a fixed rank**: per-GPU median collective duration
is identical (~108 µs on all 4 GPUs) and per-GPU totals are balanced (75–140 ms);
only the outlier maxes rotate (all-gather max on GPU:0 = 23.7 ms, reduce-scatter max
on GPU:2 = 20.9 ms, all-reduce max on GPU:3 = 22.9 ms). So the tax is barrier-sync
JITTER (whichever rank arrives early spin-waits), NOT a load imbalance from uneven
μ/ν padding or a bad rank/topology — which means COUNT-reduction (fewer barriers),
not balance-fixing, is the lever.

**The residual serialization is ACROSS trials, not within.** The stack scan is a
`while` loop; its loop-carried induction var serializes iterations even though the
trials are independent (`carry=None`). Evidence: the collective FRACTION falls
72.8% → 32.1% going nt1 → nt8 (more compute to hide under at nt8), and trial-to-trial
spacing in the timeline is ~3 ms (one collective-wait apiece). ⇒ the live levers are
(a) **bounded-unroll** the scan (expose trial i+1's all-gather under trial i's
FFT/GEMM) and (b) **batch trials onto one collective** (ring-style, one all-gather
for all trials vs n_trials all-gathers) — NOT "turn on overlap."

---

## 2. HLO evidence

### 2a. Stack W-term: two materialized T-scale transposes bracket the FFT

From `hlo/stack_inflated_1x1_nt1.hlo` (`_matvec` compiled `as_text()`), with
XLA source-line metadata:

| op | src_line | HLO | shape (1×1 infl) | bytes | kind |
|----|---------:|-----|------------------|------:|------|
| encode GEMM `kctM,cksN→MNtsk` | 99 | `custom-call.9.0 __cublas$gemm` | out `c128[16,1600,1600]` | 655 MB | cuBLAS |
| **encode→FFT transpose (L2)** | 99 | `transpose.196/.3` dims `{3,2,1,0}` | `c128[800,1600,2,16]` | **655 MB** | **materialized kInput fusion** |
| IFFT | 313 | `fft.10.0 IFFT len={4,4,1}` | `c128[800,800,2,2,4,4,1]` | 655 MB r+w | cuFFT |
| W_R × T_R | 105 | `loop_multiply_fusion` | same | 655 MB | fusion |
| FFT | 322 | `fft.11.0 FFT` | same | 655 MB r+w | cuFFT |
| **FFT→decode transpose (L3)** | 106 | `transpose.197/.2` dims `{4,1,3,2,0}` | `c128[16,800,2,2,800]` | **655 MB** | **materialized** (fused w/ ortho-norm ×0.25) |
| decode GEMM `kctM,MNtsk→cNsk` | 113 | `custom-call.10.0 __cublas$gemm` | out `c128[16,1600,48]` | — | cuBLAS |
| decode GEMM `kvsN,cNsk→cvk` | 116 | `custom-call.11.0 __cublas$gemm` | out `c128[16,48,48]` | — | cuBLAS |

Both transposes are real `transpose` ops inside `kInput` transpose fusions — **not
free `bitcast`s** (a dimension permutation `{4,1,3,2,0}` cannot be a bitcast).
At 2×2 they persist per-rank (`transpose.196.1 c128[400,800,2,16]`,
`transpose.197.1 c128[16,400,2,2,400]`, 164 MB/rank each). **Structurally
irreducible without changing the encode/decode**: a batched GEMM over k emits k as
the outer batch dim; cuFFT wants k as the three minor axes → the reorder is forced.
Levers: (i) fold the norm/scale into the decode GEMM (already partly done — the
ortho ×0.25 is fused into the L3 transpose); (ii) a k-minor "gathered" encode
(all_gather variant) that outputs k-minor directly — trades a transpose for a
larger encode collective (measure at inflated, §4/COMMS #5).

### 2b. Stack V-term (dense B1 exchange) — GSPMD collective + hidden M cost

* GSPMD compiles the k-summed exchange to **2 all-reduce** (async start/done):
  `all-reduce src_line=139` (S=`c128[8,400]` reshard to `sh.S_k0`) + `src_line=141`
  (U=V_q0·S). Both on TINY tensors → cheap. (Confirms COMMS H1: stack exchange is
  ~2 all-reduces vs the ring's ppermute-heavy `apply_V_ring`.)
* **Hidden cost (refines LAYOUT L1):** `compute_pair_amplitude` (`kcsm,kvsm→kcvm`,
  src_line=14) materializes **M_X and M_Y each = `c128[16,48,48,800]` = 472 MB**
  at inflated (custom-call.6.0 / .7.0). Two of them per matvec. Tiny at fixture
  (2×2×9×400 = 92 KB) but 2×472 MB at 48-band inflated — a real inflated-regime term.

### 2c. Collective schedule (2×2) — per matvec

`hlo/stack_inflated_2x2_nt8.hlo`, `hlo/ring_*_2x2_nt1.hlo`, `hlo/full_fixture_2x2_nt1.hlo`.
All collectives emit as async `-start`/`-done` pairs (overlap-capable). "Launches"
= runtime count (fori_loop/while bodies execute px|py|nt times).

| matvec | W-term (encode/decode) | V-term (exchange) | per-matvec total (2×2) | scales with nt? |
|--------|------------------------|-------------------|------------------------|-----------------|
| **stack** nt1 | 2 all-gather (src96 `all_gather('y')`, src98 `all_gather('x')`) + 2 reduce-scatter (src112 `psum_scatter('x')`, src115 `psum_scatter('y')`) | 2 all-reduce (src139, src141) | ~6 | **W ×nt** (in scan) |
| **stack** nt8 | 4/trial × 8 = 32 (in while body) | 2 all-reduce (once) | ~34 | yes |
| **ring** nt1/nt8 | 4 ppermute (src93 valence×py, src123 cond×px) + W-decode reshard | 4 ppermute (src235 step_y×py, src246 step_x×px) + 1 all-reduce (src258 `psum('y')`) + 1 reduce-scatter (src266 `psum_scatter('x')`) | ~11 (8 ppermute + 1 AR + 2 RS) | **NO (batched)** |
| **full** nt1 | encode_T A + B: src93,123 (A) + src154,185 (B) ppermutes | apply_V_ring ×4 sub-applies: src235×4, src246×4 ppermutes + 1 AR + 3 RS | ~36 (16 ppermute ops ×~2 rt + 3 RS + 1 AR) | (batched) |

The full non-TDA matvec (A(X),A(Y),B(X),B(Y) = 4 sub-applies) is the **~20 ms
SOLVE** operator (warm 16.1 ms min / 20.1 med @ fixture 2×2) — matches PHASE2_LOG.
Its ~32 ppermute launches over tiny fixture messages are the latency floor.

### 2d. GEMM operand layouts & fusions

All cuBLAS gemms are batched with **k (or k·μ) as batch dim 0**, operand layouts
uniformly row-major `{2,1,0}` / `{3,2,1,0}` — no `{0,1}` col-major mismatches,
no surprise operand transposes beyond the two T-scale reorders above. The
encode/decode chains are captured into **CUDA command-buffers** (`command_buffer`
calls) at nt1 → the launch latency of the GEMM/transpose sub-chain is amortized as
a CUDA graph. Fusion count (stack inflated 1×1): 18 (15 kInput reductions,
3 kLoop) — the elementwise/reduction glue is well-fused; the cost centers are the
2 FFTs + 2 T-transposes + the cuBLAS gemms, not fusion overhead.

---

## 3. Buffer / donation report

`compiled.memory_analysis()` peak per rank (temp+arg+out−alias):

| matvec | regime | nt | mesh | temp MB | peak/rank MB | one-T bound |
|--------|--------|----|------|--------:|-------------:|------------:|
| stack | inflated | 1 | 1×1 | 1350 | 1604 | 655 MB (μ²ns²nk·16) |
| stack | inflated | 8 | 1×1 | **1359** | 1622 | 655 MB (FLAT in nt ✓) |
| stack | inflated | 1 | 2×2 | 819 | 902 | 164 MB/rank |
| stack | inflated | 8 | 2×2 | **578** | 663 | 164 MB/rank (nt8 < nt1!) |
| ring | inflated | 1 | 2×2 | 347 | 430 | 164 MB/rank |

* **Stack temp is FLAT in n_trials** (1350→1359 MB @1×1) — the scan reuses one T
  slot; the design's central promise holds at the buffer level. Peak ≈ 2× the
  one-T bound (the FFT `T_R`/`U_R` scratch), as PHASE2 predicted.
* **Curiosity:** at 2×2 the nt8 while-loop peak (578 MB) is *lower* than the nt1
  unrolled peak (819 MB). At nt1 XLA unrolls the length-1 scan and hoists the FFT
  to the entry, materializing the full-T + norm scratch at once; the nt8 while body
  keeps strictly one T. So the "single matvec" is not the memory-minimal shape.
* **`alias_size = 0` everywhere** in these direct-matvec compiles → no input buffer
  is donated/aliased into an output; every argument is copied in. The matvec keeps
  all 9 args live (psi ×4, eps ×2, W_R, V_q0, X) — expected (persistent read-only
  arrays), but see donation defects below for the *solver-level* jits.

### Donation "not usable" warnings (perf-only, copy fallback)

| # | shape | origin | site | regime |
|---|-------|--------|------|--------|
| a | `c128[1,400,400,2,2,9]` | **T** | ring `apply_W_from_T` `donate_argnums=(0,)` (`bse_ring_comm.py:361`) | fixture |
| a' | `c128[1,800,800,2,2,16]` | **T** | same | inflated |
| b | `c128[200,200,3,3,1]` | **W_q** | `bse_lanczos.solve_bse_sharded` `donate_argnums=(6,)` (`bse_lanczos.py:240`) | fixture 2×2 |

(a/a') The ring's intent was to donate the bulky T into the W-term output, but the
output `WX` has shape `(nt,c,v,k)` ≠ T's `(nt,μ,ν,ns,ns,k)` — no same-shape output
to alias into, so donation is DECLINED. **Decisive HLO check:** the optimized ring
HLO contains **NO full-T copy** (`c128[1,400,400,2,2,16]` per-rank) — the declined
donation just frees the buffer, it does NOT emit a fallback copy. So the warning is
**cosmetic** (perf-neutral), NOT an extra full-T round trip. What IS present at the
`encode_T_ring`(shard_map)→`apply_W_from_T`(separate jit) boundary: 5 SMALL
(~1–5 MB/rank) chunk-layout copies (`c128[1,24,24,16]`, `c128[1,24,400,16]`,
`c128[1,24,16,2,400]`) — real, but ~40× smaller than a full-T pass. (b) W_q→W_R:
`ensure_W_R` builds `W_R = ifftn(W_q)` (a *new* buffer); the `c128[200,200,3,3,1]`
donation at `bse_lanczos.py:240` is the same declined-donation class (solver-path,
not reproduced here — needs a real `solve_bse_sharded` run; mechanism identical to
the ring T, i.e. expected cosmetic, no full-W_q copy). Net: both donation warnings
are perf-cosmetic; the fix is to DROP the unusable `donate_argnums` (removes the
warning, changes nothing measurable).

---

## 4. Per-einsum / per-category attribution at inflated size

Static (HLO byte accounting, inflated 1×1 nt1, T-tensor = 655 MB):

| stage | op | HBM touched | note |
|-------|----|-----:|------|
| V-term encode | M_X + M_Y gemms + S reduce | 2×472 MB + small | pair amplitudes |
| W encode GEMM1/2 | cuBLAS ×2 | ~655 MB out | k-batched |
| **W L2 transpose** | materialized | **655 r+w** | k-batch→k-minor |
| W IFFT | cuFFT | 655 r+w | |
| W_R mult | fusion | 655 r+w | |
| W FFT | cuFFT | 655 r+w | |
| **W L3 transpose** | materialized | **655 r+w** | k-minor→k-batch |
| W decode GEMM1/2 | cuBLAS ×2 | ~655 r | |
| **total** | | **≈8.96 GB** (matches `bytes_accessed`) | AI=0.039 F/B |

The two L2/L3 transposes are ~2 of ~7 full-T round trips ≈ **~19–28% of W-term
traffic** = the concrete avoidable-bandwidth number. FFT (2×) + W_R-mult are ~3
round trips and are algorithmically required. GEMMs are compute-negligible
(flops 349 M) — the whole matvec is memory-motion.

### Device-measured split (xprof, stack inflated 1×1 nt1 = bandwidth regime, NO
collective masking; per-call over 10, device time 14.07 ms ≈ warm 15.13 ms):

| category | %  | ms/call | top kernels |
|----------|---:|--------:|-------------|
| gemm | 35.3% | 4.97 | cutlass z884gemm 3.98 + scal (complex scale) 0.99 |
| fft | 28.4% | 4.00 | vector_fft 2.01 + regular_fft 1.99 (the IFFT+FFT pair) |
| elementwise/fusion | 19.3% | 2.71 | `loop_multiply_fusion` (W_R×T_R) 1.18 + norm/V-term |
| **copy/transpose** | **17.0%** | **2.39** | **`input_transpose_fusion_2` 1.07 + `_3` 1.07 = the two T-transposes (L2+L3)** |

**The two T-transposes = 2.14 ms = 15.2% of device wall at inflated.** Removing one
(k-minor encode or fused norm) recovers ~7–8% of the matvec at inflated, more where
the FFT/GEMM are smaller (fixture 1×1: copy/transpose is **36.7%**, the #1 category).
Note GEMM (35%) is the largest inflated category — but it is *bandwidth*, not FLOP,
bound (349 MFLOP total): the z884gemm reads/writes the 655 MB operands.

### Stack-vs-ring crossover (§5 evidence): warm min ms / peak-MB-per-rank

| regime·mesh | ring nt1 | stack nt1 | ring nt8 | stack nt8 | crossover |
|-------------|---------:|----------:|---------:|----------:|-----------|
| inflated 2×2 | **12.35 / 347** | 18.80 / 819 | 55.13 / 2647 | **33.93 / 578** | ~nt 2–3 |
| inflated 1×1 | **14.09 / 1604** | 15.13 / 1604 | 185.30 / **10530** | **90.93 / 1359** | low |
| fixture 2×2 | **9.23 / 76** | 15.35 / 76 | 19.41 / 553 | **15.18 / 69** | ~nt 2–3 |

**Ring wins nt1 on BOTH time and memory; stack wins nt8 on BOTH.** Ring nt1→nt8
grows 4.5× (time) / linear (mem: 347→2647 MB, ×7.6 at 1×1 → 10.5 GB); stack nt1→nt8
grows 1.8× (time) / flat (mem 578 MB). Crossover ≈ **nt 2–3**. So: the production
wiring is CORRECT to use the stack for block solvers (nt ≥ 4: FEAST subspaces,
block-Lanczos), and CORRECT to keep the ring live for single-vector
(`estimate_spectral_bounds_sharded`, the resolvent SOLVE) — that is the optimal
choice at nt1, not tech debt. The retirement plan should preserve a small-nt path,
not blanket-replace with the stack.

---

## 5. Efficiency thesis (for the synthesis owner)

1. **Bandwidth lever (inflated):** kill one of the two T-transposes by having the
   encode emit k-minor (all_gather-encode variant) or by fusing the FFT norm/scale
   to remove a pass. ~20–28% W-term HBM saving where W dominates.
2. **Latency lever (fixture / any small per-device problem):** overlap is ALREADY
   ON *within* a trial (max concurrency 7–9; NCCL on its own streams). The residual
   cost is (a) collective COUNT × wait-latency and (b) the `while`-loop serializing
   the *independent* trials. Two levers, both memory-vs-latency trades LAYOUT scopes:
   **bounded-unroll (=2–4)** the scan (pipeline trial i+1's all-gather under trial i's
   FFT/GEMM; ≤ unroll× peak-T), or **batch trials onto one collective** (ring-style —
   fewer, larger NCCL calls). A pure `--xla_gpu_enable_latency_hiding_scheduler=true`
   probe won't help (overlap already scheduled); the ceiling is the while-loop
   iteration barrier, which only source changes (unroll/batch) remove.
3. **Memory-vs-time dispatch:** the ring (batched, fixed collectives, linear mem)
   strictly beats the stack at nt1 and — pending ring-nt8 numbers — may beat it at
   nt8 in TIME wherever the linear-T memory fits. A size/nt-aware chooser (ring for
   small nt·μ that fits; stack when memory-bound) is the structural option, against
   the no-redundancy cost of keeping both.
