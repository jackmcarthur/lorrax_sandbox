# GWJAX FFI Boundary Profile on LORRAX_A

Date: 2026-05-13

Branch: `sources/lorrax_A`, `agent/ffi-boundary-profile-a`.

Initial profile base: `origin/agent/zeta-ibz-header` at `991623a`.

Follow-up profile base: fast-forwarded to
`lorrax_D/agent/cusolvermp-ffi-profile` at `c21d855`.

## Goal

Profile the real MoS2 3x3 GWJAX path on 4 GPUs, not a toy null FFI, to answer
whether the JAX/FFI boundary and surrounding host orchestration are a primary
runtime cost.  This run intentionally used the sandbox profiling stack around
`gw.gw_jax` so that conclusions transfer directly to in-situ GWJAX execution.

No source files were manually edited for this checkpoint.  After the first
two profiles, `lorrax_A` was fast-forwarded to the D usage branch because that
stack already contained fixes for the top retrace locations exposed by the
first profile.

## Runs

Both runs reuse the completed D-side MoS2 3x3 inputs from
`runs/MoS2/00_mos2_3x3_cohsex/D_perf_after_2026-05-12`.

| Variant | Run directory | Linear algebra path |
|---|---|---|
| Baseline A | `runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_profile_2026-05-13` | charge `sharded_cholesky`; transverse JAX/CUDA LU |
| cuSOLVERMp A | `runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_cusolvermp_profile_2026-05-13` | `LORRAX_USE_CUSOLVERMP_CHARGE_FACTOR=1`, `LORRAX_USE_CUSOLVERMP_LU=1` |
| A fast-forwarded to D usage stack | `runs/MoS2/00_mos2_3x3_cohsex/A_rebased_D_usage_profile_2026-05-13` | default-on cuSOLVERMp on true 2D mesh, G-flat usage stack |

Launch pattern:

```bash
module load lorrax_A
export SLURM_JOBID=52886424
cd "$RUN"
LORRAX_NGPU=4 lxrun python3 -u /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
  --out profile -m gw.gw_jax -i cohsex.in 2>&1 | tee logs/profile_launch.log
```

The cuSOLVERMp variant also exported:

```bash
export LORRAX_USE_CUSOLVERMP_CHARGE_FACTOR=1
export LORRAX_USE_CUSOLVERMP_LU=1
```

Post-run analyzers were saved in each run's `logs/` directory:

```bash
python3 scripts/profiling/analyze_hlo_dump.py "$RUN/profile" > "$RUN/logs/analyze_hlo_dump.log"
python3 scripts/profiling/analyze_compile_log.py "$RUN/profile" > "$RUN/logs/analyze_compile_log.log"
python3 scripts/profiling/analyze_trace.py "$RUN/profile" > "$RUN/logs/analyze_trace.log"
```

## Headline Results

| Metric | Baseline A | cuSOLVERMp A | Delta |
|---|---:|---:|---:|
| `run_module:gw.gw_jax` wall | 101.25 s | 101.40 s | +0.15 s |
| Profiled section total | 85.422 s | 85.844 s | +0.422 s |
| `V_q_compute` | 38.771 s | 38.384 s | -0.387 s |
| Charge zeta fit | 10.870 s | 11.290 s | +0.420 s |
| Transverse mu1 fit | 8.514 s | 8.859 s | +0.345 s |
| Transverse mu2 fit | 8.287 s | 8.376 s | +0.089 s |
| Transverse mu3 fit | 8.434 s | 8.639 s | +0.205 s |
| `chi0_W` | 1.852 s | 1.821 s | -0.031 s |
| `sigma` | 4.209 s | 4.157 s | -0.052 s |

After fast-forwarding A to the D usage stack, the same high-level profile gave:

| Metric | A at `c21d855` | Delta vs cuSOLVERMp A |
|---|---:|---:|
| `run_module:gw.gw_jax` wall | 150.30 s | +48.90 s |
| Profiled section total | 133.750 s | +47.906 s |
| `V_q_compute` | 39.492 s | +1.108 s |
| Charge zeta fit | 22.937 s | +11.647 s |
| Transverse mu1 fit | 20.664 s | +11.805 s |
| Transverse mu2 fit | 19.776 s | +11.400 s |
| Transverse mu3 fit | 20.677 s | +12.038 s |
| `chi0_W` | 1.848 s | +0.027 s |
| `sigma` | 4.054 s | -0.103 s |

This follow-up is not an apples-to-apples "same algorithm, fewer retraces"
comparison.  The D usage stack changes the zeta/V_q storage path and reports a
G-flat memory model.  It reduces compile/retrace counts, but the total runtime
regresses because each zeta fit now spends much more time in HDF5 write/close
work.

The cuSOLVERMp opt-in changes the actual linear algebra paths:

- Baseline prints `path=sharded_cholesky` for the charge channel and `path=lu`
  for transverse indefinite channels.
- cuSOLVERMp prints `path=cusolvermp_cholesky` and `path=cusolvermp_lu`.

Despite that, total wall time is effectively unchanged.  The cuSOLVERMp path
slows the zeta-fit compute chunks by roughly 1.0-1.6 s in the transverse fits,
but HDF5 write/close timings drop by a similar amount in this one run.  Treat
that cancellation as noisy pipeline timing, not evidence that cuSOLVERMp helps
I/O.

## JAX Compile / Retrace Overhead

| Metric | Baseline A | cuSOLVERMp A | A at `c21d855` |
|---|---:|---:|---:|
| HLO modules dumped | 1077 | 1033 | 849 |
| Sum of per-module peak HBM | 206.26 GiB | 186.85 GiB | 219.11 GiB |
| XLA compile count | 582 | 562 | 478 |
| XLA compile time | 21.509 s | 16.378 s | 15.484 s |
| Trace+transform count | 1526 | 1444 | 1350 |
| Trace+transform time | 2.164 s | 1.880 s | 1.624 s |
| Cache misses | 629 | 602 | 525 |

The FFI choice is not the only, or main, source of first-run JAX overhead.  The
profile shows hundreds of retraces/compiles either way.  Top cache-miss
locations are nearly identical:

| Location | Baseline misses | cuSOLVERMp misses | Likely fix |
|---|---:|---:|---|
| `src/file_io/_slab_io_ffi.py:697` | 20 | 20 | Hoist `read_slab.<locals>._per_rank` out of the method or cache the jitted callable by static shape/backend key. |
| `src/file_io/_slab_io_ffi.py:585` | 17 | 17 | Same for `write_slab.<locals>._per_rank`. |
| `src/file_io/wfn_loader.py:956` | 19 | 19 | Repeated small `_where` with context/signature changes; inspect shape/context stability. |
| `src/common/gamma_matrices.py:122` | 19 | 19 | Repeated tiny `_where` over rank metadata; likely host/precompute candidate. |
| `src/common/gamma_matrices.py:99` | 18 | 18 | Repeated `dynamic_slice` over small metadata. |
| `src/common/fft_helpers.py:343` | 19 | 17 | FFT shape/context churn across chunk variants. |

This is the strongest "JAX boundary" finding: before chasing the cost of
entering one CustomCall, there are many avoidable compilations from local
closures and shape/context churn around the file I/O and transform pipeline.

The D usage stack moves this in the right direction but does not eliminate it.
The `_slab_io_ffi.py` sites are now `_get_read_sm.<locals>._per_rank` and
`_get_write_sm.<locals>._per_rank`, still with 20 and 17 misses respectively.
So the factory/cache change helped total compile count, but the callable
identity is still not stable enough from JAX's perspective.

## FFI / CustomCall Counts

HLO custom-call counts show the actual number of GWJAX-level FFI call sites is
small for the cuSOLVERMp solve/factorization path:

| Target | Baseline A | cuSOLVERMp A | A at `c21d855` |
|---|---:|---:|---:|
| `__cublas$gemm` | 144 | 137 | 233 |
| `xla_python_gpu_callback` | 12 | 12 | 60 |
| `lorrax_phdf5_write` | 49 | 50 | 50 |
| `lorrax_phdf5_read` | 32 | 33 | 33 |
| `lorrax_phdf5_read_kchunk_union` | 13 | 13 | 11 |
| `__cublas$triangularSolve` | 34 | 8 | 8 |
| `cusolver_getrf_ffi` | 13 | 4 | 4 |
| `cu_lu_pivots_to_permutation` | 13 | 4 | 4 |
| `lorrax_cusolvermp_batched_solve_lu` | 0 | 9 | 9 |
| `lorrax_cusolvermp_batched_potrf` | 0 | 3 | 3 |
| `lorrax_cusolvermp_batched_potrs` | 0 | 3 | 3 |

Interpretation: the steady-state Python-to-JAX-to-CustomCall boundary for these
linear algebra calls is not itself exploding.  There are 15 cuSOLVERMp
operation custom calls in the full profiled run.  Optimizing descriptor setup
inside those calls may still be tidy, but it cannot explain tens of seconds at
this scale.

## Device Trace Findings

| Metric | Baseline A | cuSOLVERMp A | A at `c21d855` |
|---|---:|---:|---:|
| Trace duration | 99.552 s | 100.026 s | 148.479 s |
| GPU events | 12,338 | 241,980 | not recorded in summary line, trace file present |
| GPU compute streams | 8 | 654 | 654 |
| H2D copies | 1,711 | 4,861 | 5,325 |
| H2D bytes / time | 11.11 GiB / 646.63 ms | 11.12 GiB / 726.94 ms | 11.14 GiB / 539.70 ms |
| H2D overlap | 0.000 | 0.000 | 0.000 |
| D2H copies | 588 | 2,334 | 2,798 |
| D2H bytes / time | 4.42 GiB / 183.03 ms | 4.42 GiB / 188.66 ms | 4.42 GiB / 185.11 ms |
| D2D copies | 846 | 28,173 | 28,251 |
| D2D bytes / time | 8.55 GiB / 11.71 ms | 29.07 GiB / 103.09 ms | 29.29 GiB / 104.65 ms |

The cuSOLVERMp path massively increases low-level GPU events and stream count.
Top cuSOLVERMp-related trace entries:

| Op | Count | Total | Source |
|---|---:|---:|---|
| `custom-call.53.0` | 161,784 | 5.195 s | `jit(_kernel)/.../jit(_solve)/.../ffi_call` |
| `custom-call.49.0` | 20,016 | 0.951 s | `jit(_kernel)/.../jit(_potrs)/.../ffi_call` |
| `custom-call.1.0` | 198 | 0.303 s | `jit(_potrf)/.../ffi_call` |

Those counts are GPU/internal events under a few HLO CustomCalls, not direct
Python call counts.

Host/device transfers are not a leading runtime term here: H2D+D2H time is
under 1 s in both profiles.  But the H2D overlap fraction is zero, so if larger
systems increase these transfers, scheduling/preloading could matter.

## Collectives / Sharding Findings

The dominant GPU time in both profiles remains collectives and tiled GW work,
not FFI entry overhead.

Baseline top kernels:

- `all-reduce-start`: 234 calls, 21.195 s, HLO `jit__batched_chol`.
- `all-gather-start`: 47 calls, 2.840 s.
- `all-gather-start.2`: 36 calls, 1.484 s.
- `all-to-all.5`: 1 call, 1.333 s.

cuSOLVERMp top kernels:

- `all-reduce-start`: 240 calls, 20.673 s.
- `all-gather-start`: 47 calls, 4.965 s.
- `all-to-all.5`: 1 call, 1.214 s.

Largest HLO collectives in both profiles:

| Source | Output bytes | Notes |
|---|---:|---|
| `src/common/wfn_transforms.py:291` | 2.47 GiB | repeated wavefunction transform all-gather |
| `src/common/wfn_transforms.py:354` | 2.47 GiB | repeated in kernel modules |
| `src/gw/v_q_tile.py:717` | 163.44 MiB | repeated V_q tile all-gather |
| `src/gw/v_q_tile.py:718` | 163.44 MiB | repeated V_q tile all-gather |

After the D usage fast-forward, the largest all-gathers moved to
`src/common/wfn_transforms.py:109` at 506.25 MiB each.  That is smaller per
collective than the earlier 2.47 GiB all-gathers, but it comes with larger
peak HBM modules (~10.36 GiB peak for repeated `jit__kernel`) and much larger
HDF5 write/close time in the zeta sections.

The largest user-visible wall section is still `gw_jax.V_q_compute`
(~38.4-38.8 s), followed by the zeta fits (~36-37 s combined).

## Current Conclusions

1. For this real MoS2 3x3 GWJAX run, the direct JAX FFI boundary for
   cuSOLVERMp factor/solve calls is not the main wall-time limiter.  The run
   has only 3 `potrf`, 3 `potrs`, and 9 `solve_lu` HLO CustomCalls.

2. cuSOLVERMp increases GPU-side event/kernel activity dramatically and slows
   the zeta-fit compute chunks, but total end-to-end wall time stays roughly
   flat because unrelated I/O/close timings shift in the opposite direction.
   This is not a convincing speedup path yet.

3. The most actionable JAX overhead is first-run orchestration: roughly
   525-629 cache misses and 15-22 s of XLA compile time.  Local jitted
   closures/factories in the PHDF5 slab I/O FFI remain suspicious because the
   cache-miss reason is still "never seen function" for `_per_rank` callables,
   even after the D usage stack's caching work.

4. The largest steady-state performance issues remain high-level GWJAX data
   motion and collectives: `V_q_compute`, zeta-fit chunk kernels, repeated
   2.47 GiB wavefunction all-gathers, and large all-reduces in the factorization
   path.

5. The D usage stack reduced compile count and removed the 2.47 GiB all-gather
   as the largest HLO collective, but regressed total runtime by ~49 s on this
   MoS2 profile because zeta HDF5 write/close time increased sharply.  That is
   now the most urgent follow-up if this stack is meant to be the active
   performance branch.

## q-loop Acceleration Probe

After the high-level profile, I tested the least invasive q-loop acceleration
idea: CUDA Graph replay around the existing full-mesh `for q in range(nq)`
`cusolverMpPotrf` loop.

Prototype design:

- Keep the current JAX API and current single FFI CustomCall over the full
  `(nq, n, n)` batch.
- Add an opt-in `LORRAX_CUSOLVERMP_POTRF_GRAPH=1` path inside
  `batched_potrf_ffi.cc`.
- Use ctx-owned staging/workspace buffers, because CUDA Graph capture bakes in
  device pointers and JAX/XLA may pass different input/output buffers on later
  calls.
- Capture only the cuSOLVERMp `Potrf` loop; copy input into staging before
  graph launch and copy staging back to the JAX output afterward.

Build result: the prototype compiled cleanly, but the runtime test failed
during warmup on all 4 ranks:

```text
INTERNAL: cuda graph capture cusolverMpPotrf failed at q=0: status=7
```

Baseline for the same shape immediately before the failed graph run:

```text
mode=potrf nq=9 n=640 dtype=c128 mesh=2x2
summary: mean=9.519 ms, median=9.523 ms, p90=9.706 ms, min=9.123 ms, max=9.837 ms
```

Interpretation: NCCL itself supports CUDA Graph capture in this environment,
but `cusolverMpPotrf` in the currently loaded cuSOLVERMp 0.7.2 stack does not
appear stream-capture-safe.  The failed prototype was removed from source and
the FFI shared library was rebuilt from the reverted source, so no broken
`LORRAX_CUSOLVERMP_POTRF_GRAPH` path remains in the checkout.

This rules out the simplest "graph replay the existing q loop" path on the
current library stack.  The remaining q-loop options are more invasive:

- Try a newer cuSOLVERMp stack only if the NCCL dependency can also be updated.
  The current code already warns that cuSOLVERMp 0.8 expects NCCL >= 2.27, while
  this run loaded NCCL 2.26.3.
- Revisit q-parallel layout/sub-communicators, but that changes sharding and
  communication topology rather than being a wrapper-only optimization.
- Avoid cuSOLVERMp for small `n` if a JAX/XLA or cuBLAS-backed factorization
  wins end-to-end at MoS2 scale.

## Next Optimization Queue

1. Fix the D usage stack zeta I/O regression first.  In the rebased profile,
   each zeta fit spends ~5.4-5.9 s in `zeta_fit.close_io` plus large
   `zeta_fit.chunk.h5_write` time; this dominates the regression.

2. Continue the `_per_rank` callable identity work in
   `src/file_io/_slab_io_ffi.py`.  The first D caching pass reduced global
   compile count, but the top cache misses still report new `_per_rank`
   function identities.

3. Audit the `wfn_loader.py`, `gamma_matrices.py`, and `fft_helpers.py` cache
   misses for small metadata operations that can be precomputed on host or
   folded into larger jitted regions.

4. Use the HLO collective report to inspect both generations:
   pre-rebase `wfn_transforms.py:291`, `wfn_transforms.py:354`, and
   `v_q_tile.py:717-718`; post-rebase `wfn_transforms.py:109`.  The D usage
   branch changed the collective shape, but not necessarily the total cost.

5. If cuSOLVERMp remains desired, focus on reducing its internal per-q
   fan-out/stream activity, not just CustomCall setup.  The earlier D-side FFI
   microprofile already showed descriptor setup and cross-stream waits are
   microsecond-scale on this build.

6. Repeat this same A-side profile after each change.  The important scoreboard
   is `profile/compile_summary.md`, `profile/hlo_summary.md`, and
   `profile/trace_summary.md`, plus the `gw_jax.zeta_fit_chunked*` and
   `gw_jax.V_q_compute` sections in `logs/profile_launch.log`.
