# phdf5 FFI — async/buffer-reuse sweep, read path mirrored from writes

Agent: C. Branch: `agent/C-sigma-ppm-cleanup`. Date: 2026-04-19.

## TL;DR

Applied the two write-path perf fixes (runtime-offset FFI + pooled
event) to the READ path as well. Result: **XLA compile total
18.51 s → 16.43 s** (−2.08 s, cold-start savings), and at steady
state **total gw_jax wall 24.72 s → 24.02 s** (−0.70 s) on MoS2
3×3, 4-GPU. All 10 previously-unnamed "jit__unnamed_wrapped_function"
HLO modules collapse into properly-named `jit__per_rank` entries,
one per distinct dataset/shape signature.

## Profiling flow used this session

1. Ran profiled gw_jax with the latest build (`scripts/profiling/run_profiled.py`).
2. `analyze_hlo_dump.py` showed 10 `unnamed_wrapped` modules with
   identical arg signatures but varying-offset FFI Attrs.
3. `analyze_compile_log.py` showed XLA compile = 17.9 s out of 24 s
   total — 75% of wall was compile on cold start.
4. `retrace_details.txt` pinpointed 4 writes + 4 reads of zeta_q
   + 4-5 sigma_mnk writes as separate compiles, all of identical shape.
5. Agent research confirmed our design (`xla::ffi::Future` + bg worker
   + pooled events) is canonical — matches HDF5 vol-async and
   OpenXLA's AsyncStart/Done pattern. No simpler JAX-side pattern missed.

## What landed

### Read path mirrors the write path

- `read_ffi.cc`: `offset_base` FFI Attr → `Buffer<S64>` runtime input.
  Handler D2H-copies 24 bytes and reads it.
- `ctx->h2d_event`: new pooled event on `PhdfCtx`, created once at
  open, destroyed once at close. Replaces the per-call
  `cudaEventCreateWithFlags`/`cudaEventDestroy` pair that would
  otherwise trigger the known ~800 ms stall on non-rank-0 processes
  under JAX's cuda_async allocator.
- Dropped `cudaEventSynchronize` at read-handler end. The
  `cudaStreamWaitEvent(xla_stream, h2d_event)` already orders
  subsequent ops behind the H2D, so the handler returns without
  blocking its executor thread.
- `_slab_io_ffi.read_slab` now uses the shared `_sm_cache` (keyed on
  `("read", ctx_handle, ds_id, mesh_shape, …)`) with `jax.jit`
  wrap — mirrors the write path.

### Measurements (MoS2 3×3, 4-GPU, no profiler, 3-run avg)

| metric | before (write fix only) | after (this commit) | Δ |
|---|---:|---:|---|
| Total wall | 24.72 s | **24.02 s** | **−0.70 s** |
| zeta_fit_chunked | 10.22 s | 10.03 s | −0.19 s |
| V_q_compute | 6.49 s | 6.25 s | −0.24 s |
| ppm_sigma | 5.07 s | 5.07 s | — |

From the profile:

| metric | before | after |
|---|---:|---:|
| XLA compile total | 18.51 s | **16.43 s** |
| `unnamed_wrapped_function_` HLO modules | 10 | **0** |
| Named `jit__per_rank` modules | 0 | 10 (all distinct signatures) |

## What's left — and why we're not chasing it

Explored several more ideas; none cleared the "within reason" bar:

- **Double-buffered writes** (2 pinned buffers, D2H pipelines with
  previous H5Dwrite): saving ~30 ms × 3 = ~90 ms, at a cost of
  changing the Python-worker serialisation contract. Not worth.

- **V_q_compute read prefetch** (issue read N+1 during processing of
  N): requires refactoring V_q_compute's inner loop, not just the
  SlabIO API. The reads are already bounded by H5Dread collective
  rendezvous, so the prefetch win would mostly hide ~50 ms of H2D
  per read. Not worth the blast radius.

- **`ALLOC_TIME_EARLY` removal in ctx setup**: was deliberately
  enabled to avoid zero-fill on create; removal might save H5Dcreate
  time but risks per-write extension cost. Didn't test.

- **Persistent XLA compile cache**: would eliminate the 16 s compile
  on every cold start, but that's a run configuration change, not a
  code change. Separate decision.

- **JAX async FFI alternatives**: research confirmed our
  `ffi::Future` + bg worker is the canonical pattern. No simpler
  JAX-native async primitive exists today.

## Summary of the perf improvements across this extended session

Starting from the committed "async writes (broken)" state that
regressed runtime by +10 s, ending at 24.02 s:

| commit | focus | Total |
|---|---|---:|
| (pre-session) async broken | —  | 35.16 s |
| `5eb6293` mystery solved | pooled d2h_event | 25.07 s |
| `0a41f42` startup tweaks | single SlabIO + eager MPI | 24.84 s |
| `e31970d` runtime-offset writes | no per-chunk compile (write) | 24.72 s |
| **`a5e404f` runtime-offset reads** | **no per-chunk compile (read)** | **24.02 s** |

Total improvement from regression peak: **35.16 s → 24.02 s**
(−11.14 s). Vs sync-FFI baseline: **25.28 s → 24.02 s** (−1.26 s).
`eqp0.dat` bit-identical at every step.

## Commits

- `e31970d` — write path runtime-offset + jit-wrap
- `a5e404f` — read path runtime-offset + pooled h2d_event + jit-wrap
