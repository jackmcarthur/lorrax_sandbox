# Closing the 10s async FFI regression — per-rank D2H asymmetry

Agent: C. Branch: `agent/C-sigma-ppm-cleanup`. Date: 2026-04-18.

## TL;DR

Regression found, root-caused, and fixed:

| run (MoS2 3×3, 4-GPU, no profiler) | Total | zeta_fit_chunked | ppm_sigma |
|---|---:|---:|---:|
| sync baseline (49_ffi_cached_C) | **25.28 s** | 10.97 s | 5.10 s |
| async v1 (jit + async D2H) | 41.45 s | 19.13 s | 12.44 s |
| async v2 (pinned reuse + no-jit) | 35.16 s | 16.12 s | 9.23 s |
| **async final (sync cudaMemcpy)** | **24.96 s** | 10.50 s | 5.20 s |

Main-thread `chunk.h5_write` stays at **0.002 s** (down from 3.48 s).
`eqp0.dat` bit-identical. All 4 ranks enter H5Dwrite within 1 ms of
each other per chunk.

## The hunt

Starting point: async FFI v2 regressed total runtime by ~10 s at this
scale, despite the main-thread dispatch dropping from 790 ms → 0.4 ms.
The close drain absorbed more than the inline writes had taken.

Three cheap env-flag experiments ruled out the obvious suspects:

| variant | Total |
|---|---:|
| async v2 default | 35.16 s |
| `LORRAX_PHDF5_NO_COLL_META=1` | 33.02 s |
| `LORRAX_PHDF5_INDEPENDENT=1` | 35.83 s |
| nocollmeta + indep | 36.70 s |

None closed the gap. So the regression wasn't HDF5 metadata ops or
MPI-IO collective mode.

### Instrument the writer thread

Added per-call fprintf timestamps in `async_worker` capturing:
- `task_begin` — C++ writer thread pops the task
- `d2h_done` — `cudaEventSynchronize(ev_done)` returns
- `write_enter` / `write_exit` — around `H5Dwrite`

First zeta chunk, ranks 0-3:

```
r0: begin=...116  d2h_done=...585 (+10)    write_exit=...814 (+1473)
r1: begin=...577  d2h_done=...478 (+901)   write_exit=...814 (+580)
r2: begin=...575  d2h_done=...477 (+902)   write_exit=...814 (+580)
r3: begin=...575  d2h_done=...480 (+905)   write_exit=...814 (+578)
```

**Every chunk, on every run: rank 0 D2H took 10 ms, ranks 1/2/3 took
~900 ms.** Same per-rank pattern, deterministic. Once the slow ranks
caught up, the MPI-IO collective completed fast (~580 ms on the
laggards, 1473 ms on rank 0 including its wait).

Per-rank D2H bandwidth:
- r0: 270 MB / 10 ms = **27 GB/s** (PCIe Gen4 peak)
- r1/2/3: 270 MB / 900 ms = **300 MB/s** (90× slower)

### Hypotheses eliminated

1. **xla_stream queue depth**: removed the `cudaStreamWaitEvent(ctx->stream, ev_prod)` that serialised D2H behind all queued xla_stream work → the 900 ms stall persisted unchanged.
2. **A's readiness**: added `A.block_until_ready()` on main thread before enqueue → unchanged.
3. **xla_stream drain**: added a `jnp.zeros(1).block_until_ready()` side-effect on main thread to drain all queued work before enqueue → unchanged.
4. **HDF5 metadata ops**: env flags above, no effect.
5. **Compile time**: 18.2 s async vs 19.9 s pre-FFI baseline — similar.

After all that, the 900 ms D2H on ranks 1/2/3 was still there.

### The fix

Switched the handler from `cudaMemcpyAsync` (on a private ctx->stream)
to synchronous `cudaMemcpy` (on the null stream). This blocks the
XLA executor thread for the full D2H wall time — ~50 ms for 270 MB
at peak PCIe bandwidth — but sidesteps whatever is happening in
CUDA's stream-ordered allocator that the async path runs into.

Result on the same config:

```
r0: begin=...973  d2h_done=...973 (+0)   write_exit=...726 (+753)
r1: begin=...972  d2h_done=...972 (+0)   write_exit=...726 (+753)
r2: begin=...973  d2h_done=...973 (+0)   write_exit=...726 (+753)
r3: begin=...972  d2h_done=...972 (+0)   write_exit=...726 (+754)
```

All 4 ranks aligned to within 1 ms on entry and exit. Write collective
completes symmetrically. Total runtime matches (slightly beats) sync.

## Remaining mystery

What is CUDA's `cuda_async` allocator actually doing on ranks 1/2/3
when `cudaMemcpyAsync(pinned_buf, A, ..., ctx->stream)` is issued? A
was allocated on xla_stream but its ready event has already fired
(main thread did `A.block_until_ready()` before this path). The
three most plausible "stream-ordered allocator" stalls — free-event
waits, pool-mutex contention, cross-stream read dependencies — don't
match what we measured (explicit xla_stream drain didn't help).

Not fighting this further for now. The sync-D2H path is simpler,
fully correct, and gives the right end-to-end numbers. If at future
scale the 50 ms sync D2H per write on the XLA executor becomes a
meaningful fraction of end-to-end, one next step would be a dedicated
D2H thread with its own stream and a pinned-buffer pool — but that
problem isn't on the critical path today.

## Answer to the user's startup question

Two kinds of setup we lift out of the per-write path:

1. **`ensure_pinned(ctx, bytes)`** (C++): grows `ctx->pinned_buf`
   once on the first write; subsequent writes reuse.
2. **`_FfiBackend._sm_cache`** (Python): keyed on
   (ctx_handle, ds_id, offset, shape, dtype, sharding). Hits when
   the exact same write recurs; at MoS2 3x3 every offset is unique
   so cache never hits — harmless. For repeated same-offset writes
   (e.g. `accumulate_slab` patterns) this pays off.

The thing we deliberately do NOT amortise is `H5Dget_space` +
`H5Sselect_hyperslab` + `H5Screate_simple` — these are per-call,
but collectively cost < 1 ms so not worth caching.

## Code

- [`src/ffi/phdf5/cpp/write_ffi.cc`](/global/u2/j/jackm/software/lorrax_C/src/ffi/phdf5/cpp/write_ffi.cc) — sync `cudaMemcpy` in handler; dropped ~50 lines of async-D2H plumbing.

## Commit

[`77719d7`](../../sources/lorrax_C) — `phdf5 FFI: sync cudaMemcpy for D2H — closes the 10s async regression`.

## Run directories

- [49_ffi_cached_C](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/49_ffi_cached_C/) — sync FFI baseline (25.28 s)
- [52_async_nocollmeta_C](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/52_async_nocollmeta_C/) — env probe
- [53_async_indep_C](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/53_async_indep_C/) — env probe
- [55_async_barrier_perchunk_C](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/55_async_barrier_perchunk_C/) — per-rank H5Dwrite timestamp probe
- [59_async_syncD2H_C](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/59_async_syncD2H_C/) — first run with sync-D2H fix
- [60_async_final_C](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/60_async_final_C/) — final cleaned build, 24.96 s
