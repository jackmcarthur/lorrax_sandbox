# The 800 ms cudaEventDestroy mystery — investigated and solved

Agent: C. Branch: `agent/C-sigma-ppm-cleanup`. Date: 2026-04-18.

## TL;DR

**The culprit was `cudaEventDestroy`, not `cudaMemcpyAsync`.**

At MoS2 3×3, 4-GPU, with async D2H on a private `ctx->stream`,
`cudaEventDestroy(ev_d2h_done)` on the writer thread blocked **750–775 ms**
on ranks 1/2/3 (but only ~8 μs on rank 0). The D2H itself was fast.

Fix: replace per-call event create/destroy with a **single ctx-owned
`d2h_event`** that's re-recorded on every write and destroyed once at
file close. Async D2H runs clean on all ranks, total runtime 25.07 s
(matches sync baseline, previously 31–35 s when the bug triggered).

## How we got there

Four phases of instrumentation that progressively narrowed the stall:

### Phase 1 — measure per-rank H5Dwrite wall times

```
sync mode:    r0-r3 all 766-772 ms (aligned)
async mode:   r0 1503 ms, r1/r2 734 ms, r3 771 ms
```

Initial take: rank 0 entered H5Dwrite 770 ms EARLY and idled waiting
for others at the MPI-IO collective. Phrased the earlier report as
"per-rank D2H asymmetry".

### Phase 2 — absolute timestamps on writer-thread entry

```
async chunk 0 per-rank:
 rank  t_enter       writer_sync_exit   t_wr_pre (before H5Dwrite)
  r0   2632175336513 +10 ms             +10.5 ms     ← fast
  r2   2632175336600 +10 ms             +735 ms      ← 725 ms gap
  r1   2632175336171 +10 ms             +786 ms      ← 775 ms gap
  r3   2632175336083 +10 ms             +786 ms      ← 775 ms gap
```

Handlers on all ranks fired within 500 μs of each other. cudaEventSync
exited within 10 ms on all ranks. But ranks 1/2/3 had a **~780 ms gap
between cudaEventSync exit and entering H5Dwrite** with seemingly
nothing in between.

### Phase 3 — microsecond timestamps around every C++ call

```cpp
int64_t t_worker_entry = now_us();
cudaEventSynchronize(ev_d2h_done);
int64_t t_sync_exit = now_us();
cudaEventDestroy(ev_d2h_done);
int64_t t_destroy_exit = now_us();
```

```
 rank  sync_enter→sync_exit  sync_exit→destroy_exit
  r0                +10 ms                 +8 μs
  r1                +10 ms              +753 ms   ◀
  r2                +10 ms              +775 ms   ◀
  r3                +10 ms              +775 ms   ◀
```

**`cudaEventDestroy` was the blocker.** Running with
`LORRAX_PHDF5_SKIP_DESTROY=1` (intentionally leaking events) produced:

```
sync 25.28 s   →   async (skip-destroy) 25.41 s
```

vs 35 s in the original buggy build. Confirmed.

### Phase 4 — what's special about `cudaEventDestroy`

Not reproducible in a standalone test (tried and gave up). Hypothesis:

**CUDA's stream-ordered (`cuda_async`) allocator keeps internal
bookkeeping tied to recently-recorded events.** When `cudaEventDestroy`
is called on an event whose record-stream (`ctx->stream`) has done a
D2H out of memory that was itself allocated on a DIFFERENT stream
(`xla_stream`), and that other stream still has queued work, CUDA
appears to wait for the stream-ordered allocator's "release" bookkeeping
to settle before freeing the event slot.

Rank 0's xla_stream happens to be nearly idle at the relevant moment
(less queued next-chunk compute); ranks 1-3 have 700-800 ms of next-
chunk compute queued. The destroy blocks waiting for that backlog to
drain on a per-rank basis.

Tested explicitly removing the `cudaStreamWaitEvent(xla_stream, ev)`
back-guard (which would make xla_stream wait on the event): **destroy
still blocked**. So it's not a xla_stream wait-list thing. Most likely
it's an allocator-internal side effect that I couldn't fully confirm
without reading the cuda_async driver source.

## The fix

Two sources of per-call event lifecycle become one fixed-cost event
lifecycle, both at ctx open/close:

```cpp
// ctx.h
struct PhdfCtx {
    ...
    cudaEvent_t  d2h_event           = nullptr;
};

// context.cc open_ctx
cudaEventCreateWithFlags(&ctx->d2h_event, cudaEventDisableTiming);

// context.cc close_ctx
if (ctx->d2h_event) cudaEventDestroy(ctx->d2h_event);

// write_ffi.cc handler — async, never creates a new event
cudaMemcpyAsync(pinned_buf, A, bytes, D2H, ctx->stream);
cudaEventRecord(ctx->d2h_event, ctx->stream);  // re-records existing event

// write_ffi.cc writer thread
cudaEventSynchronize(ctx->d2h_event);   // no destroy here
H5Dwrite(...);
```

Semantics: `cudaEventRecord` on an already-recorded event is well-defined
— it updates the record point to the current stream position. Subsequent
`cudaEventSynchronize` waits for the new record point. Safe because the
Python worker serializes writes, so there's never contention for the event.

## Final results — MoS2 3×3, 4-GPU, no profiler

| variant | Total | zeta.h5_write main-thread | ppm_sigma self |
|---|---:|---:|---:|
| sync baseline (`49_ffi_cached_C`) | 25.28 s | 3.48 s | 1.42 s |
| async broken (`50_ffi_async_C`) | 35.16 s | 0.002 s | 5.58 s |
| async sync-D2H workaround (`60_async_final_C`) | 24.96 s | 0.002 s | 1.53 s |
| **async pooled event (`61_async_resolved_C`)** | **25.07 s** | **0.020 s** | **1.51 s** |

- Async path **restored and working**: cudaMemcpyAsync on a private
  stream, no handler-thread block, main-thread dispatch ~0.2 ms.
- Total runtime matches sync baseline to within noise.
- `eqp0.dat` bit-identical.

## The mystery, framed

What we know for certain:
- `cudaEventDestroy` on a freshly-recorded event blocks 750-800 ms on
  3 of 4 ranks on Perlmutter when xla_stream has a compute backlog.
- The block is deterministic and rank-specific (r0 always fast, r1/2/3
  always slow, regardless of run).
- It is NOT the xla_stream back-wait (removing it doesn't help).
- It is NOT the D2H itself (cudaEventSync returns in 10 ms).
- Using a single reused event fully eliminates the problem.

What we don't know:
- Exact CUDA internals that cause the destroy to block.
- Whether this is specific to `cuda_async` allocator, or to the
  combination with the NVIDIA JAX container's CUDA driver version.
- Whether it reproduces on a non-Perlmutter system.

What this means for other projects: **if you're doing stream-ordered
async D2H out of JAX-owned memory and creating an event per call, be
aware that `cudaEventDestroy` may silently serialise on some per-rank
bookkeeping.** Pool your events.

## Commit

[`5eb6293`](../../sources/lorrax_C) — `phdf5 FFI: async D2H via pooled ctx event — solves the 800 ms mystery`.

## Run directories

All under [`runs/MoS2/02_mos2_3x3_nosym`](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/):

- `49_ffi_cached_C` — sync baseline
- `50_ffi_async_C` — async with per-call event (the bug)
- `52–54_async_*` — env-flag probes (nocollmeta / indep)
- `55_async_barrier_perchunk_C` — first per-rank H5Dwrite timestamps
- `56_async_nowait_C` — removed cross-stream-wait, no effect
- `m1–m9_*` — instrumented variants with cudaEvent timing
- `mA_async_skipdestroy_C` — **proof that destroy is the blocker**
- `60_async_final_C` — sync-cudaMemcpy workaround
- `61_async_resolved_C` — pooled event, final solution
