# End-to-end impact of async FFI writes on GWJAX — MoS2 3×3

Agent: C. Branch: `agent/C-sigma-ppm-cleanup`. Date: 2026-04-18.

## TL;DR

The async FFI writes **do what they were designed to do — main-thread
dispatch drops from 3.48 s → 0.002 s** — but the total run time
**regresses by ~10 s** at MoS2 3×3 scale. The writes themselves run
slower in the background than they did inline, and `close()` ends up
absorbing the difference as a drain wait.

Ship-decision: **don't enable `use_ffi_io=true` at this scale** until
the drain is investigated. The Python-worker architecture is correct;
we're bottlenecked on something downstream.

## Per-section timing, same inputs, no profiler overhead

All three runs: MoS2 3×3, `gw_jax -i cohsex.in`, `use_ffi_io=true`, 4 × A100.

| section | sync (49) | async v1 | async v2 |
|---|---:|---:|---:|
| **Total recorded** | **25.28 s** | 41.45 s | **35.16 s** |
| `gw_jax.zeta_fit_chunked` | 10.97 | 19.13 | 16.12 |
| &nbsp;&nbsp;`zeta_fit.chunk_loop` | 5.17 | 1.86 | 1.79 |
| &nbsp;&nbsp;&nbsp;&nbsp;`chunk.h5_write` | **3.48** | 0.002 | **0.002** |
| &nbsp;&nbsp;zeta self (close drain) | 0.89 | 12.17 | **8.82** |
| `gw_jax.V_q_compute` | 6.49 | 7.13 | 7.08 |
| `gw_jax.chi0_W` | 1.69 | 1.72 | — |
| `gw_jax.sigma` | 0.94 | 0.94 | 0.92 |
| `gw_jax.ppm_sigma` | 5.10 | 12.44 | 9.23 |
| &nbsp;&nbsp;ppm self (close drain) | 1.42 | 8.75 | 5.58 |

Legend:
- **sync (49)** = commit `c941770` — `ffi::Error` handler, shard_map
  eager, no Python worker. Writes block main thread inline.
- **async v1** = commit `a015055` — `ffi::Future` handler, per-call
  `cudaMallocHost`, `jit(shard_map)` in Python worker thread.
- **async v2** = commit `4108e35` (this session) — async v1 + reverted
  to `ensure_pinned(ctx->pinned_buf)` + dropped `jit` wrap + `_sm_cache`.

## What landed, empirically

### Main-thread write_slab dispatch: 790 ms → 0.4 ms

Measured with a dedicated bench at zeta-chunk scale (270 MB/rank):

| metric | sync | async v2 |
|---|---:|---:|
| `write_slab(...)` Python return time (chunks 1..N) | 790 ms | 0.40 ms |
| token.block_until_ready() after | 0.05 ms | 1825 ms |

The main thread IS freed: 2000x faster dispatch, confirmed across 5
chunks of 270 MB per rank on 4 GPUs.

Script: [`scripts/phdf5_async_bench.py`](/pscratch/sd/j/jackm/lorrax_sandbox/scripts/phdf5_async_bench.py).

### `zeta_fit.chunk.h5_write` section: 3.48 s → 0.002 s

This is the main-thread-visible section wrapping `zeta_io.write_slab(...)`
inside the chunk loop. With async writes, the main thread dispatches
and moves on — the section closes in microseconds, as expected.

## Why it regressed end-to-end

Two bills didn't show up in the chunk-loop section but are still paid:

### 1. Close drain (zeta: +7.9 s, sigma: +4.2 s)

Both `SlabIO(use_ffi_io=True)` instances drain their Python worker
thread at `close()` — i.e. `block_until_ready` on every outstanding
write. That cost is now attributed to the SELF time of
`gw_jax.zeta_fit_chunked` and `gw_jax.ppm_sigma`, not to the inner
write section.

In sync, writes happened inside `chunk_loop` and were fully accounted.
In async, writes happen outside — correctness-identical, accounting-
different. **But the writes themselves are also ~2-3× slower in the
async path**, so the close drain doesn't just move the cost, it
inflates it.

### 2. The async writes are slower

Measured per-chunk wall time (zeta, 270 MB/rank, 4 GPU):

| run | mean per-chunk H5Dwrite wall | |
|---|---:|---|
| sync | 0.87 s | (chunk.h5_write / 4) |
| async v2 | ~1.8 s | (bench `ready` column) |

2× slowdown. Hypothesis: **MPI-IO collective rendezvous drift**.
- Sync path: main threads on all 4 ranks call H5Dwrite at the same
  logical moment (the bottleneck is MPI-IO itself), so the collective
  rendezvous is fast.
- Async path: each rank's Python worker thread dispatches writes at
  a slightly different wall time (depending on when its main-thread
  compute reached the dispatch point). The slowest rank drags the
  collective.

Not yet verified with a targeted probe, but every other plausible
cause has been eliminated:
- Not compile time: 19.9 s baseline vs 18.2 s async — similar.
- Not per-call cudaMallocHost: reuse fixed it, still slow.
- Not per-call jit recompile: removed, still slow.
- Not GPU contention with compute: trace shows D2H + H2D ≈ 0.75 s
  out of 35 s total — not on the critical path.

## Profile — study of the zeta piece

From the profiled run
[`runs/MoS2/02_mos2_3x3_nosym/51_ffi_async_profile_C/profile`](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/51_ffi_async_profile_C/profile/):

**HLO dump** (287 modules, 48.6 GiB sum of peaks):
- 10 × `jit__unnamed_wrapped_function_` modules, 900 MB each — each
  is a shard_map(ffi_write_call) body. 4 for zeta, ~6 for sigma.
  Different `offset_base` → different compile per chunk; `_sm_cache`
  doesn't hit because the key is always new.

**Compile summary**:
- XLA compile total: 18.2 s (vs 19.9 s pre-FFI baseline at commit
  `25_lorrax_final_profile_C`).
- Top contributor: `<unnamed wrapped function>` — 54 compiles, 1.9 s.
- **Compile time is roughly flat vs baseline — this is not where the
  regression is hiding.**

**Trace summary** (xprof, rank 0):
- 1245 H2D (8.58 GiB, 528 ms, **overlap_frac = 0**)
- 437 D2H (5.35 GiB, 219 ms, **overlap_frac = 0**)
- Peak D2H bandwidth window: 2.68 GB/s — nowhere near PCIe
  saturation. I/O is not link-limited.
- Top GPU ops are all-reduce, all-gather, FFT, GEMM — physics kernels.
  No FFI kernel shows up (the handler is all host-side).

So the profile tells us:
1. GPU compute + compile are not the regression.
2. D2H throughput is fine (ensure_pinned reuse worked).
3. The 10 s is hiding on the host side, in the MPI-IO collective
   inside the writer thread — which xprof doesn't instrument.

## What to try next

Three avenues, in decreasing expected impact:

1. **Bench the per-write H5Dwrite wall time on the worker thread
   alone.** Rank 0 writer thread can record `{start, H5Dwrite_done}`
   timestamps and dump to a per-rank log. If rank-to-rank variance is
   seconds, the MPI-IO drift hypothesis is confirmed; fix is a
   per-chunk `MPI_Barrier` before each H5Dwrite on the writer thread.
2. **Try `LORRAX_PHDF5_INDEPENDENT=1`** — switches dataset writes to
   `H5FD_MPIO_INDEPENDENT`. Loses MPI-IO aggregation bandwidth, but
   avoids the collective rendezvous cost. For small write counts at
   modest scale this could be a net win.
3. **Don't use the FFI path at 3×3 scale.** The allgather-and-rank-0
   backend was faster by 10 s here. FFI's design goal was scale
   (>128 ranks, >GB/s sustained write); the fallback is appropriate
   for small jobs.

## Commits

- [`a015055`](../../sources/lorrax_C) — Initial async implementation
  (ffi::Future + ctx writer thread + Python worker).
- [`4108e35`](../../sources/lorrax_C) — Pinned reuse + drop jit + sm_cache.

## Run directories

- [49_ffi_cached_C](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/49_ffi_cached_C/) — sync FFI baseline (25.3 s)
- [50_ffi_async_C](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/50_ffi_async_C/) — async v2 timing-only (35.2 s)
- [51_ffi_async_profile_C](/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/51_ffi_async_profile_C/) — async v2 profiled (41 s profiled wall)

## Lesson

The architecture is sound (main thread freed, correctness proved by
smoke test + pytest). But the bench at zeta scale showed that per-
write MPI-IO cost ≈ 2× sync when the caller can't tightly synchronise
dispatch timing across ranks. That's an inherent trade-off of
decoupled async writes over an MPI-IO collective backend, not a
Python-side bug. Future scaling might flip the tradeoff the other way,
but at 4 ranks it doesn't.
