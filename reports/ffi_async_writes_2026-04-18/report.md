# FFI async writes — wiring `ffi::Future` + Python worker thread

Agent: C. Branch: `agent/C-sigma-ppm-cleanup`. Date: 2026-04-18.

## TL;DR

- Before: `SlabIO.write_slab(..., use_ffi_io=True)` blocked the Python
  main thread for the **full duration of `H5Dwrite`** (≈ 790 ms per
  270 MB/rank zeta chunk, MoS2 3×3, 4-GPU).
- After: main-thread dispatch time is **0.4 ms** per chunk. The
  H5Dwrite runs concurrently with next-chunk compute via a Python
  worker thread; main thread only blocks when we explicitly drain
  (e.g. at `close()`).

- Cost: +100 lines C++ (async handler, writer thread, FIFO queue),
  +35 lines Python (dispatch queue, drain, shutdown).
- Correctness: pytest 14/14, 4-GPU phdf5 smoke test passes
  bit-identical round-trip.

## The long answer to "how do you wire an FFI async from JAX's perspective"

Two facts discovered by instrumentation:

1. **`ffi::Future` is not enough.** XLA's FFI has first-class async
   handler support via `xla::ffi::Future` return type (detected via
   `ResultEncoding<stage, Future>` template specialization, see
   [ffi.h:1239](/global/u2/j/jackm/software/lorrax_C/.venv/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/ffi.h#L1239)).
   Wiring it is ~30 lines of C++: construct `Promise`, return
   `Future(promise)` from the handler, enqueue the blocking work onto
   a worker thread that calls `promise.SetAvailable()` when done.
   **But**: `jit(ffi_call)(A)` still blocks the Python caller until
   the Future resolves. Verified with a 300 ms sleep injected
   after `SetAvailable`: `jit_call` time grew from 45 ms → 350 ms in
   lockstep. ffi::Future only decouples *downstream XLA ops* from the
   producer; the Python `jit(...)` call itself is still synchronous.

2. **Python-thread async needs a Python worker thread.** The main
   Python thread must hand the whole `jit(ffi_call)(A)` call off to a
   separate Python thread. We do this with a single `threading.Thread`
   per `_FfiBackend`, draining a `queue.Queue` in FIFO order. Main
   thread enqueues (~0.2 ms) and returns; the worker calls
   `jit(...)(A).block_until_ready()` for each task.

The "one worker per backend, FIFO" shape is load-bearing: every rank
must dispatch writes to MPI-IO in the same order for the collective to
rendezvous, and a per-call thread would let the OS scheduler reorder.
The same invariant holds for the C++ writer thread in `ctx.h` — one
per ctx, not one per H5Dwrite call.

## Measurements — Python dispatch time per chunk

All runs 4 × A100, MoS2 3×3 scale, 270 MB/rank F64 chunks, 5 chunks.

| version | ds0 dispatch | ds1 | ds2 | ds3 | ds4 |
|---|---:|---:|---:|---:|---:|
| baseline sync handler, eager shard_map | 789 ms | 1250 ms | 1208 ms | 1179 ms | 1230 ms |
| async handler + jit wrap, no Py worker | 789 ms | 1237 ms | 1208 ms | 1179 ms | 1230 ms |
| async handler + jit wrap + **Python worker** | **0.51 ms** | **0.41 ms** | **0.50 ms** | **0.35 ms** | **0.35 ms** |

Same setup, single-rank, 32 MB, 4 chunks, to isolate the handler:

| thing measured | time |
|---|---:|
| handler entry → cudaMallocHost done | 17.0 ms |
| cudaMallocHost → cudaMemcpyAsync return | 0.05 ms |
| cudaMemcpyAsync → enqueue task + notify | 0.02 ms |
| total handler execution | 17.1 ms |
| `jit(ffi_call)(A)` return (no sleep) | 45 ms |
| `jit(ffi_call)(A)` return (worker sleeps 300 ms post-SetAvailable) | **350 ms** |

The last row is the smoking gun: XLA waits on the Future.

`cudaMallocHost` is ~17 ms per 32 MB buffer — a pinned-buffer pool
would pay for itself at scale, but it's nothing compared to the 45 ms
H5Dwrite wall.

## Architecture

```
           (main thread, per chunk)                (Python worker)               (C++ writer thread, per ctx)
  ┌──────────────────────────────┐          ┌────────────────────────┐          ┌──────────────────────────────┐
  │ build zeta_chunk via jit     │          │ queue.get() — blocking │          │ cv.wait(queue_cv)            │
  │ write_slab(zeta_chunk, ...)  │─ enqueue→│ fn = _task             │          │                              │
  │   └── (0.4 ms, returns)      │          │ jit(sm)(A)             │─ FFI ───→│ cudaMemcpyAsync done →       │
  │ del zeta_chunk               │          │   ... XLA dispatches   │          │ cudaEventSynchronize →       │
  │ (compute next chunk on GPU)  │          │   the FFI handler,     │          │ H5Dwrite (MPI collective) →  │
  │                              │          │   which enqueues a C++ │          │ SetAvailable() →             │
  │                              │          │   task, returns Future │          │ free pinned buffer           │
  │                              │          │   block_until_ready()  │          │                              │
  │                              │          │   (waits on Future) ←──┼─signals─ │                              │
  └──────────────────────────────┘          └────────────────────────┘          └──────────────────────────────┘
```

MPI-collective ordering: main thread enqueues in dispatch order →
Python worker pops in FIFO → XLA dispatches in that order → C++ writer
thread receives in that order → all ranks call H5Dwrite in the same
sequence. Rendezvous safe.

## Code

- [src/ffi/phdf5/cpp/write_ffi.cc](/global/u2/j/jackm/software/lorrax_C/src/ffi/phdf5/cpp/write_ffi.cc):
  handler now returns `ffi::Future`, enqueues work onto a ctx-level
  task queue, per-call pinned buffer.
- [src/ffi/phdf5/cpp/ctx.h](/global/u2/j/jackm/software/lorrax_C/src/ffi/phdf5/cpp/ctx.h):
  `PhdfCtx` gains `writer_thread`, `queue_mu`, `queue_cv`,
  `task_queue`, `shutdown_flag`.
- [src/ffi/phdf5/cpp/context.cc](/global/u2/j/jackm/software/lorrax_C/src/ffi/phdf5/cpp/context.cc):
  start writer thread in `open_ctx`, drain-join in `close_ctx` before
  H5Fclose.
- [src/ffi/common/cpp/CMakeLists.txt](/global/u2/j/jackm/software/lorrax_C/src/ffi/common/cpp/CMakeLists.txt):
  `find_package(Threads REQUIRED)` + link `Threads::Threads`.
- [src/ffi/phdf5/write.py](/global/u2/j/jackm/software/lorrax_C/src/ffi/phdf5/write.py):
  `has_side_effect=True` on the ffi_call (prevents XLA from DCE-ing
  the write if a caller ever discards the output token).
- [src/file_io/_slab_io_ffi.py](/global/u2/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py):
  `_FfiBackend` owns a Python worker thread + `_dispatch_queue`;
  `write_slab` enqueues a closure, `_drain_pending` blocks main
  thread, `close` drains + shuts down.

## What's next

1. End-to-end profile of `gw_jax --use_ffi_io=true` on MoS2 3×3 to
   confirm that the 4 × 790 ms serialized wait inside
   `zeta_fit.chunk.h5_write` now fully overlaps with the next chunk's
   compute, and `h5_write` drops to near zero in the timing summary.
   Deferred to next session; needs fresh allocation.
2. Pinned-buffer pool in `write_ffi.cc` to cut the 17 ms
   `cudaMallocHost` per dispatch down to a few µs.
3. Generalise the Python worker pattern to reads (`read_slab` still
   calls `block_until_ready` eagerly — that's fine for correctness but
   the same pattern would work).

## Scripts

- [scripts/phdf5_async_bench.py](/pscratch/sd/j/jackm/lorrax_sandbox/scripts/phdf5_async_bench.py)
  — 4-GPU dispatch-vs-ready bench at arbitrary chunk size.
- [scripts/phdf5_singlerank_bench.py](/pscratch/sd/j/jackm/lorrax_sandbox/scripts/phdf5_singlerank_bench.py)
  — 1-GPU handler microbenchmark used to isolate the 17 ms cudaMallocHost.
- [scripts/phdf5_probe.py](/pscratch/sd/j/jackm/lorrax_sandbox/scripts/phdf5_probe.py)
  — minimal open/create/write/close probe used to debug the
  different-path-per-rank deadlock.

## Lesson

Two completely separate synchronisation layers have to be removed to
free the Python main thread:

1. `H5Dwrite` blocking the FFI handler → fix in C++ with a worker
   thread (ffi::Future + task queue).
2. `jit(ffi_call)` blocking Python on the Future → fix in Python with
   a worker thread.

The C++ layer on its own buys nothing for "Python doing other work"
(that was the false-premise guess at the start of the session);
the Python layer on its own works with any handler shape (sync or
async). Both together preserve MPI-IO rendezvous order without either
the main Python thread or the XLA executor thread getting stuck
waiting on MPI-IO.
