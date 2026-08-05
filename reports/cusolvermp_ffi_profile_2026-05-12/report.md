# cuSOLVERMp FFI profiling harness and first traces (2026-05-12)

## Summary

Built a dedicated 4-GPU profiling harness at `sources/lorrax_D/src/ffi/cusolvermp/profile_batched.py` and added NVTX ranges to the cuSOLVERMp batched `potrf`/`potrs` FFI handlers. The benchmark runs under the same `lxrun`/Shifter/JAX-distributed path as GWJAX and uses the MoS2 3x3-like default shape: `nq=9`, `n=640`, `mrhs=640`, `complex128`, `2x2` GPU mesh.

Main result: the expensive part is not descriptor creation or workspace-size queries on this build. It is the serial C++ `for q` loop over cuSOLVERMp calls. At this shape, each extra `potrf` slice costs about 0.96 ms and each `potrs` slice about 2.0 ms. Descriptor setup plus buffer-size query plus destroy is single-digit microseconds per FFI call, so descriptor caching is not the first high-impact optimization for current Perlmutter `lorrax_D`.

## Artifacts

Allocation:

```bash
salloc --nodes=1 --qos=interactive --time=04:00:00 --constraint=gpu --gpus=4 --account=m2651
# job 52887312 on nid001057
```

Run directory:

```text
runs/FFI/cusolvermp_batched_profile_2026-05-12/
  logs/
  nsys/
  reports/
```

Key source edits:

```text
sources/lorrax_D/src/ffi/cusolvermp/profile_batched.py
sources/lorrax_D/src/ffi/cusolvermp/cpp/batched_potrf_ffi.cc
sources/lorrax_D/src/ffi/cusolvermp/cpp/batched_potrs_ffi.cc
sources/lorrax_D/src/ffi/common/cpp/CMakeLists.txt
```

Key Nsight stats:

```text
runs/FFI/cusolvermp_batched_profile_2026-05-12/reports/nsys_potrf_nvtx_sync_rank0_stats.txt
runs/FFI/cusolvermp_batched_profile_2026-05-12/reports/nsys_potrf_potrs_nvtx_nosync_rank0_stats.txt
```

## Timing Table

All runs used 4 MPI ranks / 4 A100 GPUs through `LORRAX_NGPU=4 lxrun`.

| Case | Command notes | Median | Mean | Notes |
|---|---:|---:|---:|---|
| `potrf`, baseline prebuilt inputs | `--mode potrf --nq 9 -n 640 --sync-each-iter` | 9.413 ms | 9.427 ms | Before NVTX rebuild; tight steady state. |
| `potrf`, NVTX prebuilt inputs | same shape | 9.480 ms | 10.368 ms | Same median class; a few 12 ms jitter outliers. |
| `potrf_potrs`, prebuilt inputs | `--mode potrf_potrs --nq 9 -n 640 --mrhs 640` | 27.976 ms | 34.580 ms | One 81 ms outlier; steady samples are ~28 ms. |
| `potrf_potrs`, Nsight no per-iter barrier | `--cuda-profiler-api`, no `--sync-each-iter` | 29.462 ms | 29.793 ms | Cleanest internal attribution trace. |

Nq sweep for `potrf`, `n=640`, `complex128`, after NVTX build:

| `nq` | Median | Mean |
|---:|---:|---:|
| 1 | 1.951 ms | 1.958 ms |
| 3 | 3.821 ms | 3.837 ms |
| 9 | 9.447 ms | 9.447 ms |
| 18 | 17.862 ms | 17.839 ms |

The near-linear `nq` scaling is the smoking gun: this FFI is behaving like a serial queue of independent distributed solves, not like a batched library call.

## Nsight Findings

From `nsys_potrf_nvtx_sync_rank0_stats.txt`, rank 0:

| NVTX range | Instances | Median / avg |
|---|---:|---:|
| XLA `jit__potrf` | 8 | 8.920 ms / 9.191 ms |
| FFI custom call | 8 | 8.857 ms / 9.131 ms |
| `potrf.Potrf[q=0]` | 8 | 1.096 ms / 1.378 ms |
| `potrf.Potrf[q=1..8]` | 8 each | ~0.958-0.973 ms |
| `potrf.cross_stream_wait` | 16 | 5.8 us median / 7.9 us avg |
| `potrf.BufferSize` | 8 | 2.5 us median / 3.4 us avg |
| `potrf.CreateMatrixDesc` | 8 | 1.7 us median / 2.3 us avg |
| `potrf.DestroyMatrixDesc` | 8 | 0.7 us median / 0.7 us avg |

From the no-per-iteration-barrier combined trace, rank 0:

| NVTX range | Instances | Median / avg |
|---|---:|---:|
| XLA `jit__potrf` | 6 | 8.966 ms / 8.949 ms |
| XLA `jit__potrs` | 6 | 18.418 ms / 18.494 ms |
| `potrf.Potrf[q=0]` | 6 | 1.115 ms / 1.099 ms |
| `potrf.Potrf[q=1..8]` | 6 each | ~0.956-0.981 ms |
| `potrs.Potrs[q=0]` | 6 | 1.474 ms / 1.558 ms |
| `potrs.Potrs[q=1..8]` | 6 each | ~1.986-2.060 ms |
| `potrs.cross_stream_wait` | 12 | 4.4 us median / 5.2 us avg |
| `potrs.BufferSize` | 6 | 0.8 us median / 1.0 us avg |
| `potrs.CreateMatrixDesc[A+B]` | 12 | ~0.5-1.4 us medians |

CUDA kernel summaries show many NCCL broadcast/allreduce/sendrecv kernels plus cuSOLVER/cuBLAS kernels inside each per-q call. The per-q library call time, not the host descriptor setup, is where the wall time lives.

## NCCL Env Tuning

Quick `potrf` checks with `--sync-each-iter`:

| Env | Median | Result |
|---|---:|---|
| Default | ~9.45 ms | Best observed steady state. |
| `NCCL_PROTO=Simple NCCL_ALGO=Ring` | 9.811 ms | Slower. |
| `NCCL_PROTO=LL128 NCCL_ALGO=Ring` | 9.983 ms | Slower and more jitter. |
| `NCCL_MAX_NCHANNELS=1` | 11.032 ms | Clearly slower. |

Conclusion: leave NCCL defaults alone for this shape unless a production GWJAX trace shows a different message regime.

## Transferable Next Steps

1. Use this harness as the first reproduction target before changing GWJAX. It is much cheaper than a full COHSEX run, uses the same wrappers and FFI handlers, and now supports `nsys --capture-range=cudaProfilerApi`.
2. Deprioritize descriptor/workspace caching for `potrf`/`potrs` unless another platform shows larger setup costs. It may still be tidy, but the expected win here is below 0.1% for `nq=9`.
3. Treat CUDA graph capture cautiously. A graph over the q loop would capture concrete device pointers from donated input/output buffers; real GWJAX buffers change across calls, so replay would need safe graph-exec updates or allocator pointer stability that we should not assume.
4. Investigate batching across independent q slices using subcommunicators or an alternate layout. The existing SLATE batched path is the obvious comparison point because it already maps batch work differently; a cuSOLVERMp row/column subgrid variant may be worth reviving if resharding can be avoided or amortized.
5. Add equivalent NVTX ranges to LU/cuBLASMp handlers before optimizing them. The result here is a good warning that plausible host-side overhead hypotheses can be wrong by two orders of magnitude.

## Verification

Build:

```bash
module load lorrax_D
export SLURM_JOBID=52887312
LORRAX_NGPU=1 LORRAX_NTASKS=1 src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
```

Result: success; rebuilt `src/ffi/common/cpp/build/liblorrax_ffi.so`.

Profile harness smoke:

```bash
LORRAX_NGPU=4 lxrun python3 -u -m ffi.cusolvermp.profile_batched \
  --mode potrf_potrs --nq 2 -n 64 --mrhs 64 --iters 1 --warmup 1
```

Result: success; `iter 000: 1.540 ms`.

Python compile:

```bash
python3 -m py_compile src/ffi/cusolvermp/profile_batched.py
```

Result: success.

Full pytest:

```bash
uv run python -m pytest -q
```

Result: failed after 9:07 with `203 passed, 20 skipped, 6 failed, 4 errors`. The failures were outside the modified FFI/profiling files:

```text
tests/test_v_q_bispinor_orchestrator.py: make_v_munu_chunked_kernel missing mesh_xy
tests/test_v_q_per_q_g_chunked.py: make_v_munu_chunked_kernel missing mesh_xy
tests/test_kmeans_sharded.py: small label mismatches vs naive reference
tests/test_gw_jax_regression.py: CUDA_ERROR_OUT_OF_MEMORY initializing backend
```

These look unrelated to the cuSOLVERMp FFI profiling edits, but the tree is not full-suite-clean at this checkpoint.
