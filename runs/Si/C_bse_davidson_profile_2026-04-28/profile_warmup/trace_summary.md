# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/C_bse_davidson_profile_2026-04-28/profile_warmup/xprof/rank_0/plugins/profile/2026_04_28_17_10_28/perfetto_trace.json.gz`
**Duration:** 12.734 s
**GPU streams:** 4 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 417 | 67.43 MiB | 19.42 ms | 3.64 |
| D2H | 342 | 3.19 MiB | 0.79 ms | 4.20 |
| D2D | 921 | 2.48 GiB | 5.03 ms | 530.14 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 417 | 19.42 ms | 19.42 ms | 0.000 |
| D2H | 342 | 0.79 ms | 0.79 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 0.80 s |
| D2H | 960.00 KiB | 0.01 | 5.10 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.6.0` | 164 | 587.64 | 3.67 | 93.75 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(_apply_W_from_T)/jit(shmap_b` |
| `fft.7.0` | 123 | 441.56 | 3.67 | 100 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(_apply_W_from_T)/jit(shmap_b` |
| `cublas-batch-gemm.2.0` | 82 | 336.26 | 4.12 | 12.5 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/kctM` |
| `all-reduce-start.1` | 44 | 291.37 | 75.59 | 0 | `jit__ritz_and_residuals` | `` |
| `loop_multiply_fusion` | 130 | 170.63 | 4.17 | 75 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(_apply_W_from_T)` |
| `input_transpose_fusion_1` | 74 | 145.42 | 3.57 | 56.25 | `` | `` |
| `input_transpose_fusion.2` | 74 | 145.02 | 3.55 | 56.25 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while` |
| `loop_broadcast_fusion.3` | 41 | 93.75 | 2.32 | 100 | `jit__matvec_impl` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1::Params)` | 41 | 78.15 | 1.93 | 12.5 | `` | `` |
| `all-reduce-start` | 131 | 68.26 | 13.59 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)` |
| `collective-permute-start.3` | 82 | 31.16 | 0.45 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/pperm` |
| `collective-permute-start.2` | 82 | 24.95 | 19.94 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/pperm` |
| `custom-call.9.0` | 2524 | 22.44 | 0.17 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `collective-permute-start.1` | 82 | 19.96 | 1.47 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/pperm` |
| `reduce-scatter.5` | 41 | 16.46 | 0.72 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)` |
| `all-gather-start` | 2 | 11.83 | 6.66 | 0 | `jit__identity_fn` | `` |
| `loop_add_fusion.3` | 82 | 3.08 | 0.04 | 75 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body` |
| `custom-call.7.0` | 264 | 2.30 | 0.07 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/lu` |
| `custom-call.10.0` | 264 | 2.30 | 0.07 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(solve)/lu` |
| `input_transpose_fusion` | 74 | 2.12 | 0.05 | 56.25 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(_apply_W_from_T)/kctM` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start.1` | 0.0 % | 75587.2 | `` |
| `collective-permute-start.2` | 0.0 % | 19943.1 | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/ppermute` |
| `all-reduce-start.1` | 0.0 % | 18983.5 | `` |
| `all-reduce-start.1` | 0.0 % | 16149.7 | `` |
| `all-reduce-start` | 0.0 % | 13588.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 13377.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start.1` | 0.0 % | 11798.8 | `` |
| `all-reduce-start.1` | 0.0 % | 11673.9 | `` |
| `all-reduce-start.1` | 0.0 % | 11413.5 | `` |
| `all-reduce-start` | 0.0 % | 10321.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start.1` | 0.0 % | 10235.8 | `` |
| `all-reduce-start.1` | 0.0 % | 10170.0 | `` |
| `all-reduce-start.1` | 0.0 % | 9506.8 | `` |
| `all-reduce-start.1` | 0.0 % | 7754.8 | `` |
| `all-reduce-start.1` | 0.0 % | 7296.1 | `` |
| `all-reduce-start.1` | 0.0 % | 7180.8 | `` |
| `all-reduce-start.1` | 0.0 % | 6887.6 | `` |
| `all-reduce-start.1` | 0.0 % | 6843.9 | `` |
| `all-gather-start` | 0.0 % | 6658.7 | `` |
| `all-reduce-start.1` | 0.0 % | 6597.7 | `` |

