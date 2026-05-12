# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_b1_rtol1e4/xprof/rank_0/plugins/profile/2026_04_27_02_27_51/perfetto_trace.json.gz`
**Duration:** 8.310 s
**GPU streams:** 3 compute, 4 H2D, 3 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 352 | 64.89 MiB | 18.62 ms | 3.65 |
| D2H | 923 | 31.31 KiB | 1.70 ms | 0.02 |
| D2D | 5407 | 193.60 MiB | 9.08 ms | 22.37 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 352 | 18.62 ms | 18.62 ms | 0.000 |
| D2H | 923 | 1.70 ms | 1.70 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 1.00 s |
| D2H | 15.00 KiB | 0.00 | 1.60 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `collective-permute-start` | 138 | 61.80 | 59.94 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `cublas-batch-gemm.5.0` | 136 | 52.69 | 0.41 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `loop_transpose_fusion.7` | 68 | 47.76 | 0.70 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `custom-call.72.0` | 1360 | 37.68 | 0.33 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/cond/branch_1_fun/jit(ei` |
| `loop_multiply_fusion` | 73 | 28.19 | 0.42 | 7.8125 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `loop_transpose_fusion.4` | 68 | 26.46 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.5` | 68 | 26.27 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.2` | 68 | 26.22 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.3` | 68 | 26.17 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.6` | 68 | 26.05 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_complex_fusion.1` | 68 | 25.61 | 0.38 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.4.0` | 68 | 23.66 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.1.0` | 68 | 23.59 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.5.0` | 68 | 23.58 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.3.0` | 68 | 23.54 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `input_transpose_fusion_3` | 68 | 23.48 | 0.35 | 56.25 | `` | `` |
| `fft.2.0` | 68 | 23.39 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.0.0` | 68 | 23.17 | 0.34 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_broadcast_fusion_3` | 68 | 15.72 | 0.23 | 100 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1::Params)` | 68 | 13.71 | 0.21 | 12.5 | `` | `` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `collective-permute-start` | 0.0 % | 59939.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `all-reduce-start.3` | 0.0 % | 5413.9 | `jit(_full_run)/jit(main)/while/body/scatter` |
| `collective-permute-start.2` | 0.0 % | 2254.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `collective-permute-start.1` | 0.0 % | 1437.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `all-reduce-start.3` | 0.0 % | 936.1 | `jit(_full_run)/jit(main)/while/body/scatter` |
| `all-reduce-start` | 0.0 % | 851.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `collective-permute-start.2` | 0.0 % | 764.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 412.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 410.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 396.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 394.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 394.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 393.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 393.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 393.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 393.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 392.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 392.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 391.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 391.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |

