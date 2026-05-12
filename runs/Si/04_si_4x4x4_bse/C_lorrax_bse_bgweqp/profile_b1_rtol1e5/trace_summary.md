# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_b1_rtol1e5/xprof/rank_0/plugins/profile/2026_04_27_02_28_59/perfetto_trace.json.gz`
**Duration:** 8.854 s
**GPU streams:** 3 compute, 4 H2D, 3 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 352 | 64.89 MiB | 19.01 ms | 3.58 |
| D2H | 1949 | 32.64 KiB | 3.60 ms | 0.01 |
| D2D | 11167 | 397.54 MiB | 18.72 ms | 22.27 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 352 | 19.01 ms | 19.01 ms | 0.000 |
| D2H | 1949 | 3.60 ms | 3.60 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 0.90 s |
| D2H | 15.00 KiB | 0.00 | 1.50 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `cublas-batch-gemm.5.0` | 280 | 108.64 | 0.40 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `loop_transpose_fusion.7` | 140 | 98.22 | 0.71 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `custom-call.72.0` | 2890 | 82.87 | 0.33 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/cond/branch_1_fun/jit(ei` |
| `loop_multiply_fusion` | 145 | 58.00 | 0.42 | 7.8125 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `loop_transpose_fusion.4` | 140 | 54.42 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.5` | 140 | 54.08 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.2` | 140 | 53.93 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.3` | 140 | 53.89 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.6` | 140 | 53.63 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_complex_fusion.1` | 140 | 52.71 | 0.38 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.4.0` | 140 | 48.70 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.1.0` | 140 | 48.59 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.5.0` | 140 | 48.57 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.3.0` | 140 | 48.51 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `input_transpose_fusion_3` | 140 | 48.32 | 0.35 | 56.25 | `` | `` |
| `fft.2.0` | 140 | 48.15 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.0.0` | 140 | 47.73 | 0.34 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_broadcast_fusion_3` | 140 | 32.37 | 0.23 | 100 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1::Params)` | 140 | 28.25 | 0.21 | 12.5 | `` | `` |
| `collective-permute-start.3` | 280 | 18.86 | 0.17 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `reduce-scatter.4.1` | 0.0 % | 11454.7 | `jit(_full_run)/jit(main)/while` |
| `all-reduce-start.3` | 0.0 % | 6221.6 | `jit(_full_run)/jit(main)/while/body/scatter` |
| `all-reduce-start.3` | 0.0 % | 2281.1 | `jit(_full_run)/jit(main)/while/body/scatter` |
| `collective-permute-start.2` | 0.0 % | 1376.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `all-reduce-start.3` | 0.0 % | 1243.8 | `jit(_full_run)/jit(main)/while/body/scatter` |
| `all-reduce-start.3` | 0.0 % | 920.4 | `jit(_full_run)/jit(main)/while/body/scatter` |
| `all-reduce-start` | 0.0 % | 745.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `collective-permute-start.2` | 0.0 % | 591.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `collective-permute-start.2` | 0.0 % | 541.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `collective-permute-start.2` | 0.0 % | 520.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `all-reduce-start.3` | 0.0 % | 484.7 | `jit(_full_run)/jit(main)/while/body/scatter` |
| `reduce-scatter.6` | 0.0 % | 448.6 | `jit(_full_run)/jit(main)/while` |
| `cublas-batch-gemm.5.0` | 12.5 % | 396.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 396.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 396.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 396.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 396.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 396.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 396.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 396.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |

