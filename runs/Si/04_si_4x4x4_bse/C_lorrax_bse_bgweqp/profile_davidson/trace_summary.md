# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_davidson/xprof/rank_0/plugins/profile/2026_04_29_11_12_47/perfetto_trace.json.gz`
**Duration:** 12.674 s
**GPU streams:** 2 compute, 8 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 386 | 68.09 MiB | 11.37 ms | 6.28 |
| D2H | 148 | 6.62 MiB | 0.60 ms | 11.62 |
| D2D | 379 | 11.67 MiB | 0.74 ms | 16.50 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 386 | 11.37 ms | 11.37 ms | 0.000 |
| D2H | 148 | 0.60 ms | 0.60 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 0.80 s |
| D2H | 2.50 MiB | 0.03 | 6.20 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.8.0` | 2480 | 871.87 | 0.37 | 93.75 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/jit(shmap` |
| `fft.9.0` | 1860 | 657.69 | 0.37 | 100 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/jit(shmap` |
| `loop_multiply_fusion` | 720 | 260.16 | 0.43 | 75 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion` | 654 | 216.72 | 0.36 | 56.25 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion.1` | 628 | 212.66 | 0.35 | 56.25 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.71.0` | 620 | 156.66 | 0.26 | 12.5 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.72.0` | 620 | 126.12 | 0.22 | 12.5 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/kctM` |
| `all-reduce-start` | 721 | 102.55 | 24.81 | 0 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/MN` |
| `custom-call.69.0` | 620 | 90.95 | 0.16 | 12.5 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/kcsM` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1::Params)` | 663 | 90.06 | 0.15 | 12.5 | `` | `` |
| `all-reduce-start.1` | 684 | 78.66 | 8.85 | 0 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/kcvM` |
| `all-gather-start` | 623 | 76.11 | 42.78 | 0 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/kctM` |
| `reduce-scatter.3` | 620 | 46.80 | 0.13 | 0 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while` |
| `all-to-all.2` | 620 | 46.49 | 0.44 | 0 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while` |
| `custom-call.9.0` | 3877 | 25.51 | 0.33 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.73.0` | 620 | 17.99 | 0.03 | 12.5 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/kvsN` |
| `reduce-scatter.1.1` | 620 | 12.74 | 0.31 | 0 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while` |
| `all-gather-start.1` | 621 | 12.49 | 0.32 | 0 | `jit__identity_fn` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/kctM` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1::Params)` | 654 | 9.50 | 0.02 | 12.5 | `` | `` |
| `custom-call.70.0` | 620 | 9.28 | 0.03 | 12.5 | `jit_matvec_scan` | `jit(matvec_scan)/jit(main)/while/body/jit(_matvec)/kctM` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start` | 0.0 % | 42776.7 | `` |
| `all-reduce-start` | 0.0 % | 24805.0 | `jit(_ritz_and_residuals)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 19905.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start.1` | 0.0 % | 8845.7 | `` |
| `all-reduce-start` | 0.0 % | 8720.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start.1` | 0.0 % | 8185.3 | `` |
| `all-reduce-start.1` | 0.0 % | 7382.6 | `` |
| `all-reduce-start` | 0.0 % | 5631.0 | `` |
| `all-reduce-start` | 0.0 % | 5535.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start.1` | 0.0 % | 5516.6 | `` |
| `all-reduce-start` | 0.0 % | 5162.8 | `` |
| `all-reduce-start` | 0.0 % | 3538.1 | `` |
| `all-reduce-start` | 0.0 % | 2435.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 2428.1 | `jit(_impl)/jit(main)/reduce_sum` |
| `all-reduce-start.1` | 0.0 % | 2226.1 | `` |
| `all-reduce-start.1` | 0.0 % | 2112.2 | `` |
| `all-reduce-start.1` | 0.0 % | 1673.1 | `` |
| `all-reduce-start` | 0.0 % | 1523.2 | `jit(_ritz_and_residuals)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 1436.8 | `` |
| `all-reduce-start.1` | 0.0 % | 1361.5 | `` |

