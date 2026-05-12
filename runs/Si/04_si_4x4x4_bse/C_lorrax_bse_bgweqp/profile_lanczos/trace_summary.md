# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_lanczos/xprof/rank_0/plugins/profile/2026_04_28_16_23_57/perfetto_trace.json.gz`
**Duration:** 7.371 s
**GPU streams:** 1 compute, 8 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 347 | 68.64 MiB | 19.18 ms | 3.75 |
| D2H | 1900 | 32.19 KiB | 3.31 ms | 0.01 |
| D2D | 1912 | 2.17 MiB | 3.19 ms | 0.71 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 347 | 19.18 ms | 19.18 ms | 0.000 |
| D2H | 1900 | 3.31 ms | 3.31 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 1.20 s |
| D2H | 15.00 KiB | 0.00 | 1.80 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.16.0` | 240 | 84.19 | 0.36 | 93.75 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `fft.17.0` | 180 | 63.54 | 0.36 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `loop_multiply_fusion.1` | 60 | 24.89 | 0.42 | 75 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion.1` | 60 | 20.84 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion.2` | 60 | 20.61 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `reduce-scatter.3` | 60 | 16.79 | 12.05 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `custom-call.173.0` | 60 | 15.80 | 0.27 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `all-reduce-start` | 1832 | 15.36 | 1.12 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `custom-call.174.0` | 60 | 12.38 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `all-to-all.2` | 60 | 10.35 | 0.21 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1::Params)` | 60 | 9.26 | 0.16 | 12.5 | `` | `` |
| `custom-call.171.0` | 60 | 8.57 | 0.15 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcsM` |
| `all-gather-start` | 60 | 4.93 | 0.12 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `input_reduce_fusion` | 1830 | 4.87 | 0.00 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `loop_multiply_fusion` | 1835 | 4.26 | 0.00 | 31.25 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `loop_subtract_fusion` | 1830 | 3.97 | 0.00 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `loop_add_fusion` | 1830 | 3.44 | 0.00 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/add` |
| `wrapped_compare` | 1890 | 3.27 | 0.00 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/cond/lt` |
| `loop_complex_transpose_fusion` | 60 | 2.01 | 0.05 | 75 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body` |
| `all-reduce-start.1` | 60 | 1.99 | 0.05 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/MN` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `reduce-scatter.3` | 0.0 % | 12053.0 | `jit(_full_run)/jit(main)` |
| `all-reduce-start` | 0.0 % | 1121.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `custom-call.173.0` | 12.5 % | 266.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 266.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 266.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 266.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 265.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 265.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 265.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 265.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 265.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 265.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 265.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 264.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 264.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 264.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 264.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 264.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 263.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.173.0` | 12.5 % | 263.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

