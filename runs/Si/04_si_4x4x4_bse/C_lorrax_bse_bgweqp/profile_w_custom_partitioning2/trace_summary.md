# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_w_custom_partitioning2/xprof/rank_0/plugins/profile/2026_04_27_10_38_17/perfetto_trace.json.gz`
**Duration:** 7.247 s
**GPU streams:** 1 compute, 8 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 347 | 64.89 MiB | 10.75 ms | 6.33 |
| D2H | 2355 | 32.40 KiB | 4.14 ms | 0.01 |
| D2D | 2367 | 1.00 MiB | 3.55 ms | 0.30 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 347 | 10.75 ms | 10.75 ms | 0.000 |
| D2H | 2355 | 4.14 ms | 4.14 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 57.17 MiB | 0.60 | 0.90 s |
| D2H | 15.00 KiB | 0.00 | 1.30 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.3.0` | 600 | 211.77 | 0.36 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `fft.2.0` | 600 | 211.66 | 0.36 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `all-reduce-start` | 2147 | 196.74 | 178.81 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `reduce-scatter.3` | 200 | 100.75 | 87.42 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `loop_multiply_fusion` | 205 | 82.94 | 0.42 | 7.8125 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion` | 200 | 69.47 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion.1` | 200 | 68.66 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `custom-call.121.0` | 200 | 50.80 | 0.26 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `all-to-all.2` | 200 | 42.38 | 31.79 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `custom-call.122.0` | 200 | 40.41 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.119.0` | 200 | 29.85 | 0.16 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcsM` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1::Params)` | 200 | 29.42 | 0.16 | 12.5 | `` | `` |
| `all-gather-start` | 200 | 14.06 | 0.15 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `input_reduce_fusion` | 2145 | 7.31 | 0.00 | 50 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `all-reduce-start.1` | 200 | 7.09 | 0.11 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/MN` |
| `custom-call.123.0` | 200 | 5.33 | 0.03 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kvsN` |
| `loop_subtract_fusion` | 2145 | 4.86 | 0.00 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `reduce-scatter.1.1` | 200 | 4.58 | 0.09 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `loop_add_fusion` | 2145 | 4.15 | 0.00 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/add` |
| `wrapped_compare` | 2345 | 3.73 | 0.00 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/cond/lt` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 178811.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `reduce-scatter.3` | 0.0 % | 87416.3 | `jit(_full_run)/jit(main)` |
| `all-to-all.2` | 0.0 % | 31788.7 | `jit(_full_run)/jit(main)` |
| `custom-call.109.0` | 25.0 % | 466.7 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.109.0` | 25.0 % | 287.4 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.109.0` | 25.0 % | 281.5 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.109.0` | 25.0 % | 279.0 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.121.0` | 12.5 % | 264.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 258.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 256.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 256.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 256.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 256.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 256.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 256.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

