# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_w_custom_partitioning1/xprof/rank_0/plugins/profile/2026_04_27_10_37_57/perfetto_trace.json.gz`
**Duration:** 7.445 s
**GPU streams:** 1 compute, 8 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 347 | 64.89 MiB | 20.60 ms | 3.30 |
| D2H | 2355 | 32.40 KiB | 4.15 ms | 0.01 |
| D2D | 2367 | 1.00 MiB | 3.55 ms | 0.30 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 347 | 20.60 ms | 20.60 ms | 0.000 |
| D2H | 2355 | 4.15 ms | 4.15 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 0.70 s |
| D2H | 15.00 KiB | 0.00 | 1.20 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.3.0` | 600 | 211.79 | 0.36 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `fft.2.0` | 600 | 211.65 | 0.36 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `all-reduce-start` | 2147 | 191.00 | 168.25 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `loop_multiply_fusion` | 205 | 82.89 | 0.42 | 7.8125 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion` | 200 | 69.47 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion.1` | 200 | 68.67 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `custom-call.121.0` | 200 | 50.89 | 0.26 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.122.0` | 200 | 40.41 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.119.0` | 200 | 29.68 | 0.18 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcsM` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1::Params)` | 200 | 29.46 | 0.15 | 12.5 | `` | `` |
| `all-gather-start` | 200 | 14.04 | 0.10 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `reduce-scatter.3` | 200 | 13.38 | 0.21 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `all-to-all.2` | 200 | 10.88 | 0.15 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `all-reduce-start.1` | 200 | 7.73 | 0.11 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/MN` |
| `input_reduce_fusion` | 2145 | 7.31 | 0.00 | 50 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `custom-call.123.0` | 200 | 5.28 | 0.03 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kvsN` |
| `reduce-scatter.1.1` | 200 | 4.88 | 0.50 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `loop_subtract_fusion` | 2145 | 4.85 | 0.00 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `loop_add_fusion` | 2145 | 4.08 | 0.00 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/add` |
| `all-gather-start.2` | 200 | 3.91 | 0.80 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 168251.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.2` | 0.0 % | 796.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `reduce-scatter.1.1` | 0.0 % | 501.6 | `jit(_full_run)/jit(main)` |
| `custom-call.109.0` | 25.0 % | 467.1 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.109.0` | 25.0 % | 285.0 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.109.0` | 25.0 % | 284.9 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.109.0` | 25.0 % | 277.7 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.121.0` | 12.5 % | 263.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 259.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 259.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 259.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 258.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 258.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 258.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.121.0` | 12.5 % | 257.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

