# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_simple/xprof/rank_0/plugins/profile/2026_04_27_02_45_53/perfetto_trace.json.gz`
**Duration:** 8.074 s
**GPU streams:** 1 compute, 8 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 347 | 64.89 MiB | 18.80 ms | 3.62 |
| D2H | 2355 | 32.40 KiB | 4.22 ms | 0.01 |
| D2D | 2367 | 1.00 MiB | 3.55 ms | 0.30 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 347 | 18.80 ms | 18.80 ms | 0.000 |
| D2H | 2355 | 4.22 ms | 4.22 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 0.90 s |
| D2H | 15.00 KiB | 0.00 | 1.40 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `loop_transpose_fusion.7` | 200 | 93.53 | 0.47 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `loop_multiply_fusion` | 205 | 83.01 | 0.42 | 7.8125 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `loop_transpose_fusion.4` | 200 | 78.09 | 0.40 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `loop_transpose_fusion.3` | 200 | 77.47 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `loop_transpose_fusion.2` | 200 | 77.25 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `loop_transpose_fusion.5` | 200 | 76.96 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `loop_transpose_fusion.6` | 200 | 76.65 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `fft.2.0` | 200 | 69.55 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `fft.3.0` | 200 | 69.37 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `fft.5.0` | 200 | 69.37 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `fft.4.0` | 200 | 69.35 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `input_transpose_fusion` | 200 | 69.19 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `fft.1.0` | 200 | 68.87 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `fft.0.0` | 200 | 68.57 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/custom_part` |
| `custom-call.140.0` | 200 | 51.91 | 0.26 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.141.0` | 200 | 40.51 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.136.0` | 200 | 33.34 | 0.17 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcsN` |
| `custom-call.138.0` | 200 | 29.61 | 0.16 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcsM` |
| `all-reduce-start` | 2147 | 22.22 | 2.29 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `all-reduce-start.1` | 200 | 19.01 | 0.12 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/MN` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start.3` | 0.0 % | 6512.9 | `jit(_full_run)/jit(main)/div` |
| `reduce-scatter.3` | 0.0 % | 2404.6 | `jit(_full_run)/jit(main)` |
| `all-reduce-start` | 0.0 % | 2294.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 503.4 | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `custom-call.128.0` | 25.0 % | 466.8 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `all-gather-start` | 0.0 % | 292.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.128.0` | 25.0 % | 286.6 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.128.0` | 25.0 % | 285.6 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `all-gather-start` | 0.0 % | 279.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.128.0` | 25.0 % | 275.5 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.140.0` | 12.5 % | 264.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.140.0` | 12.5 % | 264.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.140.0` | 12.5 % | 264.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.140.0` | 12.5 % | 264.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.140.0` | 12.5 % | 263.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.140.0` | 12.5 % | 263.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.140.0` | 12.5 % | 263.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.140.0` | 12.5 % | 263.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.140.0` | 12.5 % | 263.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.140.0` | 12.5 % | 263.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

