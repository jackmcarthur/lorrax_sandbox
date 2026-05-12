# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_simple_fused/xprof/rank_0/plugins/profile/2026_04_27_02_52_29/perfetto_trace.json.gz`
**Duration:** 7.571 s
**GPU streams:** 1 compute, 8 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 347 | 64.89 MiB | 20.43 ms | 3.33 |
| D2H | 2355 | 32.40 KiB | 4.15 ms | 0.01 |
| D2D | 2367 | 1.00 MiB | 3.57 ms | 0.29 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 347 | 20.43 ms | 20.43 ms | 0.000 |
| D2H | 2355 | 4.15 ms | 4.15 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 57.14 MiB | 0.60 | 0.70 s |
| D2H | 15.00 KiB | 0.00 | 1.30 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.14.0` | 800 | 280.66 | 0.36 | 93.75 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `fft.15.0` | 600 | 211.75 | 0.36 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `loop_multiply_fusion` | 205 | 82.96 | 0.42 | 7.8125 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion` | 200 | 69.52 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion.1` | 200 | 68.74 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 200 | 50.73 | 0.26 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.161.0` | 200 | 40.48 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.158.0` | 200 | 29.31 | 0.15 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcsM` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1::Params)` | 200 | 29.22 | 0.15 | 12.5 | `` | `` |
| `all-reduce-start` | 2147 | 19.31 | 0.08 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `reduce-scatter.3` | 200 | 14.12 | 1.80 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `all-to-all.2` | 200 | 13.58 | 0.14 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `all-gather-start` | 200 | 11.95 | 0.10 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `input_reduce_fusion` | 2145 | 7.00 | 0.00 | 50 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `all-reduce-start.1` | 200 | 5.84 | 0.07 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/MN` |
| `custom-call.162.0` | 200 | 5.41 | 0.03 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kvsN` |
| `loop_subtract_fusion` | 2145 | 4.09 | 0.00 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `loop_add_fusion` | 2145 | 4.05 | 0.01 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/add` |
| `custom-call.159.0` | 200 | 4.03 | 0.04 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `reduce-scatter.1.1` | 200 | 3.91 | 0.07 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `reduce-scatter.3` | 0.0 % | 1797.9 | `jit(_full_run)/jit(main)` |
| `custom-call.148.0` | 25.0 % | 466.8 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.148.0` | 25.0 % | 299.2 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.148.0` | 25.0 % | 291.5 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.148.0` | 25.0 % | 289.1 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.160.0` | 12.5 % | 258.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 257.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 257.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 257.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

