# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_fused_clean/xprof/rank_0/plugins/profile/2026_04_27_10_11_30/perfetto_trace.json.gz`
**Duration:** 8.276 s
**GPU streams:** 1 compute, 8 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 347 | 64.89 MiB | 18.85 ms | 3.61 |
| D2H | 2355 | 32.40 KiB | 4.23 ms | 0.01 |
| D2D | 2367 | 1.00 MiB | 3.58 ms | 0.29 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 347 | 18.85 ms | 18.85 ms | 0.000 |
| D2H | 2355 | 4.23 ms | 4.23 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 1.00 s |
| D2H | 15.00 KiB | 0.00 | 1.50 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.14.0` | 800 | 280.67 | 0.36 | 93.75 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `fft.15.0` | 600 | 211.77 | 0.36 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `all-reduce-start` | 2147 | 200.94 | 180.85 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `loop_multiply_fusion` | 205 | 82.98 | 0.42 | 7.8125 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion` | 200 | 69.48 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion.1` | 200 | 68.69 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 200 | 50.69 | 0.26 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.161.0` | 200 | 40.44 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.158.0` | 200 | 30.04 | 0.16 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcsM` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1::Params)` | 200 | 29.35 | 0.15 | 12.5 | `` | `` |
| `reduce-scatter.3` | 200 | 14.17 | 0.12 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `all-gather-start` | 200 | 12.99 | 0.11 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `all-to-all.2` | 200 | 12.88 | 0.22 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `input_reduce_fusion` | 2145 | 7.34 | 0.00 | 50 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `all-reduce-start.1` | 200 | 6.62 | 0.07 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/MN` |
| `custom-call.162.0` | 200 | 5.37 | 0.03 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kvsN` |
| `loop_subtract_fusion` | 2145 | 4.86 | 0.00 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `reduce-scatter.1.1` | 200 | 4.16 | 0.06 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `loop_add_fusion` | 2145 | 3.98 | 0.00 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/add` |
| `custom-call.159.0` | 200 | 3.73 | 0.04 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 180854.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `custom-call.148.0` | 25.0 % | 467.0 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `all-gather-start.3` | 0.0 % | 451.4 | `jit(_full_run)/jit(main)/div` |
| `all-reduce-start` | 0.0 % | 317.7 | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `custom-call.148.0` | 25.0 % | 284.6 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.148.0` | 25.0 % | 282.5 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.148.0` | 25.0 % | 280.1 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.160.0` | 12.5 % | 257.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 257.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 257.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 257.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.160.0` | 12.5 % | 256.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

