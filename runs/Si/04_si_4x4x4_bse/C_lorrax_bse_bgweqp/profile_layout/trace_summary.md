# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_layout/xprof/rank_0/plugins/profile/2026_04_27_02_58_37/perfetto_trace.json.gz`
**Duration:** 7.261 s
**GPU streams:** 1 compute, 8 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 347 | 64.89 MiB | 13.29 ms | 5.12 |
| D2H | 2355 | 32.40 KiB | 4.15 ms | 0.01 |
| D2D | 2367 | 1.00 MiB | 3.50 ms | 0.30 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 347 | 13.29 ms | 13.29 ms | 0.000 |
| D2H | 2355 | 4.15 ms | 4.15 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 0.80 s |
| D2H | 15.00 KiB | 0.00 | 1.40 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.14.0` | 800 | 280.82 | 0.36 | 93.75 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `fft.15.0` | 600 | 211.70 | 0.36 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `loop_multiply_fusion` | 205 | 83.30 | 0.42 | 7.8125 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion` | 200 | 70.04 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion.1` | 200 | 68.90 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 200 | 50.58 | 0.26 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.171.0` | 200 | 40.42 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.168.0` | 200 | 29.47 | 0.16 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcsM` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1::Params)` | 200 | 29.04 | 0.15 | 12.5 | `` | `` |
| `all-reduce-start` | 2147 | 19.44 | 0.61 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `all-to-all.2` | 200 | 16.52 | 0.13 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `reduce-scatter.3` | 200 | 13.73 | 0.12 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `all-gather-start` | 200 | 10.76 | 0.09 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `input_reduce_fusion` | 2145 | 6.96 | 0.00 | 50 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `reduce-scatter.1.1` | 200 | 6.29 | 3.46 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `custom-call.172.0` | 200 | 5.43 | 0.03 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kvsN` |
| `all-reduce-start.1` | 200 | 4.64 | 0.05 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/MN` |
| `loop_add_fusion` | 2145 | 4.24 | 0.00 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/add` |
| `loop_subtract_fusion` | 2145 | 4.05 | 0.00 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `all-reduce-start.2` | 200 | 3.77 | 0.06 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcvM` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `reduce-scatter.1.1` | 0.0 % | 3462.2 | `jit(_full_run)/jit(main)` |
| `all-reduce-start` | 0.0 % | 605.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `custom-call.158.0` | 25.0 % | 467.2 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.158.0` | 25.0 % | 303.1 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.158.0` | 25.0 % | 286.1 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.158.0` | 25.0 % | 284.1 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.170.0` | 12.5 % | 257.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 256.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 256.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 256.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.170.0` | 12.5 % | 255.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

