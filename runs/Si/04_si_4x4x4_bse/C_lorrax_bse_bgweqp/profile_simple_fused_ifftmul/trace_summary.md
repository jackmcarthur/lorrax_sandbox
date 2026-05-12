# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_simple_fused_ifftmul/xprof/rank_0/plugins/profile/2026_04_27_11_21_56/perfetto_trace.json.gz`
**Duration:** 7.909 s
**GPU streams:** 1 compute, 8 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 347 | 64.89 MiB | 9.54 ms | 7.13 |
| D2H | 2355 | 32.64 KiB | 4.24 ms | 0.01 |
| D2D | 2367 | 1.23 MiB | 3.81 ms | 0.34 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 347 | 9.54 ms | 9.54 ms | 0.000 |
| D2H | 2355 | 4.24 ms | 4.24 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 1.20 s |
| D2H | 15.00 KiB | 0.00 | 1.60 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.16.0` | 800 | 280.65 | 0.36 | 93.75 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `fft.17.0` | 600 | 211.78 | 0.36 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/jit(shmap_b` |
| `all-reduce-start` | 2147 | 184.18 | 165.30 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `loop_multiply_fusion` | 205 | 83.02 | 0.42 | 31.25 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion` | 200 | 69.46 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)` |
| `input_transpose_fusion.1` | 200 | 68.67 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 200 | 50.58 | 0.26 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.184.0` | 200 | 40.41 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.181.0` | 200 | 29.63 | 0.16 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kcsM` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tt_align1::Params)` | 200 | 29.04 | 0.15 | 12.5 | `` | `` |
| `reduce-scatter.3` | 200 | 16.23 | 2.85 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `all-to-all.2` | 200 | 15.96 | 0.15 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |
| `all-gather-start` | 200 | 11.22 | 0.11 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `input_reduce_fusion` | 2145 | 7.04 | 0.00 | 50 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `custom-call.185.0` | 200 | 5.36 | 0.03 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kvsN` |
| `all-reduce-start.1` | 200 | 5.13 | 0.06 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/MN` |
| `loop_subtract_fusion` | 2145 | 4.76 | 0.00 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body` |
| `loop_add_fusion` | 2145 | 4.14 | 0.00 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/add` |
| `wrapped_compare` | 2345 | 3.86 | 0.00 | 1.5625 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/cond/lt` |
| `custom-call.182.0` | 200 | 3.43 | 0.04 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 165298.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `reduce-scatter.3` | 0.0 % | 2852.7 | `jit(_full_run)/jit(main)` |
| `custom-call.171.0` | 25.0 % | 466.5 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.171.0` | 25.0 % | 288.4 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.171.0` | 25.0 % | 285.6 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.171.0` | 25.0 % | 274.4 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.183.0` | 12.5 % | 257.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 256.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 256.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 256.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 256.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 256.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 255.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 255.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 255.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 255.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 255.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 255.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 255.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |
| `custom-call.183.0` | 12.5 % | 255.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec)/kctM` |

