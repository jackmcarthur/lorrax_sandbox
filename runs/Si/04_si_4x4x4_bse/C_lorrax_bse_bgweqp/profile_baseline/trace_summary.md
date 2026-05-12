# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_baseline/xprof/rank_0/plugins/profile/2026_04_27_01_13_58/perfetto_trace.json.gz`
**Duration:** 11.900 s
**GPU streams:** 0 compute, 2 H2D, 2 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 70 | 510.17 MiB | 33.96 ms | 15.75 |
| D2H | 2349 | 2.63 KiB | 4.15 ms | 0.00 |
| D2D | 58 | 478.57 MiB | 0.80 ms | 627.30 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 70 | 33.96 ms | 33.96 ms | 0.000 |
| D2H | 2349 | 4.15 ms | 4.15 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 225.01 MiB | 2.36 | 0.20 s |
| D2H | 318.00 B | 0.00 | 11.80 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.13.0` | 800 | 1115.95 | 1.41 | 93.75 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `fft.14.0` | 600 | 841.50 | 1.41 | 100 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `loop_multiply_fusion` | 211 | 330.45 | 1.66 | 31.25 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion_2` | 200 | 304.42 | 1.53 | 56.25 | `` | `` |
| `input_transpose_fusion_1` | 200 | 298.92 | 1.51 | 56.25 | `` | `` |
| `fft.12.0` | 800 | 282.73 | 0.36 | 93.75 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 800 | 267.37 | 0.98 | 12.5 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1::Params)` | 200 | 160.06 | 0.81 | 12.5 | `` | `` |
| `input_reduce_fusion` | 2150 | 9.60 | 0.01 | 100 | `jit_norm` | `jit(norm)/jit(main)` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1::Params)` | 200 | 5.12 | 0.03 | 12.5 | `` | `` |
| `loop_subtract_fusion` | 2145 | 4.96 | 0.00 | 100 | `jit_scan` | `jit(scan)/jit(main)/while/body/while/body` |
| `loop_complex_transpose_fusion` | 200 | 4.85 | 0.03 | 75 | `` | `` |
| `wrapped_compare` | 2345 | 3.68 | 0.00 | 1.5625 | `jit_scan` | `jit(scan)/jit(main)/while/body/while/cond/lt` |
| `loop_add_fusion` | 2353 | 3.54 | 0.00 | 31.25 | `jit_add` | `jit(add)/jit(main)/add` |
| `loop_transpose_fusion` | 200 | 3.30 | 0.02 | 100 | `` | `` |
| `loop_multiply_fusion_1` | 200 | 2.59 | 0.01 | 100 | `` | `` |
| `wrapped_transpose_2` | 200 | 2.15 | 0.01 | 100 | `` | `` |
| `custom-call.12.0` | 69 | 2.07 | 0.47 | 100 | `jit_eigh` | `jit(eigh)/jit(main)/eigh` |
| `loop_multiply_fusion_2` | 200 | 1.58 | 0.01 | 56.25 | `` | `` |
| `wrapped_transpose_1` | 200 | 1.48 | 0.01 | 100 | `` | `` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `` | 12.5 % | 978.1 | `` |
| `` | 12.5 % | 977.9 | `` |
| `` | 12.5 % | 977.4 | `` |
| `` | 12.5 % | 977.2 | `` |
| `` | 12.5 % | 977.1 | `` |
| `` | 12.5 % | 977.1 | `` |
| `` | 12.5 % | 976.7 | `` |
| `` | 12.5 % | 976.2 | `` |
| `` | 12.5 % | 976.2 | `` |
| `` | 12.5 % | 976.1 | `` |
| `` | 12.5 % | 975.7 | `` |
| `` | 12.5 % | 975.7 | `` |
| `` | 12.5 % | 975.6 | `` |
| `` | 12.5 % | 975.5 | `` |
| `` | 12.5 % | 975.4 | `` |
| `` | 12.5 % | 975.4 | `` |
| `` | 12.5 % | 975.3 | `` |
| `` | 12.5 % | 974.9 | `` |
| `` | 12.5 % | 974.9 | `` |
| `` | 12.5 % | 974.8 | `` |

