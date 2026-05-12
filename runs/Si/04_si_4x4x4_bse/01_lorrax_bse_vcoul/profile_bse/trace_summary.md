# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/profile_bse/xprof/plugins/profile/2026_04_24_01_49_12/perfetto_trace.json.gz`
**Duration:** 11.454 s
**GPU streams:** 0 compute, 2 H2D, 2 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 57 | 510.17 MiB | 61.04 ms | 8.76 |
| D2H | 2348 | 2.61 KiB | 4.17 ms | 0.00 |
| D2D | 37 | 464.51 MiB | 0.75 ms | 647.73 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 57 | 61.04 ms | 61.04 ms | 0.000 |
| D2H | 2348 | 4.17 ms | 4.17 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 281.28 MiB | 2.95 | 0.20 s |
| D2H | 160.00 B | 0.00 | 11.30 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.13.0` | 800 | 1116.34 | 1.42 | 93.75 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `fft.14.0` | 600 | 841.80 | 1.41 | 100 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `loop_multiply_fusion` | 207 | 330.33 | 1.66 | 31.25 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion_2` | 200 | 303.65 | 1.53 | 56.25 | `` | `` |
| `input_transpose_fusion_1` | 200 | 298.69 | 1.51 | 56.25 | `` | `` |
| `fft.12.0` | 800 | 282.68 | 0.36 | 93.75 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 800 | 267.53 | 0.98 | 12.5 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1::Params)` | 200 | 160.04 | 0.81 | 12.5 | `` | `` |
| `input_reduce_fusion` | 2148 | 9.41 | 0.01 | 100 | `jit_norm` | `jit(norm)/jit(main)` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1::Params)` | 200 | 5.11 | 0.03 | 12.5 | `` | `` |
| `loop_subtract_fusion` | 2145 | 4.94 | 0.00 | 100 | `jit_scan` | `jit(scan)/jit(main)/while/body/while/body` |
| `loop_complex_transpose_fusion` | 200 | 4.77 | 0.03 | 75 | `` | `` |
| `wrapped_compare` | 2345 | 3.70 | 0.00 | 1.5625 | `jit_scan` | `jit(scan)/jit(main)/while/body/while/cond/lt` |
| `loop_add_fusion` | 2353 | 3.54 | 0.00 | 31.25 | `jit_add` | `jit(add)/jit(main)/add` |
| `loop_transpose_fusion` | 200 | 3.38 | 0.02 | 100 | `` | `` |
| `loop_multiply_fusion_1` | 200 | 2.56 | 0.01 | 100 | `` | `` |
| `wrapped_transpose_2` | 200 | 2.08 | 0.01 | 100 | `` | `` |
| `custom-call.12.0` | 69 | 2.06 | 0.47 | 100 | `jit_eigh` | `jit(eigh)/jit(main)/eigh` |
| `loop_multiply_fusion_2` | 200 | 1.58 | 0.01 | 56.25 | `` | `` |
| `input_transpose_fusion` | 202 | 1.50 | 0.34 | 100 | `jit_eigh` | `jit(eigh)/jit(main)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `` | 12.5 % | 980.2 | `` |
| `` | 12.5 % | 979.6 | `` |
| `` | 12.5 % | 978.7 | `` |
| `` | 12.5 % | 978.4 | `` |
| `` | 12.5 % | 977.6 | `` |
| `` | 12.5 % | 977.2 | `` |
| `` | 12.5 % | 977.1 | `` |
| `` | 12.5 % | 977.0 | `` |
| `` | 12.5 % | 977.0 | `` |
| `` | 12.5 % | 976.8 | `` |
| `` | 12.5 % | 976.8 | `` |
| `` | 12.5 % | 976.7 | `` |
| `` | 12.5 % | 976.6 | `` |
| `` | 12.5 % | 976.3 | `` |
| `` | 12.5 % | 976.1 | `` |
| `` | 12.5 % | 975.9 | `` |
| `` | 12.5 % | 975.9 | `` |
| `` | 12.5 % | 975.8 | `` |
| `` | 12.5 % | 975.7 | `` |
| `` | 12.5 % | 975.7 | `` |

