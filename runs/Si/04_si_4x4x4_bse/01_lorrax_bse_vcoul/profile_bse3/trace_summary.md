# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/profile_bse3/xprof/plugins/profile/2026_04_24_01_56_43/perfetto_trace.json.gz`
**Duration:** 10.659 s
**GPU streams:** 0 compute, 2 H2D, 2 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 57 | 510.17 MiB | 36.16 ms | 14.79 |
| D2H | 2348 | 2.61 KiB | 4.20 ms | 0.00 |
| D2D | 37 | 464.51 MiB | 0.76 ms | 643.76 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 57 | 36.16 ms | 36.16 ms | 0.000 |
| D2H | 2348 | 4.20 ms | 4.20 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 281.28 MiB | 2.95 | 0.10 s |
| D2H | 160.00 B | 0.00 | 10.50 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.8.0` | 800 | 1107.22 | 1.41 | 93.75 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `fft.9.0` | 600 | 833.37 | 1.41 | 100 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `loop_multiply_fusion` | 208 | 324.12 | 1.63 | 31.25 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion_2` | 200 | 281.49 | 1.41 | 56.25 | `` | `` |
| `input_transpose_fusion_1` | 200 | 275.31 | 1.38 | 56.25 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 600 | 207.34 | 0.98 | 12.5 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1::Params)` | 200 | 147.66 | 0.75 | 12.5 | `` | `` |
| `input_reduce_fusion` | 2148 | 9.50 | 0.01 | 100 | `jit_norm` | `jit(norm)/jit(main)` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1::Params)` | 200 | 5.11 | 0.03 | 12.5 | `` | `` |
| `loop_subtract_fusion` | 2145 | 4.99 | 0.00 | 100 | `jit_scan` | `jit(scan)/jit(main)/while/body/while/body` |
| `wrapped_compare` | 2345 | 3.72 | 0.00 | 1.5625 | `jit_scan` | `jit(scan)/jit(main)/while/body/while/cond/lt` |
| `loop_add_fusion` | 2353 | 3.52 | 0.00 | 31.25 | `jit_add` | `jit(add)/jit(main)/add` |
| `custom-call.12.0` | 69 | 2.07 | 0.47 | 100 | `jit_eigh` | `jit(eigh)/jit(main)/eigh` |
| `wrapped_transpose_2` | 200 | 2.04 | 0.01 | 100 | `` | `` |
| `input_reduce_fusion_1` | 200 | 2.04 | 0.01 | 50 | `` | `` |
| `loop_multiply_fusion_1` | 200 | 1.73 | 0.01 | 56.25 | `` | `` |
| `fft.0.0` | 4 | 1.40 | 0.35 | 93.75 | `jit__lambda_` | `jit(<lambda>)/jit(main)/jit(fft)/fft` |
| `wrapped_transpose_1` | 200 | 1.39 | 0.01 | 100 | `` | `` |
| `input_transpose_fusion` | 202 | 1.38 | 0.34 | 100 | `jit_eigh` | `jit(eigh)/jit(main)` |
| `input_reduce_fusion_3` | 200 | 0.95 | 0.01 | 56.25 | `` | `` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `` | 12.5 % | 979.6 | `` |
| `` | 12.5 % | 976.8 | `` |
| `` | 12.5 % | 975.9 | `` |
| `` | 12.5 % | 974.9 | `` |
| `` | 12.5 % | 974.8 | `` |
| `` | 12.5 % | 974.6 | `` |
| `` | 12.5 % | 974.4 | `` |
| `` | 12.5 % | 973.9 | `` |
| `` | 12.5 % | 973.8 | `` |
| `` | 12.5 % | 973.3 | `` |
| `` | 12.5 % | 973.2 | `` |
| `` | 12.5 % | 973.1 | `` |
| `` | 12.5 % | 972.7 | `` |
| `` | 12.5 % | 972.7 | `` |
| `` | 12.5 % | 972.6 | `` |
| `` | 12.5 % | 972.6 | `` |
| `` | 12.5 % | 972.5 | `` |
| `` | 12.5 % | 972.5 | `` |
| `` | 12.5 % | 972.4 | `` |
| `` | 12.5 % | 972.3 | `` |

