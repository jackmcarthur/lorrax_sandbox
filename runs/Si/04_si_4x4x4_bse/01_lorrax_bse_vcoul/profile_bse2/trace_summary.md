# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/profile_bse2/xprof/plugins/profile/2026_04_24_01_52_48/perfetto_trace.json.gz`
**Duration:** 10.195 s
**GPU streams:** 0 compute, 2 H2D, 2 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 57 | 510.17 MiB | 35.26 ms | 15.17 |
| D2H | 2348 | 2.61 KiB | 4.07 ms | 0.00 |
| D2D | 37 | 464.51 MiB | 0.75 ms | 651.31 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 57 | 35.26 ms | 35.26 ms | 0.000 |
| D2H | 2348 | 4.07 ms | 4.07 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 281.28 MiB | 2.95 | 0.10 s |
| D2H | 319.00 B | 0.00 | 10.10 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.8.0` | 800 | 1107.25 | 1.41 | 93.75 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `fft.9.0` | 600 | 833.33 | 1.41 | 100 | `jit_scan` | `jit(scan)/jit(main)/while/body/jit(_matvec_impl)/jit(fft)/ff` |
| `loop_multiply_fusion` | 208 | 324.07 | 1.63 | 31.25 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `input_transpose_fusion_2` | 200 | 281.50 | 1.41 | 56.25 | `` | `` |
| `input_transpose_fusion_1` | 200 | 275.28 | 1.38 | 56.25 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 600 | 206.98 | 0.98 | 12.5 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1::Params)` | 200 | 147.44 | 0.74 | 12.5 | `` | `` |
| `input_reduce_fusion` | 2148 | 9.47 | 0.01 | 100 | `jit_norm` | `jit(norm)/jit(main)` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_tn_align1::Params)` | 200 | 5.11 | 0.03 | 12.5 | `` | `` |
| `loop_subtract_fusion` | 2145 | 4.94 | 0.00 | 100 | `jit_scan` | `jit(scan)/jit(main)/while/body/while/body` |
| `wrapped_compare` | 2345 | 3.74 | 0.00 | 1.5625 | `jit_scan` | `jit(scan)/jit(main)/while/body/while/cond/lt` |
| `loop_add_fusion` | 2353 | 3.54 | 0.00 | 31.25 | `jit_add` | `jit(add)/jit(main)/add` |
| `loop_transpose_fusion` | 201 | 2.18 | 0.02 | 100 | `jit_compute_pair_amplitude` | `jit(compute_pair_amplitude)/jit(main)/kcsm` |
| `custom-call.12.0` | 69 | 2.04 | 0.47 | 100 | `jit_eigh` | `jit(eigh)/jit(main)/eigh` |
| `wrapped_transpose_1` | 200 | 2.04 | 0.01 | 100 | `` | `` |
| `loop_multiply_fusion_1` | 200 | 1.73 | 0.01 | 56.25 | `` | `` |
| `wrapped_transpose_2` | 200 | 1.57 | 0.01 | 100 | `` | `` |
| `fft.0.0` | 4 | 1.40 | 0.35 | 93.75 | `jit__lambda_` | `jit(<lambda>)/jit(main)/jit(fft)/fft` |
| `wrapped_transpose` | 201 | 1.39 | 0.01 | 56.25 | `jit_transpose` | `` |
| `input_transpose_fusion` | 202 | 1.38 | 0.34 | 100 | `jit_eigh` | `jit(eigh)/jit(main)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `` | 12.5 % | 975.9 | `` |
| `` | 12.5 % | 974.0 | `` |
| `` | 12.5 % | 973.4 | `` |
| `` | 12.5 % | 973.4 | `` |
| `` | 12.5 % | 973.2 | `` |
| `` | 12.5 % | 973.2 | `` |
| `` | 12.5 % | 973.1 | `` |
| `` | 12.5 % | 973.0 | `` |
| `` | 12.5 % | 972.8 | `` |
| `` | 12.5 % | 972.7 | `` |
| `` | 12.5 % | 972.6 | `` |
| `` | 12.5 % | 972.5 | `` |
| `` | 12.5 % | 972.5 | `` |
| `` | 12.5 % | 972.4 | `` |
| `` | 12.5 % | 972.4 | `` |
| `` | 12.5 % | 972.3 | `` |
| `` | 12.5 % | 972.2 | `` |
| `` | 12.5 % | 972.1 | `` |
| `` | 12.5 % | 972.1 | `` |
| `` | 12.5 % | 971.9 | `` |

