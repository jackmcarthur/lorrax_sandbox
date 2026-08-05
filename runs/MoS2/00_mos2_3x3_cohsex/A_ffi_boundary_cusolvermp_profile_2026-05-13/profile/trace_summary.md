# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_cusolvermp_profile_2026-05-13/profile/xprof/rank_0/plugins/profile/2026_05_12_20_15_52/perfetto_trace.json.gz`
**Duration:** 100.026 s
**GPU streams:** 654 compute, 23 H2D, 18 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 4861 | 11.12 GiB | 726.94 ms | 16.42 |
| D2H | 2334 | 4.42 GiB | 188.66 ms | 25.16 |
| D2D | 28173 | 29.07 GiB | 103.09 ms | 302.77 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 4861 | 726.94 ms | 726.94 ms | 0.000 |
| D2H | 2334 | 188.66 ms | 162.54 ms | 0.138 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1.01 GiB | 10.88 | 84.30 s |
| D2H | 129.77 MiB | 1.36 | 25.60 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 240 | 20673.12 | 2985.61 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/qii->q/reduce_sum` |
| `custom-call.53.0` | 161784 | 5195.00 | 85.89 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve)/jit(shmap_body)/ffi_call` |
| `all-gather-start` | 47 | 4964.99 | 1591.86 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(fft)/fft` |
| `all-to-all.5` | 1 | 1213.51 | 1213.51 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `custom-call.49.0` | 20016 | 951.16 | 20.49 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_potrs)/jit(shmap_body)/ffi_call` |
| `loop_transpose_fusion_3` | 48 | 498.34 | 10.45 | 100 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 76 | 392.10 | 6.09 | 12.5 | `` | `` |
| `fft.11.0` | 152 | 331.94 | 2.63 | 50 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/jit(_convolve)/jit(shmap_body)/jit(f` |
| `loop_reduce_fusion` | 78 | 320.76 | 10.26 | 62.5 | `jit__einsum` | `jit(_einsum)/jit(main)/dot_general` |
| `custom-call.1.0` | 198 | 303.09 | 276.12 | 0 | `jit__potrf` | `jit(_potrf)/jit(main)/jit(shmap_body)/ffi_call` |
| `fft.13.0` | 96 | 262.09 | 2.91 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_right_ifft_contract_fft)/jit(shm` |
| `fft.12.0` | 96 | 261.62 | 2.88 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_left_ifft_conj)/jit(shmap_body)/` |
| `loop_transpose_fusion_2` | 16 | 165.20 | 10.40 | 100 | `` | `` |
| `all-to-all.1.1` | 33 | 131.80 | 91.71 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `input_transpose_fusion` | 95 | 118.68 | 3.03 | 56.25 | `jit_eigh` | `jit(sigma_sx)/jit(main)/jit(_convolve)` |
| `loop_transpose_fusion.4` | 32 | 94.80 | 3.03 | 100 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(fft)/fft` |
| `all-reduce-start.1` | 13 | 71.72 | 58.18 | 0 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/kmty` |
| `Memset 3` | 19247 | 42.09 | 0.02 | - | `` | `` |
| `all-to-all.3` | 32 | 37.91 | 1.23 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `loop_gather_fusion` | 52 | 25.32 | 0.73 | 14.0625 | `jit_inv` | `jit(inv)/jit(main)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 2985614.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 1591858.8 | `jit(fn)/jit(main)/jit(fft)/fft` |
| `all-reduce-start` | 0.0 % | 1435938.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1434077.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1419428.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1369113.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-to-all.5` | 0.0 % | 1213515.0 | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-reduce-start` | 0.0 % | 1165701.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1105808.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1105513.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1073460.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 755523.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 691155.3 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 594923.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 540203.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 429166.3 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-gather-start` | 0.0 % | 362715.5 | `jit(fn)/jit(main)/jit(fft)/fft` |
| `all-reduce-start` | 0.0 % | 343149.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 334028.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 312550.7 | `jit(_kernel)/jit(main)/sharding_constraint` |

