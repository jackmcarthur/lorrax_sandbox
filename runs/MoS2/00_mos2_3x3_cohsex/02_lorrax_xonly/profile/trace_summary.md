# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/02_lorrax_xonly/profile/xprof/rank_0/plugins/profile/2026_05_11_17_57_50/perfetto_trace.json.gz`
**Duration:** 33.864 s
**GPU streams:** 4 compute, 10 H2D, 7 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 834 | 1.14 GiB | 58.68 ms | 20.94 |
| D2H | 180 | 1.12 GiB | 45.87 ms | 26.16 |
| D2D | 338 | 2.31 GiB | 3.96 ms | 625.19 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 834 | 58.68 ms | 58.68 ms | 0.000 |
| D2H | 180 | 45.87 ms | 45.87 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 781.46 MiB | 8.19 | 22.80 s |
| D2H | 585.68 MiB | 6.14 | 14.30 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 82 | 10960.24 | 4453.92 | 0 | `jit__batched_chol` | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/whil` |
| `all-gather-start` | 11 | 1065.56 | 375.47 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/sharding_cons` |
| `all-gather-start.1` | 3 | 248.22 | 245.20 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(fft)/fft` |
| `all-reduce-start.1` | 22 | 60.29 | 44.77 | 0 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/kmty` |
| `loop_transpose_fusion_2` | 2 | 38.24 | 19.14 | 100 | `` | `` |
| `loop_reduce_fusion` | 41 | 26.35 | 25.65 | 62.5 | `jit__einsum` | `jit(_einsum)/jit(main)/dot_general` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 4 | 25.17 | 12.57 | 12.5 | `` | `` |
| `fft.13.0` | 3 | 19.09 | 6.56 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_right_ifft_contract_fft)/jit(shm` |
| `fft.12.0` | 3 | 19.09 | 6.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_left_ifft_conj)/jit(shmap_body)/` |
| `triangular-solve.9.0` | 39 | 18.17 | 3.83 | 62.5 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/jit(shmap_bod` |
| `triangular-solve.8.0` | 39 | 17.43 | 3.23 | 62.5 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/jit(shmap_bod` |
| `input_transpose_fusion_3` | 3 | 15.32 | 15.29 | 56.25 | `` | `` |
| `custom-call.1` | 1764 | 13.22 | 0.33 | 100 | `jit_eigh` | `jit(eigh)/jit(main)/eigh` |
| `all-to-all.1.1` | 1 | 10.39 | 10.39 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_z)/sharding_constraint` |
| `all-to-all.3` | 1 | 9.05 | 9.05 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_z)/sharding_constraint` |
| `fft.11.0` | 10 | 6.74 | 1.56 | 50 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/jit(_convolve)/jit(shmap_body)/jit(f` |
| `fft.1.0` | 12 | 6.23 | 0.63 | 93.75 | `jit_fn` | `jit(fn)/jit(main)/jit(fft)/fft` |
| `fft.2.0` | 6 | 4.97 | 1.57 | 46.875 | `jit__f` | `jit(_f)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `custom-call.1.0` | 103 | 4.41 | 1.32 | 43.75 | `jit__einsum` | `jit(_einsum)/jit(main)/dot_general` |
| `collective-permute-start` | 5 | 4.23 | 3.98 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_z)/sharding_constraint` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 4453923.4 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 2827643.2 | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/while/body` |
| `all-reduce-start` | 0.0 % | 1029025.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 447958.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 375472.5 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 277834.4 | `jit(trace)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 258204.4 | `jit(gather)/jit(main)/gather` |
| `all-gather-start.1` | 0.0 % | 245195.8 | `jit(_kernel)/jit(main)/jit(fn)/jit(fft)/fft` |
| `all-reduce-start` | 0.0 % | 241188.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 236136.4 | `jit(hartree)/jit(main)/xy` |
| `all-reduce-start` | 0.0 % | 202696.7 | `jit(sigma_coh)/jit(main)/kmsx` |
| `all-gather-start` | 0.0 % | 150623.6 | `jit(gather)/jit(main)/gather` |
| `all-reduce-start` | 0.0 % | 128733.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 127985.2 | `jit(fn)/jit(main)/jit(fft)/fft` |
| `all-reduce-start` | 0.0 % | 123549.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 81074.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 75016.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 69240.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 68726.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 67889.5 | `jit(_psum)/jit(main)/reduce_sum` |

