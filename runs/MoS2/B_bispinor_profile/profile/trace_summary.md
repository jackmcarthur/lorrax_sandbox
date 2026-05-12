# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/B_bispinor_profile/profile/xprof/rank_0/plugins/profile/2026_05_03_03_34_12/perfetto_trace.json.gz`
**Duration:** 48.189 s
**GPU streams:** 8 compute, 9 H2D, 10 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 1162 | 3.79 GiB | 253.01 ms | 16.10 |
| D2H | 189 | 1.79 GiB | 73.30 ms | 26.19 |
| D2D | 721 | 2.17 GiB | 4.66 ms | 499.35 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 1162 | 253.01 ms | 253.01 ms | 0.000 |
| D2H | 189 | 73.30 ms | 65.94 ms | 0.101 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1012.50 MiB | 10.62 | 41.80 s |
| D2H | 225.01 MiB | 2.36 | 43.70 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 114 | 5794.11 | 2245.29 | 0 | `jit__batched_chol` | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/whil` |
| `all-gather-start` | 18 | 3044.62 | 2850.47 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/sharding_cons` |
| `all-to-all.3` | 12 | 2866.12 | 856.86 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constra` |
| `all-gather-start.2` | 13 | 25.58 | 10.13 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/pjit` |
| `all-gather-start.1` | 14 | 17.71 | 8.07 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/sharding_cons` |
| `triangular-solve.4.0` | 756 | 17.31 | 0.04 | 62.5 | `jit__batched_chol` | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/whil` |
| `triangular-solve.8.0` | 504 | 17.23 | 0.24 | 62.5 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/jit(shmap_bod` |
| `triangular-solve.9.0` | 504 | 16.98 | 0.21 | 62.5 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/jit(shmap_bod` |
| `fft.12.0` | 48 | 15.11 | 0.33 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_fft_and_rslice)/jit(shmap_body)/` |
| `custom-call.4.0` | 144 | 14.75 | 0.22 | 0 | `jit__batched_chol` | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/whil` |
| `fft.2.0` | 11 | 7.23 | 1.57 | 46.875 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `all-to-all.1.1` | 12 | 6.46 | 0.60 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 24 | 5.16 | 0.24 | 12.5 | `` | `` |
| `all-reduce-start.1` | 72 | 4.44 | 0.10 | 0 | `jit__batched_chol` | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/whil` |
| `collective-permute-start` | 15 | 3.99 | 0.33 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `input_transpose_fusion` | 117 | 3.93 | 1.59 | 56.25 | `jit_gather` | `jit(gather)/jit(main)/gather` |
| `fft.14.0` | 36 | 2.70 | 0.09 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_right_ifft_mul_fft)/jit(shmap_bo` |
| `fft.13.0` | 36 | 2.67 | 0.08 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_left_ifft_conj)/jit(shmap_body)/` |
| `custom-call.6.0` | 72 | 2.58 | 0.04 | 12.5 | `jit__batched_chol` | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/whil` |
| `input_transpose_fusion.6` | 12 | 2.58 | 0.24 | 56.25 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start` | 0.0 % | 2850471.5 | `jit(_fft_gather_reshard)/jit(main)/jit(_take)/gather` |
| `all-reduce-start` | 0.0 % | 2245288.0 | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/while/body` |
| `all-reduce-start` | 0.0 % | 2244016.0 | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/while/body` |
| `all-to-all.3` | 0.0 % | 856861.6 | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constraint` |
| `all-to-all.3` | 0.0 % | 824484.9 | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constraint` |
| `all-to-all.3` | 0.0 % | 813648.7 | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 486297.6 | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/while/body` |
| `all-reduce-start` | 0.0 % | 150735.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-to-all.3` | 0.0 % | 131524.8 | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constraint` |
| `all-to-all.3` | 0.0 % | 129237.2 | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 125292.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 118349.2 | `jit(_fft_gather_reshard)/jit(main)/jit(_take)/gather` |
| `all-reduce-start` | 0.0 % | 84930.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 66376.5 | `jit(gather)/jit(main)/gather` |
| `all-reduce-start` | 0.0 % | 64233.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 62958.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 58759.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 58253.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 47256.0 | `jit(trace)/jit(main)/reduce_sum` |
| `all-to-all.3` | 0.0 % | 41556.9 | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constraint` |

