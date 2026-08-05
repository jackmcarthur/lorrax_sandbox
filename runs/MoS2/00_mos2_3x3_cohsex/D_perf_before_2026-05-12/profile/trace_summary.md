# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_perf_before_2026-05-12/profile/xprof/rank_0/plugins/profile/2026_05_12_16_53_07/perfetto_trace.json.gz`
**Duration:** 157.905 s
**GPU streams:** 8 compute, 15 H2D, 18 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 2535 | 11.13 GiB | 539.46 ms | 22.16 |
| D2H | 1052 | 4.42 GiB | 182.56 ms | 25.99 |
| D2D | 1254 | 8.84 GiB | 12.89 ms | 736.53 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 2535 | 539.46 ms | 539.46 ms | 0.000 |
| D2H | 1052 | 182.56 ms | 182.54 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1.01 GiB | 10.88 | 115.70 s |
| D2H | 129.73 MiB | 1.36 | 67.90 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 306 | 26242.20 | 4583.49 | 0 | `jit__batched_chol` | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/whil` |
| `all-gather-start` | 53 | 3456.83 | 1538.88 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/sharding_cons` |
| `all-to-all.5` | 33 | 1647.98 | 1602.19 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_z)/sharding_constraint` |
| `all-gather-start.5` | 32 | 1259.30 | 223.94 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `custom-call.135.0` | 48 | 1098.70 | 47.14 | 18.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/jit(shmap_bod` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 332 | 914.74 | 3.42 | 12.5 | `` | `` |
| `loop_transpose_fusion_5` | 64 | 661.71 | 10.47 | 100 | `` | `` |
| `wrapped_add_2` | 128 | 536.71 | 4.33 | 100 | `` | `` |
| `loop_reduce_fusion` | 78 | 317.71 | 10.39 | 62.5 | `jit__einsum` | `jit(_einsum)/jit(main)/dot_general` |
| `fft.30.0` | 96 | 259.58 | 2.93 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_right_ifft_contract_fft)/jit(shm` |
| `fft.29.0` | 96 | 258.42 | 2.88 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_left_ifft_conj)/jit(shmap_body)/` |
| `all-reduce-start.1` | 31 | 168.25 | 130.13 | 0 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/kmty` |
| `loop_transpose_fusion_10` | 128 | 99.16 | 0.87 | 100 | `` | `` |
| `all-to-all.1.1` | 33 | 96.43 | 54.46 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_z)/sharding_constraint` |
| `all-gather-start.4` | 32 | 85.54 | 2.76 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `all-to-all.2.1` | 33 | 79.21 | 16.44 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-to-all.3.1` | 33 | 79.12 | 2.89 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `fft.25.0` | 128 | 65.13 | 0.53 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.24.0` | 128 | 65.07 | 0.53 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.27.0` | 128 | 65.07 | 0.53 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 4583490.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 2918063.7 | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/while/body` |
| `all-to-all.5` | 0.0 % | 1602191.5 | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-gather-start` | 0.0 % | 1538876.4 | `jit(_reshard)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 1304043.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1292928.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1273898.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1266645.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1155430.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1105710.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1088626.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1064284.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 819438.1 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 803379.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 566875.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 538982.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 521922.9 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 501085.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 411036.1 | `jit(_reshard)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 326862.1 | `jit(_psum)/jit(main)/reduce_sum` |

