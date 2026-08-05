# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_perf_take_2026-05-12_profile/xprof/rank_0/plugins/profile/2026_05_12_20_34_21/perfetto_trace.json.gz`
**Duration:** 153.406 s
**GPU streams:** 654 compute, 23 H2D, 18 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 5325 | 11.14 GiB | 562.01 ms | 21.28 |
| D2H | 2798 | 4.42 GiB | 185.14 ms | 25.64 |
| D2D | 28251 | 29.29 GiB | 104.19 ms | 301.81 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 5325 | 562.01 ms | 562.01 ms | 0.000 |
| D2H | 2798 | 185.14 ms | 185.11 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1.01 GiB | 10.88 | 115.50 s |
| D2H | 129.73 MiB | 1.36 | 87.90 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 312 | 17117.82 | 3032.34 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/qii->q/reduce_sum` |
| `custom-call.128.0` | 161784 | 5137.51 | 34.27 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve)/jit(shmap_body)/ffi_call` |
| `all-gather-start` | 53 | 2599.63 | 775.59 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start.3` | 32 | 2096.24 | 942.49 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 332 | 921.05 | 3.48 | 12.5 | `` | `` |
| `custom-call.124.0` | 20016 | 913.28 | 23.43 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_potrs)/jit(shmap_body)/ffi_call` |
| `loop_transpose_fusion_3` | 64 | 662.01 | 10.48 | 100 | `` | `` |
| `wrapped_add_2` | 128 | 537.38 | 4.34 | 100 | `` | `` |
| `loop_reduce_fusion` | 78 | 318.54 | 10.39 | 62.5 | `jit__einsum` | `jit(_einsum)/jit(main)/dot_general` |
| `fft.30.0` | 96 | 260.24 | 2.92 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_right_ifft_contract_fft)/jit(shm` |
| `fft.29.0` | 96 | 259.12 | 2.88 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_left_ifft_conj)/jit(shmap_body)/` |
| `custom-call.1.0` | 210 | 192.35 | 164.19 | 0 | `jit__potrf` | `jit(_potrf)/jit(main)/jit(shmap_body)/ffi_call` |
| `loop_transpose_fusion_8` | 128 | 99.53 | 0.89 | 100 | `` | `` |
| `all-to-all.1.1` | 1 | 86.99 | 86.99 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/sharding_constraint` |
| `all-gather-start.2` | 36 | 86.62 | 2.79 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/squeeze` |
| `all-reduce-start.1` | 13 | 84.05 | 80.03 | 0 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/kmty` |
| `all-gather-start.1` | 40 | 71.36 | 2.43 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `fft.25.0` | 128 | 65.39 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.24.0` | 128 | 65.39 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.28.0` | 128 | 65.38 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 3032344.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1143343.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.3` | 0.0 % | 942486.7 | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `all-reduce-start` | 0.0 % | 903336.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 875388.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 854110.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 809801.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 775591.8 | `jit(_reshard)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 667090.1 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 561680.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 560725.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 518764.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 513346.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 512747.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 505055.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 411320.5 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-gather-start` | 0.0 % | 407470.4 | `jit(_reshard)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 358432.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 347085.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 322821.8 | `jit(_kernel)/jit(main)/sharding_constraint` |

