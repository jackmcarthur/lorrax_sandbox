# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_rebased_D_usage_profile_2026-05-13/profile/xprof/rank_0/plugins/profile/2026_05_12_20_27_15/perfetto_trace.json.gz`
**Duration:** 148.479 s
**GPU streams:** 654 compute, 23 H2D, 18 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 5325 | 11.14 GiB | 539.70 ms | 22.16 |
| D2H | 2798 | 4.42 GiB | 185.11 ms | 25.65 |
| D2D | 28251 | 29.29 GiB | 104.65 ms | 300.50 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 5325 | 539.70 ms | 539.70 ms | 0.000 |
| D2H | 2798 | 185.11 ms | 185.09 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1.01 GiB | 10.88 | 109.30 s |
| D2H | 129.73 MiB | 1.36 | 84.00 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 312 | 17034.26 | 3085.93 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/qii->q/reduce_sum` |
| `custom-call.128.0` | 161784 | 5149.85 | 38.57 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve)/jit(shmap_body)/ffi_call` |
| `all-gather-start.3` | 32 | 3930.40 | 787.12 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `all-gather-start` | 53 | 2730.87 | 932.36 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 332 | 922.91 | 3.48 | 12.5 | `` | `` |
| `custom-call.124.0` | 20016 | 909.94 | 20.71 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_potrs)/jit(shmap_body)/ffi_call` |
| `loop_transpose_fusion_3` | 64 | 662.91 | 10.47 | 100 | `` | `` |
| `wrapped_add_2` | 128 | 537.03 | 4.33 | 100 | `` | `` |
| `loop_reduce_fusion` | 78 | 318.47 | 10.39 | 62.5 | `jit__einsum` | `jit(_einsum)/jit(main)/dot_general` |
| `fft.30.0` | 96 | 260.20 | 2.93 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_right_ifft_contract_fft)/jit(shm` |
| `fft.29.0` | 96 | 259.13 | 2.88 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_left_ifft_conj)/jit(shmap_body)/` |
| `custom-call.1.0` | 210 | 180.00 | 151.90 | 0 | `jit__potrf` | `jit(_potrf)/jit(main)/jit(shmap_body)/ffi_call` |
| `all-to-all.1.1` | 1 | 124.29 | 124.29 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/sharding_constraint` |
| `loop_transpose_fusion_8` | 128 | 99.38 | 0.88 | 100 | `` | `` |
| `all-gather-start.2` | 36 | 86.89 | 2.90 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/squeeze` |
| `all-gather-start.1` | 40 | 76.65 | 6.78 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start.1` | 13 | 65.40 | 49.60 | 0 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/kmty` |
| `fft.28.0` | 128 | 65.39 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.24.0` | 128 | 65.36 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.25.0` | 128 | 65.33 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 3085927.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1114812.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 932355.6 | `jit(_reshard)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 890605.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 860582.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 827141.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.3` | 0.0 % | 787121.5 | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `all-reduce-start` | 0.0 % | 766888.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.3` | 0.0 % | 750807.4 | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `all-gather-start.3` | 0.0 % | 704902.5 | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `all-reduce-start` | 0.0 % | 656507.2 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 577749.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 550963.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 516067.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 515043.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 507060.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 500748.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.3` | 0.0 % | 433098.6 | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `all-reduce-start` | 0.0 % | 415759.9 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-gather-start` | 0.0 % | 392110.3 | `jit(_reshard)/jit(main)/sharding_constraint` |

