# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_perf_after_2026-05-12/profile/xprof/rank_0/plugins/profile/2026_05_12_21_43_22/perfetto_trace.json.gz`
**Duration:** 148.143 s
**GPU streams:** 654 compute, 23 H2D, 18 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 5339 | 11.14 GiB | 559.63 ms | 21.38 |
| D2H | 2798 | 4.42 GiB | 185.26 ms | 25.63 |
| D2D | 28259 | 29.29 GiB | 104.19 ms | 301.82 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 5339 | 559.63 ms | 559.63 ms | 0.000 |
| D2H | 2798 | 185.26 ms | 185.24 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1.01 GiB | 10.88 | 117.70 s |
| D2H | 129.73 MiB | 1.36 | 42.80 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 291 | 17079.31 | 2858.05 | 0 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `custom-call.128.0` | 161784 | 5177.10 | 50.39 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve)/jit(shmap_body)/ffi_call` |
| `all-gather-start` | 53 | 2540.95 | 754.28 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start.3` | 32 | 2322.02 | 660.69 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 332 | 935.34 | 3.57 | 12.5 | `` | `` |
| `custom-call.124.0` | 20016 | 898.30 | 19.09 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_potrs)/jit(shmap_body)/ffi_call` |
| `loop_transpose_fusion_3` | 64 | 664.24 | 10.50 | 100 | `` | `` |
| `wrapped_add_2` | 128 | 540.00 | 4.36 | 100 | `` | `` |
| `loop_reduce_fusion` | 78 | 322.30 | 10.55 | 62.5 | `jit__einsum` | `jit(_einsum)/jit(main)/dot_general` |
| `fft.30.0` | 96 | 261.10 | 2.92 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_right_ifft_contract_fft)/jit(shm` |
| `fft.29.0` | 96 | 260.09 | 2.88 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_left_ifft_conj)/jit(shmap_body)/` |
| `custom-call.1.0` | 210 | 174.19 | 146.04 | 0 | `jit__potrf` | `jit(_potrf)/jit(main)/jit(shmap_body)/ffi_call` |
| `all-to-all.1.1` | 1 | 118.45 | 118.45 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/sharding_constraint` |
| `loop_transpose_fusion_8` | 128 | 100.18 | 0.88 | 100 | `` | `` |
| `all-gather-start.2` | 36 | 86.32 | 2.87 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/squeeze` |
| `all-gather-start.1` | 40 | 71.96 | 3.00 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `fft.28.0` | 128 | 66.17 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.25.0` | 128 | 66.15 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.24.0` | 128 | 66.11 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.27.0` | 128 | 66.03 | 0.54 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 2858045.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1127753.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 942978.8 | `jit(_einsum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 863810.9 | `jit(_einsum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 794425.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 784808.2 | `jit(_einsum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 754275.3 | `jit(_reshard)/jit(main)/sharding_constraint` |
| `all-gather-start.3` | 0.0 % | 660686.0 | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `all-reduce-start` | 0.0 % | 645197.5 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 553214.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 547258.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 527923.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 525857.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 518863.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 498751.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.3` | 0.0 % | 479515.7 | `jit(_kernel)/jit(main)/jit(fn)/transpose` |
| `all-reduce-start` | 0.0 % | 416838.4 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-gather-start` | 0.0 % | 400459.5 | `jit(_reshard)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 357311.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 314442.8 | `jit(_psum)/jit(main)/reduce_sum` |

