# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_wfn_rchunk_shardmap_2026-05-13/profile/xprof/rank_0/plugins/profile/2026_05_13_00_30_33/perfetto_trace.json.gz`
**Duration:** 149.354 s
**GPU streams:** 654 compute, 23 H2D, 18 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 5339 | 11.14 GiB | 536.43 ms | 22.30 |
| D2H | 2689 | 4.41 GiB | 184.57 ms | 25.65 |
| D2D | 28259 | 29.29 GiB | 104.00 ms | 302.37 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 5339 | 536.43 ms | 536.43 ms | 0.000 |
| D2H | 2689 | 184.57 ms | 184.44 ms | 0.001 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1.01 GiB | 10.88 | 111.00 s |
| D2H | 129.73 MiB | 1.36 | 85.40 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 291 | 17042.60 | 2860.42 | 0 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `custom-call.233.0` | 161784 | 5211.74 | 49.38 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve)/jit(shmap_body)/ffi_call` |
| `all-gather-start` | 53 | 2570.84 | 795.61 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start.3` | 32 | 2125.71 | 946.34 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/shard_map` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 332 | 984.75 | 4.00 | 12.5 | `` | `` |
| `custom-call.229.0` | 20016 | 912.64 | 19.75 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_potrs)/jit(shmap_body)/ffi_call` |
| `loop_transpose_fusion_3` | 64 | 784.47 | 12.44 | 100 | `` | `` |
| `wrapped_add_2` | 128 | 656.65 | 5.33 | 100 | `` | `` |
| `loop_reduce_fusion` | 78 | 405.07 | 13.23 | 62.5 | `jit__einsum` | `jit(_einsum)/jit(main)/dot_general` |
| `fft.30.0` | 96 | 313.60 | 3.48 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_right_ifft_contract_fft)/jit(shm` |
| `fft.29.0` | 96 | 313.24 | 3.47 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_left_ifft_conj)/jit(shmap_body)/` |
| `custom-call.1.0` | 210 | 221.36 | 193.22 | 0 | `jit__potrf` | `jit(_potrf)/jit(main)/jit(shmap_body)/ffi_call` |
| `fft.27.0` | 128 | 81.04 | 0.66 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.25.0` | 128 | 81.04 | 0.66 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.24.0` | 128 | 80.97 | 0.66 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.28.0` | 128 | 80.85 | 0.66 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `fft.26.0` | 128 | 80.77 | 0.66 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(shmap_body)/jit(fft)/fft` |
| `all-reduce-start.1` | 13 | 52.82 | 42.93 | 0 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/kmty` |
| `all-to-all.1.1` | 1 | 44.63 | 44.63 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/sharding_constraint` |
| `Memset 3` | 19248 | 44.33 | 0.02 | - | `` | `` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 2860420.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1118181.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.3` | 0.0 % | 946341.2 | `jit(_kernel)/jit(main)/jit(fn)/shard_map` |
| `all-reduce-start` | 0.0 % | 938208.3 | `jit(_einsum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 936510.5 | `jit(_einsum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 891694.9 | `jit(_einsum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 844485.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 795606.9 | `jit(_reshard)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 649853.3 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 564298.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 545182.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 522743.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 517170.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 513967.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 508904.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 409772.8 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-gather-start` | 0.0 % | 398186.4 | `jit(_reshard)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 343277.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 310788.2 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 307834.9 | `jit(_psum)/jit(main)/reduce_sum` |

