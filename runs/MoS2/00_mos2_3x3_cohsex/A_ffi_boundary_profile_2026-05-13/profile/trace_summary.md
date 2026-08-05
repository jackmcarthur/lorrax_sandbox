# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_profile_2026-05-13/profile/xprof/rank_0/plugins/profile/2026_05_12_20_12_06/perfetto_trace.json.gz`
**Duration:** 99.552 s
**GPU streams:** 8 compute, 15 H2D, 18 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 1711 | 11.11 GiB | 646.63 ms | 18.45 |
| D2H | 588 | 4.42 GiB | 183.03 ms | 25.92 |
| D2D | 846 | 8.55 GiB | 11.71 ms | 784.13 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 1711 | 646.63 ms | 646.63 ms | 0.000 |
| D2H | 588 | 183.03 ms | 167.60 ms | 0.084 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1.01 GiB | 10.88 | 68.30 s |
| D2H | 129.73 MiB | 1.36 | 25.10 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 234 | 21194.66 | 2970.28 | 0 | `jit__batched_chol` | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/whil` |
| `all-gather-start` | 47 | 2840.12 | 1640.62 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/sharding_cons` |
| `all-gather-start.2` | 36 | 1483.73 | 331.80 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(fft)/fft` |
| `all-to-all.5` | 1 | 1332.87 | 1332.87 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `custom-call.60.0` | 48 | 1090.06 | 46.10 | 18.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/jit(shmap_bod` |
| `loop_transpose_fusion_2` | 64 | 662.08 | 10.46 | 100 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 76 | 392.97 | 6.08 | 12.5 | `` | `` |
| `fft.11.0` | 152 | 330.65 | 2.67 | 50 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/jit(_convolve)/jit(shmap_body)/jit(f` |
| `loop_reduce_fusion` | 78 | 317.42 | 10.25 | 62.5 | `jit__einsum` | `jit(_einsum)/jit(main)/dot_general` |
| `fft.13.0` | 96 | 260.71 | 2.91 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_right_ifft_contract_fft)/jit(shm` |
| `fft.12.0` | 96 | 259.75 | 2.87 | 93.75 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_left_ifft_conj)/jit(shmap_body)/` |
| `input_transpose_fusion_3` | 43 | 97.63 | 3.25 | 56.25 | `` | `` |
| `loop_transpose_fusion.3` | 32 | 95.29 | 3.06 | 100 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(fn)/jit(fft)/fft` |
| `triangular-solve.11.0` | 1032 | 64.64 | 0.51 | 62.5 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/jit(shmap_bod` |
| `all-to-all.1.1` | 33 | 64.30 | 21.61 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_z)/sharding_constraint` |
| `triangular-solve.10.0` | 1032 | 60.83 | 0.58 | 62.5 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/jit(shmap_bod` |
| `all-to-all.3` | 32 | 48.40 | 2.00 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_z)/sharding_constraint` |
| `all-gather-start.1` | 40 | 33.47 | 1.85 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/sharding_cons` |
| `input_transpose_fusion` | 113 | 28.10 | 2.64 | 56.25 | `jit_eigh` | `jit(sigma_sx)/jit(main)/jit(_convolve)` |
| `all-reduce-start.1` | 31 | 27.69 | 21.95 | 0 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/kmty` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 2970280.7 | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/while/body` |
| `all-reduce-start` | 0.0 % | 2855632.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 1640622.5 | `jit(fn)/jit(main)/jit(fft)/fft` |
| `all-reduce-start` | 0.0 % | 1414666.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1397471.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1367955.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1342032.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-to-all.5` | 0.0 % | 1332865.1 | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-reduce-start` | 0.0 % | 1211427.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1096002.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1054876.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 732872.8 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 499007.3 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 380632.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 356198.6 | `jit(fn)/jit(main)/jit(fft)/fft` |
| `all-gather-start.2` | 0.0 % | 331799.8 | `jit(_kernel)/jit(main)/jit(fn)/jit(fft)/fft` |
| `all-gather-start.2` | 0.0 % | 307898.9 | `jit(_kernel)/jit(main)/jit(fn)/jit(fft)/fft` |
| `all-reduce-start` | 0.0 % | 304945.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.2` | 0.0 % | 297819.7 | `jit(_kernel)/jit(main)/jit(fn)/jit(fft)/fft` |
| `all-reduce-start` | 0.0 % | 267485.8 | `jit(sigma_sx)/jit(main)/kmsx` |

