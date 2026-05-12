# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/C_profile_sigma_2026-04-23/profile_v2/xprof/rank_0/plugins/profile/2026_04_23_13_05_37/perfetto_trace.json.gz`
**Duration:** 30.045 s
**GPU streams:** 4 compute, 9 H2D, 8 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 976 | 1.59 GiB | 90.55 ms | 18.84 |
| D2H | 728 | 1.62 GiB | 67.43 ms | 25.85 |
| D2D | 604 | 2.41 GiB | 4.69 ms | 551.32 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 976 | 90.55 ms | 90.55 ms | 0.000 |
| D2H | 728 | 67.43 ms | 65.41 ms | 0.030 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1012.50 MiB | 10.62 | 15.10 s |
| D2H | 1012.50 MiB | 10.62 | 9.50 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 78 | 5128.83 | 1291.33 | 0 | `jit__prepare_sigma_state` | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `all-gather-start` | 10 | 2102.35 | 818.09 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/sharding_cons` |
| `all-to-all.5` | 2 | 1096.51 | 1083.98 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `reduce-scatter.14` | 277 | 607.59 | 356.89 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `all-to-all.3` | 1 | 581.89 | 581.89 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constra` |
| `all-to-all.1.1` | 3 | 142.28 | 126.70 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/sharding_constraint` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 556 | 107.42 | 0.21 | 12.5 | `` | `` |
| `reduce-scatter.15` | 277 | 105.76 | 33.86 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `loop_transpose_fusion.1` | 300 | 61.61 | 3.18 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_convert_fusion` | 300 | 55.83 | 0.21 | 1.5625 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `loop_transpose_fusion.7` | 284 | 55.77 | 2.13 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `custom-call.27.0` | 277 | 48.69 | 0.18 | 12.5 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/ksxn` |
| `loop_transpose_fusion` | 307 | 43.88 | 1.73 | 100 | `jit__identity_fn` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion.2` | 298 | 42.00 | 3.31 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)` |
| `loop_transpose_fusion.6` | 284 | 41.06 | 1.66 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion.8` | 282 | 32.28 | 3.21 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.0.0` | 300 | 30.94 | 1.55 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.1.0` | 300 | 30.56 | 1.55 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.2.0` | 300 | 29.37 | 1.55 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.7.0` | 284 | 28.87 | 1.64 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 1291334.6 | `jit(_mean)/jit(main)/reduce_sum` |
| `all-to-all.5` | 0.0 % | 1083983.8 | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-reduce-start` | 0.0 % | 919680.7 | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/while/body` |
| `all-gather-start` | 0.0 % | 818088.6 | `jit(_fft_gather_reshard)/jit(main)/jit(_take)/gather` |
| `all-to-all.3` | 0.0 % | 581893.8 | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 509288.0 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 380113.8 | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `reduce-scatter.14` | 0.0 % | 356887.5 | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_body)` |
| `all-reduce-start` | 0.0 % | 349825.2 | `jit(trace)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 321847.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 215907.5 | `jit(gather)/jit(main)/gather` |
| `all-gather-start` | 0.0 % | 214299.1 | `` |
| `all-reduce-start` | 0.0 % | 212334.1 | `jit(_reduce_min)/jit(main)/reduce_min` |
| `all-reduce-start` | 0.0 % | 210934.5 | `jit(hartree)/jit(main)/xy` |
| `all-reduce-start` | 0.0 % | 177802.5 | `jit(_reduce_sum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 134473.7 | `jit(gather)/jit(main)/gather` |
| `all-gather-start` | 0.0 % | 133415.4 | `` |
| `all-to-all.1.1` | 0.0 % | 126697.1 | `jit(_solve_w)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 119678.5 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-reduce-start` | 0.0 % | 115409.2 | `jit(_psum)/jit(main)/reduce_sum` |

