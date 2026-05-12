# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/C_profile_sigma_2026-04-23/profile/xprof/rank_0/plugins/profile/2026_04_23_12_09_27/perfetto_trace.json.gz`
**Duration:** 30.738 s
**GPU streams:** 5 compute, 5 H2D, 8 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 957 | 1.59 GiB | 82.59 ms | 20.65 |
| D2H | 720 | 1.62 GiB | 67.44 ms | 25.84 |
| D2D | 594 | 2.41 GiB | 4.67 ms | 553.96 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 957 | 82.59 ms | 82.59 ms | 0.000 |
| D2H | 720 | 67.44 ms | 65.99 ms | 0.021 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 642.34 MiB | 6.74 | 14.40 s |
| D2H | 1012.50 MiB | 10.62 | 8.80 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 71 | 4702.75 | 1988.42 | 0 | `jit__prepare_sigma_state` | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `all-to-all.5` | 2 | 1133.10 | 1132.42 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-gather-start` | 10 | 1083.99 | 449.32 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_solve_all_at_once)/sharding_cons` |
| `reduce-scatter.14` | 277 | 634.99 | 339.62 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `all-to-all.3` | 1 | 514.37 | 514.37 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constra` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 556 | 107.33 | 0.21 | 12.5 | `` | `` |
| `loop_transpose_fusion.1` | 300 | 61.69 | 3.18 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_convert_fusion` | 300 | 55.83 | 0.21 | 1.5625 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `loop_transpose_fusion.7` | 284 | 55.81 | 2.13 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `reduce-scatter.15` | 277 | 51.51 | 6.91 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `custom-call.27.0` | 277 | 48.70 | 0.18 | 12.5 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/ksxn` |
| `loop_transpose_fusion` | 307 | 43.86 | 1.73 | 100 | `jit__identity_fn` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion.2` | 298 | 41.95 | 3.31 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)` |
| `loop_transpose_fusion.6` | 284 | 41.08 | 1.66 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion.8` | 282 | 32.29 | 3.20 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.0.0` | 300 | 30.96 | 1.55 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.1.0` | 300 | 30.66 | 1.55 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.2.0` | 300 | 29.38 | 1.55 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.7.0` | 284 | 28.96 | 1.64 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.6.0` | 284 | 28.78 | 1.64 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 1988419.8 | `jit(_mean)/jit(main)/reduce_sum` |
| `all-to-all.5` | 0.0 % | 1132418.1 | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-to-all.3` | 0.0 % | 514366.0 | `jit(_kernel)/jit(main)/jit(_reshard_rchunk)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 449323.1 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 438972.1 | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `reduce-scatter.14` | 0.0 % | 339616.9 | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_body)` |
| `all-reduce-start` | 0.0 % | 328266.6 | `jit(trace)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 306630.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 209007.7 | `` |
| `all-reduce-start` | 0.0 % | 204538.6 | `jit(_reduce_min)/jit(main)/reduce_min` |
| `all-gather-start` | 0.0 % | 203944.1 | `jit(gather)/jit(main)/gather` |
| `all-reduce-start` | 0.0 % | 202904.9 | `jit(hartree)/jit(main)/xy` |
| `all-reduce-start` | 0.0 % | 153078.2 | `jit(_reduce_sum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 124769.7 | `jit(gather)/jit(main)/gather` |
| `all-reduce-start` | 0.0 % | 121826.8 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-reduce-start` | 0.0 % | 117835.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `reduce-scatter.14` | 0.0 % | 104089.6 | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_body)` |
| `all-reduce-start` | 0.0 % | 100749.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 95891.7 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-gather-start` | 0.0 % | 69867.5 | `` |

