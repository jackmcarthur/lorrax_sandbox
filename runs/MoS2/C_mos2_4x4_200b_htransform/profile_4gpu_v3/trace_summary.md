# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v3/xprof/rank_0/plugins/profile/2026_04_25_16_51_59/perfetto_trace.json.gz`
**Duration:** 20.408 s
**GPU streams:** 3 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 441 | 693.08 MiB | 39.21 ms | 18.53 |
| D2H | 629 | 28.23 MiB | 2.06 ms | 14.39 |
| D2D | 98 | 8.06 GiB | 13.04 ms | 664.00 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 441 | 39.21 ms | 39.21 ms | 0.000 |
| D2H | 629 | 2.06 ms | 2.06 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 337.50 MiB | 3.54 | 1.00 s |
| D2H | 14.06 MiB | 0.15 | 5.10 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `custom-call.2.0` | 80110 | 3268.78 | 0.78 | 12.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(eigvalsh))/jit(eigh)/ei` |
| `custom-call.1.0` | 5040 | 559.60 | 0.52 | 12.5 | `jit_eigvalsh` | `jit(eigvalsh)/jit(main)/jit(eigh)/eigh` |
| `all-reduce-start` | 5 | 221.31 | 193.32 | 0 | `jit__reduce_max` | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-gather-start` | 12 | 97.67 | 42.33 | 0 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `triangular-solve.4.0` | 174 | 62.71 | 1.31 | 62.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `triangular-solve.5.0` | 174 | 62.49 | 1.30 | 62.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `all-to-all.2` | 10 | 31.53 | 10.02 | 0 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/add` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1::Params)` | 1 | 17.83 | 17.83 | 0 | `` | `` |
| `Memset 3` | 6639 | 10.89 | 0.01 | - | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1::Params)` | 1 | 10.48 | 10.48 | 0 | `` | `` |
| `input_transpose_fusion.2` | 9 | 7.25 | 1.04 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |
| `input_transpose_fusion.3` | 7 | 5.10 | 0.77 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(eigvalsh))/jit(eigh)` |
| `wrapped_transpose.1` | 6 | 4.25 | 0.71 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `input_transpose_fusion` | 17 | 4.07 | 0.53 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |
| `input_transpose_fusion.1` | 13 | 3.28 | 0.51 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |
| `loop_transpose_fusion` | 4 | 2.40 | 1.13 | 100 | `jit__build_fH` | `jit(_build_fH)/jit(main)/custom_partitioning` |
| `custom-call.6.0` | 6 | 2.08 | 0.35 | 12.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/dot_general` |
| `loop_transpose_fusion.1` | 3 | 1.92 | 1.14 | 100 | `jit__build_fH` | `jit(_build_fH)/jit(main)/custom_partitioning` |
| `custom-call.1` | 15 | 1.43 | 0.65 | 18.75 | `jit_sort` | `jit(sort)/jit(main)/sort` |
| `loop_complex_fusion` | 15 | 1.32 | 1.12 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 193323.4 | `jit(_reduce_min)/jit(main)/reduce_min` |
| `all-gather-start` | 0.0 % | 42325.6 | `jit(eigvalsh)/jit(main)/jit(eigh)/eigh` |
| `all-reduce-start` | 0.0 % | 23020.2 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-gather-start` | 0.0 % | 20094.1 | `jit(reshape)/jit(main)/reshape` |
| `` | 0.0 % | 17827.5 | `` |
| `` | 0.0 % | 10484.1 | `` |
| `all-to-all.2` | 0.0 % | 10018.9 | `jit(_kpath_batch)/jit(main)/add` |
| `all-gather-start` | 0.0 % | 7858.3 | `` |
| `all-to-all.2` | 0.0 % | 5430.5 | `jit(_accum)/jit(main)/reshape` |
| `all-to-all.2` | 0.0 % | 4764.4 | `jit(_reshard_rchunk)/jit(main)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 4510.2 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-gather-start` | 0.0 % | 4492.0 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-gather-start` | 0.0 % | 4455.0 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-gather-start` | 0.0 % | 4431.4 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-gather-start` | 0.0 % | 4417.1 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-gather-start` | 0.0 % | 4392.1 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-reduce-start` | 0.0 % | 3386.7 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-to-all.2` | 0.0 % | 3081.5 | `jit(_kpath_batch)/jit(main)/add` |
| `all-to-all.2` | 0.0 % | 2024.5 | `jit(_kpath_batch)/jit(main)/add` |
| `all-to-all.2` | 0.0 % | 2024.2 | `jit(_kpath_batch)/jit(main)/add` |

