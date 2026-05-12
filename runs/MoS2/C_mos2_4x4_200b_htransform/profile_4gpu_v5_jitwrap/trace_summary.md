# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v5_jitwrap/xprof/rank_0/plugins/profile/2026_04_25_20_29_12/perfetto_trace.json.gz`
**Duration:** 15.245 s
**GPU streams:** 4 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 398 | 693.08 MiB | 40.91 ms | 17.76 |
| D2H | 629 | 28.23 MiB | 2.08 ms | 14.23 |
| D2D | 74 | 8.04 GiB | 12.92 ms | 668.43 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 398 | 40.91 ms | 40.91 ms | 0.000 |
| D2H | 629 | 2.08 ms | 2.08 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 337.50 MiB | 3.54 | 5.40 s |
| D2H | 14.06 MiB | 0.15 | 4.90 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `custom-call.2.0` | 80527 | 3230.40 | 0.78 | 12.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(eigvalsh))/jit(eigh)/ei` |
| `custom-call.1.0` | 4621 | 543.37 | 0.17 | 12.5 | `jit__build_fH` | `jit(_build_fH)/jit(main)/kim` |
| `all-gather-start` | 12 | 127.11 | 72.78 | 0 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `triangular-solve.4.0` | 174 | 62.76 | 1.31 | 62.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `triangular-solve.5.0` | 174 | 62.57 | 1.30 | 62.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `all-to-all.2` | 10 | 33.01 | 12.86 | 0 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/add` |
| `all-reduce-start` | 3 | 21.34 | 21.08 | 0 | `jit__diag_stats` | `jit(_diag_stats)/jit(main)/reduce_min` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1::Params)` | 1 | 17.84 | 17.84 | 0 | `` | `` |
| `Memset 3` | 6639 | 11.08 | 0.01 | - | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1::Params)` | 1 | 10.49 | 10.49 | 0 | `` | `` |
| `input_transpose_fusion.2` | 10 | 7.26 | 1.04 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |
| `input_transpose_fusion.3` | 7 | 5.07 | 0.77 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(eigvalsh))/jit(eigh)` |
| `wrapped_transpose.1` | 6 | 4.25 | 0.71 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `input_transpose_fusion` | 17 | 4.08 | 0.53 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |
| `input_transpose_fusion.1` | 13 | 3.27 | 0.51 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |
| `loop_transpose_fusion` | 4 | 2.40 | 1.13 | 100 | `jit__build_fH` | `jit(_build_fH)/jit(main)/custom_partitioning` |
| `custom-call.6.0` | 6 | 2.08 | 0.35 | 12.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/dot_general` |
| `loop_transpose_fusion.1` | 3 | 1.93 | 1.14 | 100 | `jit__build_fH` | `jit(_build_fH)/jit(main)/custom_partitioning` |
| `custom-call.1` | 15 | 1.43 | 0.66 | 18.75 | `jit__post_kpath` | `jit(_post_kpath)/jit(main)/jit(sort)/sort` |
| `loop_complex_fusion` | 13 | 1.29 | 1.11 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start` | 0.0 % | 72776.6 | `jit(_diag_stats)/jit(main)/jit(eigvalsh)/jit(eigh)/eigh` |
| `all-reduce-start` | 0.0 % | 21078.5 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-gather-start` | 0.0 % | 20726.9 | `jit(reshape)/jit(main)/reshape` |
| `` | 0.0 % | 17836.3 | `` |
| `all-to-all.2` | 0.0 % | 12861.4 | `jit(_kpath_batch)/jit(main)/add` |
| `` | 0.0 % | 10485.1 | `` |
| `all-gather-start` | 0.0 % | 5972.2 | `` |
| `all-to-all.2` | 0.0 % | 5432.8 | `jit(_accum)/jit(main)/reshape` |
| `all-gather-start` | 0.0 % | 4550.9 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-gather-start` | 0.0 % | 4513.5 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-gather-start` | 0.0 % | 4502.9 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-gather-start` | 0.0 % | 4479.8 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-gather-start` | 0.0 % | 4471.3 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-to-all.2` | 0.0 % | 4418.9 | `jit(_kpath_batch)/jit(main)/add` |
| `all-gather-start` | 0.0 % | 4387.3 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `all-to-all.2` | 0.0 % | 2756.7 | `jit(_reshard_rchunk)/jit(main)/sharding_constraint` |
| `all-to-all.2` | 0.0 % | 1833.3 | `jit(_kpath_batch)/jit(main)/add` |
| `all-to-all.2` | 0.0 % | 1828.6 | `jit(_kpath_batch)/jit(main)/add` |
| `all-to-all.2` | 0.0 % | 1827.6 | `jit(_kpath_batch)/jit(main)/add` |
| `all-to-all.2` | 0.0 % | 1823.1 | `jit(_kpath_batch)/jit(main)/add` |

