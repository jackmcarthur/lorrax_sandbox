# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v9_matmul/xprof/rank_0/plugins/profile/2026_04_25_20_47_33/perfetto_trace.json.gz`
**Duration:** 12.961 s
**GPU streams:** 5 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 425 | 693.08 MiB | 39.55 ms | 18.38 |
| D2H | 198 | 28.23 MiB | 1.43 ms | 20.67 |
| D2D | 69 | 6.04 GiB | 9.79 ms | 662.51 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 425 | 39.55 ms | 39.55 ms | 0.000 |
| D2H | 198 | 1.43 ms | 1.43 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 337.50 MiB | 3.54 | 5.70 s |
| D2H | 14.06 MiB | 0.15 | 5.10 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `custom-call.2.0` | 20449 | 839.57 | 0.78 | 12.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(eigvalsh))/jit(eigh)/ei` |
| `custom-call.1.0` | 4621 | 543.20 | 0.17 | 12.5 | `jit__build_fH` | `jit(_build_fH)/jit(main)/dot_general` |
| `all-gather-start` | 13 | 128.70 | 43.97 | 0 | `jit__diag_stats` | `jit(_diag_stats)/jit(main)/jit(eigvalsh)/jit(eigh)/eigh` |
| `all-reduce-start.1` | 1 | 84.91 | 84.91 | 0 | `jit__diag_stats` | `jit(_diag_stats)/jit(main)/reduce_max` |
| `all-gather-start.1` | 3 | 82.54 | 76.18 | 0 | `jit__diag_stats` | `jit(_diag_stats)/jit(main)/jit(eigvalsh)/jit(eigh)/eigh` |
| `triangular-solve.4.0` | 174 | 63.14 | 1.31 | 62.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `triangular-solve.5.0` | 174 | 62.94 | 1.30 | 62.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1::Params)` | 1 | 17.80 | 17.80 | 0 | `` | `` |
| `all-to-all.2` | 3 | 10.70 | 5.41 | 0 | `jit__accum` | `jit(_accum)/jit(main)/reshape` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1::Params)` | 1 | 10.48 | 10.48 | 0 | `` | `` |
| `input_transpose_fusion` | 17 | 5.48 | 0.77 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/res` |
| `all-reduce-start` | 3 | 3.22 | 2.98 | 0 | `jit__diag_stats` | `jit(_diag_stats)/jit(main)/reduce_min` |
| `Memset 3` | 1889 | 3.05 | 0.01 | - | `` | `` |
| `loop_transpose_fusion` | 6 | 2.78 | 1.13 | 100 | `jit__post_kpath` | `batches[5]` |
| `input_transpose_fusion.2` | 10 | 2.63 | 0.51 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 6 | 2.00 | 0.34 | 12.5 | `` | `` |
| `loop_transpose_fusion.1` | 4 | 1.93 | 1.14 | 100 | `jit__post_kpath` | `batches[4]` |
| `input_transpose_fusion.3` | 7 | 1.65 | 0.51 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(eigvalsh))/jit(eigh)` |
| `input_transpose_fusion.1` | 8 | 1.53 | 0.51 | 31.25 | `jit__gamma_rt` | `jit(_gamma_rt)/jit(main)` |
| `custom-call.1` | 15 | 1.40 | 0.66 | 18.75 | `jit__post_kpath` | `jit(_post_kpath)/jit(main)/jit(sort)/sort` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start.1` | 0.0 % | 84914.7 | `jit(_diag_stats)/jit(main)/reduce_max` |
| `all-gather-start.1` | 0.0 % | 76177.3 | `` |
| `all-gather-start` | 0.0 % | 43974.0 | `jit(_diag_stats)/jit(main)/jit(eigvalsh)/jit(eigh)/eigh` |
| `all-gather-start` | 0.0 % | 34120.5 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `` | 0.0 % | 17798.3 | `` |
| `all-gather-start` | 0.0 % | 14255.1 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `all-gather-start` | 0.0 % | 11317.4 | `jit(cholesky)/jit(main)/cholesky` |
| `` | 0.0 % | 10482.5 | `` |
| `all-gather-start` | 0.0 % | 10470.4 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `all-gather-start.1` | 0.0 % | 6161.1 | `jit(cholesky)/jit(main)/cholesky` |
| `all-gather-start` | 0.0 % | 6159.9 | `` |
| `all-to-all.2` | 0.0 % | 5412.7 | `jit(_accum)/jit(main)/reshape` |
| `all-to-all.2` | 0.0 % | 5168.6 | `jit(_reshard_rchunk)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 2984.1 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-gather-start` | 0.0 % | 2764.8 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `all-gather-start` | 0.0 % | 2070.3 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `all-gather-start` | 0.0 % | 1899.3 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `triangular-solve.4.0` | 0.0 % | 1308.5 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `triangular-solve.4.0` | 0.0 % | 1307.8 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |
| `triangular-solve.4.0` | 0.0 % | 1307.7 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/triangular_solve` |

