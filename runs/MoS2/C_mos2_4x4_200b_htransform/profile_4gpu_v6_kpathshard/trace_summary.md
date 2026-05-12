# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v6_kpathshard/xprof/rank_0/plugins/profile/2026_04_25_20_39_51/perfetto_trace.json.gz`
**Duration:** 12.562 s
**GPU streams:** 5 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 422 | 693.08 MiB | 39.78 ms | 18.27 |
| D2H | 198 | 28.23 MiB | 1.44 ms | 20.60 |
| D2D | 69 | 6.04 GiB | 9.76 ms | 664.50 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 422 | 39.78 ms | 39.78 ms | 0.000 |
| D2H | 198 | 1.44 ms | 1.44 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 337.50 MiB | 3.54 | 5.40 s |
| D2H | 14.06 MiB | 0.15 | 5.30 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `custom-call.2.0` | 20449 | 835.29 | 0.78 | 12.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(eigvalsh))/jit(eigh)/ei` |
| `custom-call.1.0` | 4621 | 543.42 | 0.17 | 12.5 | `jit__build_fH` | `jit(_build_fH)/jit(main)/kim` |
| `all-gather-start` | 12 | 82.99 | 66.48 | 0 | `jit__diag_stats` | `jit(_diag_stats)/jit(main)/jit(eigvalsh)/jit(eigh)/eigh` |
| `triangular-solve.4.0` | 174 | 62.76 | 1.31 | 62.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `triangular-solve.5.0` | 174 | 62.55 | 1.30 | 62.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/tri` |
| `all-reduce-start` | 3 | 48.19 | 47.93 | 0 | `jit__diag_stats` | `jit(_diag_stats)/jit(main)/reduce_min` |
| `all-gather-start.1` | 8 | 21.37 | 9.29 | 0 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/bk` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1::Params)` | 1 | 17.83 | 17.83 | 0 | `` | `` |
| `all-gather-start.2` | 6 | 13.45 | 3.23 | 0 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/res` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1::Params)` | 1 | 10.49 | 10.49 | 0 | `` | `` |
| `all-to-all.2` | 3 | 8.31 | 5.42 | 0 | `jit__accum` | `jit(_accum)/jit(main)/reshape` |
| `input_transpose_fusion` | 16 | 5.04 | 0.72 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/res` |
| `loop_transpose_fusion` | 11 | 4.79 | 1.13 | 100 | `jit__post_kpath` | `batches[5]` |
| `Memset 3` | 1894 | 3.07 | 0.01 | - | `` | `` |
| `all-reduce-start.1` | 1 | 2.39 | 2.39 | 0 | `jit__diag_stats` | `jit(_diag_stats)/jit(main)/reduce_max` |
| `input_transpose_fusion.1` | 13 | 2.26 | 0.51 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |
| `input_transpose_fusion.2` | 10 | 2.11 | 0.51 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/bk` |
| `loop_transpose_fusion.1` | 4 | 1.93 | 1.14 | 100 | `jit__post_kpath` | `batches[4]` |
| `custom-call.6.0` | 6 | 1.92 | 0.32 | 12.5 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)/bk` |
| `input_transpose_fusion.4` | 6 | 1.53 | 0.26 | 56.25 | `jit__kpath_batch` | `jit(_kpath_batch)/jit(main)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start` | 0.0 % | 66476.5 | `jit(_diag_stats)/jit(main)/jit(eigvalsh)/jit(eigh)/eigh` |
| `all-reduce-start` | 0.0 % | 47931.7 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `` | 0.0 % | 17830.2 | `` |
| `` | 0.0 % | 10485.4 | `` |
| `all-gather-start.1` | 0.0 % | 9293.7 | `jit(_kpath_batch)/jit(main)/bk` |
| `all-gather-start` | 0.0 % | 6191.5 | `` |
| `all-to-all.2` | 0.0 % | 5423.8 | `jit(_accum)/jit(main)/reshape` |
| `all-gather-start.2` | 0.0 % | 3231.7 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `all-gather-start.1` | 0.0 % | 2952.1 | `jit(_kpath_batch)/jit(main)/bk` |
| `all-to-all.2` | 0.0 % | 2743.1 | `jit(_reshard_rchunk)/jit(main)/sharding_constraint` |
| `all-reduce-start.1` | 0.0 % | 2391.3 | `jit(_diag_stats)/jit(main)/reduce_max` |
| `all-gather-start.1` | 0.0 % | 2250.5 | `jit(_kpath_batch)/jit(main)/bk` |
| `all-gather-start.1` | 0.0 % | 2189.7 | `jit(_kpath_batch)/jit(main)/bk` |
| `all-gather-start.2` | 0.0 % | 2157.9 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `all-gather-start.1` | 0.0 % | 2156.9 | `jit(_kpath_batch)/jit(main)/bk` |
| `all-gather-start.1` | 0.0 % | 2129.5 | `jit(_kpath_batch)/jit(main)/bk` |
| `all-gather-start.2` | 0.0 % | 2079.4 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `all-gather-start.2` | 0.0 % | 2024.5 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `all-gather-start.2` | 0.0 % | 1986.1 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |
| `all-gather-start.2` | 0.0 % | 1972.6 | `jit(_kpath_batch)/jit(main)/vmap(jit(_solve_triangular))/reshape` |

