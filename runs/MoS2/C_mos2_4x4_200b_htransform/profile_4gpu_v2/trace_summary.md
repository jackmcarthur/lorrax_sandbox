# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v2/xprof/rank_0/plugins/profile/2026_04_25_16_25_48/perfetto_trace.json.gz`
**Duration:** 22.092 s
**GPU streams:** 3 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 446 | 693.07 MiB | 41.62 ms | 17.46 |
| D2H | 628 | 28.23 MiB | 2.09 ms | 14.14 |
| D2D | 110 | 11.49 GiB | 18.53 ms | 665.77 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 446 | 41.62 ms | 41.62 ms | 0.000 |
| D2H | 628 | 2.09 ms | 2.09 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 337.50 MiB | 3.54 | 5.50 s |
| D2H | 14.06 MiB | 0.15 | 5.30 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `custom-call.1.0` | 85162 | 3766.51 | 0.81 | 12.5 | `jit_eigvalsh` | `jit(eigvalsh)/jit(main)/jit(eigh)/eigh` |
| `all-reduce-start` | 5 | 419.72 | 259.14 | 0 | `jit__reduce_max` | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-gather-start` | 6 | 126.62 | 109.69 | 0 | `jit_fft` | `jit(fft)/jit(main)/fft` |
| `triangular-solve.1.0` | 348 | 125.98 | 1.31 | 62.5 | `jit__solve_triangular` | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `input_transpose_fusion` | 35 | 20.76 | 1.06 | 56.25 | `jit_eigvalsh` | `jit(_solve_triangular)/jit(main)/reshape` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_nn_align1::Params)` | 1 | 17.83 | 17.83 | 0 | `` | `` |
| `all-to-all.2` | 3 | 15.53 | 9.96 | 0 | `jit__accum` | `jit(_accum)/jit(main)/reshape` |
| `Memset 3` | 6639 | 11.43 | 0.01 | - | `` | `` |
| `loop_complex_fusion` | 28 | 10.82 | 1.12 | 100 | `jit_conjugate` | `jit(conjugate)/jit(main)` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1::Params)` | 1 | 10.48 | 10.48 | 0 | `` | `` |
| `loop_multiply_fusion` | 35 | 10.32 | 0.77 | 100 | `jit_multiply` | `jit(multiply)/jit(main)` |
| `input_transpose_fusion.1` | 18 | 9.63 | 0.73 | 56.25 | `jit__solve_triangular` | `b` |
| `wrapped_transpose` | 41 | 9.22 | 0.72 | 56.25 | `jit_transpose` | `a` |
| `wrapped_add` | 15 | 7.76 | 1.19 | 1.5625 | `jit_scan` | `jit(scan)/jit(main)/while` |
| `all-gather-start.1` | 3 | 5.91 | 5.51 | 0 | `jit_fft` | `jit(fft)/jit(main)/fft` |
| `loop_transpose_fusion` | 4 | 2.45 | 1.13 | 100 | `jit_fft` | `jit(fft)/jit(main)/fft` |
| `fft.0.0` | 5 | 2.08 | 0.52 | 93.75 | `jit_fft` | `jit(fft)/jit(main)/fft` |
| `loop_transpose_fusion.1` | 3 | 1.81 | 1.14 | 100 | `jit_fft` | `x` |
| `custom-call.1` | 15 | 1.43 | 0.66 | 18.75 | `jit_sort` | `jit(sort)/jit(main)/sort` |
| `fft.2.0` | 2 | 1.04 | 0.52 | 46.875 | `jit__fft_and_rslice` | `jit(_fft_and_rslice)/jit(main)/custom_partitioning` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 259143.6 | `jit(_reduce_min)/jit(main)/reduce_min` |
| `all-reduce-start` | 0.0 % | 151767.7 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-gather-start` | 0.0 % | 109687.9 | `jit(fft)/jit(main)/fft` |
| `` | 0.0 % | 17829.2 | `` |
| `` | 0.0 % | 10484.2 | `` |
| `all-to-all.2` | 0.0 % | 9955.3 | `jit(_reshard_rchunk)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 8552.5 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-gather-start` | 0.0 % | 6954.1 | `` |
| `all-gather-start.1` | 0.0 % | 5509.8 | `jit(fft)/jit(main)/fft` |
| `all-to-all.2` | 0.0 % | 5444.1 | `jit(_accum)/jit(main)/reshape` |
| `all-gather-start` | 0.0 % | 4961.3 | `jit(cholesky)/jit(main)/cholesky` |
| `all-gather-start` | 0.0 % | 4680.5 | `jit(eigvalsh)/jit(main)/jit(eigh)/eigh` |
| `triangular-solve.1.0` | 0.0 % | 1306.7 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1306.6 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1306.1 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1306.0 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1305.9 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1305.8 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1301.7 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1301.4 | `jit(_solve_triangular)/jit(main)/triangular_solve` |

