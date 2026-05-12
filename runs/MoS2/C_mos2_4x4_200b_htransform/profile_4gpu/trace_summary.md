# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu/xprof/rank_0/plugins/profile/2026_04_25_13_20_07/perfetto_trace.json.gz`
**Duration:** 21.620 s
**GPU streams:** 2 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 364 | 690.90 MiB | 38.50 ms | 18.82 |
| D2H | 599 | 337.59 MiB | 15.57 ms | 22.73 |
| D2D | 139 | 12.16 GiB | 19.37 ms | 673.70 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 364 | 38.50 ms | 38.50 ms | 0.000 |
| D2H | 599 | 15.57 ms | 15.57 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 337.50 MiB | 3.54 | 1.80 s |
| D2H | 84.38 MiB | 0.88 | 1.50 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `custom-call.1.0` | 75966 | 3136.15 | 0.81 | 12.5 | `jit_eigvalsh` | `jit(eigvalsh)/jit(main)/jit(eigh)/eigh` |
| `custom-call.0.0` | 4858 | 551.21 | 0.17 | 1.5625 | `jit_svd` | `jit(svd)/jit(main)/svd` |
| `all-gather-start` | 4 | 339.51 | 281.73 | 0 | `jit__identity_fn` | `` |
| `triangular-solve.1.0` | 348 | 119.12 | 1.31 | 62.5 | `jit__solve_triangular` | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `all-reduce-start` | 1 | 103.34 | 103.34 | 0 | `jit__accum_G` | `jit(_accum_G)/jit(main)/jit(shmap_body)` |
| `custom-call.3.0` | 1 | 37.79 | 37.79 | 0 | `jit__accum_G` | `jit(_accum_G)/jit(main)/jit(shmap_body)/dot_general` |
| `input_transpose_fusion` | 31 | 19.28 | 1.06 | 56.25 | `jit_eigvalsh` | `jit(_solve_triangular)/jit(main)/reshape` |
| `loop_multiply_fusion` | 57 | 12.43 | 2.22 | 100 | `jit_multiply` | `jit(multiply)/jit(main)` |
| `loop_complex_fusion` | 26 | 11.08 | 2.23 | 100 | `jit_conjugate` | `jit(conjugate)/jit(main)` |
| `Memset 3` | 6241 | 10.41 | 0.01 | - | `` | `` |
| `custom-call.2.0` | 1 | 9.26 | 9.26 | 0 | `jit__accum_G` | `jit(_accum_G)/jit(main)/jit(shmap_body)/krb` |
| `wrapped_transpose` | 41 | 8.45 | 0.72 | 56.25 | `jit_transpose` | `a` |
| `input_transpose_fusion.1` | 14 | 8.09 | 0.73 | 56.25 | `jit__solve_triangular` | `b` |
| `wrapped_add` | 13 | 6.86 | 1.18 | 1.5625 | `jit_scan` | `jit(scan)/jit(main)/while` |
| `fft.2.0` | 48 | 2.08 | 0.06 | 93.75 | `jit_fft` | `jit(fft)/jit(main)/fft` |
| `custom-call.1` | 15 | 1.43 | 0.66 | 18.75 | `jit_sort` | `jit(sort)/jit(main)/sort` |
| `fft.0.0` | 3 | 1.04 | 0.35 | 93.75 | `jit_fft` | `jit(fft)/jit(main)/fft` |
| `loop_reduce_fusion` | 3 | 0.69 | 0.69 | 100 | `jit_norm` | `jit(norm)/jit(main)` |
| `loop_subtract_fusion` | 9 | 0.55 | 0.54 | 100 | `jit_subtract` | `jit(subtract)/jit(main)/sub` |
| `wrapped_real` | 3 | 0.54 | 0.27 | 100 | `jit_real` | `jit(real)/jit(main)/real` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start` | 0.0 % | 281733.9 | `` |
| `all-reduce-start` | 0.0 % | 103339.7 | `jit(_accum_G)/jit(main)/jit(shmap_body)` |
| `all-gather-start` | 0.0 % | 56958.1 | `` |
| `custom-call.3.0` | 0.0 % | 37786.3 | `jit(_accum_G)/jit(main)/jit(shmap_body)/dot_general` |
| `custom-call.2.0` | 0.0 % | 9258.6 | `jit(_accum_G)/jit(main)/jit(shmap_body)/krb` |
| `triangular-solve.1.0` | 0.0 % | 1307.1 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1306.7 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1306.7 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1306.4 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1306.3 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1302.7 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1302.6 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1302.1 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1301.8 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1300.5 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1097.8 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1097.4 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1097.1 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1097.0 | `jit(_solve_triangular)/jit(main)/triangular_solve` |
| `triangular-solve.1.0` | 0.0 % | 1096.6 | `jit(_solve_triangular)/jit(main)/triangular_solve` |

