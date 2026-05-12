# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/27_lorrax_enk_numpy_C/profile/xprof/rank_0/plugins/profile/2026_04_18_00_29_00/perfetto_trace.json.gz`
**Duration:** 35.621 s
**GPU streams:** 4 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 1275 | 4.62 GiB | 534.97 ms | 9.28 |
| D2H | 450 | 4.48 GiB | 183.46 ms | 26.21 |
| D2D | 975 | 9.11 GiB | 15.43 ms | 633.93 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 1275 | 534.97 ms | 534.97 ms | 0.000 |
| D2H | 450 | 183.46 ms | 183.46 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 797.69 MiB | 8.36 | 18.10 s |
| D2H | 256.00 MiB | 2.68 | 8.40 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 87 | 2260.61 | 1749.61 | 0 | `jit__prepare_sigma_state` | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `all-gather-start` | 16 | 231.49 | 90.73 | 0 | `jit__identity_fn` | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `custom-call.1.0` | 8 | 195.36 | 62.82 | 12.5 | `jit__single_chunk_proc` | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_genera` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 830 | 154.51 | 0.21 | 12.5 | `` | `` |
| `loop_transpose_fusion.1` | 302 | 65.80 | 3.36 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_convert_fusion` | 320 | 56.51 | 0.21 | 1.5625 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `fft.1.0` | 309 | 56.44 | 2.77 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion.7` | 280 | 52.52 | 0.19 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `reduce-scatter.14` | 276 | 51.69 | 2.52 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `loop_transpose_fusion` | 315 | 50.95 | 3.19 | 100 | `jit__identity_fn` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion.2` | 301 | 42.06 | 1.74 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)` |
| `reduce-scatter.15` | 276 | 39.39 | 9.33 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `loop_transpose_fusion.6` | 280 | 38.50 | 0.14 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.0.0` | 306 | 32.60 | 1.64 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.2.0` | 300 | 30.53 | 1.56 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion_8` | 278 | 28.09 | 0.10 | 100 | `` | `` |
| `fft.7.0` | 280 | 25.92 | 0.10 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.6.0` | 280 | 25.91 | 0.10 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `all-gather-start.2` | 2 | 25.63 | 25.13 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/slice` |
| `fft.8.0` | 280 | 25.22 | 0.09 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 1749606.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 90730.0 | `jit(fft)/jit(main)/fft` |
| `all-reduce-start` | 0.0 % | 80497.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 77001.8 | `jit(_mean)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 76226.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `custom-call.1.0` | 12.5 % | 62817.9 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 61826.4 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 60716.0 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `all-gather-start` | 0.0 % | 54026.8 | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-reduce-start` | 0.0 % | 50449.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 41535.4 | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 34810.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 31082.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 30722.7 | `` |
| `all-gather-start.2` | 0.0 % | 25131.3 | `jit(_solve_w)/jit(main)/slice` |
| `all-reduce-start` | 0.0 % | 24522.0 | `jit(trace)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 18508.0 | `jit(gather)/jit(main)/gather` |
| `all-to-all.2` | 0.0 % | 15751.0 | `jit(_reshard_rchunk)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 15139.1 | `jit(hartree)/jit(main)/xy` |
| `all-reduce-start` | 0.0 % | 12344.4 | `jit(_psum)/jit(main)/reduce_sum` |

