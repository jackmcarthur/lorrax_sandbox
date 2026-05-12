# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/51_ffi_async_profile_C/profile/xprof/rank_0/plugins/profile/2026_04_18_17_03_46/perfetto_trace.json.gz`
**Duration:** 49.337 s
**GPU streams:** 5 compute, 5 H2D, 7 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 1245 | 8.58 GiB | 528.01 ms | 17.45 |
| D2H | 437 | 5.35 GiB | 218.88 ms | 26.24 |
| D2D | 987 | 14.14 GiB | 23.45 ms | 647.51 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 1245 | 528.01 ms | 528.01 ms | 0.000 |
| D2H | 437 | 218.88 ms | 218.88 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1.17 GiB | 12.51 | 22.90 s |
| D2H | 256.00 MiB | 2.68 | 25.80 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 63 | 349.32 | 97.47 | 0 | `jit__prepare_sigma_state` | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `all-gather-start` | 18 | 288.66 | 122.40 | 0 | `jit__identity_fn` | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `custom-call.1.0` | 14 | 194.95 | 61.98 | 12.5 | `jit__single_chunk_proc` | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_genera` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 830 | 155.00 | 0.21 | 12.5 | `` | `` |
| `loop_transpose_fusion.1` | 314 | 65.70 | 0.85 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.1.0` | 318 | 57.46 | 2.77 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_convert_fusion` | 299 | 55.90 | 0.21 | 1.5625 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `reduce-scatter.14` | 276 | 53.49 | 0.66 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `loop_transpose_fusion.7` | 280 | 52.31 | 0.19 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion` | 326 | 50.99 | 0.81 | 100 | `jit__identity_fn` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `all-gather-start.2` | 2 | 42.94 | 42.43 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/slice` |
| `loop_transpose_fusion.2` | 310 | 41.85 | 0.44 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)` |
| `loop_transpose_fusion.6` | 280 | 38.41 | 0.14 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.0.0` | 315 | 33.72 | 0.42 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.2.0` | 309 | 31.72 | 0.40 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion_8` | 278 | 27.99 | 0.10 | 100 | `` | `` |
| `fft.7.0` | 280 | 25.95 | 0.10 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.6.0` | 280 | 25.80 | 0.10 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.8.0` | 280 | 25.26 | 0.09 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `reduce-scatter.15` | 276 | 21.93 | 1.99 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start` | 0.0 % | 122404.3 | `jit(fft)/jit(main)/fft` |
| `all-reduce-start` | 0.0 % | 97467.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 89545.0 | `jit(_mean)/jit(main)/reduce_sum` |
| `custom-call.1.0` | 12.5 % | 61984.4 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 61378.0 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 61340.5 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `all-gather-start` | 0.0 % | 60617.7 | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-gather-start` | 0.0 % | 47129.8 | `jit(_reshard_rchunk)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 44908.6 | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `all-gather-start.2` | 0.0 % | 42427.2 | `jit(_solve_w)/jit(main)/slice` |
| `all-reduce-start` | 0.0 % | 27120.0 | `jit(trace)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 20772.2 | `jit(gather)/jit(main)/gather` |
| `all-reduce-start` | 0.0 % | 17211.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.1` | 0.0 % | 12488.6 | `jit(fft)/jit(main)/fft` |
| `all-reduce-start` | 0.0 % | 11751.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 11095.1 | `jit(gather)/jit(main)/gather` |
| `all-reduce-start` | 0.0 % | 10668.0 | `jit(hartree)/jit(main)/xy` |
| `all-reduce-start` | 0.0 % | 8313.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 7492.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 6569.3 | `jit(_psum)/jit(main)/reduce_sum` |

