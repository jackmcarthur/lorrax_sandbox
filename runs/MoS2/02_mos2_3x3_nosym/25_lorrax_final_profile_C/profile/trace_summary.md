# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/25_lorrax_final_profile_C/profile/xprof/rank_0/plugins/profile/2026_04_18_00_17_19/perfetto_trace.json.gz`
**Duration:** 37.689 s
**GPU streams:** 4 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 1872 | 4.62 GiB | 559.67 ms | 8.87 |
| D2H | 450 | 4.48 GiB | 183.36 ms | 26.22 |
| D2D | 2664 | 9.23 GiB | 18.36 ms | 539.75 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 1872 | 559.67 ms | 559.67 ms | 0.000 |
| D2H | 450 | 183.36 ms | 183.36 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 751.83 MiB | 7.88 | 19.30 s |
| D2H | 256.00 MiB | 2.68 | 9.50 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 87 | 2212.01 | 1662.44 | 0 | `jit__prepare_sigma_state` | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `all-gather-start` | 16 | 223.13 | 78.21 | 0 | `jit__identity_fn` | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `custom-call.1.0` | 8 | 197.57 | 62.74 | 12.5 | `jit__single_chunk_proc` | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_genera` |
| `custom-call.27.0` | 276 | 111.52 | 0.43 | 12.5 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `reduce-scatter.4.1` | 276 | 75.55 | 10.11 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `loop_transpose_fusion.1` | 302 | 65.98 | 3.37 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.1.0` | 309 | 56.28 | 2.77 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `input_transpose_fusion.4` | 276 | 53.63 | 0.20 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `loop_transpose_fusion.7` | 280 | 52.18 | 0.19 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion` | 315 | 51.07 | 3.21 | 100 | `jit__identity_fn` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 278 | 48.43 | 0.18 | 12.5 | `` | `` |
| `loop_transpose_fusion.2` | 301 | 41.58 | 1.73 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)` |
| `loop_transpose_fusion.6` | 280 | 38.45 | 0.14 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `reduce-scatter.5.1` | 276 | 34.71 | 7.59 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `all-reduce-start.1` | 22 | 32.41 | 18.65 | 0 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/kmty` |
| `fft.0.0` | 306 | 32.41 | 1.63 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.2.0` | 300 | 30.51 | 1.56 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion_8` | 278 | 27.81 | 0.10 | 100 | `` | `` |
| `fft.7.0` | 280 | 26.24 | 0.10 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.6.0` | 280 | 25.96 | 0.10 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 1662444.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 150413.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 82047.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 78208.2 | `jit(fft)/jit(main)/fft` |
| `all-reduce-start` | 0.0 % | 71840.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 66829.7 | `jit(_mean)/jit(main)/reduce_sum` |
| `custom-call.1.0` | 12.5 % | 62735.1 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 62674.3 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 62110.5 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `all-gather-start` | 0.0 % | 52041.3 | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-reduce-start` | 0.0 % | 41894.6 | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 27998.7 | `` |
| `all-reduce-start` | 0.0 % | 26769.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.2` | 0.0 % | 25054.8 | `jit(_solve_w)/jit(main)/slice` |
| `all-reduce-start` | 0.0 % | 23506.8 | `jit(trace)/jit(main)/reduce_sum` |
| `all-reduce-start.1` | 0.0 % | 18649.4 | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/while/body` |
| `all-gather-start` | 0.0 % | 18611.5 | `jit(gather)/jit(main)/gather` |
| `all-reduce-start` | 0.0 % | 17143.1 | `jit(_reduce_min)/jit(main)/reduce_min` |
| `all-reduce-start` | 0.0 % | 16937.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 10822.2 | `jit(_psum)/jit(main)/reduce_sum` |

