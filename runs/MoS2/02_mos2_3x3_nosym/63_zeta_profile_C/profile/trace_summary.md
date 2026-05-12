# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/63_zeta_profile_C/profile/xprof/rank_0/plugins/profile/2026_04_19_01_21_22/perfetto_trace.json.gz`
**Duration:** 38.614 s
**GPU streams:** 5 compute, 5 H2D, 7 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 1245 | 8.58 GiB | 517.62 ms | 17.80 |
| D2H | 437 | 5.35 GiB | 218.94 ms | 26.23 |
| D2D | 987 | 14.14 GiB | 23.15 ms | 655.88 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 1245 | 517.62 ms | 517.62 ms | 0.000 |
| D2H | 437 | 218.94 ms | 218.94 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 1.04 GiB | 11.15 | 21.80 s |
| D2H | 256.00 MiB | 2.68 | 14.60 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 63 | 350.84 | 100.07 | 0 | `jit__prepare_sigma_state` | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `custom-call.1.0` | 14 | 197.16 | 62.47 | 12.5 | `jit__single_chunk_proc` | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_genera` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 830 | 154.71 | 0.21 | 12.5 | `` | `` |
| `all-reduce-start.1` | 22 | 92.28 | 51.51 | 0 | `jit_sigma_sx` | `jit(sigma_sx)/jit(main)/kmty` |
| `loop_transpose_fusion.1` | 314 | 66.13 | 0.86 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `all-gather-start` | 18 | 64.79 | 21.71 | 0 | `jit__identity_fn` | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `fft.1.0` | 318 | 57.66 | 2.77 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_convert_fusion` | 299 | 56.76 | 0.21 | 1.5625 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `reduce-scatter.14` | 276 | 55.57 | 0.84 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |
| `loop_transpose_fusion.7` | 280 | 52.72 | 0.19 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion` | 326 | 51.10 | 0.82 | 100 | `jit__identity_fn` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion.2` | 310 | 42.43 | 0.45 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)` |
| `loop_transpose_fusion.6` | 280 | 38.57 | 0.14 | 100 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.0.0` | 315 | 33.86 | 0.42 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.2.0` | 309 | 31.75 | 0.40 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `loop_transpose_fusion_8` | 278 | 28.11 | 0.10 | 100 | `` | `` |
| `fft.6.0` | 280 | 25.97 | 0.10 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.7.0` | 280 | 25.84 | 0.10 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `fft.8.0` | 280 | 25.17 | 0.09 | 50 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/custom_par` |
| `reduce-scatter.15` | 276 | 24.15 | 2.13 | 0 | `jit__tau_kernel` | `jit(_tau_kernel)/jit(main)/jit(_sigma_kij_kernel)/jit(shmap_` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 100074.2 | `jit(_psum)/jit(main)/reduce_sum` |
| `custom-call.1.0` | 12.5 % | 62473.5 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 62458.7 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 61950.1 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `all-reduce-start` | 0.0 % | 56160.9 | `jit(sigma_sx)/jit(main)/kmsx` |
| `all-reduce-start` | 0.0 % | 54439.3 | `jit(sigma_coh)/jit(main)/kmsx` |
| `all-reduce-start.1` | 0.0 % | 51508.8 | `jit(hartree)/jit(main)/kmsx` |
| `all-reduce-start.1` | 0.0 % | 38858.0 | `jit(sigma_sx)/jit(main)/kmty` |
| `all-reduce-start` | 0.0 % | 34358.9 | `jit(_prepare_sigma_state)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 26897.5 | `jit(_mean)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 22448.1 | `jit(trace)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 21712.4 | `jit(_reshard_rchunk)/jit(main)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 17245.4 | `jit(fft)/jit(main)/fft` |
| `all-reduce-start` | 0.0 % | 14318.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 10141.5 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 7781.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 6182.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 5967.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 5088.8 | `` |
| `all-gather-start` | 0.0 % | 3195.5 | `jit(_solve_all_at_once)/jit(main)/sharding_constraint` |

