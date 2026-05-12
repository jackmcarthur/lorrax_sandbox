# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/08_lorrax_gnppm_profile_C/profile/xprof/rank_0/plugins/profile/2026_04_17_19_31_46/perfetto_trace.json.gz`
**Duration:** 38.665 s
**GPU streams:** 4 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 2687 | 4.55 GiB | 564.48 ms | 8.66 |
| D2H | 442 | 4.34 GiB | 177.71 ms | 26.20 |
| D2D | 4023 | 9.74 GiB | 21.53 ms | 485.98 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 2687 | 564.48 ms | 564.48 ms | 0.000 |
| D2H | 442 | 177.71 ms | 177.71 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 720.16 MiB | 7.55 | 19.30 s |
| D2H | 256.00 MiB | 2.68 | 9.10 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 360 | 2282.02 | 1616.40 | 0 | `jit__batched_chol` | `jit(_batched_chol)/jit(main)/while/body/jit(shmap_body)/whil` |
| `all-gather-start` | 14 | 223.10 | 90.70 | 0 | `jit__identity_fn` | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `custom-call.1.0` | 8 | 196.43 | 62.86 | 12.5 | `jit__single_chunk_proc` | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_genera` |
| `custom-call.19.0` | 276 | 111.69 | 0.44 | 12.5 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `all-reduce-start.1` | 298 | 60.87 | 9.00 | 0 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `loop_transpose_fusion.2` | 301 | 58.79 | 1.73 | 100 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `fft.1.0` | 309 | 56.20 | 2.77 | 50 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `input_transpose_fusion.2` | 299 | 54.18 | 0.38 | 100 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `loop_transpose_fusion.8` | 278 | 51.52 | 0.19 | 100 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `loop_transpose_fusion.1` | 302 | 51.41 | 3.37 | 100 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 278 | 48.18 | 0.18 | 12.5 | `` | `` |
| `loop_transpose_fusion.3` | 297 | 41.12 | 3.20 | 100 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `loop_transpose_fusion.7` | 280 | 38.70 | 0.19 | 100 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `fft.0.0` | 306 | 32.35 | 1.64 | 50 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `fft.2.0` | 300 | 30.46 | 1.56 | 50 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `loop_transpose_fusion_9` | 276 | 27.74 | 0.10 | 100 | `` | `` |
| `fft.7.0` | 280 | 26.04 | 0.10 | 50 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `fft.6.0` | 280 | 25.81 | 0.10 | 50 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `fft.8.0` | 280 | 25.08 | 0.09 | 50 | `jit__tau_channel_step` | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline` |
| `all-gather-start.2` | 2 | 23.80 | 23.35 | 0 | `jit__solve_w` | `jit(_solve_w)/jit(main)/slice` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 1616400.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 131321.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 93748.9 | `jit(_reduce_sum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 90701.2 | `jit(fft)/jit(main)/fft` |
| `all-reduce-start` | 0.0 % | 78557.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 68249.3 | `jit(_mean)/jit(main)/reduce_sum` |
| `custom-call.1.0` | 12.5 % | 62861.5 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 61834.3 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `custom-call.1.0` | 12.5 % | 61727.7 | `jit(_single_chunk_proc)/jit(main)/vmap(mG,nG->mn)/dot_general` |
| `all-reduce-start` | 0.0 % | 50734.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 50047.7 | `jit(_solve_w)/jit(main)/jit(_pad)/pad` |
| `all-gather-start` | 0.0 % | 29035.2 | `` |
| `all-gather-start.2` | 0.0 % | 23351.1 | `jit(_solve_w)/jit(main)/slice` |
| `all-gather-start` | 0.0 % | 16836.1 | `jit(_reshard_rchunk)/jit(main)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 14806.3 | `jit(gather)/jit(main)/gather` |
| `all-reduce-start` | 0.0 % | 14549.3 | `jit(_reduce_min)/jit(main)/reduce_min` |
| `all-reduce-start` | 0.0 % | 14420.5 | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline)/kmsx` |
| `all-reduce-start` | 0.0 % | 13779.5 | `jit(trace)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 12182.5 | `jit(_tau_channel_step)/jit(main)/jit(_sigma_channel_pipeline)/kmsx` |
| `all-reduce-start` | 0.0 % | 11178.1 | `jit(sigma_sx)/jit(main)/kmsx` |

