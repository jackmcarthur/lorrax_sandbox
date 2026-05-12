# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/30_profile_kmeans_isdf/profile/xprof/plugins/profile/2026_04_16_15_58_41/perfetto_trace.json.gz`
**Duration:** 6.354 s
**GPU streams:** 0 compute, 2 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 2322 | 7.40 MiB | 107.49 ms | 0.07 |
| D2H | 421 | 71.55 MiB | 3.47 ms | 21.59 |
| D2D | 2714 | 21.78 MiB | 4.62 ms | 4.94 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 2322 | 107.49 ms | 107.49 ms | 0.000 |
| D2H | 421 | 3.47 ms | 3.47 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 5.86 MiB | 0.06 | 1.60 s |
| D2H | 8.54 MiB | 0.09 | 4.90 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `loop_dynamic_slice_fusion` | 35039 | 66.29 | 0.00 | 100 | `jit_scatter` | `jit(scatter)/jit(main)/scatter` |
| `loop_dynamic_update_slice_fusion` | 34160 | 61.31 | 0.00 | 100 | `jit_scatter` | `jit(scatter)/jit(main)/scatter` |
| `loop_add_fusion` | 33811 | 51.84 | 0.00 | 56.25 | `jit__linspace` | `jit(_linspace)/jit(main)` |
| `void magma_sgemmEx_kernel<float, float, float, false, true, 6, 4, 6, 3, 4>(int, int, int, Tensor, int, Tensor, int, Tensor, int, Tensor, int, int, int, float const*, float const*, float, float, int, cublasLtEpilogue_t, int, void const*, long)` | 288 | 28.73 | 0.10 | 43.75 | `` | `` |
| `custom-call.1.0` | 417 | 6.21 | 0.10 | 50 | `jit_pbc_distance_sq_batch` | `jit(pbc_distance_sq_batch)/jit(main)/pki` |
| `input_transpose_fusion_1` | 16 | 5.97 | 0.38 | 100 | `` | `` |
| `input_transpose_fusion` | 16 | 3.53 | 0.22 | 100 | `` | `` |
| `input_reduce_fusion` | 450 | 2.28 | 0.09 | 100 | `jit__argmin` | `jit(_reduce_max)/jit(main)/reduce_max` |
| `loop_subtract_fusion` | 400 | 1.26 | 0.15 | 100 | `jit_pbc_distance_sq_batch` | `jit(pbc_distance_sq_batch)/jit(main)` |
| `loop_reduce_fusion` | 416 | 1.24 | 0.23 | 100 | `jit_pbc_distance_sq_batch` | `jit(pbc_distance_sq_batch)/jit(main)` |
| `input_reduce_fusion_1` | 16 | 1.01 | 0.06 | 100 | `` | `` |
| `wrapped_multiply` | 415 | 0.93 | 0.00 | 100 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `wrapped_minimum` | 399 | 0.87 | 0.00 | 100 | `jit_minimum` | `jit(minimum)/jit(main)/min` |
| `input_compare_reduce_fusion` | 16 | 0.68 | 0.04 | 100 | `` | `` |
| `fft.2.0` | 64 | 0.19 | 0.00 | 93.75 | `jit_fft` | `jit(fft)/jit(main)/fft` |
| `wrapped_select` | 48 | 0.10 | 0.00 | 100 | `jit_select_n` | `jit(select_n)/jit(main)/select_n` |
| `loop_compare_fusion` | 48 | 0.09 | 0.00 | 100 | `jit_less` | `jit(less)/jit(main)` |
| `wrapped_convert` | 33 | 0.08 | 0.00 | 1.5625 | `jit_convert_element_type` | `jit(convert_element_type)/jit(main)/convert_element_type` |
| `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1::Params)` | 16 | 0.07 | 0.00 | 18.75 | `` | `` |
| `wrapped_transpose` | 17 | 0.06 | 0.00 | 100 | `jit_transpose` | `` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `` | 43.8 % | 101.2 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.1 | `` |
| `` | 43.8 % | 101.0 | `` |
| `` | 43.8 % | 101.0 | `` |
| `` | 43.8 % | 101.0 | `` |
| `` | 43.8 % | 101.0 | `` |
| `` | 43.8 % | 101.0 | `` |
| `` | 43.8 % | 101.0 | `` |
| `` | 43.8 % | 101.0 | `` |
| `` | 43.8 % | 101.0 | `` |

