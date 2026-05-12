# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/01_mos2_3x3_d3h_2026-05-09/A_bispinor_smoke/v_q_profile_120825/V_q_compute_bispinor-20260511-120947-p3/plugins/profile/2026_05_11_12_10_19/nid001108.trace.json.gz`
**Duration:** 30.338 s
**GPU streams:** 2 compute, 5 H2D, 11 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 205 | 6.85 GiB | 275.02 ms | 26.76 |
| D2H | 94 | 50.55 MiB | 2.20 ms | 24.08 |
| D2D | 82 | 4.18 MiB | 0.16 ms | 27.66 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 205 | 275.02 ms | 275.02 ms | 0.000 |
| D2H | 94 | 2.20 ms | 2.20 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 923.91 MiB | 9.69 | 5.00 s |
| D2H | 11.71 MiB | 0.12 | 5.40 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 39 | 264.93 | 105.11 | 0 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 7 | 14.79 | 4.14 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `fft.2.0` | 12 | 13.73 | 1.43 | 46.875 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `input_transpose_fusion` | 7 | 11.06 | 2.14 | 25 | `jit__kernel` | `jit(_kernel)/jit(main)` |
| `fft.5.0` | 9 | 9.75 | 1.14 | 46.875 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `fft.4.0` | 9 | 9.45 | 1.05 | 46.875 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `all-gather-start.1` | 7 | 6.75 | 1.21 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `custom-call.1.0` | 13 | 5.58 | 1.22 | 12.5 | `jit__kernel` | `jit(_kernel)/jit(main)/qmG` |
| `loop_complex_fusion` | 21 | 1.11 | 0.22 | 100 | `jit__kernel` | `jit(_kernel)/jit(main)` |
| `collective-permute-start` | 7 | 1.05 | 0.19 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `loop_gather_fusion` | 21 | 0.55 | 0.09 | 100 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_take)/gather` |
| `loop_gather_fusion.1` | 3 | 0.19 | 0.07 | 100 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_take)/gather` |
| `all-gather-start.2` | 4 | 0.14 | 0.04 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/squeeze` |
| `loop_broadcast_fusion` | 15 | 0.10 | 0.02 | 100 | `jit__zeros` | `jit(_broadcast_arrays)/jit(main)/broadcast_in_dim` |
| `wrapped_multiply` | 25 | 0.06 | 0.00 | 100 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `loop_add_fusion` | 21 | 0.05 | 0.00 | 100 | `jit_get_sqrt_v_and_phase` | `jit(get_sqrt_v_and_phase)/jit(main)` |
| `loop_dynamic_slice_fusion` | 12 | 0.03 | 0.00 | 100 | `jit_dynamic_slice` | `` |
| `loop_compare_fusion` | 7 | 0.02 | 0.00 | 100 | `jit_greater` | `jit(greater)/jit(main)/gt` |
| `wrapped_divide` | 6 | 0.01 | 0.00 | 100 | `jit_true_divide` | `jit(true_divide)/jit(main)/div` |
| `loop_reduce_fusion` | 6 | 0.01 | 0.00 | 100 | `jit__reduce_sum` | `jit(_reduce_sum)/jit(main)/reduce_sum` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 105106.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 43287.7 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 41390.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 27804.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 25518.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 13795.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 4140.3 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 3422.6 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 3206.1 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 2654.1 | `jit(_psum)/jit(main)/reduce_sum` |
| `input_transpose_fusion` | 25.0 % | 2139.5 | `jit(_kernel)/jit(main)` |
| `input_transpose_fusion` | 25.0 % | 2139.2 | `jit(_kernel)/jit(main)` |
| `input_transpose_fusion` | 25.0 % | 2129.8 | `jit(_kernel)/jit(main)` |
| `all-reduce-start` | 0.0 % | 1760.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `fft.2.0` | 46.9 % | 1430.6 | `jit(_kernel)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `all-gather-start` | 0.0 % | 1247.5 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `custom-call.1.0` | 12.5 % | 1223.4 | `jit(_kernel)/jit(main)/qmG` |
| `all-gather-start.1` | 0.0 % | 1208.3 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 1197.9 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start.1` | 0.0 % | 1175.9 | `jit(_kernel)/jit(main)/sharding_constraint` |

