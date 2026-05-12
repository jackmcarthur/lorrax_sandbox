# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/01_mos2_3x3_d3h_2026-05-09/A_bispinor_smoke/v_q_profile_115346/V_q_compute_bispinor-20260511-115510-p3/plugins/profile/2026_05_11_11_55_40/nid001105.trace.json.gz`
**Duration:** 28.473 s
**GPU streams:** 2 compute, 5 H2D, 11 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 212 | 6.85 GiB | 275.04 ms | 26.76 |
| D2H | 101 | 50.55 MiB | 2.21 ms | 24.04 |
| D2D | 103 | 9.67 MiB | 0.21 ms | 49.38 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 212 | 275.04 ms | 275.04 ms | 0.000 |
| D2H | 101 | 2.21 ms | 2.21 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 881.90 MiB | 9.25 | 4.50 s |
| D2H | 11.71 MiB | 0.12 | 5.20 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-reduce-start` | 46 | 137.58 | 60.37 | 0 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.1` | 7 | 69.12 | 63.22 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start` | 7 | 37.81 | 26.90 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `fft.2.0` | 12 | 13.72 | 1.43 | 46.875 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `input_transpose_fusion` | 7 | 11.05 | 2.15 | 25 | `jit__kernel` | `jit(_kernel)/jit(main)` |
| `fft.5.0` | 9 | 9.74 | 1.12 | 46.875 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `fft.4.0` | 9 | 9.43 | 1.05 | 46.875 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `custom-call.1.0` | 13 | 5.67 | 1.31 | 12.5 | `jit__kernel` | `jit(_kernel)/jit(main)/qmG` |
| `loop_complex_fusion` | 21 | 1.12 | 0.22 | 100 | `jit__kernel` | `jit(_kernel)/jit(main)` |
| `collective-permute-start` | 7 | 1.05 | 0.19 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/sharding_constraint` |
| `loop_gather_fusion` | 21 | 0.54 | 0.09 | 100 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_take)/gather` |
| `loop_gather_fusion.1` | 3 | 0.20 | 0.07 | 100 | `jit__kernel` | `jit(_kernel)/jit(main)/jit(_take)/gather` |
| `all-gather-start.2` | 4 | 0.17 | 0.05 | 0 | `jit__kernel` | `jit(_kernel)/jit(main)/squeeze` |
| `loop_broadcast_fusion` | 15 | 0.10 | 0.02 | 100 | `jit__zeros` | `jit(_broadcast_arrays)/jit(main)/broadcast_in_dim` |
| `loop_dynamic_slice_fusion` | 19 | 0.07 | 0.01 | 100 | `jit_gather` | `` |
| `wrapped_multiply` | 25 | 0.06 | 0.00 | 100 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `loop_add_fusion` | 21 | 0.05 | 0.00 | 100 | `jit_get_sqrt_v_and_phase` | `jit(get_sqrt_v_and_phase)/jit(main)` |
| `input_reduce_fusion` | 7 | 0.02 | 0.00 | 100 | `jit__reduce_max` | `jit(_reduce_max)/jit(main)/reduce_max` |
| `wrapped_abs` | 7 | 0.02 | 0.00 | 100 | `jit_abs` | `jit(abs)/jit(main)/abs` |
| `input_reduce_fusion.1` | 7 | 0.02 | 0.00 | 50 | `jit__reduce_max` | `jit(_reduce_max)/jit(main)/reduce_max` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start.1` | 0.0 % | 63217.3 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 60366.3 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 38923.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 26896.2 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 24880.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 3525.2 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 3286.5 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 2234.8 | `jit(_psum)/jit(main)/reduce_sum` |
| `input_transpose_fusion` | 25.0 % | 2146.7 | `jit(_kernel)/jit(main)` |
| `input_transpose_fusion` | 25.0 % | 2141.7 | `jit(_kernel)/jit(main)` |
| `input_transpose_fusion` | 25.0 % | 2139.8 | `jit(_kernel)/jit(main)` |
| `all-reduce-start` | 0.0 % | 2056.4 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start` | 0.0 % | 1909.7 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-reduce-start` | 0.0 % | 1546.9 | `jit(_psum)/jit(main)/reduce_sum` |
| `fft.2.0` | 46.9 % | 1433.3 | `jit(_kernel)/jit(main)/jit(shmap_body)/jit(fft)/fft` |
| `custom-call.1.0` | 12.5 % | 1314.0 | `jit(_kernel)/jit(main)/qmG` |
| `all-gather-start.1` | 0.0 % | 1242.6 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-reduce-start` | 0.0 % | 1187.3 | `jit(_reduce_max)/jit(main)/reduce_max` |
| `all-gather-start.1` | 0.0 % | 1183.4 | `jit(_kernel)/jit(main)/sharding_constraint` |
| `all-gather-start` | 0.0 % | 1119.0 | `jit(_kernel)/jit(main)/sharding_constraint` |

