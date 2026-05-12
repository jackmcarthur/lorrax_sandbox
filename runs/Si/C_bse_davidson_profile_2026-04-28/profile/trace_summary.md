# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/C_bse_davidson_profile_2026-04-28/profile/xprof/rank_0/plugins/profile/2026_04_28_17_06_55/perfetto_trace.json.gz`
**Duration:** 12.438 s
**GPU streams:** 4 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 406 | 65.87 MiB | 19.28 ms | 3.58 |
| D2H | 327 | 63.12 KiB | 0.65 ms | 0.10 |
| D2D | 896 | 2.48 GiB | 4.99 ms | 533.58 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 406 | 19.28 ms | 19.28 ms | 0.000 |
| D2H | 327 | 0.65 ms | 0.65 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 57.14 MiB | 0.60 | 0.90 s |
| D2H | 21.00 KiB | 0.00 | 3.40 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `fft.6.0` | 164 | 586.98 | 3.67 | 93.75 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(_apply_W_from_T)/jit(shmap_b` |
| `fft.7.0` | 123 | 441.36 | 3.67 | 100 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(_apply_W_from_T)/jit(shmap_b` |
| `cublas-batch-gemm.2.0` | 82 | 335.90 | 4.12 | 12.5 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/kctM` |
| `all-reduce-start.1` | 40 | 179.08 | 12.31 | 0 | `jit__ritz_and_residuals` | `` |
| `loop_multiply_fusion` | 126 | 170.39 | 4.17 | 75 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(_apply_W_from_T)` |
| `input_transpose_fusion_1` | 71 | 145.42 | 3.57 | 56.25 | `` | `` |
| `input_transpose_fusion.2` | 71 | 144.81 | 3.55 | 56.25 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while` |
| `loop_broadcast_fusion.3` | 41 | 94.08 | 2.32 | 100 | `jit__matvec_impl` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1::Params)` | 41 | 78.22 | 1.93 | 12.5 | `` | `` |
| `all-reduce-start` | 123 | 38.86 | 6.37 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)` |
| `collective-permute-start.3` | 82 | 31.75 | 0.45 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/pperm` |
| `collective-permute-start.1` | 82 | 29.25 | 8.33 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/pperm` |
| `custom-call.9.0` | 2378 | 21.81 | 0.17 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `reduce-scatter.5` | 41 | 15.96 | 0.52 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)` |
| `collective-permute-start.2` | 82 | 15.95 | 9.35 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/pperm` |
| `loop_add_fusion.3` | 82 | 3.14 | 0.04 | 75 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body` |
| `all-gather-start` | 2 | 3.03 | 3.01 | 0 | `jit__identity_fn` | `` |
| `collective-permute-start` | 84 | 2.28 | 0.06 | 0 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/pperm` |
| `input_transpose_fusion` | 71 | 2.11 | 0.05 | 56.25 | `jit__matvec_impl` | `jit(_matvec_impl)/jit(main)/jit(_apply_W_from_T)/kctM` |
| `custom-call.7.0` | 240 | 2.10 | 0.07 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/lu` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start.1` | 0.0 % | 12309.7 | `` |
| `all-reduce-start.1` | 0.0 % | 11009.4 | `` |
| `all-reduce-start.1` | 0.0 % | 10772.2 | `` |
| `all-reduce-start.1` | 0.0 % | 10603.6 | `` |
| `all-reduce-start.1` | 0.0 % | 10498.9 | `` |
| `all-reduce-start.1` | 0.0 % | 9857.7 | `` |
| `collective-permute-start.2` | 0.0 % | 9350.3 | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/ppermute` |
| `all-reduce-start.1` | 0.0 % | 8493.8 | `` |
| `collective-permute-start.1` | 0.0 % | 8325.0 | `jit(_matvec_impl)/jit(main)/jit(shmap_body)/while/body/ppermute` |
| `all-reduce-start.1` | 0.0 % | 7812.7 | `` |
| `all-reduce-start.1` | 0.0 % | 7611.2 | `` |
| `all-reduce-start.1` | 0.0 % | 6771.3 | `` |
| `all-reduce-start.1` | 0.0 % | 6750.3 | `` |
| `all-reduce-start` | 0.0 % | 6366.0 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-reduce-start.1` | 0.0 % | 6215.6 | `` |
| `all-reduce-start.1` | 0.0 % | 5941.3 | `` |
| `all-reduce-start.1` | 0.0 % | 5793.2 | `` |
| `all-reduce-start.1` | 0.0 % | 5397.0 | `` |
| `all-reduce-start.1` | 0.0 % | 4960.7 | `` |
| `all-reduce-start.1` | 0.0 % | 4722.6 | `` |

