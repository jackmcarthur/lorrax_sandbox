# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_sharded_v3/xprof/rank_0/plugins/profile/2026_04_27_01_54_12/perfetto_trace.json.gz`
**Duration:** 8.164 s
**GPU streams:** 2 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 351 | 64.89 MiB | 11.95 ms | 5.70 |
| D2H | 2353 | 32.63 KiB | 4.16 ms | 0.01 |
| D2D | 5765 | 567.65 MiB | 10.89 ms | 54.67 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 351 | 11.95 ms | 11.95 ms | 0.000 |
| D2H | 2353 | 4.16 ms | 4.16 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 0.80 s |
| D2H | 15.00 KiB | 0.00 | 1.30 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `cublas-batch-gemm.5.0` | 400 | 154.69 | 0.39 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `loop_transpose_fusion.7` | 200 | 140.32 | 0.70 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_multiply_fusion` | 205 | 82.88 | 0.42 | 31.25 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `loop_transpose_fusion.4` | 200 | 77.76 | 0.40 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.5` | 200 | 77.19 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.3` | 200 | 77.08 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.2` | 200 | 77.07 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.6` | 200 | 76.62 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_complex_fusion.1` | 200 | 75.34 | 0.38 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.4.0` | 200 | 69.57 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.1.0` | 200 | 69.41 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.5.0` | 200 | 69.36 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.3.0` | 200 | 69.28 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `input_transpose_fusion_1` | 200 | 69.06 | 0.35 | 56.25 | `` | `` |
| `fft.2.0` | 200 | 68.76 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.0.0` | 200 | 68.20 | 0.34 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_broadcast_fusion_2` | 200 | 46.10 | 0.23 | 100 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_16x32_16x3_nn_align1::Params)` | 200 | 40.35 | 0.21 | 12.5 | `` | `` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1>(cutlass_80_tensorop_z884gemm_32x16_16x3_nn_align1::Params)` | 200 | 26.64 | 0.13 | 12.5 | `` | `` |
| `all-reduce-start.1` | 200 | 20.58 | 0.14 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 973.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `custom-call.258.0` | 25.0 % | 467.0 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 389.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 388.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 388.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 388.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 388.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 388.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 388.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 388.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 388.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |

