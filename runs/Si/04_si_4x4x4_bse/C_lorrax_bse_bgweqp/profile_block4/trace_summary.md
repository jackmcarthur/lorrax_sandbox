# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_block4/xprof/rank_0/plugins/profile/2026_04_27_02_04_46/perfetto_trace.json.gz`
**Duration:** 14.707 s
**GPU streams:** 3 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 364 | 64.89 MiB | 19.12 ms | 3.56 |
| D2H | 553 | 30.87 KiB | 1.03 ms | 0.03 |
| D2D | 921 | 567.63 MiB | 2.38 ms | 250.24 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 364 | 19.12 ms | 19.12 ms | 0.000 |
| D2H | 553 | 1.03 ms | 1.03 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 0.90 s |
| D2H | 15.00 KiB | 0.00 | 1.50 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `cublas-batch-gemm.5.0` | 100 | 162.60 | 1.69 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `loop_transpose_fusion.12` | 50 | 145.21 | 2.91 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_multiply_fusion` | 55 | 82.72 | 1.67 | 31.25 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `loop_transpose_fusion.9` | 50 | 76.67 | 1.54 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.7` | 50 | 76.34 | 1.53 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.10` | 50 | 76.29 | 1.53 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.11` | 50 | 76.28 | 1.53 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.8` | 50 | 76.21 | 1.53 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_complex_fusion.1` | 50 | 75.89 | 1.52 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `input_transpose_fusion.1` | 50 | 75.35 | 1.52 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.4.0` | 50 | 69.08 | 1.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.1.0` | 50 | 68.80 | 1.38 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.3.0` | 50 | 68.74 | 1.38 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.5.0` | 50 | 68.66 | 1.38 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.2.0` | 50 | 68.66 | 1.38 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.0.0` | 50 | 68.54 | 1.37 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_broadcast_fusion_2` | 50 | 45.40 | 0.91 | 100 | `` | `` |
| `custom-call.294.0` | 50 | 39.04 | 0.79 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `all-reduce-start.1` | 50 | 17.84 | 0.45 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `collective-permute-start.3` | 100 | 16.55 | 1.23 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start.2` | 0.0 % | 6840.3 | `jit(_full_run)/jit(main)/while/body/jit(qr)/householder_product` |
| `collective-permute-start.2` | 0.0 % | 5091.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `collective-permute-start` | 0.0 % | 4038.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1693.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1689.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1673.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1672.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1655.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1653.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1635.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1635.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1633.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1631.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1630.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1630.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1630.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1629.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1629.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1628.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |
| `cublas-batch-gemm.5.0` | 12.5 % | 1628.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/while/body` |

