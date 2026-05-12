# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_sharded/xprof/rank_0/plugins/profile/2026_04_27_01_28_02/perfetto_trace.json.gz`
**Duration:** 13.176 s
**GPU streams:** 2 compute, 8 H2D, 2 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 364 | 64.89 MiB | 15.31 ms | 4.44 |
| D2H | 2353 | 32.63 KiB | 4.28 ms | 0.01 |
| D2D | 5768 | 848.75 MiB | 12.02 ms | 74.03 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 364 | 15.31 ms | 15.31 ms | 0.000 |
| D2H | 2353 | 4.28 ms | 4.28 ms | 0.000 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 56.26 MiB | 0.59 | 0.70 s |
| D2H | 15.00 KiB | 0.00 | 1.30 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 20 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `all-gather-start.3` | 200 | 1664.33 | 8.70 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `all-gather-start.1` | 201 | 1619.85 | 8.34 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.14.0` | 800 | 1109.73 | 1.41 | 93.75 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `all-gather-start.2` | 200 | 858.27 | 4.49 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `all-gather-start` | 201 | 843.98 | 4.89 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.15.0` | 600 | 835.19 | 1.41 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.2` | 200 | 164.33 | 0.83 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.4` | 200 | 164.25 | 0.83 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `cublas-batch-gemm.5.0` | 400 | 154.22 | 0.42 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `loop_transpose_fusion.3` | 200 | 97.47 | 0.50 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `input_transpose_fusion.1` | 200 | 70.62 | 0.36 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)` |
| `input_transpose_fusion.2` | 200 | 68.36 | 0.34 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `loop_broadcast_fusion_2` | 200 | 46.15 | 0.24 | 100 | `` | `` |
| `custom-call.230.0` | 200 | 40.48 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `custom-call.229.0` | 200 | 28.51 | 0.16 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `collective-permute-start` | 402 | 28.42 | 0.50 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `all-reduce-start.1` | 200 | 25.11 | 0.56 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `collective-permute-start.1` | 400 | 23.88 | 0.70 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `all-reduce-start` | 2147 | 16.75 | 0.35 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `reduce-scatter.8.1` | 200 | 16.58 | 0.44 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start.3` | 0.0 % | 8699.3 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8673.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8623.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8609.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8601.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8585.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8583.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8580.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8577.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8575.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8556.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8543.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8543.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8534.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8518.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8517.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8517.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8513.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8510.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8507.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |

