# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_sharded_v2/xprof/rank_0/plugins/profile/2026_04_27_01_30_52/perfetto_trace.json.gz`
**Duration:** 14.745 s
**GPU streams:** 2 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 353 | 64.89 MiB | 18.74 ms | 3.63 |
| D2H | 2353 | 32.63 KiB | 4.19 ms | 0.01 |
| D2D | 5765 | 567.50 MiB | 10.97 ms | 54.24 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 353 | 18.74 ms | 18.74 ms | 0.000 |
| D2H | 2353 | 4.19 ms | 4.19 ms | 0.000 |

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
| `all-gather-start.3` | 200 | 1656.91 | 9.49 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `all-gather-start.1` | 200 | 1622.29 | 8.42 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.15.0` | 800 | 1109.69 | 1.41 | 93.75 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `all-gather-start.2` | 200 | 853.71 | 4.94 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `all-gather-start` | 200 | 843.66 | 4.91 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.16.0` | 600 | 835.17 | 1.41 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.2` | 200 | 167.08 | 0.85 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.4` | 200 | 167.01 | 0.84 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `cublas-batch-gemm.5.0` | 400 | 154.46 | 0.39 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `loop_transpose_fusion.3` | 200 | 98.33 | 0.50 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `input_transpose_fusion.1` | 200 | 70.48 | 0.36 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)` |
| `input_transpose_fusion.2` | 200 | 68.51 | 0.34 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `loop_broadcast_fusion_2` | 200 | 46.11 | 0.23 | 100 | `` | `` |
| `custom-call.231.0` | 200 | 40.24 | 0.20 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `collective-permute-start.1` | 400 | 34.15 | 0.73 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `all-reduce-start` | 2147 | 31.67 | 11.50 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `custom-call.230.0` | 200 | 28.37 | 0.16 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `all-reduce-start.1` | 200 | 28.25 | 0.67 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `collective-permute-start` | 402 | 25.18 | 0.54 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `reduce-scatter.8.1` | 200 | 20.94 | 0.60 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-reduce-start` | 0.0 % | 11499.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start.3` | 0.0 % | 9487.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 9231.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8719.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8654.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8636.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8624.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8597.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8590.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8588.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8582.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8577.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8575.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8560.4 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8542.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8537.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8536.5 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8531.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8520.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |
| `all-gather-start.3` | 0.0 % | 8505.9 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_apply_W_from_T)/jit(f` |

