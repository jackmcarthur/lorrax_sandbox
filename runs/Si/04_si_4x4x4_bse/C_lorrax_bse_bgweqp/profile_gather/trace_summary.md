# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_gather/xprof/rank_0/plugins/profile/2026_04_27_02_31_46/perfetto_trace.json.gz`
**Duration:** 7.862 s
**GPU streams:** 2 compute, 4 H2D, 3 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 351 | 64.89 MiB | 11.04 ms | 6.16 |
| D2H | 2354 | 32.40 KiB | 4.08 ms | 0.01 |
| D2D | 3566 | 190.07 MiB | 5.96 ms | 33.45 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 351 | 11.04 ms | 11.04 ms | 0.000 |
| D2H | 2354 | 4.08 ms | 4.08 ms | 0.000 |

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
| `loop_transpose_fusion.7` | 200 | 141.08 | 0.71 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_multiply_fusion` | 205 | 82.93 | 0.42 | 7.8125 | `jit_multiply` | `jit(multiply)/jit(main)/mul` |
| `loop_transpose_fusion.4` | 200 | 78.16 | 0.40 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.3` | 200 | 77.46 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.2` | 200 | 77.30 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.5` | 200 | 77.30 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `loop_transpose_fusion.6` | 200 | 76.64 | 0.39 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.2.0` | 200 | 69.54 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.3.0` | 200 | 69.33 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.4.0` | 200 | 69.28 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.1.0` | 200 | 69.18 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `input_transpose_fusion.1` | 200 | 69.04 | 0.35 | 56.25 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.5.0` | 200 | 68.91 | 0.35 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `fft.0.0` | 200 | 68.23 | 0.34 | 100 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `custom-call.270.0` | 200 | 50.41 | 0.26 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `custom-call.271.0` | 200 | 40.47 | 0.21 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(_a` |
| `custom-call.268.0` | 200 | 33.53 | 0.17 | 12.5 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `all-reduce-start` | 2147 | 24.71 | 0.17 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/while/body/dot_general` |
| `all-reduce-start.1` | 200 | 19.99 | 0.47 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |
| `all-gather-start.1` | 200 | 19.40 | 7.80 | 0 | `jit__full_run` | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(sh` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start.1` | 0.0 % | 7795.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/all_gather` |
| `reduce-scatter.10` | 0.0 % | 615.4 | `jit(_full_run)/jit(main)` |
| `all-reduce-start.1` | 0.0 % | 473.0 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)` |
| `custom-call.258.0` | 25.0 % | 467.1 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.258.0` | 25.0 % | 295.2 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.258.0` | 25.0 % | 290.9 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.258.0` | 25.0 % | 284.9 | `jit(_full_run)/jit(main)/jit(eigh)/eigh` |
| `custom-call.270.0` | 12.5 % | 255.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 254.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 254.6 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 254.2 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 254.1 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 253.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 253.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 253.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 253.8 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 253.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 253.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 253.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |
| `custom-call.270.0` | 12.5 % | 253.7 | `jit(_full_run)/jit(main)/while/body/jit(_matvec_impl)/jit(shmap_body)/kctM` |

