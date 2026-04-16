# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/02_davidson_4gpu/profile/xprof/rank_0/plugins/profile/2026_04_16_11_21_50/perfetto_trace.json.gz`
**Duration:** 21.286 s
**GPU streams:** 6 compute, 4 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 606 | 26.66 MiB | 17.89 ms | 1.56 |
| D2H | 683 | 32.80 MiB | 2.32 ms | 14.85 |
| D2D | 838 | 26.22 MiB | 1.63 ms | 16.82 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 606 | 17.89 ms | 17.89 ms | 0.000 |
| D2H | 683 | 2.32 ms | 2.24 ms | 0.032 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 12.42 MiB | 0.13 | 21.10 s |
| D2H | 31.06 MiB | 0.33 | 21.10 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 15 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `custom-call.4.0` | 9680 | 67.30 | 0.21 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `all-gather-start` | 2 | 9.16 | 7.83 | 0 | `jit__identity_fn` | `` |
| `custom-call.5.0` | 816 | 8.52 | 0.08 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/lu` |
| `custom-call.2.0` | 816 | 8.48 | 0.08 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/lu` |
| `fft.2.0` | 545 | 8.33 | 0.08 | 93.75 | `jit__apply_H_sparse` | `jit(_apply_H_sparse)/jit(main)/jit(apply_H_k)/jit(fft)/fft` |
| `custom-call.12.0` | 102 | 7.32 | 0.20 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/msG` |
| `fft.3.0` | 399 | 6.12 | 0.02 | 56.25 | `jit__apply_H_sparse` | `jit(_apply_H_sparse)/jit(main)/jit(apply_H_k)/jit(fft)/fft` |
| `custom-call.7.0` | 272 | 4.84 | 0.05 | 0 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(cholesky)/cholesky` |
| `all-reduce-start` | 2 | 4.70 | 4.69 | 0 | `jit__psum` | `jit(_psum)/jit(main)/reduce_sum` |
| `loop_multiply_fusion` | 327 | 4.45 | 0.11 | 100 | `jit__apply_H_sparse` | `jit(_apply_H_sparse)/jit(main)/jit(apply_H_k)` |
| `input_transpose_fusion` | 241 | 2.75 | 0.09 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)` |
| `loop_broadcast_fusion` | 138 | 2.52 | 0.05 | 100 | `jit_species_structure_factors` | `jit(species_structure_factors)/jit(main)/while/body/broadcas` |
| `triangular-solve.11.0` | 170 | 2.39 | 0.02 | 62.5 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/vmap(` |
| `triangular-solve.13.0` | 170 | 2.38 | 0.02 | 62.5 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/vmap(` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1::Params)` | 205 | 2.30 | 0.02 | 0 | `` | `` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `all-gather-start` | 0.0 % | 7829.7 | `` |
| `all-reduce-start` | 0.0 % | 4694.6 | `jit(_psum)/jit(main)/reduce_sum` |
| `all-gather-start` | 0.0 % | 1331.7 | `` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.1 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.1 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.1 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.0 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.0 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.0 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |

