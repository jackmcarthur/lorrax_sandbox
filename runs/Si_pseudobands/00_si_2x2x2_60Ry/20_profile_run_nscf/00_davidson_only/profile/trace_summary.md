# Trace summary

**Trace:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/00_davidson_only/profile/xprof/plugins/profile/2026_04_16_11_18_43/perfetto_trace.json.gz`
**Duration:** 22.603 s
**GPU streams:** 6 compute, 2 H2D, 4 D2H

_Companion:_ [`trace_details.txt`](trace_details.txt) — dense per-event dump of the top copies + top kernels.

## Host ↔ device transfers

| Direction | Count | Total bytes | Total time | Avg GB/s |
|---|---:|---:|---:|---:|
| H2D | 999 | 33.47 MiB | 43.35 ms | 0.81 |
| D2H | 2650 | 6.91 MiB | 4.16 ms | 1.74 |
| D2D | 2923 | 79.96 MiB | 5.69 ms | 14.74 |

_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as one stream. With multiple streams, instantaneous bandwidth can be higher; see the peak table below._

## Async overlap — were copies hidden behind compute?

| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 999 | 43.35 ms | 43.35 ms | 0.000 |
| D2H | 2650 | 4.16 ms | 3.88 ms | 0.066 |

_overlap_frac = (total − exposed) / total. **Close to 1 is good** (copy happened while the GPU was busy with compute, so it's free). **Below ~0.3 means the copy is blocking the pipeline** — either the issuer is waiting on the data (legitimate stall) or the copy was dispatched too late (schedulable bug)._

## Bandwidth saturation (window = 100 ms)

| Direction | Peak window bytes | Peak window GB/s | At t |
|---|---:|---:|---:|
| H2D | 3.89 MiB | 0.04 | 1.00 s |
| D2H | 884.45 KiB | 0.01 | 21.30 s |

_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. Sustained > ~20 GB/s in a window means the link is saturated; combine with the overlap table above — saturated + low overlap = real bottleneck._

## Top 15 GPU kernels by total time

| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |
|---|---:|---:|---:|---:|---|---|
| `custom-call.4.0` | 36989 | 260.62 | 0.21 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.5.0` | 3102 | 32.30 | 0.08 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/lu` |
| `custom-call.2.0` | 3102 | 32.17 | 0.08 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/lu` |
| `fft.2.0` | 2063 | 31.53 | 0.08 | 93.75 | `jit__apply_H_sparse` | `jit(_apply_H_sparse)/jit(main)/jit(apply_H_k)/jit(fft)/fft` |
| `custom-call.12.0` | 390 | 28.01 | 0.20 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/msG` |
| `fft.3.0` | 1533 | 23.56 | 0.02 | 56.25 | `jit__apply_H_sparse` | `jit(_apply_H_sparse)/jit(main)/jit(apply_H_k)/jit(fft)/fft` |
| `custom-call.7.0` | 1034 | 18.42 | 0.05 | 0 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(cholesky)/cholesky` |
| `loop_multiply_fusion` | 1182 | 16.38 | 0.11 | 100 | `jit__apply_H_sparse` | `jit(_apply_H_sparse)/jit(main)/jit(apply_H_k)` |
| `input_transpose_fusion` | 916 | 10.07 | 0.09 | 100 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)` |
| `loop_broadcast_fusion` | 522 | 9.58 | 0.05 | 100 | `jit_species_structure_factors` | `jit(species_structure_factors)/jit(main)/while/body/broadcas` |
| `triangular-solve.11.0` | 647 | 9.05 | 0.02 | 62.5 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/vmap(` |
| `triangular-solve.13.0` | 647 | 9.03 | 0.02 | 62.5 | `jit__ritz_and_residuals` | `jit(_ritz_and_residuals)/jit(main)/jit(inv)/jit(solve)/vmap(` |
| `void cutlass::Kernel2<cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1>(cutlass_80_tensorop_z884gemm_32x32_16x4_tn_align1::Params)` | 778 | 8.69 | 0.02 | 0 | `` | `` |
| `input_reduce_fusion` | 1058 | 6.55 | 0.01 | 98.4375 | `jit_precond_fn` | `jit(precond_fn)/jit(main)` |
| `custom-call.16.0` | 984 | 6.32 | 0.21 | 100 | `jit_eigh` | `jit(eigh)/jit(main)/eigh` |

## Low-occupancy compute kernels (theoretical < 50 %, ranked by wasted time)

| Op | Occupancy | µs | Source |
|---|---:|---:|---|
| `custom-call.4.0` | 25.0 % | 208.4 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.3 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.3 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.2 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.1 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.1 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |
| `custom-call.4.0` | 25.0 % | 208.1 | `jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh` |

