# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/02_davidson_4gpu/profile/xla_dump`
**Modules dumped:** 319
**Sum of per-module peak live HBM:** 1.08 GiB (upper bound; peaks happen at different times)

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0225.jit__apply_H_sparse` | 214.12 MiB | 205.03 MiB — preallocated-temp: |
| `module_0223.jit__apply_H_sparse` | 161.31 MiB | 153.77 MiB — preallocated-temp: |
| `module_0221.jit__apply_H_sparse` | 110.05 MiB | 104.07 MiB — preallocated-temp: |
| `module_0219.jit__apply_H_sparse` | 56.47 MiB | 52.03 MiB — preallocated-temp: |
| `module_0335.jit__identity_fn` | 31.05 MiB | 24.84 MiB — output shape is \|c128[4,8,12,2,2120]\|, maybe-live-out: |
| `module_0421.jit__identity_fn` | 31.05 MiB | 24.84 MiB — output shape is \|c128[4,8,12,2,2120]\|, maybe-live-out: |
| `module_0087.jit__ritz_and_residuals` | 21.86 MiB | 13.32 MiB — preallocated-temp: |
| `module_0083.jit__ritz_and_residuals` | 17.98 MiB | 10.99 MiB — preallocated-temp: |
| `module_0079.jit__ritz_and_residuals` | 14.10 MiB | 8.66 MiB — preallocated-temp: |
| `module_0065.jit_compute_V_H_and_V_xc` | 13.88 MiB | 10.68 MiB — preallocated-temp: |

## Compute — aggregate op counts

| Op | Count |
|---|---:|
| `fusion` | 405 |
| `transpose` | 74 |
| `copy` | 60 |
| `concatenate` | 53 |
| `gather` | 42 |
| `reduce` | 30 |
| `fft` | 22 |
| `reshape` | 14 |
| `scatter` | 8 |
| `all-gather-start` | 4 |

### Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 49 |
| `__cublas$triangularSolve` | 16 |
| `cusolver_getrf_ffi` | 8 |
| `cu_lu_pivots_to_permutation` | 8 |
| `cusolver_syevd_ffi` | 7 |
| `__cusolver$cholesky` | 4 |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Output type |
|---|---|---:|---|
| `module_0335.jit__identity_fn` | `all-gather-start` | 31.05 MiB | `(c128[1,8,12,2,2120]{4,3,2,1,0}, c128[4,8,12,2,2120]{4,3,2,1,0})` |
| `module_0421.jit__identity_fn` | `all-gather-start` | 31.05 MiB | `(c128[1,8,12,2,2120]{4,3,2,1,0}, c128[4,8,12,2,2120]{4,3,2,1,0})` |
| `module_0333.jit__identity_fn` | `all-gather-start` | 3.75 KiB | `(f64[1,8,12]{2,1,0}, f64[4,8,12]{2,1,0})` |
| `module_0419.jit__identity_fn` | `all-gather-start` | 3.75 KiB | `(f64[1,8,12]{2,1,0}, f64[4,8,12]{2,1,0})` |

## Rematerialization warnings

_None._ 🎉

## Retrace groups — jit() name → module count

_Multiple modules per jit name mean XLA recompiled from scratch under a new signature. >2 is worth investigating; >5 is almost always a shape-polymorphism bug._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_multiply` | 43 | 6.60 MiB | 92.88 MiB |
| `jit_broadcast_in_dim` | 32 | 3.11 MiB | 20.03 MiB |
| `jit_add` | 26 | 2.14 MiB | 4.57 MiB |
| `jit_convert_element_type` | 19 | 99.38 KiB | 578.73 KiB |
| `jit_concatenate` | 17 | 7.76 MiB | 65.23 MiB |
| `jit_matmul` | 15 | 6.74 MiB | 40.32 MiB |
| `jit__reduce_sum` | 12 | 2.23 MiB | 13.23 MiB |
| `jit_conjugate` | 11 | 6.21 MiB | 40.78 MiB |
| `jit_integer_pow` | 11 | 729.00 KiB | 1.36 MiB |
| `jit_gather` | 10 | 1.34 MiB | 12.99 MiB |

## Per-module file index

_(for deeper inspection — each module writes up to 10 files; the memory-usage-report is the most agent-readable)_

| Module | Files |
|---|---|
| `module_0225.jit__apply_H_sparse` | module_0225.jit__apply_H_sparse.autotune_results.pbtxt, module_0225.jit__apply_H_sparse.before_optimizations.txt, module_0225.jit__apply_H_sparse.gpu_target_config.pbtxt, … (+7) |
| `module_0223.jit__apply_H_sparse` | module_0223.jit__apply_H_sparse.autotune_results.pbtxt, module_0223.jit__apply_H_sparse.before_optimizations.txt, module_0223.jit__apply_H_sparse.gpu_target_config.pbtxt, … (+7) |
| `module_0221.jit__apply_H_sparse` | module_0221.jit__apply_H_sparse.autotune_results.pbtxt, module_0221.jit__apply_H_sparse.before_optimizations.txt, module_0221.jit__apply_H_sparse.gpu_target_config.pbtxt, … (+7) |
| `module_0219.jit__apply_H_sparse` | module_0219.jit__apply_H_sparse.autotune_results.pbtxt, module_0219.jit__apply_H_sparse.before_optimizations.txt, module_0219.jit__apply_H_sparse.gpu_target_config.pbtxt, … (+7) |
| `module_0335.jit__identity_fn` | module_0335.jit__identity_fn.0000.spmd-partitioner.after_hlo-constant-splitter.before_sharding-propagation.txt, module_0335.jit__identity_fn.0001.spmd-partitioner.after_sharding-propagation.before_spmd-partitioning.txt, module_0335.jit__identity_fn.autotune_results.pbtxt, … (+9) |
| `module_0421.jit__identity_fn` | module_0421.jit__identity_fn.0000.spmd-partitioner.after_hlo-constant-splitter.before_sharding-propagation.txt, module_0421.jit__identity_fn.0001.spmd-partitioner.after_sharding-propagation.before_spmd-partitioning.txt, module_0421.jit__identity_fn.autotune_results.pbtxt, … (+9) |
| `module_0087.jit__ritz_and_residuals` | module_0087.jit__ritz_and_residuals.autotune_results.pbtxt, module_0087.jit__ritz_and_residuals.before_optimizations.txt, module_0087.jit__ritz_and_residuals.gpu_target_config.pbtxt, … (+7) |
| `module_0083.jit__ritz_and_residuals` | module_0083.jit__ritz_and_residuals.autotune_results.pbtxt, module_0083.jit__ritz_and_residuals.before_optimizations.txt, module_0083.jit__ritz_and_residuals.gpu_target_config.pbtxt, … (+7) |
| `module_0079.jit__ritz_and_residuals` | module_0079.jit__ritz_and_residuals.autotune_results.pbtxt, module_0079.jit__ritz_and_residuals.before_optimizations.txt, module_0079.jit__ritz_and_residuals.gpu_target_config.pbtxt, … (+7) |
| `module_0065.jit_compute_V_H_and_V_xc` | module_0065.jit_compute_V_H_and_V_xc.autotune_results.pbtxt, module_0065.jit_compute_V_H_and_V_xc.before_optimizations.txt, module_0065.jit_compute_V_H_and_V_xc.gpu_target_config.pbtxt, … (+7) |
