# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/00_davidson_only/profile/xla_dump`
**Modules dumped:** 215
**Sum of per-module peak live HBM:** 869.94 MiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0225.jit__apply_H_sparse` | 214.12 MiB | 205.03 MiB — preallocated-temp: |
| `module_0223.jit__apply_H_sparse` | 161.31 MiB | 153.77 MiB — preallocated-temp: |
| `module_0221.jit__apply_H_sparse` | 110.05 MiB | 104.07 MiB — preallocated-temp: |
| `module_0219.jit__apply_H_sparse` | 56.47 MiB | 52.03 MiB — preallocated-temp: |
| `module_0087.jit__ritz_and_residuals` | 21.86 MiB | 13.32 MiB — preallocated-temp: |
| `module_0083.jit__ritz_and_residuals` | 17.98 MiB | 10.99 MiB — preallocated-temp: |
| `module_0079.jit__ritz_and_residuals` | 14.10 MiB | 8.66 MiB — preallocated-temp: |
| `module_0065.jit_compute_V_H_and_V_xc` | 13.88 MiB | 10.68 MiB — preallocated-temp: |
| `module_0229.jit__einsum` | 10.25 MiB | 4.00 MiB — preallocated-temp: |
| `module_0075.jit__ritz_and_residuals` | 8.54 MiB | 4.66 MiB — preallocated-temp: |
| `module_0253.jit_concatenate` | 7.76 MiB | 3.88 MiB — output shape is \|c128[60,2,2120]\|, maybe-live-out: |
| `module_0243.jit__einsum` | 7.01 MiB | 3.11 MiB — preallocated-temp: |
| `module_0317.jit_matmul` | 6.74 MiB | 2.27 MiB — preallocated-temp: |
| `module_0193.jit_matmul` | 6.71 MiB | 2.26 MiB — preallocated-temp: |
| `module_0397.jit_matmul` | 6.68 MiB | 2.25 MiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

_No collective ops found (single-device or pure-SPMD-free)._

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_multiply` | 31 | 6.60 MiB | 65.13 MiB |
| `jit_broadcast_in_dim` | 23 | 3.11 MiB | 15.11 MiB |
| `jit_add` | 16 | 2.14 MiB | 3.94 MiB |
| `jit_convert_element_type` | 15 | 99.38 KiB | 346.85 KiB |
| `jit_matmul` | 9 | 6.74 MiB | 24.14 MiB |
| `jit_concatenate` | 8 | 7.76 MiB | 21.75 MiB |
| `jit__reduce_sum` | 8 | 2.23 MiB | 8.64 MiB |
| `jit_precond_fn` | 8 | 1.57 MiB | 12.55 MiB |
| `jit_integer_pow` | 7 | 729.00 KiB | 1.10 MiB |
| `jit__pad` | 6 | 4.39 MiB | 8.86 MiB |
| `jit_gather` | 6 | 1.34 MiB | 7.78 MiB |
| `jit_conjugate` | 5 | 6.21 MiB | 19.42 MiB |
| `jit_transpose` | 5 | 4.40 MiB | 13.21 MiB |
| `jit_true_divide` | 5 | 1.78 MiB | 1.88 MiB |
| `jit_dynamic_slice` | 5 | 361.28 KiB | 604.12 KiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 39 |
| `__cublas$triangularSolve` | 16 |
| `cusolver_getrf_ffi` | 8 |
| `cu_lu_pivots_to_permutation` | 8 |
| `cusolver_syevd_ffi` | 5 |
| `__cusolver$cholesky` | 4 |

