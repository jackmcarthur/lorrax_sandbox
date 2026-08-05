# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_perf_scan_2026-05-12_profile/xla_dump`
**Modules dumped:** 849
**Sum of per-module peak live HBM:** 219.93 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0403.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0465.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0467.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0566.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0639.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0641.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0729.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0813.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0815.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0211.jit__kernel` | 10.11 GiB | 9.94 GiB — preallocated-temp: |
| `module_0247.jit__kernel` | 10.11 GiB | 9.94 GiB — preallocated-temp: |
| `module_0249.jit__kernel` | 10.11 GiB | 9.94 GiB — preallocated-temp: |
| `module_1081.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_1083.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_1105.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_1107.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_1007.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |
| `module_1009.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |
| `module_1049.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |
| `module_1051.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0211.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0211.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0211.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0211.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0211.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0247.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0247.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0247.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0247.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0247.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0249.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0249.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0249.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0249.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0249.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0403.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0403.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0403.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0403.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0403.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit__per_rank` | 102 | 1.01 GiB | 11.44 GiB |
| `jit_broadcast_in_dim` | 80 | 28.83 MiB | 296.77 MiB |
| `jit_convert_element_type` | 73 | 6.00 MiB | 21.31 MiB |
| `jit__psum` | 67 | 4.75 MiB | 26.61 MiB |
| `jit__kernel` | 34 | 10.36 GiB | 168.41 GiB |
| `jit__take` | 34 | 28.83 MiB | 120.72 MiB |
| `jit_true_divide` | 28 | 28.83 MiB | 491.59 MiB |
| `jit_dynamic_slice` | 28 | 10.00 MiB | 37.54 MiB |
| `jit_squeeze` | 26 | 4.00 MiB | 8.77 MiB |
| `jit_multiply` | 25 | 28.83 MiB | 177.69 MiB |
| `jit_transpose` | 24 | 259.45 MiB | 2.34 GiB |
| `jit_concatenate` | 16 | 2.11 MiB | 8.44 MiB |
| `jit_gather` | 14 | 112.50 MiB | 371.04 MiB |
| `jit__init_V` | 14 | 14.77 MiB | 205.42 MiB |
| `jit__identity_fn` | 12 | 28.83 MiB | 228.27 MiB |
| `jit__squeeze` | 12 | 5.77 MiB | 53.56 MiB |
| `jit_add` | 12 | 2.64 MiB | 11.86 MiB |
| `jit__multi_slice` | 11 | 43.24 MiB | 341.72 MiB |
| `jit_subtract` | 11 | 8.00 MiB | 24.78 MiB |
| `jit__ifft_contract_fft` | 10 | 960.34 MiB | 9.24 GiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 233 |
| `xla_python_gpu_callback` | 60 |
| `lorrax_phdf5_write` | 49 |
| `lorrax_phdf5_read` | 32 |
| `lorrax_phdf5_read_kchunk_union` | 11 |
| `lorrax_cusolvermp_batched_solve_lu` | 9 |
| `__cublas$triangularSolve` | 8 |
| `cusolver_getrf_ffi` | 4 |
| `cu_lu_pivots_to_permutation` | 4 |
| `lorrax_cusolvermp_batched_potrf` | 3 |
| `lorrax_cusolvermp_batched_potrs` | 3 |
| `cusolver_syevd_ffi` | 2 |

