# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_perf_scan_auto_2026-05-12_profile/xla_dump`
**Modules dumped:** 879
**Sum of per-module peak live HBM:** 572.50 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0508.jit__kernel` | 21.76 GiB | 21.45 GiB — preallocated-temp: |
| `module_0571.jit__kernel` | 21.76 GiB | 21.45 GiB — preallocated-temp: |
| `module_0573.jit__kernel` | 21.76 GiB | 21.45 GiB — preallocated-temp: |
| `module_0778.jit__kernel` | 21.76 GiB | 21.45 GiB — preallocated-temp: |
| `module_0851.jit__kernel` | 21.76 GiB | 21.45 GiB — preallocated-temp: |
| `module_0853.jit__kernel` | 21.76 GiB | 21.45 GiB — preallocated-temp: |
| `module_1048.jit__kernel` | 21.76 GiB | 21.45 GiB — preallocated-temp: |
| `module_1131.jit__kernel` | 21.76 GiB | 21.45 GiB — preallocated-temp: |
| `module_1133.jit__kernel` | 21.76 GiB | 21.45 GiB — preallocated-temp: |
| `module_0210.jit__kernel` | 21.23 GiB | 20.93 GiB — preallocated-temp: |
| `module_0247.jit__kernel` | 21.23 GiB | 20.93 GiB — preallocated-temp: |
| `module_0249.jit__kernel` | 21.23 GiB | 20.93 GiB — preallocated-temp: |
| `module_0617.jit__kernel` | 17.41 GiB | 17.15 GiB — preallocated-temp: |
| `module_0679.jit__kernel` | 17.41 GiB | 17.15 GiB — preallocated-temp: |
| `module_0681.jit__kernel` | 17.41 GiB | 17.15 GiB — preallocated-temp: |
| `module_0887.jit__kernel` | 17.41 GiB | 17.15 GiB — preallocated-temp: |
| `module_0959.jit__kernel` | 17.41 GiB | 17.15 GiB — preallocated-temp: |
| `module_0961.jit__kernel` | 17.41 GiB | 17.15 GiB — preallocated-temp: |
| `module_1157.jit__kernel` | 17.41 GiB | 17.15 GiB — preallocated-temp: |
| `module_1239.jit__kernel` | 17.41 GiB | 17.15 GiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0210.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0210.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0210.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0210.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0210.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
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
| `module_0319.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0319.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0319.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0319.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0319.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit__per_rank` | 114 | 1.01 GiB | 16.64 GiB |
| `jit_broadcast_in_dim` | 80 | 28.83 MiB | 296.77 MiB |
| `jit_convert_element_type` | 73 | 6.00 MiB | 21.33 MiB |
| `jit__psum` | 67 | 4.75 MiB | 26.61 MiB |
| `jit__kernel` | 46 | 21.76 GiB | 511.98 GiB |
| `jit__take` | 34 | 28.83 MiB | 120.72 MiB |
| `jit_transpose` | 32 | 546.29 MiB | 7.96 GiB |
| `jit_true_divide` | 28 | 28.83 MiB | 491.59 MiB |
| `jit_dynamic_slice` | 28 | 10.00 MiB | 37.54 MiB |
| `jit_squeeze` | 26 | 4.00 MiB | 8.77 MiB |
| `jit_multiply` | 25 | 28.83 MiB | 177.69 MiB |
| `jit_concatenate` | 16 | 2.11 MiB | 8.44 MiB |
| `jit_gather` | 14 | 112.50 MiB | 371.04 MiB |
| `jit__init_V` | 14 | 14.77 MiB | 205.42 MiB |
| `jit__identity_fn` | 12 | 28.83 MiB | 228.27 MiB |
| `jit__squeeze` | 12 | 5.77 MiB | 53.56 MiB |
| `jit_add` | 12 | 2.64 MiB | 11.86 MiB |
| `jit__multi_slice` | 11 | 43.24 MiB | 341.72 MiB |
| `jit_subtract` | 11 | 8.00 MiB | 24.78 MiB |
| `jit_scatter` | 10 | 34.59 MiB | 305.38 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 353 |
| `xla_python_gpu_callback` | 120 |
| `lorrax_phdf5_write` | 60 |
| `lorrax_phdf5_read` | 33 |
| `lorrax_cusolvermp_batched_solve_lu` | 18 |
| `lorrax_phdf5_read_kchunk_union` | 11 |
| `__cublas$triangularSolve` | 8 |
| `lorrax_cusolvermp_batched_potrs` | 6 |
| `cusolver_getrf_ffi` | 4 |
| `cu_lu_pivots_to_permutation` | 4 |
| `lorrax_cusolvermp_batched_potrf` | 3 |
| `cusolver_syevd_ffi` | 2 |

