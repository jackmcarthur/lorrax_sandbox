# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_perf_before_2026-05-12/profile/xla_dump`
**Modules dumped:** 1087
**Sum of per-module peak live HBM:** 219.40 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0563.jit__kernel` | 10.40 GiB | 10.23 GiB — preallocated-temp: |
| `module_0643.jit__kernel` | 10.40 GiB | 10.23 GiB — preallocated-temp: |
| `module_0645.jit__kernel` | 10.40 GiB | 10.23 GiB — preallocated-temp: |
| `module_0793.jit__kernel` | 10.40 GiB | 10.23 GiB — preallocated-temp: |
| `module_0899.jit__kernel` | 10.40 GiB | 10.23 GiB — preallocated-temp: |
| `module_0901.jit__kernel` | 10.40 GiB | 10.23 GiB — preallocated-temp: |
| `module_1022.jit__kernel` | 10.40 GiB | 10.23 GiB — preallocated-temp: |
| `module_1155.jit__kernel` | 10.40 GiB | 10.23 GiB — preallocated-temp: |
| `module_1157.jit__kernel` | 10.40 GiB | 10.23 GiB — preallocated-temp: |
| `module_0311.jit__kernel` | 10.11 GiB | 9.94 GiB — preallocated-temp: |
| `module_0349.jit__kernel` | 10.11 GiB | 9.94 GiB — preallocated-temp: |
| `module_0351.jit__kernel` | 10.11 GiB | 9.94 GiB — preallocated-temp: |
| `module_1439.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_1441.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_1463.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_1465.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_1365.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |
| `module_1367.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |
| `module_1407.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |
| `module_1409.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0311.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0311.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0311.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0311.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0311.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0349.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0349.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0349.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0349.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0349.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0351.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0351.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0351.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0351.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0351.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0563.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0563.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0563.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0563.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0563.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit__per_rank` | 134 | 1.01 GiB | 11.36 GiB |
| `jit_broadcast_in_dim` | 113 | 28.83 MiB | 312.94 MiB |
| `jit_convert_element_type` | 87 | 6.00 MiB | 22.17 MiB |
| `jit__psum` | 67 | 4.75 MiB | 26.61 MiB |
| `jit_dynamic_slice` | 50 | 10.00 MiB | 42.94 MiB |
| `jit_squeeze` | 48 | 4.00 MiB | 11.46 MiB |
| `jit_concatenate` | 46 | 8.63 MiB | 127.05 MiB |
| `jit_multiply` | 45 | 28.83 MiB | 226.22 MiB |
| `jit__take` | 34 | 28.83 MiB | 120.72 MiB |
| `jit_add` | 32 | 2.64 MiB | 28.04 MiB |
| `jit_true_divide` | 28 | 28.83 MiB | 491.59 MiB |
| `jit_transpose` | 26 | 259.45 MiB | 2.84 GiB |
| `jit__kernel` | 24 | 10.40 GiB | 168.68 GiB |
| `jit_subtract` | 21 | 8.00 MiB | 32.87 MiB |
| `jit__einsum` | 18 | 40.00 MiB | 224.11 MiB |
| `jit_matmul` | 17 | 16.00 MiB | 47.57 MiB |
| `jit_negative` | 16 | 552.09 KiB | 6.15 MiB |
| `jit_gather` | 14 | 112.50 MiB | 371.04 MiB |
| `jit__init_V` | 14 | 14.77 MiB | 205.42 MiB |
| `jit__identity_fn` | 12 | 28.83 MiB | 228.27 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 235 |
| `xla_python_gpu_callback` | 60 |
| `lorrax_phdf5_write` | 49 |
| `lorrax_phdf5_read_kchunk_union` | 43 |
| `__cublas$triangularSolve` | 34 |
| `lorrax_phdf5_read` | 32 |
| `cusolver_getrf_ffi` | 13 |
| `cu_lu_pivots_to_permutation` | 13 |
| `__cusolver$cholesky` | 2 |
| `cusolver_syevd_ffi` | 2 |

