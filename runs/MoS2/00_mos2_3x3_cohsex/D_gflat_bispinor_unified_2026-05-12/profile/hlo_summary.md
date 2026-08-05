# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_gflat_bispinor_unified_2026-05-12/profile/xla_dump`
**Modules dumped:** 1352
**Sum of per-module peak live HBM:** 495.08 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0815.jit__kernel` | 26.39 GiB | 26.03 GiB — preallocated-temp: |
| `module_0817.jit__kernel` | 26.39 GiB | 26.03 GiB — preallocated-temp: |
| `module_0911.jit__kernel` | 26.39 GiB | 26.03 GiB — preallocated-temp: |
| `module_1257.jit__kernel` | 26.39 GiB | 26.03 GiB — preallocated-temp: |
| `module_1259.jit__kernel` | 26.39 GiB | 26.03 GiB — preallocated-temp: |
| `module_1314.jit__kernel` | 26.39 GiB | 26.03 GiB — preallocated-temp: |
| `module_1679.jit__kernel` | 26.39 GiB | 26.03 GiB — preallocated-temp: |
| `module_1697.jit__kernel` | 26.39 GiB | 26.03 GiB — preallocated-temp: |
| `module_1699.jit__kernel` | 26.39 GiB | 26.03 GiB — preallocated-temp: |
| `module_0351.jit__kernel` | 25.71 GiB | 25.35 GiB — preallocated-temp: |
| `module_0353.jit__kernel` | 25.71 GiB | 25.35 GiB — preallocated-temp: |
| `module_0446.jit__kernel` | 25.71 GiB | 25.35 GiB — preallocated-temp: |
| `module_0971.jit__kernel` | 3.88 GiB | 3.79 GiB — preallocated-temp: |
| `module_0973.jit__kernel` | 3.88 GiB | 3.79 GiB — preallocated-temp: |
| `module_1052.jit__kernel` | 3.88 GiB | 3.79 GiB — preallocated-temp: |
| `module_1411.jit__kernel` | 3.88 GiB | 3.79 GiB — preallocated-temp: |
| `module_1413.jit__kernel` | 3.88 GiB | 3.79 GiB — preallocated-temp: |
| `module_1436.jit__kernel` | 3.88 GiB | 3.79 GiB — preallocated-temp: |
| `module_1801.jit__kernel` | 3.88 GiB | 3.79 GiB — preallocated-temp: |
| `module_1851.jit__kernel` | 3.88 GiB | 3.79 GiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0935.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_0937.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_0953.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_0955.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_1031.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_1375.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_1377.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_1393.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_1395.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_1815.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_1817.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_1833.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_1835.jit_fn` | `all-gather-start` | 1.62 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,656,3673]{1,0,2}, c128[9,656,14692]{1,0,2})` |
| `module_0455.jit_fn` | `all-gather-start` | 1.58 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,640,3673]{1,0,2}, c128[9,640,14692]{1,0,2})` |
| `module_0457.jit_fn` | `all-gather-start` | 1.58 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,640,3673]{1,0,2}, c128[9,640,14692]{1,0,2})` |
| `module_0473.jit_fn` | `all-gather-start` | 1.58 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,640,3673]{1,0,2}, c128[9,640,14692]{1,0,2})` |
| `module_0475.jit_fn` | `all-gather-start` | 1.58 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,640,3673]{1,0,2}, c128[9,640,14692]{1,0,2})` |
| `module_0550.jit_fn` | `all-gather-start` | 1.58 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:627` | `(c128[9,640,3673]{1,0,2}, c128[9,640,14692]{1,0,2})` |
| `module_0351.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |
| `module_0351.jit__kernel` | `all-gather-start` | 506.25 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,4,9,24,24,80]{5,4,3,2,1,0}, c128[16,4,9,24,24,80]{5,` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 163 | 44.21 MiB | 897.75 MiB |
| `jit__per_rank` | 136 | 56.25 MiB | 1.46 GiB |
| `jit_convert_element_type` | 104 | 6.00 MiB | 28.23 MiB |
| `jit__psum` | 83 | 4.75 MiB | 34.76 MiB |
| `jit_dynamic_slice` | 56 | 10.00 MiB | 59.96 MiB |
| `jit_squeeze` | 53 | 4.00 MiB | 14.97 MiB |
| `jit_concatenate` | 51 | 8.63 MiB | 130.48 MiB |
| `jit_multiply` | 50 | 28.83 MiB | 309.62 MiB |
| `jit_fn` | 43 | 3.53 GiB | 88.65 GiB |
| `jit__take` | 38 | 28.83 MiB | 178.91 MiB |
| `jit__lambda_` | 35 | 44.21 MiB | 571.13 MiB |
| `jit_true_divide` | 35 | 28.83 MiB | 620.36 MiB |
| `jit_add` | 32 | 2.64 MiB | 30.17 MiB |
| `jit__kernel` | 24 | 26.39 GiB | 360.90 GiB |
| `jit_transpose` | 24 | 28.83 MiB | 527.63 MiB |
| `jit_subtract` | 24 | 8.00 MiB | 44.92 MiB |
| `jit_gather` | 20 | 112.50 MiB | 441.57 MiB |
| `jit__einsum` | 20 | 40.00 MiB | 283.55 MiB |
| `jit_iota` | 19 | 7.67 KiB | 69.17 KiB |
| `jit_matmul` | 18 | 16.00 MiB | 61.35 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 382 |
| `xla_python_gpu_callback` | 120 |
| `__cublas$triangularSolve` | 61 |
| `lorrax_phdf5_write` | 49 |
| `lorrax_phdf5_read_kchunk_union` | 43 |
| `lorrax_phdf5_read` | 33 |
| `cusolver_getrf_ffi` | 23 |
| `cu_lu_pivots_to_permutation` | 23 |
| `__cusolver$cholesky` | 3 |
| `cusolver_syevd_ffi` | 3 |

