# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_phase3b_661prime_2026-05-09/profile/xla_dump`
**Modules dumped:** 925
**Sum of per-module peak live HBM:** 185.04 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0664.jit__kernel` | 12.76 GiB | 12.48 GiB — preallocated-temp: |
| `module_0735.jit__kernel` | 12.76 GiB | 12.48 GiB — preallocated-temp: |
| `module_0737.jit__kernel` | 12.76 GiB | 12.48 GiB — preallocated-temp: |
| `module_0753.jit__kernel` | 12.76 GiB | 12.48 GiB — preallocated-temp: |
| `module_0859.jit__kernel` | 12.76 GiB | 12.48 GiB — preallocated-temp: |
| `module_0861.jit__kernel` | 12.76 GiB | 12.48 GiB — preallocated-temp: |
| `module_0575.jit__kernel` | 12.76 GiB | 12.48 GiB — preallocated-temp: |
| `module_0611.jit__kernel` | 12.76 GiB | 12.48 GiB — preallocated-temp: |
| `module_0613.jit__kernel` | 12.76 GiB | 12.48 GiB — preallocated-temp: |
| `module_0332.jit__kernel` | 12.13 GiB | 11.87 GiB — preallocated-temp: |
| `module_0349.jit__kernel` | 12.13 GiB | 11.87 GiB — preallocated-temp: |
| `module_0351.jit__kernel` | 12.13 GiB | 11.87 GiB — preallocated-temp: |
| `module_1061.jit__kernel` | 1.32 GiB | 811.30 MiB — preallocated-temp: |
| `module_1063.jit__kernel` | 1.32 GiB | 811.30 MiB — preallocated-temp: |
| `module_1085.jit__kernel` | 1.32 GiB | 811.30 MiB — preallocated-temp: |
| `module_1087.jit__kernel` | 1.32 GiB | 811.30 MiB — preallocated-temp: |
| `module_0979.jit__kernel` | 807.92 MiB | 531.56 MiB — preallocated-temp: |
| `module_0981.jit__kernel` | 807.92 MiB | 531.56 MiB — preallocated-temp: |
| `module_1021.jit__kernel` | 807.92 MiB | 531.56 MiB — preallocated-temp: |
| `module_1023.jit__kernel` | 807.92 MiB | 531.56 MiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0332.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0349.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0351.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0575.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0611.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0613.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0664.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0735.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0737.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0753.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0859.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0861.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0575.jit__kernel` | `all-to-all` | 354.38 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:884` | `c128[4,3,168,11520]{3,2,1,0}` |
| `module_0575.jit__kernel` | `all-to-all` | 354.38 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:885` | `c128[3,672,1,4,2880]{4,2,1,0,3}` |
| `module_0611.jit__kernel` | `all-to-all` | 354.38 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:884` | `c128[4,3,168,11520]{3,2,1,0}` |
| `module_0611.jit__kernel` | `all-to-all` | 354.38 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:885` | `c128[3,672,1,4,2880]{4,2,1,0,3}` |
| `module_0613.jit__kernel` | `all-to-all` | 354.38 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:884` | `c128[4,3,168,11520]{3,2,1,0}` |
| `module_0613.jit__kernel` | `all-to-all` | 354.38 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:885` | `c128[3,672,1,4,2880]{4,2,1,0,3}` |
| `module_0664.jit__kernel` | `all-to-all` | 354.38 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:884` | `c128[4,3,168,11520]{3,2,1,0}` |
| `module_0664.jit__kernel` | `all-to-all` | 354.38 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:885` | `c128[3,672,1,4,2880]{4,2,1,0,3}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 114 | 29.53 MiB | 307.93 MiB |
| `jit__per_rank` | 86 | 265.78 MiB | 6.31 GiB |
| `jit_convert_element_type` | 56 | 510.26 KiB | 1.72 MiB |
| `jit__psum` | 54 | 4.75 MiB | 10.96 MiB |
| `jit_dynamic_slice` | 44 | 920.18 KiB | 12.79 MiB |
| `jit_add` | 43 | 14.64 MiB | 105.39 MiB |
| `jit__take` | 40 | 3.16 MiB | 29.64 MiB |
| `jit_concatenate` | 36 | 2.11 MiB | 12.71 MiB |
| `jit_multiply` | 34 | 1.49 MiB | 13.67 MiB |
| `jit_squeeze` | 32 | 340.17 KiB | 1.06 MiB |
| `jit_true_divide` | 30 | 14.77 MiB | 234.45 MiB |
| `jit__kernel` | 24 | 12.76 GiB | 162.63 GiB |
| `jit_iota` | 19 | 5.25 KiB | 34.86 KiB |
| `jit_transpose` | 18 | 531.56 MiB | 4.24 GiB |
| `jit_gather` | 14 | 77.34 MiB | 265.54 MiB |
| `jit__init_V` | 14 | 3.88 MiB | 53.54 MiB |
| `jit__einsum` | 14 | 3.16 MiB | 24.83 MiB |
| `jit_matmul` | 14 | 1.50 MiB | 4.59 MiB |
| `jit_subtract` | 14 | 340.18 KiB | 1.55 MiB |
| `jit_negative` | 14 | 340.17 KiB | 1.02 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 92 |
| `lorrax_phdf5_write` | 47 |
| `__cublas$triangularSolve` | 26 |
| `lorrax_phdf5_read` | 16 |
| `xla_python_gpu_callback` | 12 |
| `cusolver_getrf_ffi` | 9 |
| `cu_lu_pivots_to_permutation` | 9 |
| `lorrax_phdf5_read_kchunk_union` | 8 |
| `__cusolver$cholesky` | 2 |

