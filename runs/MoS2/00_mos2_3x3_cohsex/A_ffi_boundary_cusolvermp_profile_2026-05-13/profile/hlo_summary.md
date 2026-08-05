# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_cusolvermp_profile_2026-05-13/profile/xla_dump`
**Modules dumped:** 1033
**Sum of per-module peak live HBM:** 186.85 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0437.jit__kernel` | 6.25 GiB | 6.08 GiB — preallocated-temp: |
| `module_0517.jit__kernel` | 6.25 GiB | 6.08 GiB — preallocated-temp: |
| `module_0519.jit__kernel` | 6.25 GiB | 6.08 GiB — preallocated-temp: |
| `module_0590.jit__kernel` | 6.25 GiB | 6.08 GiB — preallocated-temp: |
| `module_0697.jit__kernel` | 6.25 GiB | 6.08 GiB — preallocated-temp: |
| `module_0699.jit__kernel` | 6.25 GiB | 6.08 GiB — preallocated-temp: |
| `module_0743.jit__kernel` | 6.25 GiB | 6.08 GiB — preallocated-temp: |
| `module_0877.jit__kernel` | 6.25 GiB | 6.08 GiB — preallocated-temp: |
| `module_0879.jit__kernel` | 6.25 GiB | 6.08 GiB — preallocated-temp: |
| `module_0272.jit__kernel` | 6.10 GiB | 5.93 GiB — preallocated-temp: |
| `module_0309.jit__kernel` | 6.10 GiB | 5.93 GiB — preallocated-temp: |
| `module_0311.jit__kernel` | 6.10 GiB | 5.93 GiB — preallocated-temp: |
| `module_1093.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_1095.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_1117.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_1119.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_0365.jit_fn` | 3.98 GiB | 3.96 GiB — preallocated-temp: |
| `module_0367.jit_fn` | 3.98 GiB | 3.96 GiB — preallocated-temp: |
| `module_0111.jit_fn` | 3.98 GiB | 3.96 GiB — preallocated-temp: |
| `module_0113.jit_fn` | 3.98 GiB | 3.96 GiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0111.jit_fn` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:291` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0113.jit_fn` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:291` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0272.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0309.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0311.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0365.jit_fn` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0367.jit_fn` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0437.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0517.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0519.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0590.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0697.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0699.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0743.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0877.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0879.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_1019.jit__kernel` | `all-gather-start` | 163.44 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:717` | `(c128[9,164,2419]{2,0,1}, c128[9,328,2419]{2,0,1})` |
| `module_1019.jit__kernel` | `all-gather-start` | 163.44 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:718` | `(c128[9,164,2419]{2,0,1}, c128[9,328,2419]{2,0,1})` |
| `module_1021.jit__kernel` | `all-gather-start` | 163.44 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:717` | `(c128[9,164,2419]{2,0,1}, c128[9,328,2419]{2,0,1})` |
| `module_1021.jit__kernel` | `all-gather-start` | 163.44 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:718` | `(c128[9,164,2419]{2,0,1}, c128[9,328,2419]{2,0,1})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit__per_rank` | 106 | 1.01 GiB | 12.74 GiB |
| `jit_broadcast_in_dim` | 106 | 28.83 MiB | 312.91 MiB |
| `jit_convert_element_type` | 85 | 6.00 MiB | 22.01 MiB |
| `jit__psum` | 67 | 4.75 MiB | 26.61 MiB |
| `jit_dynamic_slice` | 50 | 10.00 MiB | 42.74 MiB |
| `jit_squeeze` | 48 | 4.00 MiB | 11.37 MiB |
| `jit_concatenate` | 46 | 43.13 MiB | 472.11 MiB |
| `jit_multiply` | 45 | 28.83 MiB | 397.86 MiB |
| `jit__take` | 34 | 28.83 MiB | 120.72 MiB |
| `jit_add` | 32 | 2.64 MiB | 27.87 MiB |
| `jit_true_divide` | 28 | 28.83 MiB | 491.44 MiB |
| `jit__kernel` | 24 | 6.25 GiB | 119.25 GiB |
| `jit_transpose` | 24 | 259.45 MiB | 2.34 GiB |
| `jit_subtract` | 21 | 8.00 MiB | 32.77 MiB |
| `jit__einsum` | 18 | 40.00 MiB | 490.56 MiB |
| `jit_matmul` | 17 | 16.00 MiB | 47.13 MiB |
| `jit_negative` | 16 | 552.09 KiB | 6.06 MiB |
| `jit_gather` | 14 | 112.50 MiB | 371.02 MiB |
| `jit__init_V` | 14 | 14.77 MiB | 205.42 MiB |
| `jit__identity_fn` | 12 | 28.83 MiB | 228.27 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 137 |
| `lorrax_phdf5_write` | 50 |
| `lorrax_phdf5_read` | 33 |
| `lorrax_phdf5_read_kchunk_union` | 13 |
| `xla_python_gpu_callback` | 12 |
| `lorrax_cusolvermp_batched_solve_lu` | 9 |
| `__cublas$triangularSolve` | 8 |
| `cusolver_getrf_ffi` | 4 |
| `cu_lu_pivots_to_permutation` | 4 |
| `lorrax_cusolvermp_batched_potrf` | 3 |
| `lorrax_cusolvermp_batched_potrs` | 3 |
| `cusolver_syevd_ffi` | 2 |

