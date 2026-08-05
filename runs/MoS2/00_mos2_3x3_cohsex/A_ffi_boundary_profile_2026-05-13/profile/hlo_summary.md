# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_profile_2026-05-13/profile/xla_dump`
**Modules dumped:** 1077
**Sum of per-module peak live HBM:** 206.26 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0457.jit__kernel` | 6.30 GiB | 6.12 GiB — preallocated-temp: |
| `module_0537.jit__kernel` | 6.30 GiB | 6.12 GiB — preallocated-temp: |
| `module_0539.jit__kernel` | 6.30 GiB | 6.12 GiB — preallocated-temp: |
| `module_0611.jit__kernel` | 6.30 GiB | 6.12 GiB — preallocated-temp: |
| `module_0717.jit__kernel` | 6.30 GiB | 6.12 GiB — preallocated-temp: |
| `module_0719.jit__kernel` | 6.30 GiB | 6.12 GiB — preallocated-temp: |
| `module_0764.jit__kernel` | 6.30 GiB | 6.12 GiB — preallocated-temp: |
| `module_0897.jit__kernel` | 6.30 GiB | 6.12 GiB — preallocated-temp: |
| `module_0899.jit__kernel` | 6.30 GiB | 6.12 GiB — preallocated-temp: |
| `module_0291.jit__kernel` | 6.10 GiB | 5.93 GiB — preallocated-temp: |
| `module_0329.jit__kernel` | 6.10 GiB | 5.93 GiB — preallocated-temp: |
| `module_0331.jit__kernel` | 6.10 GiB | 5.93 GiB — preallocated-temp: |
| `module_1015.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_1041.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_1113.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_1115.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_1137.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_1139.jit__kernel` | 5.14 GiB | 3.09 GiB — preallocated-temp: |
| `module_0385.jit_fn` | 3.98 GiB | 3.96 GiB — preallocated-temp: |
| `module_0387.jit_fn` | 3.98 GiB | 3.96 GiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0111.jit_fn` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:291` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0113.jit_fn` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:291` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0291.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0329.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0331.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0385.jit_fn` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0387.jit_fn` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0457.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0537.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0539.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0611.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0717.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0719.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0764.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0897.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0899.jit__kernel` | `all-gather-start` | 2.47 GiB | `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:354` | `(c128[9,20,4,24,24,80]{5,4,3,0,2,1}, c128[9,80,4,24,24,80]{5` |
| `module_0921.jit__kernel` | `all-gather-start` | 163.44 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:717` | `(c128[9,164,2419]{2,0,1}, c128[9,328,2419]{2,0,1})` |
| `module_0921.jit__kernel` | `all-gather-start` | 163.44 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:718` | `(c128[9,164,2419]{2,0,1}, c128[9,328,2419]{2,0,1})` |
| `module_0977.jit__kernel` | `all-gather-start` | 163.44 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:717` | `(c128[9,164,2419]{2,0,1}, c128[9,328,2419]{2,0,1})` |
| `module_0977.jit__kernel` | `all-gather-start` | 163.44 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:718` | `(c128[9,164,2419]{2,0,1}, c128[9,328,2419]{2,0,1})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 111 | 28.83 MiB | 313.04 MiB |
| `jit__per_rank` | 103 | 1.01 GiB | 11.68 GiB |
| `jit_convert_element_type` | 86 | 6.00 MiB | 22.49 MiB |
| `jit__psum` | 67 | 4.75 MiB | 26.61 MiB |
| `jit_dynamic_slice` | 51 | 10.00 MiB | 43.40 MiB |
| `jit_squeeze` | 49 | 4.00 MiB | 11.70 MiB |
| `jit_multiply` | 48 | 28.83 MiB | 400.85 MiB |
| `jit_concatenate` | 46 | 43.13 MiB | 472.11 MiB |
| `jit__take` | 34 | 28.83 MiB | 120.72 MiB |
| `jit_add` | 34 | 2.64 MiB | 28.44 MiB |
| `jit__kernel` | 29 | 6.30 GiB | 138.92 GiB |
| `jit_true_divide` | 29 | 28.83 MiB | 491.94 MiB |
| `jit_transpose` | 26 | 259.45 MiB | 2.84 GiB |
| `jit_subtract` | 22 | 8.00 MiB | 33.10 MiB |
| `jit__einsum` | 18 | 40.00 MiB | 490.56 MiB |
| `jit_matmul` | 18 | 16.00 MiB | 48.62 MiB |
| `jit_negative` | 17 | 552.09 KiB | 6.39 MiB |
| `jit_gather` | 15 | 112.50 MiB | 372.14 MiB |
| `jit__init_V` | 14 | 14.77 MiB | 205.42 MiB |
| `jit__identity_fn` | 12 | 28.83 MiB | 228.27 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 144 |
| `lorrax_phdf5_write` | 49 |
| `__cublas$triangularSolve` | 34 |
| `lorrax_phdf5_read` | 32 |
| `lorrax_phdf5_read_kchunk_union` | 13 |
| `cusolver_getrf_ffi` | 13 |
| `cu_lu_pivots_to_permutation` | 13 |
| `xla_python_gpu_callback` | 12 |
| `__cusolver$cholesky` | 2 |
| `cusolver_syevd_ffi` | 2 |

