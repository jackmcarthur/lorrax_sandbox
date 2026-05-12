# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_phase0_hlo_2026-05-08/profile/xla_dump`
**Modules dumped:** 805
**Sum of per-module peak live HBM:** 48.96 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0506.jit__kernel` | 1.77 GiB | 1.50 GiB — preallocated-temp: |
| `module_0557.jit__kernel` | 1.77 GiB | 1.50 GiB — preallocated-temp: |
| `module_0559.jit__kernel` | 1.77 GiB | 1.50 GiB — preallocated-temp: |
| `module_0561.jit__kernel` | 1.77 GiB | 1.50 GiB — preallocated-temp: |
| `module_0608.jit__kernel` | 1.77 GiB | 1.50 GiB — preallocated-temp: |
| `module_0629.jit__kernel` | 1.77 GiB | 1.50 GiB — preallocated-temp: |
| `module_0631.jit__kernel` | 1.77 GiB | 1.50 GiB — preallocated-temp: |
| `module_0699.jit__kernel` | 1.77 GiB | 1.50 GiB — preallocated-temp: |
| `module_0701.jit__kernel` | 1.77 GiB | 1.50 GiB — preallocated-temp: |
| `module_0251.jit__kernel` | 1.75 GiB | 1.48 GiB — preallocated-temp: |
| `module_0277.jit__kernel` | 1.75 GiB | 1.48 GiB — preallocated-temp: |
| `module_0279.jit__kernel` | 1.75 GiB | 1.48 GiB — preallocated-temp: |
| `module_0865.jit__kernel` | 1.29 GiB | 791.98 MiB — preallocated-temp: |
| `module_0867.jit__kernel` | 1.29 GiB | 791.98 MiB — preallocated-temp: |
| `module_0889.jit__kernel` | 1.29 GiB | 791.98 MiB — preallocated-temp: |
| `module_0891.jit__kernel` | 1.29 GiB | 791.98 MiB — preallocated-temp: |
| `module_0791.jit__kernel` | 788.75 MiB | 518.91 MiB — preallocated-temp: |
| `module_0793.jit__kernel` | 788.75 MiB | 518.91 MiB — preallocated-temp: |
| `module_0833.jit__kernel` | 788.75 MiB | 518.91 MiB — preallocated-temp: |
| `module_0835.jit__kernel` | 788.75 MiB | 518.91 MiB — preallocated-temp: |
| `module_0747.jit__kernel` | 769.59 MiB | 506.25 MiB — preallocated-temp: |
| `module_0749.jit__kernel` | 769.59 MiB | 506.25 MiB — preallocated-temp: |
| `module_0771.jit__kernel` | 769.59 MiB | 506.25 MiB — preallocated-temp: |
| `module_0773.jit__kernel` | 769.59 MiB | 506.25 MiB — preallocated-temp: |
| `module_0561.jit_transpose` | 518.91 MiB | 259.45 MiB — output shape is \|c128[9,2880,656]\|, maybe-live-out: |
| `module_0563.jit_transpose` | 518.91 MiB | 259.45 MiB — output shape is \|c128[9,2880,656]\|, maybe-live-out: |
| `module_0631.jit_transpose` | 518.91 MiB | 259.45 MiB — output shape is \|c128[9,2880,656]\|, maybe-live-out: |
| `module_0633.jit_transpose` | 518.91 MiB | 259.45 MiB — output shape is \|c128[9,2880,656]\|, maybe-live-out: |
| `module_0701.jit_transpose` | 518.91 MiB | 259.45 MiB — output shape is \|c128[9,2880,656]\|, maybe-live-out: |
| `module_0703.jit_transpose` | 518.91 MiB | 259.45 MiB — output shape is \|c128[9,2880,656]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0251.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0277.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0279.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0506.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0557.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0559.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0561.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0608.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0629.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0631.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0699.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0701.jit__kernel` | `all-gather-start` | 632.81 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[9,20,4,11520]{3,2,0,1}, c128[9,80,4,11520]{3,2,0,1})` |
| `module_0506.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,3,164,11520]{3,2,1,0}` |
| `module_0506.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[3,656,1,4,2880]{4,2,1,0,3}` |
| `module_0557.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,3,164,11520]{3,2,1,0}` |
| `module_0557.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[3,656,1,4,2880]{4,2,1,0,3}` |
| `module_0559.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,3,164,11520]{3,2,1,0}` |
| `module_0559.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[3,656,1,4,2880]{4,2,1,0,3}` |
| `module_0561.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,3,164,11520]{3,2,1,0}` |
| `module_0561.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[3,656,1,4,2880]{4,2,1,0,3}` |
| `module_0608.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,3,164,11520]{3,2,1,0}` |
| `module_0608.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[3,656,1,4,2880]{4,2,1,0,3}` |
| `module_0629.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,3,164,11520]{3,2,1,0}` |
| `module_0629.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[3,656,1,4,2880]{4,2,1,0,3}` |
| `module_0631.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,3,164,11520]{3,2,1,0}` |
| `module_0631.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[3,656,1,4,2880]{4,2,1,0,3}` |
| `module_0699.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,3,164,11520]{3,2,1,0}` |
| `module_0699.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[3,656,1,4,2880]{4,2,1,0,3}` |
| `module_0701.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,3,164,11520]{3,2,1,0}` |
| `module_0701.jit__kernel` | `all-to-all` | 345.94 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[3,656,1,4,2880]{4,2,1,0,3}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 94 | 28.83 MiB | 219.84 MiB |
| `jit__per_rank` | 85 | 259.45 MiB | 6.00 GiB |
| `jit__psum` | 54 | 4.75 MiB | 10.96 MiB |
| `jit_convert_element_type` | 49 | 510.26 KiB | 1.64 MiB |
| `jit__take` | 40 | 3.16 MiB | 29.64 MiB |
| `jit_add` | 36 | 567.16 KiB | 2.89 MiB |
| `jit_multiply` | 34 | 1.49 MiB | 13.67 MiB |
| `jit_concatenate` | 32 | 2.11 MiB | 12.71 MiB |
| `jit_true_divide` | 30 | 14.41 MiB | 230.23 MiB |
| `jit_dynamic_slice` | 30 | 920.18 KiB | 12.79 MiB |
| `jit__kernel` | 24 | 1.77 GiB | 32.41 GiB |
| `jit_squeeze` | 18 | 340.17 KiB | 1.06 MiB |
| `jit__init_V` | 14 | 3.69 MiB | 51.35 MiB |
| `jit__einsum` | 14 | 3.16 MiB | 24.83 MiB |
| `jit_matmul` | 14 | 1.50 MiB | 4.59 MiB |
| `jit_subtract` | 14 | 340.18 KiB | 1.55 MiB |
| `jit_negative` | 14 | 340.17 KiB | 1.02 MiB |
| `jit_transpose` | 13 | 518.91 MiB | 4.10 GiB |
| `jit__identity_fn` | 12 | 14.41 MiB | 114.32 MiB |
| `jit__multi_slice` | 11 | 36.04 MiB | 284.77 MiB |
| `jit_gather` | 10 | 77.34 MiB | 249.11 MiB |
| `jit__squeeze` | 10 | 14.41 MiB | 113.91 MiB |
| `jit__broadcast_arrays` | 10 | 1.06 MiB | 4.26 MiB |
| `jit_iota` | 10 | 640.00 B | 3.31 KiB |
| `jit__local_fftn` | 9 | 115.31 MiB | 857.81 MiB |
| `jit__expand` | 8 | 204.39 MiB | 1.60 GiB |
| `jit_scatter` | 8 | 21.62 MiB | 170.86 MiB |
| `jit__compute_CCT_LR` | 8 | 14.77 MiB | 116.77 MiB |
| `jit_reshape` | 8 | 7.03 MiB | 32.34 MiB |
| `jit__init_g0` | 8 | 23.06 KiB | 183.38 KiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 103 |
| `lorrax_phdf5_write` | 47 |
| `__cublas$triangularSolve` | 26 |
| `lorrax_phdf5_read` | 15 |
| `xla_python_gpu_callback` | 12 |
| `lorrax_phdf5_read_kchunk_union` | 9 |
| `cusolver_getrf_ffi` | 9 |
| `cu_lu_pivots_to_permutation` | 9 |
| `__cusolver$cholesky` | 2 |

