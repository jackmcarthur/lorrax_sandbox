# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/C_profile_sigma_2026-04-23/profile_v2/xla_dump`
**Modules dumped:** 433
**Sum of per-module peak live HBM:** 46.73 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0179.jit__kernel` | 4.02 GiB | 3.01 GiB — preallocated-temp: |
| `module_0191.jit__kernel` | 4.02 GiB | 3.01 GiB — preallocated-temp: |
| `module_0193.jit__kernel` | 4.02 GiB | 3.01 GiB — preallocated-temp: |
| `module_0249.jit__kernel` | 3.85 GiB | 2.84 GiB — preallocated-temp: |
| `module_0259.jit__kernel` | 3.85 GiB | 2.84 GiB — preallocated-temp: |
| `module_0261.jit__kernel` | 3.85 GiB | 2.84 GiB — preallocated-temp: |
| `module_0217.jit_transpose` | 1.98 GiB | 1012.50 MiB — output shape is \|c128[9,11520,640]\|, maybe-live-out: |
| `module_0219.jit_transpose` | 1.98 GiB | 1012.50 MiB — output shape is \|c128[9,11520,640]\|, maybe-live-out: |
| `module_0067.jit__fft_gather_reshard` | 1.00 GiB | 759.38 MiB — preallocated-temp: |
| `module_0068.jit__fft_gather_reshard` | 1.00 GiB | 759.38 MiB — preallocated-temp: |
| `module_0069.jit__fft_gather_reshard` | 1.00 GiB | 759.38 MiB — preallocated-temp: |
| `module_0209.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0221.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0223.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0234.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0245.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0247.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0021.jit__local_fft` | 1012.50 MiB | 506.25 MiB — preallocated-temp: |
| `module_0023.jit__local_fft` | 1012.50 MiB | 506.25 MiB — preallocated-temp: |
| `module_0187.jit__per_rank` | 265.89 MiB | 253.12 MiB — output shape is \|c128[9,20,2,24,24,80]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0249.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1540` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0249.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1541` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0259.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1540` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0259.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1541` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0261.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1540` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0261.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1541` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0179.jit__kernel` | `all-to-all` | 1012.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:748` | `c128[9,320,1,2,11520]{4,2,1,0,3}` |
| `module_0191.jit__kernel` | `all-to-all` | 1012.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:748` | `c128[9,320,1,2,11520]{4,2,1,0,3}` |
| `module_0193.jit__kernel` | `all-to-all` | 1012.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:748` | `c128[9,320,1,2,11520]{4,2,1,0,3}` |
| `module_0179.jit__kernel` | `all-gather-start` | 759.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,40,2,23040]{3,2,0,1}, c128[9,80,2,23040]{3,2,0,1})` |
| `module_0191.jit__kernel` | `all-gather-start` | 759.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,40,2,23040]{3,2,0,1}, c128[9,80,2,23040]{3,2,0,1})` |
| `module_0193.jit__kernel` | `all-gather-start` | 759.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,40,2,23040]{3,2,0,1}, c128[9,80,2,23040]{3,2,0,1})` |
| `module_0179.jit__kernel` | `all-to-all` | 253.12 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,20,2,2,23040]{5,3,2,1,0,4}` |
| `module_0191.jit__kernel` | `all-to-all` | 253.12 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,20,2,2,23040]{5,3,2,1,0,4}` |
| `module_0193.jit__kernel` | `all-to-all` | 253.12 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,20,2,2,23040]{5,3,2,1,0,4}` |
| `module_0401.jit__solve_w` | `all-gather-start` | 93.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:258` | `(c128[3,640,640]{2,1,0}, c128[12,640,640]{2,1,0})` |
| `module_0403.jit__solve_w` | `all-gather-start` | 93.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:258` | `(c128[3,640,640]{2,1,0}, c128[12,640,640]{2,1,0})` |
| `module_0179.jit__kernel` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:507` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0191.jit__kernel` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:507` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0193.jit__kernel` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:507` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 59 | 28.12 MiB | 312.15 MiB |
| `jit_convert_element_type` | 51 | 21.09 MiB | 58.02 MiB |
| `jit__per_rank` | 32 | 1012.50 MiB | 6.72 GiB |
| `jit__psum` | 25 | 72.07 MiB | 153.03 MiB |
| `jit_gather` | 20 | 112.50 MiB | 453.16 MiB |
| `jit_multiply` | 19 | 72.07 MiB | 272.48 MiB |
| `jit__identity_fn` | 18 | 50.76 MiB | 204.16 MiB |
| `jit_true_divide` | 16 | 42.19 MiB | 140.64 MiB |
| `jit_add` | 12 | 72.95 MiB | 280.84 MiB |
| `jit_concatenate` | 12 | 12.66 MiB | 39.01 MiB |
| `jit_reshape` | 11 | 28.12 MiB | 197.05 MiB |
| `jit__where` | 8 | 29.00 MiB | 174.02 MiB |
| `jit__take` | 8 | 3.16 MiB | 7.41 MiB |
| `jit_transpose` | 7 | 1.98 GiB | 4.02 GiB |
| `jit__multi_slice` | 7 | 21.09 MiB | 84.38 MiB |
| `jit__kernel` | 6 | 4.02 GiB | 23.62 GiB |
| `jit_sigma_sx` | 6 | 226.76 MiB | 1.21 GiB |
| `jit_minimax_tau_integrate_chi` | 6 | 196.88 MiB | 1.15 GiB |
| `jit_subtract` | 6 | 84.38 MiB | 168.76 MiB |
| `jit_iota` | 6 | 640.00 B | 1.66 KiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 76 |
| `lorrax_phdf5_write` | 22 |
| `__cublas$triangularSolve` | 12 |
| `lorrax_phdf5_read_kchunk_union` | 3 |
| `xla_python_gpu_callback` | 3 |
| `lorrax_phdf5_read` | 3 |
| `__cusolver$cholesky` | 2 |
| `cusolver_getrf_ffi` | 2 |
| `cu_lu_pivots_to_permutation` | 2 |
| `cusolver_syevd_ffi` | 2 |

