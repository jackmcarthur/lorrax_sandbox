# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/C_profile_sigma_2026-04-23/profile/xla_dump`
**Modules dumped:** 232
**Sum of per-module peak live HBM:** 30.03 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0159.jit__kernel` | 4.02 GiB | 3.01 GiB — preallocated-temp: |
| `module_0171.jit__kernel` | 4.02 GiB | 3.01 GiB — preallocated-temp: |
| `module_0229.jit__kernel` | 3.85 GiB | 2.84 GiB — preallocated-temp: |
| `module_0239.jit__kernel` | 3.85 GiB | 2.84 GiB — preallocated-temp: |
| `module_0197.jit_transpose` | 1.98 GiB | 1012.50 MiB — output shape is \|c128[9,11520,640]\|, maybe-live-out: |
| `module_0047.jit__fft_gather_reshard` | 1.00 GiB | 759.38 MiB — preallocated-temp: |
| `module_0048.jit__fft_gather_reshard` | 1.00 GiB | 759.38 MiB — preallocated-temp: |
| `module_0189.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0201.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0214.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0225.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0001.jit__local_fft` | 1012.50 MiB | 506.25 MiB — preallocated-temp: |
| `module_0167.jit__per_rank` | 265.89 MiB | 253.12 MiB — output shape is \|c128[9,20,2,24,24,80]\|, maybe-live-out: |
| `module_0406.jit_sigma_coh` | 239.94 MiB | 140.62 MiB — preallocated-temp: |
| `module_0409.jit_sigma_coh` | 239.94 MiB | 140.62 MiB — preallocated-temp: |
| `module_0385.jit_sigma_sx` | 226.76 MiB | 140.62 MiB — preallocated-temp: |
| `module_0389.jit_sigma_sx` | 226.76 MiB | 140.62 MiB — preallocated-temp: |
| `module_0521.jit__tau_kernel` | 219.30 MiB | 168.75 MiB — preallocated-temp: |
| `module_0573.jit__tau_kernel` | 219.30 MiB | 168.75 MiB — preallocated-temp: |
| `module_0574.jit__tau_kernel` | 219.30 MiB | 168.75 MiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0229.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1540` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0229.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1541` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0239.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1540` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0239.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1541` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0159.jit__kernel` | `all-to-all` | 1012.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:748` | `c128[9,320,1,2,11520]{4,2,1,0,3}` |
| `module_0171.jit__kernel` | `all-to-all` | 1012.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:748` | `c128[9,320,1,2,11520]{4,2,1,0,3}` |
| `module_0159.jit__kernel` | `all-gather-start` | 759.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,40,2,23040]{3,2,0,1}, c128[9,80,2,23040]{3,2,0,1})` |
| `module_0171.jit__kernel` | `all-gather-start` | 759.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,40,2,23040]{3,2,0,1}, c128[9,80,2,23040]{3,2,0,1})` |
| `module_0159.jit__kernel` | `all-to-all` | 253.12 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,20,2,2,23040]{5,3,2,1,0,4}` |
| `module_0171.jit__kernel` | `all-to-all` | 253.12 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,20,2,2,23040]{5,3,2,1,0,4}` |
| `module_0381.jit__solve_w` | `all-gather-start` | 93.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:258` | `(c128[3,640,640]{2,1,0}, c128[12,640,640]{2,1,0})` |
| `module_0159.jit__kernel` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:507` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0171.jit__kernel` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:507` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0311.jit_gather` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224` | `(c128[1,1,1,3,3,1,320,640]{7,5,4,3,2,1,0,6}, c128[1,1,1,3,3,` |
| `module_0159.jit__kernel` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:507` | `(c128[9,320,320]{1,0,2}, c128[9,320,640]{1,0,2})` |
| `module_0171.jit__kernel` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:507` | `(c128[9,320,320]{1,0,2}, c128[9,320,640]{1,0,2})` |
| `module_0311.jit_gather` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224` | `(c128[1,1,1,3,3,1,320,320]{6,5,4,3,2,1,0,7}, c128[1,1,1,3,3,` |
| `module_0603.jit__identity_fn` | `all-gather-start` | 27.69 MiB | `` | `(c128[21,9,40,80]{3,1,0,2}, c128[21,9,80,80]{3,1,0,2})` |
| `module_0605.jit__identity_fn` | `all-gather-start` | 26.37 MiB | `` | `(c128[20,9,40,80]{3,1,0,2}, c128[20,9,80,80]{3,1,0,2})` |
| `module_0315.jit_gather` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:239` | `(c128[9,80,2,320]{2,1,0,3}, c128[9,80,2,640]{2,1,0,3})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 30 | 28.12 MiB | 163.11 MiB |
| `jit_convert_element_type` | 25 | 21.09 MiB | 29.01 MiB |
| `jit__per_rank` | 23 | 1012.50 MiB | 4.43 GiB |
| `jit__psum` | 13 | 72.07 MiB | 79.15 MiB |
| `jit_gather` | 10 | 112.50 MiB | 226.58 MiB |
| `jit_multiply` | 10 | 72.07 MiB | 172.28 MiB |
| `jit__identity_fn` | 10 | 50.76 MiB | 127.47 MiB |
| `jit_true_divide` | 8 | 42.19 MiB | 70.32 MiB |
| `jit_add` | 7 | 72.95 MiB | 178.21 MiB |
| `jit_reshape` | 6 | 28.12 MiB | 112.59 MiB |
| `jit_concatenate` | 6 | 12.66 MiB | 19.50 MiB |
| `jit__kernel` | 4 | 4.02 GiB | 15.75 GiB |
| `jit_transpose` | 4 | 1.98 GiB | 2.02 GiB |
| `jit_sigma_sx` | 4 | 226.76 MiB | 822.66 MiB |
| `jit_minimax_tau_integrate_chi` | 4 | 196.88 MiB | 787.53 MiB |
| `jit_subtract` | 4 | 84.38 MiB | 126.57 MiB |
| `jit__where` | 4 | 29.00 MiB | 87.01 MiB |
| `jit__take` | 4 | 3.16 MiB | 3.70 MiB |
| `jit__tau_kernel` | 3 | 219.30 MiB | 657.89 MiB |
| `jit_iota` | 3 | 640.00 B | 848.00 B |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 55 |
| `lorrax_phdf5_write` | 17 |
| `__cublas$triangularSolve` | 7 |
| `lorrax_phdf5_read_kchunk_union` | 2 |
| `xla_python_gpu_callback` | 2 |
| `lorrax_phdf5_read` | 2 |
| `__cusolver$cholesky` | 1 |
| `cusolver_getrf_ffi` | 1 |
| `cu_lu_pivots_to_permutation` | 1 |
| `cusolver_syevd_ffi` | 1 |

