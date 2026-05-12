# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/03_si_10x10x10_nosym_timing/C_si10_mem35_vqsharded/profile/xla_dump`
**Modules dumped:** 453
**Sum of per-module peak live HBM:** 895.12 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0303.jit__kernel` | 49.01 GiB | 33.78 GiB — preallocated-temp: |
| `module_0384.jit__kernel` | 49.01 GiB | 33.78 GiB — preallocated-temp: |
| `module_0333.jit__kernel` | 39.82 GiB | 28.34 GiB — preallocated-temp: |
| `module_0422.jit__kernel` | 39.82 GiB | 28.34 GiB — preallocated-temp: |
| `module_0147.jit__kernel` | 29.29 GiB | 18.54 GiB — preallocated-temp: |
| `module_0182.jit__kernel` | 29.29 GiB | 18.54 GiB — preallocated-temp: |
| `module_0179.jit__kernel` | 28.32 GiB | 18.54 GiB — preallocated-temp: |
| `module_0217.jit__kernel` | 28.32 GiB | 18.54 GiB — preallocated-temp: |
| `module_0505.jit_sigma_coh` | 18.08 GiB | 14.59 GiB — preallocated-temp: |
| `module_0641.jit_sigma_coh` | 18.08 GiB | 14.59 GiB — preallocated-temp: |
| `module_0485.jit_sigma_sx` | 16.42 GiB | 13.73 GiB — preallocated-temp: |
| `module_0620.jit_sigma_sx` | 16.42 GiB | 13.73 GiB — preallocated-temp: |
| `module_0273.jit__per_rank` | 14.21 GiB | 0.00 B —  |
| `module_0340.jit__per_rank` | 14.21 GiB | 0.00 B —  |
| `module_0657.jit__tau_kernel` | 13.81 GiB | 10.73 GiB — preallocated-temp: |
| `module_0836.jit__tau_kernel` | 13.81 GiB | 10.73 GiB — preallocated-temp: |
| `module_0451.jit__chi_scan` | 12.02 GiB | 10.30 GiB — preallocated-temp: |
| `module_0581.jit__chi_scan` | 12.02 GiB | 10.30 GiB — preallocated-temp: |
| `module_0547.jit__chi_scan` | 12.02 GiB | 10.30 GiB — preallocated-temp: |
| `module_0686.jit__chi_scan` | 12.02 GiB | 10.30 GiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0303.jit__kernel` | `all-gather-start` | 14.48 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1533` | `(c128[575,120,4693]{2,0,1}, c128[575,240,4693]{2,0,1})` |
| `module_0303.jit__kernel` | `all-gather-start` | 14.48 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1534` | `(c128[575,120,4693]{2,0,1}, c128[575,240,4693]{2,0,1})` |
| `module_0384.jit__kernel` | `all-gather-start` | 14.48 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1533` | `(c128[575,120,4693]{2,0,1}, c128[575,240,4693]{2,0,1})` |
| `module_0384.jit__kernel` | `all-gather-start` | 14.48 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1534` | `(c128[575,120,4693]{2,0,1}, c128[575,240,4693]{2,0,1})` |
| `module_0333.jit__kernel` | `all-gather-start` | 10.70 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1533` | `(c128[425,120,4693]{2,0,1}, c128[425,240,4693]{2,0,1})` |
| `module_0333.jit__kernel` | `all-gather-start` | 10.70 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1534` | `(c128[425,120,4693]{2,0,1}, c128[425,240,4693]{2,0,1})` |
| `module_0422.jit__kernel` | `all-gather-start` | 10.70 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1533` | `(c128[425,120,4693]{2,0,1}, c128[425,240,4693]{2,0,1})` |
| `module_0422.jit__kernel` | `all-gather-start` | 10.70 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1534` | `(c128[425,120,4693]{2,0,1}, c128[425,240,4693]{2,0,1})` |
| `module_0147.jit__kernel` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:541` | `(c128[1000,240,480]{2,0,1}, c128[1000,480,480]{2,0,1})` |
| `module_0179.jit__kernel` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:541` | `(c128[1000,240,480]{2,0,1}, c128[1000,480,480]{2,0,1})` |
| `module_0182.jit__kernel` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:541` | `(c128[1000,240,480]{2,0,1}, c128[1000,480,480]{2,0,1})` |
| `module_0217.jit__kernel` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:541` | `(c128[1000,240,480]{2,0,1}, c128[1000,480,480]{2,0,1})` |
| `module_0405.jit_gather` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224` | `(c128[1,1,1,10,10,10,240,480]{7,5,4,3,2,1,0,6}, c128[1,1,1,1` |
| `module_0505.jit_sigma_coh` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:407` | `(c128[1000,240,480]{2,0,1}, c128[1000,480,480]{2,0,1})` |
| `module_0514.jit_gather` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224` | `(c128[1,1,1,10,10,10,240,480]{7,5,4,3,2,1,0,6}, c128[1,1,1,1` |
| `module_0567.jit_subtract` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:633` | `(c128[1000,240,480]{2,0,1}, c128[1000,480,480]{2,0,1})` |
| `module_0641.jit_sigma_coh` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:407` | `(c128[1000,240,480]{2,0,1}, c128[1000,480,480]{2,0,1})` |
| `module_0707.jit_subtract` | `all-gather-start` | 5.15 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:633` | `(c128[1000,240,480]{2,0,1}, c128[1000,480,480]{2,0,1})` |
| `module_0485.jit_sigma_sx` | `all-gather-start` | 4.29 GiB | `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:296` | `(c128[250,480,480]{2,1,0}, c128[1000,480,480]{2,1,0})` |
| `module_0505.jit_sigma_coh` | `all-gather-start` | 4.29 GiB | `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:296` | `(c128[250,480,480]{2,1,0}, c128[1000,480,480]{2,1,0})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 70 | 1.72 GiB | 23.80 GiB |
| `jit_concatenate` | 37 | 242.58 MiB | 1.60 GiB |
| `jit_convert_element_type` | 27 | 1.29 GiB | 3.55 GiB |
| `jit__per_rank` | 24 | 14.21 GiB | 70.51 GiB |
| `jit__identity_fn` | 24 | 5.15 GiB | 31.66 GiB |
| `jit_gather` | 22 | 6.87 GiB | 27.48 GiB |
| `jit_add` | 18 | 4.45 GiB | 21.76 GiB |
| `jit__psum` | 14 | 4.40 GiB | 17.81 GiB |
| `jit_multiply` | 13 | 4.40 GiB | 21.03 GiB |
| `jit_reshape` | 13 | 1.72 GiB | 13.75 GiB |
| `jit_subtract` | 12 | 7.72 GiB | 20.96 GiB |
| `jit_true_divide` | 11 | 2.57 GiB | 8.58 GiB |
| `jit_transpose` | 10 | 5.71 GiB | 24.12 GiB |
| `jit__kernel` | 8 | 49.01 GiB | 292.90 GiB |
| `jit__where` | 8 | 1.77 GiB | 10.62 GiB |
| `jit__multi_slice` | 8 | 1.41 GiB | 10.65 GiB |
| `jit_scatter` | 8 | 1.38 GiB | 10.30 GiB |
| `jit__squeeze` | 8 | 529.10 MiB | 3.43 GiB |
| `jit_greater` | 6 | 494.38 MiB | 2.90 GiB |
| `jit_iota` | 6 | 28.12 KiB | 112.70 KiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 68 |
| `lorrax_phdf5_write` | 20 |
| `__cublas$triangularSolve` | 14 |
| `lorrax_phdf5_read` | 4 |
| `__cusolver$cholesky` | 2 |
| `cusolver_getrf_ffi` | 2 |
| `cu_lu_pivots_to_permutation` | 2 |
| `cusolver_syevd_ffi` | 2 |

