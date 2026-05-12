# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v4_rebased/xla_dump`
**Modules dumped:** 183
**Sum of per-module peak live HBM:** 9.90 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0101.jit__accum` | 2.33 GiB | 1.65 GiB — preallocated-temp: |
| `module_0097.jit__reshard_rchunk` | 1.65 GiB | 675.00 MiB — preallocated-temp: |
| `module_0367.jit__kpath_batch` | 1.44 GiB | 1.32 GiB — preallocated-temp: |
| `module_0041.jit__fft_gather_reshard` | 1.33 GiB | 1012.50 MiB — preallocated-temp: |
| `module_0089.jit__fft_and_rslice` | 1012.80 MiB | 337.80 MiB — preallocated-temp: |
| `module_0317.jit_reshape` | 281.25 MiB | 112.50 MiB — preallocated-temp: |
| `module_0221.jit__build_fH` | 182.82 MiB | 56.25 MiB — preallocated-temp: |
| `module_0331.jit_matmul` | 119.53 MiB | 112.50 MiB — parameter 1, shape \|c128[16,460800]\| at ShapeIndex {}: |
| `module_0347.jit_subtract` | 119.53 MiB | 56.25 MiB — output shape is \|c128[16,480,480]\|, maybe-live-out: |
| `module_0229.jit_real` | 84.38 MiB | 56.25 MiB — parameter 0, shape \|c128[16,480,480]\| at ShapeIndex {}: |
| `module_0235.jit_imag` | 84.38 MiB | 56.25 MiB — parameter 0, shape \|c128[16,480,480]\| at ShapeIndex {}: |
| `module_0349.jit_abs` | 84.38 MiB | 56.25 MiB — parameter 0, shape \|c128[16,480,480]\| at ShapeIndex {}: |
| `module_0239.jit_gather` | 59.77 MiB | 56.25 MiB — parameter 0, shape \|c128[16,480,480]\| at ShapeIndex {}: |
| `module_0067.jit__svd_replicated` | 56.26 MiB | 28.12 MiB — preallocated-temp: |
| `module_0237.jit_abs` | 56.25 MiB | 28.12 MiB — output shape is \|f64[16,480,480]\|, maybe-live-out: |
| `module_0105.jit_matmul` | 46.19 MiB | 14.06 MiB — output shape is \|c128[960,960]\|, maybe-live-out: |
| `module_0103.jit_cholesky` | 45.70 MiB | 28.12 MiB — preallocated-temp: |
| `module_0287.jit_cholesky` | 42.24 MiB | 14.12 MiB — preallocated-temp: |
| `module_0273.jit_add` | 42.19 MiB | 14.06 MiB — output shape is \|c128[960,960]\|, maybe-live-out: |
| `module_0065.jit_reshape` | 35.16 MiB | 14.06 MiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0097.jit__reshard_rchunk` | `all-gather-start` | 1012.50 MiB | `` | `(c128[16,30,2,23040]{3,2,0,1}, c128[16,60,2,23040]{3,2,0,1})` |
| `module_0101.jit__accum` | `all-to-all` | 675.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:147` | `c128[16,60,2,1,23040]{4,3,1,0,2}` |
| `module_0367.jit__kpath_batch` | `all-gather-start` | 675.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:613` | `(c128[480,30720]{1,0}, c128[960,30720]{1,0})` |
| `module_0097.jit__reshard_rchunk` | `all-to-all` | 337.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[16,1,15,2,2,23040]{5,3,2,1,0,4}` |
| `module_0367.jit__kpath_batch` | `all-to-all` | 225.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:610` | `c128[32,2,480,480]{3,2,0,1}` |
| `module_0317.jit_reshape` | `all-gather-start` | 168.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:602` | `(c128[16,480,480]{1,0,2}, c128[16,480,960]{1,0,2})` |
| `module_0065.jit_reshape` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:98` | `(c128[16,60,2,240]{2,1,0,3}, c128[16,60,2,480]{2,1,0,3})` |
| `module_0103.jit_cholesky` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:184` | `(c128[480,960]{1,0}, c128[960,960]{1,0})` |
| `module_0243.jit_eigvalsh` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:584` | `(c128[480,960]{1,0}, c128[960,960]{1,0})` |
| `module_0041.jit__fft_gather_reshard` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720` | `(c128[16,15,2,480]{3,2,0,1}, c128[16,30,2,480]{3,2,0,1})` |
| `module_0103.jit_cholesky` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:184` | `(c128[480,480]{0,1}, c128[480,960]{0,1})` |
| `module_0243.jit_eigvalsh` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:584` | `(c128[480,480]{0,1}, c128[480,960]{0,1})` |
| `module_0041.jit__fft_gather_reshard` | `all-to-all` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:727` | `c128[16,30,2,2,240]{4,2,1,0,3}` |
| `module_0341.jit_add` | `all-to-all` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:623` | `c128[1,2,480,480]{3,2,0,1}` |
| `module_0101.jit__accum` | `reduce-scatter` | 3.52 MiB | `` | `c128[480,480]{0,1}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 22 | 14.06 MiB | 67.89 MiB |
| `jit_multiply` | 12 | 28.13 MiB | 98.50 MiB |
| `jit_add` | 11 | 42.19 MiB | 87.15 MiB |
| `jit_subtract` | 10 | 119.53 MiB | 122.39 MiB |
| `jit_iota` | 8 | 7.03 MiB | 14.12 MiB |
| `jit_reshape` | 7 | 281.25 MiB | 414.84 MiB |
| `jit_gather` | 7 | 59.77 MiB | 92.93 MiB |
| `jit_convert_element_type` | 7 | 14.94 MiB | 15.00 MiB |
| `jit_transpose` | 6 | 28.12 MiB | 58.04 MiB |
| `jit__reduce_max` | 5 | 28.16 MiB | 56.32 MiB |
| `jit__where` | 5 | 23.44 KiB | 55.45 KiB |
| `jit_dynamic_slice` | 5 | 15.02 KiB | 36.42 KiB |
| `jit_true_divide` | 5 | 15.01 KiB | 32.04 KiB |
| `jit_matmul` | 4 | 119.53 MiB | 169.29 MiB |
| `jit_abs` | 4 | 84.38 MiB | 140.71 MiB |
| `jit_conjugate` | 4 | 28.12 MiB | 72.07 MiB |
| `jit_real` | 3 | 84.38 MiB | 87.05 MiB |
| `jit__reduce_min` | 3 | 28.13 MiB | 28.13 MiB |
| `jit__multi_slice` | 3 | 21.09 MiB | 59.77 MiB |
| `jit_sort` | 3 | 7.96 MiB | 7.97 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 7 |
| `__cusolver$cholesky` | 2 |
| `cusolver_syevd_ffi` | 2 |
| `__cublas$triangularSolve` | 2 |
| `cusolver_gesvdj_ffi` | 1 |
| `__cub$DeviceRadixSort` | 1 |

