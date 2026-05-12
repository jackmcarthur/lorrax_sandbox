# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v10/xla_dump`
**Modules dumped:** 90
**Sum of per-module peak live HBM:** 9.95 GiB (upper bound; peaks occur at different times)

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
| `module_0197.jit__kpath_batch` | 1.57 GiB | 1.33 GiB — preallocated-temp: |
| `module_0041.jit__fft_gather_reshard` | 1.33 GiB | 1012.50 MiB — preallocated-temp: |
| `module_0089.jit__fft_and_rslice` | 1012.80 MiB | 337.80 MiB — preallocated-temp: |
| `module_0169.jit__identity_fn` | 618.75 MiB | 337.50 MiB — preallocated-temp: |
| `module_0175.jit__gamma_rt` | 253.13 MiB | 225.00 MiB — parameter 0, shape \|c128[16,960,960]\| at ShapeIndex {}: |
| `module_0135.jit__build_fH` | 182.82 MiB | 56.25 MiB — preallocated-temp: |
| `module_0177.jit_subtract` | 126.56 MiB | 56.25 MiB — output shape is \|c128[16,480,480]\|, maybe-live-out: |
| `module_0179.jit_abs` | 84.38 MiB | 56.25 MiB — parameter 0, shape \|c128[16,480,480]\| at ShapeIndex {}: |
| `module_0149.jit_gather` | 59.77 MiB | 56.25 MiB — parameter 0, shape \|c128[16,480,480]\| at ShapeIndex {}: |
| `module_0143.jit__diag_stats_fast` | 56.27 MiB | 56.25 MiB — parameter 0, shape \|c128[16,480,480]\| at ShapeIndex {}: |
| `module_0067.jit__svd_replicated` | 56.26 MiB | 28.12 MiB — preallocated-temp: |
| `module_0167.jit__build_S_chol` | 56.25 MiB | 28.12 MiB — preallocated-temp: |
| `module_0105.jit__finalize` | 46.19 MiB | 14.06 MiB — maybe-live-out: |
| `module_0103.jit_cholesky` | 45.70 MiB | 28.12 MiB — preallocated-temp: |
| `module_0065.jit_reshape` | 35.16 MiB | 14.06 MiB — preallocated-temp: |
| `module_0181.jit__reduce_max` | 28.16 MiB | 28.12 MiB — parameter 0, shape \|f64[16,480,480]\| at ShapeIndex {}: |
| `module_0159.jit__diag_eig_at_gamma` | 28.13 MiB | 14.06 MiB — preallocated-temp: |
| `module_0071.jit_multiply` | 28.13 MiB | 14.06 MiB — output shape is \|c128[960,960]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0097.jit__reshard_rchunk` | `all-gather-start` | 1012.50 MiB | `` | `(c128[16,30,2,23040]{3,2,0,1}, c128[16,60,2,23040]{3,2,0,1})` |
| `module_0101.jit__accum` | `all-to-all` | 675.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:147` | `c128[16,60,2,1,23040]{4,3,1,0,2}` |
| `module_0197.jit__kpath_batch` | `all-gather-start` | 562.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:664` | `(c128[960,960,8]{1,0,2}, c128[960,960,32]{1,0,2})` |
| `module_0097.jit__reshard_rchunk` | `all-to-all` | 337.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[16,1,15,2,2,23040]{5,3,2,1,0,4}` |
| `module_0169.jit__identity_fn` | `all-gather-start` | 337.50 MiB | `` | `(c128[16,480,960]{2,0,1}, c128[16,960,960]{2,0,1})` |
| `module_0169.jit__identity_fn` | `all-gather-start` | 168.75 MiB | `` | `(c128[16,480,480]{1,0,2}, c128[16,480,960]{1,0,2})` |
| `module_0065.jit_reshape` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:98` | `(c128[16,60,2,240]{2,1,0,3}, c128[16,60,2,480]{2,1,0,3})` |
| `module_0103.jit_cholesky` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:184` | `(c128[480,960]{1,0}, c128[960,960]{1,0})` |
| `module_0153.jit__identity_fn` | `all-gather-start` | 21.09 MiB | `` | `(c128[480,960]{1,0}, c128[960,960]{1,0})` |
| `module_0041.jit__fft_gather_reshard` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720` | `(c128[16,15,2,480]{3,2,0,1}, c128[16,30,2,480]{3,2,0,1})` |
| `module_0103.jit_cholesky` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:184` | `(c128[480,480]{0,1}, c128[480,960]{0,1})` |
| `module_0153.jit__identity_fn` | `all-gather-start` | 10.55 MiB | `` | `(c128[480,480]{0,1}, c128[480,960]{0,1})` |
| `module_0041.jit__fft_gather_reshard` | `all-to-all` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:727` | `c128[16,30,2,2,240]{4,2,1,0,3}` |
| `module_0101.jit__accum` | `reduce-scatter` | 3.52 MiB | `` | `c128[480,480]{0,1}` |
| `module_0199.jit__post_kpath` | `all-gather-start` | 1.79 MiB | `` | `((f64[46,960]{1,0}, f64[46,60]{1,0}), (f64[184,960]{1,0}, f6` |
| `module_0199.jit__post_kpath` | `all-to-all` | 360.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:715` | `f64[4,48,240]{2,1,0}` |
| `module_0199.jit__post_kpath` | `all-to-all` | 60.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:715` | `f64[8,4,240]{2,0,1}` |
| `module_0199.jit__post_kpath` | `all-to-all` | 60.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:715` | `f64[8,4,240]{2,0,1}` |
| `module_0199.jit__post_kpath` | `all-to-all` | 60.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:715` | `f64[8,4,240]{2,0,1}` |
| `module_0199.jit__post_kpath` | `all-to-all` | 60.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:715` | `f64[8,4,240]{2,0,1}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 13 | 14.06 MiB | 49.25 MiB |
| `jit_convert_element_type` | 5 | 14.94 MiB | 14.94 MiB |
| `jit__identity_fn` | 4 | 618.75 MiB | 671.48 MiB |
| `jit_subtract` | 4 | 126.56 MiB | 129.22 MiB |
| `jit_add` | 4 | 14.06 MiB | 14.08 MiB |
| `jit_iota` | 4 | 7.03 MiB | 14.06 MiB |
| `jit_dynamic_slice` | 4 | 15.02 KiB | 35.91 KiB |
| `jit_true_divide` | 4 | 15.01 KiB | 17.03 KiB |
| `jit_gather` | 3 | 59.77 MiB | 59.77 MiB |
| `jit_reshape` | 3 | 35.16 MiB | 91.41 MiB |
| `jit_transpose` | 3 | 28.12 MiB | 28.15 MiB |
| `jit__multi_slice` | 3 | 21.09 MiB | 59.77 MiB |
| `jit_squeeze` | 3 | 7.50 KiB | 8.69 KiB |
| `jit__reduce_max` | 2 | 28.16 MiB | 28.16 MiB |
| `jit_multiply` | 2 | 28.13 MiB | 28.14 MiB |
| `jit_scatter` | 2 | 21.09 MiB | 42.19 MiB |
| `jit__squeeze` | 2 | 14.06 MiB | 28.12 MiB |
| `jit__reduce_min` | 2 | 160.00 B | 320.00 B |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 7 |
| `__cusolver$cholesky` | 2 |
| `cusolver_syevd_ffi` | 2 |
| `__cublas$triangularSolve` | 2 |
| `cusolver_gesvdj_ffi` | 1 |
| `__cub$DeviceRadixSort` | 1 |

