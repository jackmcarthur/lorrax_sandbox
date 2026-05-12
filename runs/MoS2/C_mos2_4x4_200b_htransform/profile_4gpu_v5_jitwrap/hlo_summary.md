# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v5_jitwrap/xla_dump`
**Modules dumped:** 97
**Sum of per-module peak live HBM:** 9.39 GiB (upper bound; peaks occur at different times)

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
| `module_0207.jit__kpath_batch` | 1.44 GiB | 1.32 GiB — preallocated-temp: |
| `module_0041.jit__fft_gather_reshard` | 1.33 GiB | 1012.50 MiB — preallocated-temp: |
| `module_0089.jit__fft_and_rslice` | 1012.80 MiB | 337.80 MiB — preallocated-temp: |
| `module_0157.jit_reshape` | 281.25 MiB | 112.50 MiB — preallocated-temp: |
| `module_0135.jit__build_fH` | 182.82 MiB | 56.25 MiB — preallocated-temp: |
| `module_0171.jit_matmul` | 119.53 MiB | 112.50 MiB — parameter 1, shape \|c128[16,460800]\| at ShapeIndex {}: |
| `module_0187.jit_subtract` | 119.53 MiB | 56.25 MiB — output shape is \|c128[16,480,480]\|, maybe-live-out: |
| `module_0143.jit__diag_stats` | 84.40 MiB | 56.25 MiB — parameter 0, shape \|c128[16,480,480]\| at ShapeIndex {}: |
| `module_0189.jit_abs` | 84.38 MiB | 56.25 MiB — parameter 0, shape \|c128[16,480,480]\| at ShapeIndex {}: |
| `module_0067.jit__svd_replicated` | 56.26 MiB | 28.12 MiB — preallocated-temp: |
| `module_0155.jit__build_S_chol` | 56.25 MiB | 28.12 MiB — preallocated-temp: |
| `module_0105.jit__finalize` | 46.19 MiB | 14.06 MiB — maybe-live-out: |
| `module_0103.jit_cholesky` | 45.70 MiB | 28.12 MiB — preallocated-temp: |
| `module_0065.jit_reshape` | 35.16 MiB | 14.06 MiB — preallocated-temp: |
| `module_0191.jit__reduce_max` | 28.16 MiB | 28.12 MiB — parameter 0, shape \|f64[16,480,480]\| at ShapeIndex {}: |
| `module_0071.jit_multiply` | 28.13 MiB | 14.06 MiB — output shape is \|c128[960,960]\|, maybe-live-out: |
| `module_0181.jit_add` | 28.13 MiB | 7.03 MiB — preallocated-temp: |
| `module_0083.jit__psum` | 28.13 MiB | 14.06 MiB — output shape is \|c128[960,960]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0097.jit__reshard_rchunk` | `all-gather-start` | 1012.50 MiB | `` | `(c128[16,30,2,23040]{3,2,0,1}, c128[16,60,2,23040]{3,2,0,1})` |
| `module_0101.jit__accum` | `all-to-all` | 675.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:147` | `c128[16,60,2,1,23040]{4,3,1,0,2}` |
| `module_0207.jit__kpath_batch` | `all-gather-start` | 675.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:641` | `(c128[480,30720]{1,0}, c128[960,30720]{1,0})` |
| `module_0097.jit__reshard_rchunk` | `all-to-all` | 337.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[16,1,15,2,2,23040]{5,3,2,1,0,4}` |
| `module_0207.jit__kpath_batch` | `all-to-all` | 225.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:638` | `c128[32,2,480,480]{3,2,0,1}` |
| `module_0157.jit_reshape` | `all-gather-start` | 168.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:630` | `(c128[16,480,480]{1,0,2}, c128[16,480,960]{1,0,2})` |
| `module_0065.jit_reshape` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:98` | `(c128[16,60,2,240]{2,1,0,3}, c128[16,60,2,480]{2,1,0,3})` |
| `module_0103.jit_cholesky` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:184` | `(c128[480,960]{1,0}, c128[960,960]{1,0})` |
| `module_0143.jit__diag_stats` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:597` | `(c128[480,960]{1,0}, c128[960,960]{1,0})` |
| `module_0041.jit__fft_gather_reshard` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720` | `(c128[16,15,2,480]{3,2,0,1}, c128[16,30,2,480]{3,2,0,1})` |
| `module_0103.jit_cholesky` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:184` | `(c128[480,480]{0,1}, c128[480,960]{0,1})` |
| `module_0143.jit__diag_stats` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:597` | `(c128[480,480]{0,1}, c128[480,960]{0,1})` |
| `module_0041.jit__fft_gather_reshard` | `all-to-all` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:727` | `c128[16,30,2,2,240]{4,2,1,0,3}` |
| `module_0181.jit_add` | `all-to-all` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:651` | `c128[1,2,480,480]{3,2,0,1}` |
| `module_0101.jit__accum` | `reduce-scatter` | 3.52 MiB | `` | `c128[480,480]{0,1}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 13 | 14.06 MiB | 56.28 MiB |
| `jit_reshape` | 5 | 281.25 MiB | 386.72 MiB |
| `jit_add` | 5 | 28.13 MiB | 42.21 MiB |
| `jit_convert_element_type` | 5 | 14.94 MiB | 14.94 MiB |
| `jit_subtract` | 4 | 119.53 MiB | 122.19 MiB |
| `jit_multiply` | 4 | 28.13 MiB | 42.20 MiB |
| `jit_transpose` | 4 | 28.12 MiB | 28.16 MiB |
| `jit_iota` | 4 | 7.03 MiB | 14.06 MiB |
| `jit_dynamic_slice` | 4 | 15.02 KiB | 35.91 KiB |
| `jit_true_divide` | 4 | 15.01 KiB | 17.03 KiB |
| `jit__multi_slice` | 3 | 21.09 MiB | 59.77 MiB |
| `jit_gather` | 3 | 14.06 MiB | 14.07 MiB |
| `jit_squeeze` | 3 | 7.50 KiB | 8.69 KiB |
| `jit_matmul` | 2 | 119.53 MiB | 119.53 MiB |
| `jit__reduce_max` | 2 | 28.16 MiB | 28.16 MiB |
| `jit_conjugate` | 2 | 28.12 MiB | 42.19 MiB |
| `jit_scatter` | 2 | 21.09 MiB | 42.19 MiB |
| `jit__identity_fn` | 2 | 14.06 MiB | 28.12 MiB |
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

