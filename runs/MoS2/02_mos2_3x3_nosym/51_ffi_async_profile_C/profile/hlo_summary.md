# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/51_ffi_async_profile_C/profile/xla_dump`
**Modules dumped:** 285
**Sum of per-module peak live HBM:** 42.85 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0269.jit__single_chunk_proc` | 5.31 GiB | 3.52 GiB — preallocated-temp: |
| `module_0291.jit_concatenate` | 3.52 GiB | 1.76 GiB — output shape is \|c128[4,640,46080]\|, maybe-live-out: |
| `module_0289.jit_reshape` | 2.64 GiB | 1.32 GiB — output shape is \|c128[3,640,46080]\|, maybe-live-out: |
| `module_0287.jit_broadcast_in_dim` | 1.76 GiB | 1.32 GiB — output shape is \|c128[3,1,1,640,1,46080]\|, maybe-live-out: |
| `module_0191.jit__solve_all_at_once` | 1.06 GiB | 562.50 MiB — preallocated-temp: |
| `module_0039.jit__fft_gather_reshard` | 1.00 GiB | 759.38 MiB — preallocated-temp: |
| `module_0189.jit__identity_fn` | 1012.50 MiB | 506.25 MiB — preallocated-temp: |
| `module_0239.jit__unnamed_wrapped_function_` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,640,46080]\|, maybe-live-out: |
| `module_0243.jit__unnamed_wrapped_function_` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,640,46080]\|, maybe-live-out: |
| `module_0247.jit__unnamed_wrapped_function_` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,640,46080]\|, maybe-live-out: |
| `module_0251.jit__unnamed_wrapped_function_` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,640,46080]\|, maybe-live-out: |
| `module_0261.jit__unnamed_wrapped_function_` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,640,46080]\|, maybe-live-out: |
| `module_0271.jit__unnamed_wrapped_function_` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,640,46080]\|, maybe-live-out: |
| `module_0275.jit__unnamed_wrapped_function_` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,640,46080]\|, maybe-live-out: |
| `module_0279.jit__unnamed_wrapped_function_` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,640,46080]\|, maybe-live-out: |
| `module_0283.jit__unnamed_wrapped_function_` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,640,46080]\|, maybe-live-out: |
| `module_0285.jit_reshape` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,1,1,640,1,46080]\|, maybe-live-out: |
| `module_0143.jit__fft_and_rslice` | 822.66 MiB | 506.25 MiB — preallocated-temp: |
| `module_0173.jit__right_ifft_mul_fft` | 759.38 MiB | 253.12 MiB — parameter 0, shape \|c128[3,3,1,320,5760]\| at ShapeIndex {}, output shape is \|c128[3,3,1,320,5760]\|, maybe-live-out: |
| `module_0163.jit_reshape` | 506.25 MiB | 253.12 MiB — output shape is \|c128[3,3,1,320,5760]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0189.jit__identity_fn` | `all-to-all` | 253.12 MiB | `` | `c128[9,320,1,2,2880]{4,2,1,0,3}` |
| `module_0151.jit__reshard_rchunk` | `all-gather-start` | 189.84 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:358` | `(c128[9,20,2,11520]{3,2,0,1}, c128[9,40,2,11520]{3,2,0,1})` |
| `module_0151.jit__reshard_rchunk` | `all-to-all` | 126.56 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:359` | `c128[9,40,2,2,5760]{4,2,1,0,3}` |
| `module_0465.jit__solve_w` | `all-gather-start` | 93.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:231` | `(c128[3,640,640]{2,1,0}, c128[12,640,640]{2,1,0})` |
| `module_0191.jit__solve_all_at_once` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:560` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0371.jit_gather` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:190` | `(c128[1,1,1,3,3,1,320,640]{7,5,4,3,2,1,0,6}, c128[1,1,1,3,3,` |
| `module_0439.jit_fft` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:120` | `(c128[3,1,320,640,3]{4,3,1,0,2}, c128[3,1,640,640,3]{4,3,1,0` |
| `module_0465.jit__solve_w` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:210` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0191.jit__solve_all_at_once` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:560` | `(c128[9,320,320]{1,0,2}, c128[9,320,640]{1,0,2})` |
| `module_0371.jit_gather` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:190` | `(c128[1,1,1,3,3,1,320,320]{6,5,4,3,2,1,0,7}, c128[1,1,1,3,3,` |
| `module_0439.jit_fft` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:120` | `(c128[3,1,320,320,3]{4,2,1,0,3}, c128[3,1,320,640,3]{4,2,1,0` |
| `module_0465.jit__solve_w` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:210` | `(c128[9,320,320]{1,0,2}, c128[9,320,640]{1,0,2})` |
| `module_0661.jit__identity_fn` | `all-gather-start` | 27.69 MiB | `` | `(c128[21,9,40,80]{3,1,0,2}, c128[21,9,80,80]{3,1,0,2})` |
| `module_0663.jit__identity_fn` | `all-gather-start` | 26.37 MiB | `` | `(c128[20,9,40,80]{3,1,0,2}, c128[20,9,80,80]{3,1,0,2})` |
| `module_0375.jit_gather` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:205` | `(c128[9,80,2,320]{2,1,0,3}, c128[9,80,2,640]{2,1,0,3})` |
| `module_0661.jit__identity_fn` | `all-gather-start` | 13.84 MiB | `` | `(c128[21,9,40,40]{2,1,0,3}, c128[21,9,40,80]{2,1,0,3})` |
| `module_0663.jit__identity_fn` | `all-gather-start` | 13.18 MiB | `` | `(c128[20,9,40,40]{2,1,0,3}, c128[20,9,40,80]{2,1,0,3})` |
| `module_0039.jit__fft_gather_reshard` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:567` | `(c128[9,20,2,640]{3,2,0,1}, c128[9,40,2,640]{3,2,0,1})` |
| `module_0039.jit__fft_gather_reshard` | `all-to-all` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:574` | `c128[9,40,2,2,320]{4,2,1,0,3}` |
| `module_0623.jit__tau_kernel` | `reduce-scatter` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:411` | `(c128[9,40,2,320]{3,2,0,1}, c128[9,40,2,320]{3,2,0,1})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit__unnamed_wrapped_function_` | 54 | 900.00 MiB | 15.11 GiB |
| `jit_broadcast_in_dim` | 37 | 1.76 GiB | 2.21 GiB |
| `jit__identity_fn` | 15 | 1012.50 MiB | 1.42 GiB |
| `jit_convert_element_type` | 15 | 21.09 MiB | 29.11 MiB |
| `jit_reshape` | 13 | 2.64 GiB | 4.86 GiB |
| `jit_true_divide` | 11 | 253.13 MiB | 323.44 MiB |
| `jit_gather` | 10 | 112.50 MiB | 226.58 MiB |
| `jit_concatenate` | 9 | 3.52 GiB | 3.64 GiB |
| `jit_add` | 8 | 72.95 MiB | 178.31 MiB |
| `jit__psum` | 8 | 72.07 MiB | 145.92 MiB |
| `jit_transpose` | 7 | 112.50 MiB | 604.69 MiB |
| `jit_multiply` | 7 | 72.07 MiB | 172.28 MiB |
| `jit_iota` | 7 | 50.00 KiB | 101.64 KiB |
| `jit__multi_slice` | 6 | 379.69 MiB | 537.23 MiB |
| `jit_dynamic_slice` | 5 | 31.25 MiB | 31.32 MiB |
| `jit_subtract` | 4 | 84.38 MiB | 126.57 MiB |
| `jit__where` | 4 | 29.00 MiB | 87.01 MiB |
| `jit_scatter` | 3 | 379.69 MiB | 421.88 MiB |
| `jit__squeeze` | 3 | 253.12 MiB | 281.25 MiB |
| `jit_fft` | 3 | 126.56 MiB | 351.56 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 26 |
| `lorrax_phdf5_write` | 12 |
| `lorrax_phdf5_read` | 9 |
| `__cublas$triangularSolve` | 5 |
| `__cusolver$cholesky` | 1 |
| `cusolver_getrf_ffi` | 1 |
| `cu_lu_pivots_to_permutation` | 1 |
| `cusolver_syevd_ffi` | 1 |

