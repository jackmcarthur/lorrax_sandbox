# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/09_lorrax_baseline_C/profile/xla_dump`
**Modules dumped:** 496
**Sum of per-module peak live HBM:** 63.23 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0195.jit__identity_fn` | 8.90 GiB | 3.96 GiB — preallocated-temp: |
| `module_0275.jit__single_chunk_proc` | 5.31 GiB | 3.52 GiB — preallocated-temp: |
| `module_0193.jit__solve_all_at_once` | 4.02 GiB | 2.03 GiB — preallocated-temp: |
| `module_0191.jit__identity_fn` | 3.96 GiB | 1.98 GiB — preallocated-temp: |
| `module_0283.jit_concatenate` | 3.52 GiB | 1.76 GiB — output shape is \|c128[4,640,46080]\|, maybe-live-out: |
| `module_0175.jit__right_ifft_mul_fft` | 2.97 GiB | 1012.50 MiB — parameter 0, shape \|c128[3,3,1,320,23040]\| at ShapeIndex {}, output shape is \|c128[3,3,1,320,23040]\|, maybe-live-out: |
| `module_0281.jit_reshape` | 2.64 GiB | 1.32 GiB — output shape is \|c128[3,640,46080]\|, maybe-live-out: |
| `module_0165.jit_reshape` | 1.98 GiB | 1012.50 MiB — output shape is \|c128[3,3,1,320,23040]\|, maybe-live-out: |
| `module_0167.jit__left_ifft_conj` | 1.98 GiB | 1012.50 MiB — parameter 0, shape \|c128[3,3,1,320,23040]\| at ShapeIndex {}, output shape is \|c128[3,3,1,320,23040]\|, maybe-live-out: |
| `module_0189.jit_reshape` | 1.98 GiB | 1012.50 MiB — output shape is \|c128[9,320,23040]\|, maybe-live-out: |
| `module_0279.jit_broadcast_in_dim` | 1.76 GiB | 1.32 GiB — output shape is \|c128[3,1,1,640,1,46080]\|, maybe-live-out: |
| `module_0153.jit__reshard_rchunk` | 1.73 GiB | 1012.50 MiB — preallocated-temp: |
| `module_0163.jit__compute_P_traced` | 1.49 GiB | 1012.50 MiB — output shape is \|c128[9,320,23040]\|, maybe-live-out: |
| `module_0127.jit__multi_slice` | 1.48 GiB | 1012.50 MiB — parameter 0, shape \|c128[9,80,2,46080]\| at ShapeIndex {}: |
| `module_0157.jit_scatter` | 1.48 GiB | 506.25 MiB — output shape is \|c128[9,80,2,23040]\|, maybe-live-out: |
| `module_0039.jit__fft_gather_reshard` | 1.00 GiB | 759.38 MiB — preallocated-temp: |
| `module_0161.jit_true_divide` | 1012.50 MiB | 506.25 MiB — output shape is \|c128[9,80,2,23040]\|, maybe-live-out: |
| `module_0123.jit_broadcast_in_dim` | 1012.50 MiB | 0.00 B —  |
| `module_0125.jit__identity_fn` | 1012.50 MiB | 506.25 MiB — output shape is \|c128[9,80,2,23040]\|, maybe-live-out: |
| `module_0155.jit__squeeze` | 1012.50 MiB | 506.25 MiB — output shape is \|c128[9,80,2,23040]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0195.jit__identity_fn` | `all-gather-start` | 4.94 GiB | `` | `(c128[9,640,11520]{1,0,2}, c128[9,640,46080]{1,0,2})` |
| `module_0191.jit__identity_fn` | `all-to-all` | 1012.50 MiB | `` | `c128[9,320,1,2,11520]{4,2,1,0,3}` |
| `module_0153.jit__reshard_rchunk` | `all-gather-start` | 759.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:340` | `(c128[9,20,2,46080]{3,2,0,1}, c128[9,40,2,46080]{3,2,0,1})` |
| `module_0153.jit__reshard_rchunk` | `all-to-all` | 506.25 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:341` | `c128[9,40,2,2,23040]{4,2,1,0,3}` |
| `module_0731.jit__identity_fn` | `all-gather-start` | 180.18 MiB | `` | `(c128[1,41,9,80,80]{4,3,2,1,0}, c128[4,41,9,80,80]{4,3,2,1,0` |
| `module_0751.jit__identity_fn` | `all-gather-start` | 180.18 MiB | `` | `(c128[1,41,9,80,80]{4,3,2,1,0}, c128[4,41,9,80,80]{4,3,2,1,0` |
| `module_0481.jit__solve_w` | `all-gather-start` | 93.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:231` | `(c128[3,640,640]{2,1,0}, c128[12,640,640]{2,1,0})` |
| `module_0501.jit__solve_w` | `all-gather-start` | 93.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:231` | `(c128[3,640,640]{2,1,0}, c128[12,640,640]{2,1,0})` |
| `module_0193.jit__solve_all_at_once` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:560` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0381.jit__identity_fn` | `all-gather-start` | 84.38 MiB | `` | `(c128[1,1,1,3,3,1,320,640]{7,5,4,3,2,1,0,6}, c128[1,1,1,3,3,` |
| `module_0389.jit_gather` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:190` | `(c128[1,1,1,3,3,1,320,640]{7,5,4,3,2,1,0,6}, c128[1,1,1,3,3,` |
| `module_0401.jit__identity_fn` | `all-gather-start` | 84.38 MiB | `` | `(c128[1,1,1,3,3,1,320,640]{7,5,4,3,2,1,0,6}, c128[1,1,1,3,3,` |
| `module_0455.jit_fft` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:120` | `(c128[3,1,320,640,3]{4,3,1,0,2}, c128[3,1,640,640,3]{4,3,1,0` |
| `module_0475.jit_fft` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:120` | `(c128[3,1,320,640,3]{4,3,1,0,2}, c128[3,1,640,640,3]{4,3,1,0` |
| `module_0481.jit__solve_w` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:210` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0501.jit__solve_w` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:210` | `(c128[9,320,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0193.jit__solve_all_at_once` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:560` | `(c128[9,320,320]{1,0,2}, c128[9,320,640]{1,0,2})` |
| `module_0381.jit__identity_fn` | `all-gather-start` | 42.19 MiB | `` | `(c128[1,1,1,3,3,1,320,320]{6,5,4,3,2,1,0,7}, c128[1,1,1,3,3,` |
| `module_0389.jit_gather` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:190` | `(c128[1,1,1,3,3,1,320,320]{6,5,4,3,2,1,0,7}, c128[1,1,1,3,3,` |
| `module_0401.jit__identity_fn` | `all-gather-start` | 42.19 MiB | `` | `(c128[1,1,1,3,3,1,320,320]{6,5,4,3,2,1,0,7}, c128[1,1,1,3,3,` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 71 | 1.76 GiB | 3.07 GiB |
| `jit__identity_fn` | 33 | 8.90 GiB | 14.85 GiB |
| `jit_convert_element_type` | 30 | 21.09 MiB | 58.24 MiB |
| `jit_gather` | 23 | 112.50 MiB | 317.84 MiB |
| `jit_true_divide` | 22 | 1012.50 MiB | 1.10 GiB |
| `jit_add` | 22 | 72.95 MiB | 574.16 MiB |
| `jit_reshape` | 21 | 2.64 GiB | 7.99 GiB |
| `jit_iota` | 17 | 50.00 KiB | 223.98 KiB |
| `jit_subtract` | 16 | 84.38 MiB | 253.24 MiB |
| `jit_multiply` | 16 | 72.07 MiB | 344.54 MiB |
| `jit_transpose` | 14 | 112.50 MiB | 1.18 GiB |
| `jit__where` | 14 | 29.00 MiB | 145.99 MiB |
| `jit_concatenate` | 12 | 3.52 GiB | 3.64 GiB |
| `jit_scatter` | 11 | 1.48 GiB | 1.53 GiB |
| `jit__squeeze` | 11 | 1012.50 MiB | 1.02 GiB |
| `jit_dynamic_slice` | 9 | 31.25 MiB | 31.34 MiB |
| `jit__accumulate_window_channels_jit` | 8 | 39.55 MiB | 309.39 MiB |
| `jit__reduce_max` | 8 | 28.13 MiB | 70.36 MiB |
| `jit_sqrt` | 8 | 14.06 MiB | 28.15 MiB |
| `jit_greater` | 8 | 7.91 MiB | 31.65 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 48 |
| `__cublas$triangularSolve` | 7 |
| `cusolver_getrf_ffi` | 2 |
| `cu_lu_pivots_to_permutation` | 2 |
| `cusolver_syevd_ffi` | 2 |
| `__cusolver$cholesky` | 1 |

