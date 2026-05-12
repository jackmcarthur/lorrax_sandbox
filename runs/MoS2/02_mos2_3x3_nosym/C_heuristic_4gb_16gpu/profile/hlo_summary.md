# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/C_heuristic_4gb_16gpu/profile/xla_dump`
**Modules dumped:** 308
**Sum of per-module peak live HBM:** 19.77 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0181.jit__single_chunk_proc` | 5.30 GiB | 3.52 GiB — preallocated-temp: |
| `module_0189.jit_concatenate` | 3.52 GiB | 1.76 GiB — output shape is \|c128[4,640,46080]\|, maybe-live-out: |
| `module_0187.jit_reshape` | 2.64 GiB | 1.32 GiB — output shape is \|c128[3,640,46080]\|, maybe-live-out: |
| `module_0185.jit_broadcast_in_dim` | 1.76 GiB | 1.32 GiB — output shape is \|c128[3,1,1,640,1,46080]\|, maybe-live-out: |
| `module_0121.jit__kernel` | 1.08 GiB | 776.96 MiB — preallocated-temp: |
| `module_0123.jit__kernel` | 1.08 GiB | 776.96 MiB — preallocated-temp: |
| `module_0183.jit_reshape` | 900.00 MiB | 450.00 MiB — output shape is \|c128[1,1,1,640,1,46080]\|, maybe-live-out: |
| `module_0147.jit_transpose` | 506.25 MiB | 253.12 MiB — output shape is \|c128[9,2880,640]\|, maybe-live-out: |
| `module_0151.jit__per_rank` | 253.13 MiB | 0.00 B —  |
| `module_0154.jit__per_rank` | 253.13 MiB | 0.00 B —  |
| `module_0499.jit_norm` | 216.00 MiB | 162.00 MiB — parameter 0, shape \|f64[262144,27,3]\| at ShapeIndex {}: |
| `module_0039.jit__fft_gather_reshard` | 196.88 MiB | 126.56 MiB — preallocated-temp: |
| `module_0497.jit_subtract` | 168.00 MiB | 162.00 MiB — output shape is \|f64[262144,27,3]\|, maybe-live-out: |
| `module_0327.jit__solve_w` | 163.28 MiB | 100.00 MiB — preallocated-temp: |
| `module_0195.jit_concatenate` | 112.50 MiB | 56.25 MiB — output shape is \|c128[9,640,640]\|, maybe-live-out: |
| `module_0197.jit_reshape` | 112.50 MiB | 56.25 MiB — output shape is \|c128[3,3,1,640,640]\|, maybe-live-out: |
| `module_0623.jit_sigma_coh` | 109.86 MiB | 56.25 MiB — parameter 4, shape \|c128[9,640,640]\| at ShapeIndex {}: |
| `module_0633.jit_sigma_coh` | 109.86 MiB | 56.25 MiB — parameter 4, shape \|c128[9,640,640]\| at ShapeIndex {}: |
| `module_0603.jit_sigma_sx` | 107.23 MiB | 56.25 MiB — parameter 5, shape \|c128[9,640,640]\| at ShapeIndex {}: |
| `module_0612.jit_sigma_sx` | 107.23 MiB | 56.25 MiB — parameter 5, shape \|c128[9,640,640]\| at ShapeIndex {}: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0121.jit__kernel` | `all-gather-start` | 316.41 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,20,2,11520]{3,2,0,1}, c128[9,80,2,11520]{3,2,0,1})` |
| `module_0123.jit__kernel` | `all-gather-start` | 316.41 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,20,2,11520]{3,2,0,1}, c128[9,80,2,11520]{3,2,0,1})` |
| `module_0121.jit__kernel` | `all-to-all` | 253.12 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:776` | `c128[9,160,1,4,2880]{4,2,1,0,3}` |
| `module_0123.jit__kernel` | `all-to-all` | 253.12 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:776` | `c128[9,160,1,4,2880]{4,2,1,0,3}` |
| `module_0327.jit__solve_w` | `all-gather-start` | 106.25 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:280` | `(c128[1,640,640]{2,1,0}, c128[16,640,640]{2,1,0})` |
| `module_0121.jit__kernel` | `all-gather-start` | 70.31 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:541` | `(c128[9,160,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0123.jit__kernel` | `all-gather-start` | 70.31 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:541` | `(c128[9,160,640]{2,0,1}, c128[9,640,640]{2,0,1})` |
| `module_0259.jit_gather` | `all-gather-start` | 70.31 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224` | `(c128[1,1,1,3,3,1,160,640]{7,5,4,3,2,1,0,6}, c128[1,1,1,3,3,` |
| `module_0121.jit__kernel` | `all-to-all` | 63.28 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,5,2,4,11520]{5,3,2,1,0,4}` |
| `module_0123.jit__kernel` | `all-to-all` | 63.28 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,5,2,4,11520]{5,3,2,1,0,4}` |
| `module_0121.jit__kernel` | `all-gather-start` | 17.58 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:541` | `(c128[9,160,160]{1,0,2}, c128[9,160,640]{1,0,2})` |
| `module_0123.jit__kernel` | `all-gather-start` | 17.58 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:541` | `(c128[9,160,160]{1,0,2}, c128[9,160,640]{1,0,2})` |
| `module_0259.jit_gather` | `all-gather-start` | 17.58 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224` | `(c128[1,1,1,3,3,1,160,160]{6,5,4,3,2,1,0,7}, c128[1,1,1,3,3,` |
| `module_0263.jit_gather` | `all-gather-start` | 17.58 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:239` | `(c128[9,80,2,160]{2,1,0,3}, c128[9,80,2,640]{2,1,0,3})` |
| `module_0327.jit__solve_w` | `all-to-all` | 6.25 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:261` | `c128[1,4,1,640,160]{4,3,2,0,1}` |
| `module_0327.jit__solve_w` | `all-to-all` | 6.25 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:263` | `c128[1,4,1,640,160]{4,3,2,0,1}` |
| `module_0327.jit__solve_w` | `all-to-all` | 4.69 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:259` | `c128[4,3,160,160]{3,2,1,0}` |
| `module_0327.jit__solve_w` | `all-to-all` | 4.69 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:259` | `c128[4,3,160,160]{3,2,1,0}` |
| `module_0039.jit__fft_gather_reshard` | `all-gather-start` | 4.39 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720` | `(c128[9,5,2,640]{3,2,0,1}, c128[9,20,2,640]{3,2,0,1})` |
| `module_0039.jit__fft_gather_reshard` | `all-to-all` | 3.52 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:727` | `c128[9,20,2,4,160]{4,2,1,0,3}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 56 | 1.76 GiB | 1.86 GiB |
| `jit_convert_element_type` | 28 | 6.00 MiB | 9.23 MiB |
| `jit_multiply` | 16 | 12.00 MiB | 29.32 MiB |
| `jit_gather` | 15 | 77.34 MiB | 141.28 MiB |
| `jit_concatenate` | 13 | 3.52 GiB | 3.64 GiB |
| `jit_true_divide` | 12 | 12.00 MiB | 34.55 MiB |
| `jit_add` | 12 | 4.00 MiB | 12.12 MiB |
| `jit__per_rank` | 10 | 253.13 MiB | 516.84 MiB |
| `jit_reshape` | 9 | 2.64 GiB | 3.65 GiB |
| `jit_subtract` | 9 | 168.00 MiB | 198.40 MiB |
| `jit_transpose` | 8 | 506.25 MiB | 551.34 MiB |
| `jit_dynamic_slice` | 8 | 31.25 MiB | 49.32 MiB |
| `jit_iota` | 8 | 50.00 KiB | 201.05 KiB |
| `jit__psum` | 7 | 1.76 MiB | 3.56 MiB |
| `jit__identity_fn` | 5 | 7.03 MiB | 22.60 MiB |
| `jit_select_n` | 5 | 6.25 MiB | 6.25 MiB |
| `jit_squeeze` | 5 | 4.00 MiB | 4.01 MiB |
| `jit__broadcast_arrays` | 5 | 4.00 MiB | 5.08 MiB |
| `jit_less` | 5 | 2.25 MiB | 2.25 MiB |
| `jit_sigma_sx` | 4 | 107.23 MiB | 323.44 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 43 |
| `lorrax_phdf5_write` | 10 |
| `__cublas$triangularSolve` | 9 |
| `cusolver_getrf_ffi` | 2 |
| `cu_lu_pivots_to_permutation` | 2 |
| `__cusolver$cholesky` | 1 |
| `cusolver_syevd_ffi` | 1 |

