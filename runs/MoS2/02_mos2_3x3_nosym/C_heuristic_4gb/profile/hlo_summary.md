# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/C_heuristic_4gb/profile/xla_dump`
**Modules dumped:** 304
**Sum of per-module peak live HBM:** 21.05 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0195.jit__kernel` | 3.85 GiB | 2.84 GiB — preallocated-temp: |
| `module_0200.jit__kernel` | 3.85 GiB | 2.84 GiB — preallocated-temp: |
| `module_0121.jit__kernel` | 2.33 GiB | 1.56 GiB — preallocated-temp: |
| `module_0123.jit__kernel` | 2.33 GiB | 1.56 GiB — preallocated-temp: |
| `module_0039.jit__fft_gather_reshard` | 1.00 GiB | 759.38 MiB — preallocated-temp: |
| `module_0181.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0185.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0153.jit_transpose` | 1012.50 MiB | 506.25 MiB — output shape is \|c128[9,5760,640]\|, maybe-live-out: |
| `module_0157.jit__per_rank` | 506.25 MiB | 0.00 B —  |
| `module_0160.jit__per_rank` | 506.25 MiB | 0.00 B —  |
| `module_0631.jit_sigma_coh` | 239.94 MiB | 140.62 MiB — preallocated-temp: |
| `module_0643.jit_sigma_coh` | 239.94 MiB | 140.62 MiB — preallocated-temp: |
| `module_0611.jit_sigma_sx` | 226.76 MiB | 140.62 MiB — preallocated-temp: |
| `module_0622.jit_sigma_sx` | 226.76 MiB | 140.62 MiB — preallocated-temp: |
| `module_0507.jit_norm` | 216.00 MiB | 162.00 MiB — parameter 0, shape \|f64[262144,27,3]\| at ShapeIndex {}: |
| `module_0313.jit__chi_scan` | 196.88 MiB | 168.75 MiB — preallocated-temp: |
| `module_0323.jit__chi_scan` | 196.88 MiB | 168.75 MiB — preallocated-temp: |
| `module_0677.jit_sigma_sx` | 184.57 MiB | 140.62 MiB — preallocated-temp: |
| `module_0690.jit_sigma_sx` | 184.57 MiB | 140.62 MiB — preallocated-temp: |
| `module_0335.jit__solve_w` | 178.13 MiB | 93.75 MiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0195.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1532` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0195.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1533` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0200.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1532` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0200.jit__kernel` | `all-gather-start` | 1.11 GiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1533` | `(c128[9,160,17252]{2,0,1}, c128[9,320,17252]{2,0,1})` |
| `module_0121.jit__kernel` | `all-to-all` | 506.25 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:776` | `c128[9,320,1,2,5760]{4,2,1,0,3}` |
| `module_0123.jit__kernel` | `all-to-all` | 506.25 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:776` | `c128[9,320,1,2,5760]{4,2,1,0,3}` |
| `module_0121.jit__kernel` | `all-gather-start` | 189.84 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,20,2,11520]{3,2,0,1}, c128[9,40,2,11520]{3,2,0,1})` |
| `module_0121.jit__kernel` | `all-gather-start` | 189.84 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,20,2,11520]{3,2,0,1}, c128[9,40,2,11520]{3,2,0,1})` |
| `module_0123.jit__kernel` | `all-gather-start` | 189.84 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,20,2,11520]{3,2,0,1}, c128[9,40,2,11520]{3,2,0,1})` |
| `module_0123.jit__kernel` | `all-gather-start` | 189.84 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:416` | `(c128[9,20,2,11520]{3,2,0,1}, c128[9,40,2,11520]{3,2,0,1})` |
| `module_0335.jit__solve_w` | `all-gather-start` | 93.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:280` | `(c128[3,640,640]{2,1,0}, c128[12,640,640]{2,1,0})` |
| `module_0267.jit_gather` | `all-gather-start` | 84.38 MiB | `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224` | `(c128[1,1,1,3,3,1,320,640]{7,5,4,3,2,1,0,6}, c128[1,1,1,3,3,` |
| `module_0121.jit__kernel` | `all-gather-start` | 75.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:609` | `(c128[8,320,640]{2,0,1}, c128[8,640,640]{2,0,1})` |
| `module_0123.jit__kernel` | `all-gather-start` | 75.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:609` | `(c128[8,320,640]{2,0,1}, c128[8,640,640]{2,0,1})` |
| `module_0121.jit__kernel` | `all-to-all` | 63.28 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,10,2,2,11520]{5,3,2,1,0,4}` |
| `module_0121.jit__kernel` | `all-to-all` | 63.28 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,10,2,2,11520]{5,3,2,1,0,4}` |
| `module_0123.jit__kernel` | `all-to-all` | 63.28 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,10,2,2,11520]{5,3,2,1,0,4}` |
| `module_0123.jit__kernel` | `all-to-all` | 63.28 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[9,1,10,2,2,11520]{5,3,2,1,0,4}` |
| `module_0121.jit__kernel` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:609` | `((c128[1,320,320]{1,0,2}, c128[8,320,320]{1,0,2}), (c128[1,3` |
| `module_0123.jit__kernel` | `all-gather-start` | 42.19 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:609` | `((c128[1,320,320]{1,0,2}, c128[8,320,320]{1,0,2}), (c128[1,3` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 55 | 28.12 MiB | 192.20 MiB |
| `jit_convert_element_type` | 30 | 6.00 MiB | 9.23 MiB |
| `jit_multiply` | 16 | 12.00 MiB | 29.32 MiB |
| `jit_gather` | 15 | 112.50 MiB | 237.37 MiB |
| `jit__per_rank` | 12 | 1012.50 MiB | 3.01 GiB |
| `jit_true_divide` | 12 | 14.06 MiB | 48.61 MiB |
| `jit_add` | 12 | 4.00 MiB | 12.12 MiB |
| `jit_concatenate` | 10 | 12.66 MiB | 19.53 MiB |
| `jit_subtract` | 9 | 168.00 MiB | 198.40 MiB |
| `jit_transpose` | 8 | 1012.50 MiB | 1.05 GiB |
| `jit_iota` | 8 | 50.00 KiB | 201.03 KiB |
| `jit_reshape` | 7 | 28.12 MiB | 112.59 MiB |
| `jit__psum` | 7 | 1.76 MiB | 3.57 MiB |
| `jit_dynamic_slice` | 6 | 10.00 MiB | 18.03 MiB |
| `jit_select_n` | 5 | 6.25 MiB | 6.25 MiB |
| `jit_squeeze` | 5 | 4.00 MiB | 4.01 MiB |
| `jit__broadcast_arrays` | 5 | 4.00 MiB | 5.08 MiB |
| `jit_less` | 5 | 2.25 MiB | 2.25 MiB |
| `jit__kernel` | 4 | 3.85 GiB | 12.36 GiB |
| `jit_sigma_sx` | 4 | 226.76 MiB | 822.66 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 48 |
| `__cublas$triangularSolve` | 13 |
| `lorrax_phdf5_write` | 10 |
| `lorrax_phdf5_read` | 2 |
| `cusolver_getrf_ffi` | 2 |
| `cu_lu_pivots_to_permutation` | 2 |
| `__cusolver$cholesky` | 1 |
| `cusolver_syevd_ffi` | 1 |

