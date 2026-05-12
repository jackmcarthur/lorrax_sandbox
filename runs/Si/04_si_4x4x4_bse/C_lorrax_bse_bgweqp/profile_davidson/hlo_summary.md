# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_davidson/xla_dump`
**Modules dumped:** 99
**Sum of per-module peak live HBM:** 1.55 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0115.jit_matvec_scan` | 528.23 MiB | 463.44 MiB — preallocated-temp: |
| `module_0117.jit_matvec_scan` | 528.23 MiB | 463.44 MiB — preallocated-temp: |
| `module_0035.jit_scatter-add` | 113.38 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0037.jit_scatter-add` | 113.38 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0041.jit__local_ifftn` | 112.50 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0043.jit__local_ifftn` | 112.50 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0001.jit__identity_fn` | 3.75 MiB | 1.88 MiB — output shape is \|c128[64,4,2,240]\|, maybe-live-out: |
| `module_0099.jit__ritz_and_residuals` | 2.93 MiB | 2.07 MiB — preallocated-temp: |
| `module_0101.jit__ritz_and_residuals` | 2.93 MiB | 2.07 MiB — preallocated-temp: |
| `module_0025.jit_add` | 2.64 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0027.jit_add` | 2.64 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0095.jit__psum` | 2.50 MiB | 1.25 MiB — output shape is \|c128[80,4,4,64]\|, maybe-live-out: |
| `module_0097.jit__psum` | 2.50 MiB | 1.25 MiB — output shape is \|c128[80,4,4,64]\|, maybe-live-out: |
| `module_0083.jit__ritz_and_residuals` | 2.22 MiB | 1.52 MiB — preallocated-temp: |
| `module_0085.jit__ritz_and_residuals` | 2.22 MiB | 1.52 MiB — preallocated-temp: |
| `module_0079.jit__psum` | 1.88 MiB | 960.00 KiB — output shape is \|c128[60,4,4,64]\|, maybe-live-out: |
| `module_0081.jit__psum` | 1.88 MiB | 960.00 KiB — output shape is \|c128[60,4,4,64]\|, maybe-live-out: |
| `module_0023.jit_multiply` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0025.jit_multiply` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0033.jit__squeeze` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0115.jit_matvec_scan` | `all-to-all` | 3.75 MiB | `` | `c128[64,2,240,4,2]{4,3,2,0,1}` |
| `module_0117.jit_matvec_scan` | `all-to-all` | 3.75 MiB | `` | `c128[64,2,240,4,2]{4,3,2,0,1}` |
| `module_0115.jit_matvec_scan` | `all-gather-start` | 2.81 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:141` | `(c128[2,64,2,240]{3,2,1,0}, c128[4,64,2,240]{3,2,1,0})` |
| `module_0117.jit_matvec_scan` | `all-gather-start` | 2.81 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:141` | `(c128[2,64,2,240]{3,2,1,0}, c128[4,64,2,240]{3,2,1,0})` |
| `module_0115.jit_matvec_scan` | `reduce-scatter` | 960.00 KiB | `` | `c128[64,240,2,2]{2,1,0,3}` |
| `module_0117.jit_matvec_scan` | `reduce-scatter` | 960.00 KiB | `` | `c128[64,240,2,2]{2,1,0,3}` |
| `module_0153.jit__identity_fn` | `all-gather-start` | 480.00 KiB | `` | `(c128[20,2,4,64]{3,2,0,1}, c128[20,4,4,64]{3,2,0,1})` |
| `module_0155.jit__identity_fn` | `all-gather-start` | 480.00 KiB | `` | `(c128[20,2,4,64]{3,2,0,1}, c128[20,4,4,64]{3,2,0,1})` |
| `module_0153.jit__identity_fn` | `all-gather-start` | 240.00 KiB | `` | `(c128[20,2,2,64]{3,1,0,2}, c128[20,2,4,64]{3,1,0,2})` |
| `module_0155.jit__identity_fn` | `all-gather-start` | 240.00 KiB | `` | `(c128[20,2,2,64]{3,1,0,2}, c128[20,2,4,64]{3,1,0,2})` |
| `module_0115.jit_matvec_scan` | `all-gather-start` | 24.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:141` | `((c128[2,2,64]{2,1,0}, c128[2,64,2]{1,2,0}), (c128[4,2,64]{2` |
| `module_0117.jit_matvec_scan` | `all-gather-start` | 24.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:141` | `((c128[2,2,64]{2,1,0}, c128[2,64,2]{1,2,0}), (c128[4,2,64]{2` |
| `module_0043.jit__identity_fn` | `all-gather-start` | 10.00 KiB | `` | `(f64[1,64,4]{2,1,0}, f64[4,64,4]{2,1,0})` |
| `module_0045.jit__identity_fn` | `all-gather-start` | 10.00 KiB | `` | `(f64[1,64,4]{2,1,0}, f64[4,64,4]{2,1,0})` |
| `module_0115.jit_matvec_scan` | `reduce-scatter` | 4.00 KiB | `` | `c128[64,2,2]{1,0,2}` |
| `module_0117.jit_matvec_scan` | `reduce-scatter` | 4.00 KiB | `` | `c128[64,2,2]{1,0,2}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 13 | 1.25 MiB | 6.27 MiB |
| `jit__multi_slice` | 10 | 1.56 MiB | 7.83 MiB |
| `jit_concatenate` | 10 | 800.00 KiB | 4.38 MiB |
| `jit__psum` | 9 | 2.50 MiB | 12.52 MiB |
| `jit__ritz_and_residuals` | 8 | 2.93 MiB | 15.02 MiB |
| `jit__ortho_expand` | 8 | 1.28 MiB | 7.32 MiB |
| `jit__identity_fn` | 7 | 3.75 MiB | 5.49 MiB |
| `jit_convert_element_type` | 6 | 32.00 B | 112.00 B |
| `jit_multiply` | 5 | 1.76 MiB | 5.29 MiB |
| `jit_matvec_scan` | 2 | 528.23 MiB | 1.03 GiB |
| `jit_scatter-add` | 2 | 113.38 MiB | 226.76 MiB |
| `jit__local_ifftn` | 2 | 112.50 MiB | 225.00 MiB |
| `jit_add` | 2 | 2.64 MiB | 5.27 MiB |
| `jit__squeeze` | 2 | 1.76 MiB | 3.52 MiB |
| `jit__orthonormalise_batch` | 2 | 326.95 KiB | 653.89 KiB |
| `jit__impl` | 2 | 164.49 KiB | 328.98 KiB |
| `jit_conjugate` | 2 | 7.50 KiB | 15.00 KiB |
| `jit_dynamic_slice` | 2 | 40.00 B | 80.00 B |
| `jit_squeeze` | 2 | 32.00 B | 64.00 B |
| `jit_sqrt` | 2 | 16.00 B | 32.00 B |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 98 |
| `__cublas$triangularSolve` | 52 |
| `cusolver_getrf_ffi` | 26 |
| `cu_lu_pivots_to_permutation` | 26 |
| `__cusolver$cholesky` | 18 |
| `cusolver_syevd_ffi` | 8 |

