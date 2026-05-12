# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/C_bse_davidson_profile_2026-04-28/profile/xla_dump`
**Modules dumped:** 73
**Sum of per-module peak live HBM:** 9.95 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0063.jit__matvec_impl` | 4.46 GiB | 4.40 GiB — preallocated-temp: |
| `module_0065.jit__matvec_impl` | 4.46 GiB | 4.40 GiB — preallocated-temp: |
| `module_0037.jit_scatter-add` | 113.38 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0039.jit_scatter-add` | 113.38 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0053.jit__unnamed_wrapped_function_` | 112.50 MiB | 56.25 MiB — output shape is \|c128[1,240,240,4,4,4]\|, maybe-live-out: |
| `module_0039.jit__unnamed_wrapped_function_` | 112.50 MiB | 56.25 MiB — output shape is \|c128[1,240,240,4,4,4]\|, maybe-live-out: |
| `module_0041.jit__unnamed_wrapped_function_` | 112.50 MiB | 56.25 MiB — output shape is \|c128[1,240,240,4,4,4]\|, maybe-live-out: |
| `module_0043.jit__unnamed_wrapped_function_` | 112.50 MiB | 56.25 MiB — output shape is \|c128[1,240,240,4,4,4]\|, maybe-live-out: |
| `module_0055.jit__unnamed_wrapped_function_` | 112.50 MiB | 56.25 MiB — output shape is \|c128[1,240,240,4,4,4]\|, maybe-live-out: |
| `module_0057.jit__unnamed_wrapped_function_` | 112.50 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0059.jit__unnamed_wrapped_function_` | 112.50 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0003.jit__identity_fn` | 7.50 MiB | 3.75 MiB — output shape is \|c128[64,8,2,240]\|, maybe-live-out: |
| `module_0027.jit_add` | 2.64 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0029.jit_add` | 2.64 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0001.jit__identity_fn` | 1.88 MiB | 960.00 KiB — output shape is \|c128[64,2,2,240]\|, maybe-live-out: |
| `module_0025.jit_multiply` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0027.jit_multiply` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0035.jit__squeeze` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0037.jit__squeeze` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0107.jit__ritz_and_residuals` | 1.42 MiB | 1011.90 KiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0063.jit__matvec_impl` | `reduce-scatter` | 18.79 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:298` | `(c128[10,4,1,64]{2,0,3,1}, c128[64,4,240,2,10]{3,2,4,0,1})` |
| `module_0065.jit__matvec_impl` | `reduce-scatter` | 18.79 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:298` | `(c128[10,4,1,64]{2,0,3,1}, c128[64,4,240,2,10]{3,2,4,0,1})` |
| `module_0063.jit__matvec_impl` | `reduce-scatter` | 40.00 KiB | `` | `c128[64,10,4,1]{2,1,0,3}` |
| `module_0065.jit__matvec_impl` | `reduce-scatter` | 40.00 KiB | `` | `c128[64,10,4,1]{2,1,0,3}` |
| `module_0059.jit__identity_fn` | `all-gather-start` | 20.00 KiB | `` | `(f64[1,64,8]{2,1,0}, f64[4,64,8]{2,1,0})` |
| `module_0061.jit__identity_fn` | `all-gather-start` | 5.00 KiB | `` | `(f64[1,64,2]{2,1,0}, f64[4,64,2]{2,1,0})` |
| `module_0063.jit__identity_fn` | `all-gather-start` | 5.00 KiB | `` | `(f64[1,64,2]{2,1,0}, f64[4,64,2]{2,1,0})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit__unnamed_wrapped_function_` | 11 | 112.50 MiB | 787.50 MiB |
| `jit_concatenate` | 10 | 400.00 KiB | 2.19 MiB |
| `jit__ritz_and_residuals` | 8 | 1.42 MiB | 7.31 MiB |
| `jit__identity_fn` | 7 | 7.50 MiB | 9.40 MiB |
| `jit_multiply` | 5 | 1.76 MiB | 5.29 MiB |
| `jit_broadcast_in_dim` | 5 | 7.50 KiB | 22.52 KiB |
| `jit_convert_element_type` | 4 | 32.00 B | 80.00 B |
| `jit__matvec_impl` | 2 | 4.46 GiB | 8.92 GiB |
| `jit_scatter-add` | 2 | 113.38 MiB | 226.76 MiB |
| `jit_add` | 2 | 2.64 MiB | 5.27 MiB |
| `jit__squeeze` | 2 | 1.76 MiB | 3.52 MiB |
| `jit__impl` | 2 | 85.29 KiB | 170.57 KiB |
| `jit_reshape` | 2 | 80.00 KiB | 160.00 KiB |
| `jit__multi_slice` | 2 | 11.25 KiB | 22.50 KiB |
| `jit_conjugate` | 2 | 7.50 KiB | 15.00 KiB |
| `jit_gather` | 2 | 92.00 B | 184.00 B |
| `jit_dynamic_slice` | 2 | 40.00 B | 80.00 B |
| `jit_squeeze` | 2 | 32.00 B | 64.00 B |
| `jit__psum` | 1 | 15.05 KiB | 15.05 KiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 42 |
| `__cublas$triangularSolve` | 32 |
| `cusolver_getrf_ffi` | 16 |
| `cu_lu_pivots_to_permutation` | 16 |
| `__cusolver$cholesky` | 8 |
| `cusolver_syevd_ffi` | 8 |

