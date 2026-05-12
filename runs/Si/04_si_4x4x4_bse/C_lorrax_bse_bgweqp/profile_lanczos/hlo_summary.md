# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_lanczos/xla_dump`
**Modules dumped:** 45
**Sum of per-module peak live HBM:** 1.44 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0041.jit__full_run` | 608.04 MiB | 534.65 MiB — preallocated-temp: |
| `module_0043.jit__full_run` | 608.04 MiB | 534.65 MiB — preallocated-temp: |
| `module_0035.jit_scatter-add` | 113.38 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0037.jit_scatter-add` | 113.38 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0001.jit__identity_fn` | 7.50 MiB | 3.75 MiB — output shape is \|c128[64,8,2,240]\|, maybe-live-out: |
| `module_0019.jit_add` | 2.64 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0021.jit_add` | 2.64 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0081.jit_reshape` | 2.50 MiB | 1.25 MiB — output shape is \|c128[20,1,8,8,64]\|, maybe-live-out: |
| `module_0082.jit_reshape` | 2.50 MiB | 1.25 MiB — output shape is \|c128[20,1,8,8,64]\|, maybe-live-out: |
| `module_0083.jit_reshape` | 2.50 MiB | 1.25 MiB — output shape is \|c128[20,1,8,8,64]\|, maybe-live-out: |
| `module_0017.jit_multiply` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0019.jit_multiply` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0033.jit__squeeze` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0035.jit__squeeze` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0013.jit_multiply` | 907.50 KiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0015.jit_multiply` | 907.50 KiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0003.jit__psum` | 15.05 KiB | 7.50 KiB — output shape is \|c128[480]\|, maybe-live-out: |
| `module_0005.jit__multi_slice` | 11.25 KiB | 7.50 KiB — parameter 0, shape \|c128[480]\| at ShapeIndex {}: |
| `module_0007.jit__multi_slice` | 11.25 KiB | 7.50 KiB — parameter 0, shape \|c128[480]\| at ShapeIndex {}: |
| `module_0007.jit_conjugate` | 7.50 KiB | 3.80 KiB — output shape is \|c128[240]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0041.jit__full_run` | `all-to-all` | 15.00 MiB | `` | `c128[64,2,240,8,4]{4,3,2,0,1}` |
| `module_0043.jit__full_run` | `all-to-all` | 15.00 MiB | `` | `c128[64,2,240,8,4]{4,3,2,0,1}` |
| `module_0041.jit__full_run` | `all-gather-start` | 5.62 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:141` | `(c128[4,64,2,240]{3,2,1,0}, c128[8,64,2,240]{3,2,1,0})` |
| `module_0043.jit__full_run` | `all-gather-start` | 5.62 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:141` | `(c128[4,64,2,240]{3,2,1,0}, c128[8,64,2,240]{3,2,1,0})` |
| `module_0041.jit__full_run` | `reduce-scatter` | 1.88 MiB | `` | `c128[64,240,2,4]{2,1,0,3}` |
| `module_0041.jit__full_run` | `all-gather-start` | 1.88 MiB | `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:298` | `(c128[20,2048]{0,1}, c128[20,4096]{0,1})` |
| `module_0043.jit__full_run` | `reduce-scatter` | 1.88 MiB | `` | `c128[64,240,2,4]{2,1,0,3}` |
| `module_0043.jit__full_run` | `all-gather-start` | 1.88 MiB | `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:298` | `(c128[20,2048]{0,1}, c128[20,4096]{0,1})` |
| `module_0041.jit__full_run` | `all-gather-start` | 96.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:141` | `((c128[4,4,64]{2,1,0}, c128[4,64,4]{1,2,0}), (c128[8,4,64]{2` |
| `module_0043.jit__full_run` | `all-gather-start` | 96.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:141` | `((c128[4,4,64]{2,1,0}, c128[4,64,4]{1,2,0}), (c128[8,4,64]{2` |
| `module_0041.jit__full_run` | `all-gather-start` | 48.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:291` | `(c128[1,4,4,64]{3,1,0,2}, c128[1,4,8,64]{3,1,0,2})` |
| `module_0043.jit__full_run` | `all-gather-start` | 48.00 KiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:291` | `(c128[1,4,4,64]{3,1,0,2}, c128[1,4,8,64]{3,1,0,2})` |
| `module_0041.jit__full_run` | `reduce-scatter` | 16.00 KiB | `` | `c128[64,4,4]{1,0,2}` |
| `module_0043.jit__full_run` | `reduce-scatter` | 16.00 KiB | `` | `c128[64,4,4]{1,0,2}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_multiply` | 9 | 1.76 MiB | 5.29 MiB |
| `jit_convert_element_type` | 6 | 32.00 B | 112.00 B |
| `jit_broadcast_in_dim` | 5 | 7.50 KiB | 22.52 KiB |
| `jit_reshape` | 3 | 2.50 MiB | 7.50 MiB |
| `jit__full_run` | 2 | 608.04 MiB | 1.19 GiB |
| `jit_scatter-add` | 2 | 113.38 MiB | 226.76 MiB |
| `jit_add` | 2 | 2.64 MiB | 5.27 MiB |
| `jit__squeeze` | 2 | 1.76 MiB | 3.52 MiB |
| `jit__multi_slice` | 2 | 11.25 KiB | 22.50 KiB |
| `jit_conjugate` | 2 | 7.50 KiB | 15.00 KiB |
| `jit_dynamic_slice` | 2 | 40.00 B | 80.00 B |
| `jit_squeeze` | 2 | 32.00 B | 64.00 B |
| `jit_concatenate` | 2 | 24.00 B | 48.00 B |
| `jit_sqrt` | 2 | 16.00 B | 32.00 B |
| `jit__identity_fn` | 1 | 7.50 MiB | 7.50 MiB |
| `jit__psum` | 1 | 15.05 KiB | 15.05 KiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 16 |
| `cusolver_syevd_ffi` | 2 |

