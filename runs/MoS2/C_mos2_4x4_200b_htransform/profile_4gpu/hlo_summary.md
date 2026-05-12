# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu/xla_dump`
**Modules dumped:** 221
**Sum of per-module peak live HBM:** 33.72 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0063.jit__accum_G` | 2.99 GiB | 2.64 GiB — preallocated-temp: |
| `module_0375.jit__solve_triangular` | 1.35 GiB | 464.06 MiB — preallocated-temp: |
| `module_0377.jit__solve_triangular` | 1.35 GiB | 464.06 MiB — preallocated-temp: |
| `module_0387.jit_eigvalsh` | 1.32 GiB | 900.00 MiB — preallocated-temp: |
| `module_0373.jit_add` | 1.32 GiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0383.jit_add` | 1.32 GiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0409.jit__solve_triangular` | 914.06 MiB | 309.38 MiB — preallocated-temp: |
| `module_0411.jit__solve_triangular` | 914.06 MiB | 309.38 MiB — preallocated-temp: |
| `module_0367.jit_multiply` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0385.jit_multiply` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0365.jit_reshape` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0369.jit_swapaxes` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0371.jit_conjugate` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0379.jit_conjugate` | 900.00 MiB | 450.00 MiB — output shape is \|c128[960,960,32]\|, maybe-live-out: |
| `module_0381.jit_transpose` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0421.jit_eigvalsh` | 886.09 MiB | 590.62 MiB — preallocated-temp: |
| `module_0407.jit_add` | 885.94 MiB | 295.31 MiB — output shape is \|c128[21,960,960]\|, maybe-live-out: |
| `module_0417.jit_add` | 885.94 MiB | 295.31 MiB — output shape is \|c128[21,960,960]\|, maybe-live-out: |
| `module_0363.jit_matmul` | 679.01 MiB | 450.00 MiB — output shape is \|c128[32,921600]\|, maybe-live-out: |
| `module_0401.jit_multiply` | 590.63 MiB | 295.31 MiB — output shape is \|c128[21,960,960]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0209.jit__identity_fn` | `all-gather-start` | 281.25 MiB | `` | `(c128[4,960,960]{2,1,0}, c128[16,960,960]{2,1,0})` |
| `module_0027.jit__identity_fn` | `all-gather-start` | 17.58 MiB | `` | `(c128[240,960]{1,0}, c128[960,960]{1,0})` |
| `module_0085.jit_gather` | `all-gather-start` | 17.58 MiB | `` | `(c128[4,60,960]{2,1,0}, c128[16,60,960]{2,1,0})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_multiply` | 24 | 900.00 MiB | 3.33 GiB |
| `jit_add` | 15 | 1.32 GiB | 4.63 GiB |
| `jit_broadcast_in_dim` | 15 | 28.12 MiB | 31.50 MiB |
| `jit_reshape` | 12 | 900.00 MiB | 2.96 GiB |
| `jit_transpose` | 11 | 900.00 MiB | 2.42 GiB |
| `jit_conjugate` | 10 | 900.00 MiB | 3.11 GiB |
| `jit_subtract` | 10 | 464.06 MiB | 466.92 MiB |
| `jit_matmul` | 8 | 679.01 MiB | 1.46 GiB |
| `jit_gather` | 7 | 239.06 MiB | 287.08 MiB |
| `jit__reduce_max` | 6 | 112.56 MiB | 225.11 MiB |
| `jit_convert_element_type` | 6 | 14.94 MiB | 15.00 MiB |
| `jit_iota` | 6 | 7.03 MiB | 14.12 MiB |
| `jit__where` | 6 | 23.44 KiB | 71.40 KiB |
| `jit_exp` | 5 | 16.00 KiB | 42.02 KiB |
| `jit_dynamic_slice` | 5 | 7.98 KiB | 25.90 KiB |
| `jit__solve_triangular` | 4 | 1.35 GiB | 4.48 GiB |
| `jit_swapaxes` | 4 | 900.00 MiB | 1.59 GiB |
| `jit__moveaxis` | 4 | 450.00 MiB | 998.44 MiB |
| `jit_abs` | 4 | 337.50 MiB | 562.58 MiB |
| `jit__identity_fn` | 4 | 281.25 MiB | 333.98 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 9 |
| `__cublas$triangularSolve` | 4 |
| `cusolver_syevd_ffi` | 3 |
| `__cusolver$cholesky` | 2 |
| `cusolver_gesvdj_ffi` | 1 |
| `__cub$DeviceRadixSort` | 1 |

