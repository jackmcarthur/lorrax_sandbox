# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_wfn_rchunk_shardmap_2026-05-13/profile/xla_dump`
**Modules dumped:** 860
**Sum of per-module peak live HBM:** 219.69 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0306.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0385.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0387.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0388.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0469.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0493.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0495.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0601.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0603.jit__kernel` | 10.36 GiB | 10.19 GiB — preallocated-temp: |
| `module_0214.jit__kernel` | 10.11 GiB | 9.94 GiB — preallocated-temp: |
| `module_0251.jit__kernel` | 10.11 GiB | 9.94 GiB — preallocated-temp: |
| `module_0253.jit__kernel` | 10.11 GiB | 9.94 GiB — preallocated-temp: |
| `module_0785.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_0787.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_0809.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_0811.jit__kernel` | 5.14 GiB | 3.10 GiB — preallocated-temp: |
| `module_0711.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |
| `module_0713.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |
| `module_0753.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |
| `module_0755.jit__kernel` | 3.06 GiB | 2.03 GiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0711.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0711.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0713.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0713.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0753.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0753.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0755.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0755.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0785.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0785.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0787.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0787.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0809.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0809.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0811.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0811.jit__kernel` | `all-gather-start` | 187.50 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,164,2775]{2,0,1}, c128[9,328,2775]{2,0,1})` |
| `module_0667.jit__kernel` | `all-gather-start` | 182.92 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,160,2775]{2,0,1}, c128[9,320,2775]{2,0,1})` |
| `module_0667.jit__kernel` | `all-gather-start` | 182.92 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,160,2775]{2,0,1}, c128[9,320,2775]{2,0,1})` |
| `module_0669.jit__kernel` | `all-gather-start` | 182.92 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,160,2775]{2,0,1}, c128[9,320,2775]{2,0,1})` |
| `module_0669.jit__kernel` | `all-gather-start` | 182.92 MiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,160,2775]{2,0,1}, c128[9,320,2775]{2,0,1})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit__per_rank` | 104 | 1.01 GiB | 12.44 GiB |
| `jit_broadcast_in_dim` | 82 | 28.83 MiB | 296.77 MiB |
| `jit_convert_element_type` | 74 | 6.00 MiB | 21.34 MiB |
| `jit__psum` | 67 | 4.75 MiB | 26.61 MiB |
| `jit__kernel` | 34 | 10.36 GiB | 168.41 GiB |
| `jit__take` | 34 | 28.83 MiB | 120.72 MiB |
| `jit_true_divide` | 28 | 28.83 MiB | 491.59 MiB |
| `jit_dynamic_slice` | 28 | 10.00 MiB | 37.54 MiB |
| `jit_transpose` | 26 | 259.45 MiB | 2.84 GiB |
| `jit_squeeze` | 26 | 4.00 MiB | 8.77 MiB |
| `jit_multiply` | 25 | 28.83 MiB | 177.69 MiB |
| `jit_concatenate` | 16 | 2.11 MiB | 8.44 MiB |
| `jit_gather` | 14 | 112.50 MiB | 371.04 MiB |
| `jit__einsum` | 14 | 40.00 MiB | 205.82 MiB |
| `jit__init_V` | 14 | 14.77 MiB | 205.42 MiB |
| `jit__identity_fn` | 12 | 28.83 MiB | 228.27 MiB |
| `jit__squeeze` | 12 | 5.77 MiB | 53.56 MiB |
| `jit_add` | 12 | 2.64 MiB | 11.86 MiB |
| `jit__multi_slice` | 11 | 43.24 MiB | 341.72 MiB |
| `jit_subtract` | 11 | 8.00 MiB | 24.78 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 233 |
| `xla_python_gpu_callback` | 60 |
| `lorrax_phdf5_write` | 50 |
| `lorrax_phdf5_read` | 33 |
| `lorrax_phdf5_read_kchunk_union` | 11 |
| `lorrax_cusolvermp_batched_solve_lu` | 9 |
| `__cublas$triangularSolve` | 8 |
| `cusolver_getrf_ffi` | 4 |
| `cu_lu_pivots_to_permutation` | 4 |
| `lorrax_cusolvermp_batched_potrf` | 3 |
| `lorrax_cusolvermp_batched_potrs` | 3 |
| `cusolver_syevd_ffi` | 2 |

