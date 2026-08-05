# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/D_gflat_cri3_3x3_muchunk_2026-05-12/profile/xla_dump`
**Modules dumped:** 385
**Sum of per-module peak live HBM:** 496.89 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0240.jit__kernel` | 42.83 GiB | 40.69 GiB — preallocated-temp: |
| `module_0277.jit__kernel` | 42.83 GiB | 40.69 GiB — preallocated-temp: |
| `module_0279.jit__kernel` | 42.83 GiB | 40.69 GiB — preallocated-temp: |
| `module_0710.jit__f` | 37.58 GiB | 24.51 GiB — preallocated-temp: |
| `module_0731.jit__f` | 37.58 GiB | 24.51 GiB — preallocated-temp: |
| `module_0733.jit__f` | 37.58 GiB | 24.51 GiB — preallocated-temp: |
| `module_0682.jit__kernel` | 36.84 GiB | 24.51 GiB — preallocated-temp: |
| `module_0709.jit__kernel` | 36.84 GiB | 24.51 GiB — preallocated-temp: |
| `module_0711.jit__kernel` | 36.84 GiB | 24.51 GiB — preallocated-temp: |
| `module_0707.jit__per_rank` | 12.25 GiB | 0.00 B —  |
| `module_0729.jit__per_rank` | 12.25 GiB | 0.00 B —  |
| `module_0731.jit__per_rank` | 12.25 GiB | 0.00 B —  |
| `module_0450.jit__kernel` | 7.49 GiB | 7.27 GiB — preallocated-temp: |
| `module_0485.jit__kernel` | 7.49 GiB | 7.27 GiB — preallocated-temp: |
| `module_0487.jit__kernel` | 7.49 GiB | 7.27 GiB — preallocated-temp: |
| `module_0723.jit__kernel` | 5.84 GiB | 4.94 GiB — preallocated-temp: |
| `module_0743.jit__kernel` | 5.84 GiB | 4.94 GiB — preallocated-temp: |
| `module_0745.jit__kernel` | 5.84 GiB | 4.94 GiB — preallocated-temp: |
| `module_0679.jit__local_fftn` | 5.45 GiB | 2.72 GiB — output shape is \|c128[2,376,45,45,120]\|, maybe-live-out: |
| `module_0707.jit__local_fftn` | 5.45 GiB | 2.72 GiB — output shape is \|c128[2,376,45,45,120]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0682.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0682.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0709.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0709.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0711.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0711.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:718` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0723.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:876` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0723.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:877` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0743.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:876` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0743.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:877` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0745.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:876` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0745.jit__kernel` | `all-gather-start` | 2.47 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:877` | `(c128[9,376,16329]{2,0,1}, c128[9,752,16329]{2,0,1})` |
| `module_0240.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1090` | `c128[2,5,752,20084]{3,2,1,0}` |
| `module_0240.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1091` | `c128[5,1504,1,2,10042]{4,2,1,0,3}` |
| `module_0277.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1090` | `c128[2,5,752,20084]{3,2,1,0}` |
| `module_0277.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1091` | `c128[5,1504,1,2,10042]{4,2,1,0,3}` |
| `module_0279.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1090` | `c128[2,5,752,20084]{3,2,1,0}` |
| `module_0279.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1091` | `c128[5,1504,1,2,10042]{4,2,1,0,3}` |
| `module_0240.jit__kernel` | `all-to-all` | 2.03 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:818` | `c128[9,2,752,1,10042]{4,3,2,0,1}` |
| `module_0240.jit__kernel` | `all-to-all` | 2.03 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:820` | `c128[9,1,2,376,20084]{4,3,1,0,2}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 48 | 155.32 MiB | 523.28 MiB |
| `jit__per_rank` | 42 | 12.25 GiB | 43.61 GiB |
| `jit__psum` | 35 | 25.03 MiB | 35.91 MiB |
| `jit_convert_element_type` | 33 | 2.50 KiB | 4.52 KiB |
| `jit__kernel` | 12 | 42.83 GiB | 278.99 GiB |
| `jit__take` | 12 | 3.79 MiB | 15.15 MiB |
| `jit_transpose` | 11 | 4.05 GiB | 12.92 GiB |
| `jit_concatenate` | 11 | 11.12 MiB | 33.37 MiB |
| `jit_multiply` | 10 | 6.73 MiB | 23.70 MiB |
| `jit_gather` | 8 | 621.28 MiB | 1.51 GiB |
| `jit__identity_fn` | 8 | 33.05 MiB | 133.24 MiB |
| `jit_add` | 8 | 2.64 MiB | 10.55 MiB |
| `jit__multi_slice` | 7 | 49.57 MiB | 198.28 MiB |
| `jit__broadcast_arrays` | 7 | 5.56 MiB | 16.69 MiB |
| `jit_reshape` | 6 | 155.32 MiB | 621.28 MiB |
| `jit_true_divide` | 6 | 33.05 MiB | 132.19 MiB |
| `jit__squeeze` | 6 | 6.61 MiB | 26.44 MiB |
| `jit__local_fftn` | 5 | 5.45 GiB | 21.78 GiB |
| `jit__moveaxis` | 4 | 155.32 MiB | 621.28 MiB |
| `jit_scatter` | 4 | 39.66 MiB | 158.63 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 95 |
| `xla_python_gpu_callback` | 30 |
| `lorrax_phdf5_write` | 18 |
| `lorrax_phdf5_read_kchunk_union` | 15 |
| `__cublas$triangularSolve` | 14 |
| `lorrax_phdf5_read` | 6 |
| `__cusolver$cholesky` | 2 |
| `cusolver_syevd_ffi` | 2 |

