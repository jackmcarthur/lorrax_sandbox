# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v2/xla_dump`
**Modules dumped:** 219
**Sum of per-module peak live HBM:** 27.38 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0101.jit__accum` | 2.33 GiB | 1.65 GiB — preallocated-temp: |
| `module_0097.jit__reshard_rchunk` | 1.65 GiB | 675.00 MiB — preallocated-temp: |
| `module_0421.jit__solve_triangular` | 1.35 GiB | 464.06 MiB — preallocated-temp: |
| `module_0423.jit__solve_triangular` | 1.35 GiB | 464.06 MiB — preallocated-temp: |
| `module_0041.jit__fft_gather_reshard` | 1.33 GiB | 1012.50 MiB — preallocated-temp: |
| `module_0433.jit_eigvalsh` | 1.32 GiB | 900.00 MiB — preallocated-temp: |
| `module_0419.jit_add` | 1.32 GiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0429.jit_add` | 1.32 GiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0089.jit__fft_and_rslice` | 1012.80 MiB | 337.80 MiB — preallocated-temp: |
| `module_0413.jit_multiply` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0431.jit_multiply` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0411.jit_reshape` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0415.jit_swapaxes` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0417.jit_conjugate` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0425.jit_conjugate` | 900.00 MiB | 450.00 MiB — output shape is \|c128[960,960,32]\|, maybe-live-out: |
| `module_0427.jit_transpose` | 900.00 MiB | 450.00 MiB — output shape is \|c128[32,960,960]\|, maybe-live-out: |
| `module_0409.jit_matmul` | 679.01 MiB | 450.00 MiB — output shape is \|c128[32,921600]\|, maybe-live-out: |
| `module_0245.jit_add` | 675.00 MiB | 225.00 MiB — output shape is \|c128[16,960,960]\|, maybe-live-out: |
| `module_0317.jit_fft` | 618.75 MiB | 337.50 MiB — preallocated-temp: |
| `module_0247.jit_multiply` | 450.00 MiB | 225.00 MiB — output shape is \|c128[16,960,960]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0097.jit__reshard_rchunk` | `all-gather-start` | 1012.50 MiB | `` | `(c128[16,30,2,23040]{3,2,0,1}, c128[16,60,2,23040]{3,2,0,1})` |
| `module_0101.jit__accum` | `all-to-all` | 675.00 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:146` | `c128[16,60,2,1,23040]{4,3,1,0,2}` |
| `module_0097.jit__reshard_rchunk` | `all-to-all` | 337.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:412` | `c128[16,1,15,2,2,23040]{5,3,2,1,0,4}` |
| `module_0317.jit_fft` | `all-gather-start` | 337.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:592` | `(c128[480,960,4,4,1]{4,3,2,1,0}, c128[960,960,4,4,1]{4,3,2,1` |
| `module_0317.jit_fft` | `all-gather-start` | 168.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:592` | `(c128[480,480,4,4,1]{4,3,2,0,1}, c128[480,960,4,4,1]{4,3,2,0` |
| `module_0065.jit_reshape` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:97` | `(c128[16,60,2,240]{2,1,0,3}, c128[16,60,2,480]{2,1,0,3})` |
| `module_0103.jit_cholesky` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:183` | `(c128[480,960]{1,0}, c128[960,960]{1,0})` |
| `module_0265.jit_eigvalsh` | `all-gather-start` | 21.09 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:571` | `(c128[480,960]{1,0}, c128[960,960]{1,0})` |
| `module_0041.jit__fft_gather_reshard` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720` | `(c128[16,15,2,480]{3,2,0,1}, c128[16,30,2,480]{3,2,0,1})` |
| `module_0103.jit_cholesky` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:183` | `(c128[480,480]{0,1}, c128[480,960]{0,1})` |
| `module_0265.jit_eigvalsh` | `all-gather-start` | 10.55 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:571` | `(c128[480,480]{0,1}, c128[480,960]{0,1})` |
| `module_0041.jit__fft_gather_reshard` | `all-to-all` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:727` | `c128[16,30,2,2,240]{4,2,1,0,3}` |
| `module_0101.jit__accum` | `reduce-scatter` | 3.52 MiB | `` | `c128[480,480]{0,1}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 23 | 28.12 MiB | 81.97 MiB |
| `jit_multiply` | 17 | 900.00 MiB | 2.33 GiB |
| `jit_add` | 14 | 1.32 GiB | 3.39 GiB |
| `jit_reshape` | 10 | 900.00 MiB | 2.01 GiB |
| `jit_subtract` | 10 | 126.56 MiB | 129.42 MiB |
| `jit_transpose` | 9 | 900.00 MiB | 1.48 GiB |
| `jit_conjugate` | 8 | 900.00 MiB | 2.31 GiB |
| `jit_iota` | 8 | 7.03 MiB | 14.12 MiB |
| `jit_gather` | 7 | 59.77 MiB | 106.99 MiB |
| `jit_convert_element_type` | 7 | 14.94 MiB | 15.00 MiB |
| `jit_matmul` | 6 | 679.01 MiB | 967.84 MiB |
| `jit__where` | 6 | 23.44 KiB | 71.40 KiB |
| `jit__reduce_max` | 5 | 28.16 MiB | 56.32 MiB |
| `jit_dynamic_slice` | 5 | 15.02 KiB | 36.42 KiB |
| `jit_true_divide` | 5 | 15.01 KiB | 32.04 KiB |
| `jit_abs` | 4 | 84.38 MiB | 140.71 MiB |
| `jit_exp` | 4 | 16.00 KiB | 31.52 KiB |
| `jit_swapaxes` | 3 | 900.00 MiB | 1.35 GiB |
| `jit_negative` | 3 | 450.00 MiB | 450.03 MiB |
| `jit__identity_fn` | 3 | 281.25 MiB | 309.38 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 7 |
| `__cusolver$cholesky` | 2 |
| `cusolver_syevd_ffi` | 2 |
| `__cublas$triangularSolve` | 2 |
| `cusolver_gesvdj_ffi` | 1 |
| `__cub$DeviceRadixSort` | 1 |

