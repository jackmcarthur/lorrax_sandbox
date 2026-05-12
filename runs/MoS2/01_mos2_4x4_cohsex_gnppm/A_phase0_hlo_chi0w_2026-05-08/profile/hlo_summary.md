# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/01_mos2_4x4_cohsex_gnppm/A_phase0_hlo_chi0w_2026-05-08/profile/xla_dump`
**Modules dumped:** 325
**Sum of per-module peak live HBM:** 23.25 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0173.jit__kernel` | 1.78 GiB | 1.32 GiB — preallocated-temp: |
| `module_0175.jit__kernel` | 1.78 GiB | 1.32 GiB — preallocated-temp: |
| `module_0209.jit__kernel` | 1.78 GiB | 1.32 GiB — preallocated-temp: |
| `module_0199.jit__kernel` | 1.34 GiB | 900.00 MiB — preallocated-temp: |
| `module_0201.jit__kernel` | 1.34 GiB | 900.00 MiB — preallocated-temp: |
| `module_0223.jit__kernel` | 1.34 GiB | 900.00 MiB — preallocated-temp: |
| `module_0225.jit__kernel` | 1.34 GiB | 900.00 MiB — preallocated-temp: |
| `module_0237.jit__kernel` | 1.34 GiB | 900.00 MiB — preallocated-temp: |
| `module_0267.jit__kernel` | 1.34 GiB | 900.00 MiB — preallocated-temp: |
| `module_0175.jit_transpose` | 900.00 MiB | 450.00 MiB — output shape is \|c128[16,2880,640]\|, maybe-live-out: |
| `module_0177.jit_transpose` | 900.00 MiB | 450.00 MiB — output shape is \|c128[16,2880,640]\|, maybe-live-out: |
| `module_0211.jit_transpose` | 900.00 MiB | 450.00 MiB — output shape is \|c128[16,2880,640]\|, maybe-live-out: |
| `module_0179.jit__per_rank` | 450.00 MiB | 0.00 B —  |
| `module_0181.jit__per_rank` | 450.00 MiB | 0.00 B —  |
| `module_0216.jit__per_rank` | 450.00 MiB | 0.00 B —  |
| `module_0219.jit__per_rank` | 450.00 MiB | 0.00 B —  |
| `module_0221.jit__per_rank` | 450.00 MiB | 0.00 B —  |
| `module_0262.jit__per_rank` | 450.00 MiB | 0.00 B —  |
| `module_0021.jit__local_fft` | 450.00 MiB | 225.00 MiB — preallocated-temp: |
| `module_0023.jit__local_fft` | 450.00 MiB | 225.00 MiB — preallocated-temp: |
| `module_0067.jit__fft_gather_reshard` | 237.51 MiB | 112.50 MiB — preallocated-temp: |
| `module_0069.jit__fft_gather_reshard` | 237.51 MiB | 112.50 MiB — preallocated-temp: |
| `module_0074.jit__fft_gather_reshard` | 237.51 MiB | 112.50 MiB — preallocated-temp: |
| `module_0279.jit_gather` | 137.50 MiB | 125.00 MiB — preallocated-temp: |
| `module_0281.jit_gather` | 137.50 MiB | 125.00 MiB — preallocated-temp: |
| `module_0336.jit_gather` | 137.50 MiB | 125.00 MiB — preallocated-temp: |
| `module_0169.jit__per_rank` | 120.81 MiB | 112.50 MiB — output shape is \|c128[16,5,2,24,24,80]\|, maybe-live-out: |
| `module_0171.jit__per_rank` | 120.81 MiB | 112.50 MiB — output shape is \|c128[16,5,2,24,24,80]\|, maybe-live-out: |
| `module_0204.jit__per_rank` | 120.81 MiB | 112.50 MiB — output shape is \|c128[16,5,2,24,24,80]\|, maybe-live-out: |
| `module_0197.jit__local_fftn` | 112.50 MiB | 56.25 MiB — output shape is \|c128[2,40,24,24,80]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0173.jit__kernel` | `all-gather-start` | 562.50 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[16,20,2,11520]{3,2,0,1}, c128[16,80,2,11520]{3,2,0,1})` |
| `module_0175.jit__kernel` | `all-gather-start` | 562.50 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[16,20,2,11520]{3,2,0,1}, c128[16,80,2,11520]{3,2,0,1})` |
| `module_0209.jit__kernel` | `all-gather-start` | 562.50 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:433` | `(c128[16,20,2,11520]{3,2,0,1}, c128[16,80,2,11520]{3,2,0,1})` |
| `module_0173.jit__kernel` | `all-to-all` | 450.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,4,160,11520]{3,2,1,0}` |
| `module_0173.jit__kernel` | `all-to-all` | 450.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[4,640,1,4,2880]{4,2,1,0,3}` |
| `module_0175.jit__kernel` | `all-to-all` | 450.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,4,160,11520]{3,2,1,0}` |
| `module_0175.jit__kernel` | `all-to-all` | 450.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[4,640,1,4,2880]{4,2,1,0,3}` |
| `module_0209.jit__kernel` | `all-to-all` | 450.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:790` | `c128[4,4,160,11520]{3,2,1,0}` |
| `module_0209.jit__kernel` | `all-to-all` | 450.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:791` | `c128[4,640,1,4,2880]{4,2,1,0,3}` |
| `module_0173.jit__kernel` | `all-gather-start` | 125.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:738` | `(c128[16,160,640]{2,0,1}, c128[16,640,640]{2,0,1})` |
| `module_0175.jit__kernel` | `all-gather-start` | 125.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:738` | `(c128[16,160,640]{2,0,1}, c128[16,640,640]{2,0,1})` |
| `module_0209.jit__kernel` | `all-gather-start` | 125.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:738` | `(c128[16,160,640]{2,0,1}, c128[16,640,640]{2,0,1})` |
| `module_0279.jit_gather` | `all-gather-start` | 125.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/file_io/tagged_arrays.py:294` | `(c128[16,160,640]{2,0,1}, c128[16,640,640]{2,0,1})` |
| `module_0281.jit_gather` | `all-gather-start` | 125.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/file_io/tagged_arrays.py:294` | `(c128[16,160,640]{2,0,1}, c128[16,640,640]{2,0,1})` |
| `module_0336.jit_gather` | `all-gather-start` | 125.00 MiB | `/global/u2/j/jackm/software/lorrax_A/src/file_io/tagged_arrays.py:294` | `(c128[16,160,640]{2,0,1}, c128[16,640,640]{2,0,1})` |
| `module_0199.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:703` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0199.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:704` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0201.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:703` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0201.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:704` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0223.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:703` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0223.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:704` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0225.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:703` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0225.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:704` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0237.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:703` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0237.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:704` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0267.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:703` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0267.jit__kernel` | `all-gather-start` | 113.92 MiB | `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:704` | `(c128[16,40,2333]{2,0,1}, c128[16,160,2333]{2,0,1})` |
| `module_0173.jit__kernel` | `all-to-all` | 112.50 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:429` | `c128[16,1,5,2,4,11520]{5,3,2,1,0,4}` |
| `module_0175.jit__kernel` | `all-to-all` | 112.50 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:429` | `c128[16,1,5,2,4,11520]{5,3,2,1,0,4}` |
| `module_0209.jit__kernel` | `all-to-all` | 112.50 MiB | `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:429` | `c128[16,1,5,2,4,11520]{5,3,2,1,0,4}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 41 | 25.00 MiB | 178.04 MiB |
| `jit__psum` | 36 | 8.44 MiB | 20.97 MiB |
| `jit__per_rank` | 30 | 450.00 MiB | 3.10 GiB |
| `jit_convert_element_type` | 20 | 288.00 B | 1.12 KiB |
| `jit_gather` | 14 | 137.50 MiB | 538.37 MiB |
| `jit_true_divide` | 12 | 12.50 MiB | 75.01 MiB |
| `jit__identity_fn` | 12 | 12.50 MiB | 76.11 MiB |
| `jit__take` | 12 | 5.63 MiB | 19.76 MiB |
| `jit_transpose` | 11 | 900.00 MiB | 2.73 GiB |
| `jit_reshape` | 11 | 12.50 MiB | 112.50 MiB |
| `jit__kernel` | 9 | 1.78 GiB | 13.34 GiB |
| `jit__multi_slice` | 8 | 31.25 MiB | 156.25 MiB |
| `jit_scatter` | 6 | 18.75 MiB | 112.50 MiB |
| `jit__squeeze` | 6 | 12.50 MiB | 75.00 MiB |
| `jit__moveaxis` | 6 | 12.50 MiB | 75.00 MiB |
| `jit_xn` | 6 | 10.47 MiB | 56.25 MiB |
| `jit_iota` | 6 | 640.00 B | 1.69 KiB |
| `jit_yr` | 5 | 10.47 MiB | 47.97 MiB |
| `jit_multiply` | 5 | 1.71 MiB | 5.15 MiB |
| `jit_subtract` | 5 | 13.51 KiB | 53.54 KiB |
| `jit_concatenate` | 4 | 2.11 MiB | 4.22 MiB |
| `jit_sum` | 4 | 40.00 B | 160.00 B |
| `jit__fft_gather_reshard` | 3 | 237.51 MiB | 712.52 MiB |
| `jit__local_fftn` | 3 | 112.50 MiB | 281.25 MiB |
| `jit_minimax_tau_integrate_chi` | 3 | 93.76 MiB | 281.28 MiB |
| `jit__solve_w` | 3 | 35.25 MiB | 105.75 MiB |
| `jit__compute_CCT_LR` | 3 | 25.00 MiB | 75.00 MiB |
| `jit__compute_P_traced` | 3 | 22.75 MiB | 68.25 MiB |
| `jit__batched_chol` | 3 | 15.24 MiB | 45.71 MiB |
| `jit_get_sqrt_v_and_phase` | 3 | 12.88 MiB | 38.65 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 32 |
| `lorrax_phdf5_write` | 18 |
| `__cublas$triangularSolve` | 15 |
| `__cusolver$cholesky` | 3 |
| `lorrax_phdf5_read_kchunk_union` | 3 |
| `xla_python_gpu_callback` | 3 |
| `lorrax_phdf5_read` | 3 |
| `cusolver_getrf_ffi` | 3 |
| `cu_lu_pivots_to_permutation` | 3 |

