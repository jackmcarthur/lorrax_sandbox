# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/D_gflat_cri3_3x3_muchunk_2026-05-12/profile/xla_dump`
**Modules dumped:** 395
**Sum of per-module peak live HBM:** 266.25 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0241.jit__kernel` | 42.83 GiB | 40.69 GiB — preallocated-temp: |
| `module_0279.jit__kernel` | 42.83 GiB | 40.69 GiB — preallocated-temp: |
| `module_0281.jit__kernel` | 42.83 GiB | 40.69 GiB — preallocated-temp: |
| `module_0409.jit__kernel` | 27.23 GiB | 24.51 GiB — preallocated-temp: |
| `module_0513.jit__kernel` | 25.30 GiB | 24.51 GiB — preallocated-temp: |
| `module_0515.jit__kernel` | 25.30 GiB | 24.51 GiB — preallocated-temp: |
| `module_0373.jit__kernel` | 7.58 GiB | 7.36 GiB — preallocated-temp: |
| `module_0411.jit__kernel` | 7.58 GiB | 7.36 GiB — preallocated-temp: |
| `module_0413.jit__kernel` | 7.58 GiB | 7.36 GiB — preallocated-temp: |
| `module_0021.jit__local_fft` | 5.21 GiB | 2.61 GiB — preallocated-temp: |
| `module_0023.jit__local_fft` | 5.21 GiB | 2.61 GiB — preallocated-temp: |
| `module_0527.jit__where` | 2.09 GiB | 711.90 MiB — output shape is \|c128[9,376,13787]\|, maybe-live-out: |
| `module_0529.jit__where` | 2.09 GiB | 711.90 MiB — output shape is \|c128[9,376,13787]\|, maybe-live-out: |
| `module_0201.jit__ifft_contract_fft` | 1.29 GiB | 621.28 MiB — preallocated-temp: |
| `module_0203.jit__ifft_contract_fft` | 1.29 GiB | 621.28 MiB — preallocated-temp: |
| `module_0199.jit__ifft_conj` | 931.92 MiB | 621.28 MiB — preallocated-temp: |
| `module_0201.jit__ifft_conj` | 931.92 MiB | 621.28 MiB — preallocated-temp: |
| `module_0639.jit_sigma_coh` | 921.23 MiB | 698.94 MiB — preallocated-temp: |
| `module_0641.jit_sigma_coh` | 921.23 MiB | 698.94 MiB — preallocated-temp: |
| `module_0637.jit_sigma_sx` | 844.45 MiB | 698.94 MiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0241.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1061` | `c128[2,5,752,20084]{3,2,1,0}` |
| `module_0241.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1062` | `c128[5,1504,1,2,10042]{4,2,1,0,3}` |
| `module_0279.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1061` | `c128[2,5,752,20084]{3,2,1,0}` |
| `module_0279.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1062` | `c128[5,1504,1,2,10042]{4,2,1,0,3}` |
| `module_0281.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1061` | `c128[2,5,752,20084]{3,2,1,0}` |
| `module_0281.jit__kernel` | `all-to-all` | 2.25 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1062` | `c128[5,1504,1,2,10042]{4,2,1,0,3}` |
| `module_0241.jit__kernel` | `all-to-all` | 2.03 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:789` | `c128[9,2,752,1,10042]{4,3,2,0,1}` |
| `module_0241.jit__kernel` | `all-to-all` | 2.03 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:791` | `c128[9,1,2,376,20084]{4,3,1,0,2}` |
| `module_0279.jit__kernel` | `all-to-all` | 2.03 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:789` | `c128[9,2,752,1,10042]{4,3,2,0,1}` |
| `module_0279.jit__kernel` | `all-to-all` | 2.03 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:791` | `c128[9,1,2,376,20084]{4,3,1,0,2}` |
| `module_0281.jit__kernel` | `all-to-all` | 2.03 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:789` | `c128[9,2,752,1,10042]{4,3,2,0,1}` |
| `module_0281.jit__kernel` | `all-to-all` | 2.03 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:791` | `c128[9,1,2,376,20084]{4,3,1,0,2}` |
| `module_0241.jit__kernel` | `all-gather-start` | 1.30 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,2,9,45,45,120]{5,4,3,2,1,0}, c128[16,2,9,45,45,120]{` |
| `module_0241.jit__kernel` | `all-gather-start` | 1.30 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,2,9,45,45,120]{5,4,3,2,1,0}, c128[16,2,9,45,45,120]{` |
| `module_0241.jit__kernel` | `all-gather-start` | 1.30 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,2,9,45,45,120]{5,4,3,2,1,0}, c128[16,2,9,45,45,120]{` |
| `module_0241.jit__kernel` | `all-gather-start` | 1.30 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,2,9,45,45,120]{5,4,3,2,1,0}, c128[16,2,9,45,45,120]{` |
| `module_0241.jit__kernel` | `all-gather-start` | 1.30 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,2,9,45,45,120]{5,4,3,2,1,0}, c128[16,2,9,45,45,120]{` |
| `module_0279.jit__kernel` | `all-gather-start` | 1.30 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,2,9,45,45,120]{5,4,3,2,1,0}, c128[16,2,9,45,45,120]{` |
| `module_0279.jit__kernel` | `all-gather-start` | 1.30 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,2,9,45,45,120]{5,4,3,2,1,0}, c128[16,2,9,45,45,120]{` |
| `module_0279.jit__kernel` | `all-gather-start` | 1.30 GiB | `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109` | `(c128[4,2,9,45,45,120]{5,4,3,2,1,0}, c128[16,2,9,45,45,120]{` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 62 | 711.90 MiB | 1.89 GiB |
| `jit__per_rank` | 38 | 711.90 MiB | 2.71 GiB |
| `jit__psum` | 37 | 25.03 MiB | 43.48 MiB |
| `jit_convert_element_type` | 33 | 2.50 KiB | 4.52 KiB |
| `jit_multiply` | 13 | 1.76 MiB | 3.57 MiB |
| `jit__take` | 12 | 3.79 MiB | 15.15 MiB |
| `jit_gather` | 10 | 621.28 MiB | 1.52 GiB |
| `jit_true_divide` | 10 | 33.05 MiB | 132.20 MiB |
| `jit_iota` | 10 | 53.86 KiB | 110.34 KiB |
| `jit__kernel` | 9 | 42.83 GiB | 229.05 GiB |
| `jit__identity_fn` | 8 | 33.05 MiB | 133.24 MiB |
| `jit_add` | 8 | 2.64 MiB | 10.55 MiB |
| `jit_concatenate` | 8 | 512.00 B | 1.41 KiB |
| `jit__multi_slice` | 7 | 49.57 MiB | 198.28 MiB |
| `jit_reshape` | 6 | 155.32 MiB | 621.28 MiB |
| `jit__squeeze` | 6 | 6.61 MiB | 26.44 MiB |
| `jit_less` | 6 | 175.07 KiB | 350.51 KiB |
| `jit_dynamic_slice` | 6 | 336.00 B | 1.44 KiB |
| `jit_squeeze` | 6 | 144.00 B | 608.00 B |
| `jit__lambda_` | 5 | 711.90 MiB | 1.47 GiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 119 |
| `xla_python_gpu_callback` | 30 |
| `lorrax_phdf5_read_kchunk_union` | 15 |
| `lorrax_phdf5_write` | 15 |
| `__cublas$triangularSolve` | 14 |
| `lorrax_phdf5_read` | 5 |
| `__cusolver$cholesky` | 2 |
| `cusolver_syevd_ffi` | 2 |

