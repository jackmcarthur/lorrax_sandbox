# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/B_bispinor_profile/profile/xla_dump`
**Modules dumped:** 1101
**Sum of per-module peak live HBM:** 42.23 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0929.jit__kernel` | 2.99 GiB | 1.98 GiB — preallocated-temp: |
| `module_0931.jit__kernel` | 2.99 GiB | 1.98 GiB — preallocated-temp: |
| `module_1078.jit__kernel` | 2.99 GiB | 1.98 GiB — preallocated-temp: |
| `module_0997.jit_gather` | 1.76 GiB | 1.32 GiB — preallocated-temp: |
| `module_0999.jit_gather` | 1.76 GiB | 1.32 GiB — preallocated-temp: |
| `module_1173.jit_gather` | 1.76 GiB | 1.32 GiB — preallocated-temp: |
| `module_0923.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0925.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_1070.jit__per_rank` | 1012.50 MiB | 0.00 B —  |
| `module_0023.jit__local_fft` | 810.00 MiB | 405.00 MiB — preallocated-temp: |
| `module_0025.jit__local_fft` | 810.00 MiB | 405.00 MiB — preallocated-temp: |
| `module_0645.jit__kernel` | 473.57 MiB | 405.04 MiB — preallocated-temp: |
| `module_0647.jit__kernel` | 473.57 MiB | 405.04 MiB — preallocated-temp: |
| `module_0769.jit__kernel` | 473.57 MiB | 405.04 MiB — preallocated-temp: |
| `module_0771.jit__kernel` | 473.57 MiB | 405.04 MiB — preallocated-temp: |
| `module_0819.jit__kernel` | 473.57 MiB | 405.04 MiB — preallocated-temp: |
| `module_0893.jit__kernel` | 473.57 MiB | 405.04 MiB — preallocated-temp: |
| `module_0895.jit__kernel` | 473.57 MiB | 405.04 MiB — preallocated-temp: |
| `module_0925.jit__kernel` | 473.57 MiB | 405.04 MiB — preallocated-temp: |
| `module_1030.jit__kernel` | 473.57 MiB | 405.04 MiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0997.jit_gather` | `all-gather-start` | 1.32 GiB | `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:271` | `(c128[1,4,4,3,3,1,320,640]{7,5,4,3,2,1,0,6}, c128[1,4,4,3,3,` |
| `module_0999.jit_gather` | `all-gather-start` | 1.32 GiB | `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:271` | `(c128[1,4,4,3,3,1,320,640]{7,5,4,3,2,1,0,6}, c128[1,4,4,3,3,` |
| `module_1173.jit_gather` | `all-gather-start` | 1.32 GiB | `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:271` | `(c128[1,4,4,3,3,1,320,640]{7,5,4,3,2,1,0,6}, c128[1,4,4,3,3,` |
| `module_0997.jit_gather` | `all-gather-start` | 675.00 MiB | `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:271` | `(c128[1,4,4,3,3,1,320,320]{6,5,4,3,2,1,0,7}, c128[1,4,4,3,3,` |
| `module_0999.jit_gather` | `all-gather-start` | 675.00 MiB | `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:271` | `(c128[1,4,4,3,3,1,320,320]{6,5,4,3,2,1,0,7}, c128[1,4,4,3,3,` |
| `module_1173.jit_gather` | `all-gather-start` | 675.00 MiB | `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:271` | `(c128[1,4,4,3,3,1,320,320]{6,5,4,3,2,1,0,7}, c128[1,4,4,3,3,` |
| `module_0929.jit__kernel` | `all-gather-start` | 159.46 MiB | `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1546` | `(c128[9,160,2419]{2,0,1}, c128[9,320,2419]{2,0,1})` |
| `module_0929.jit__kernel` | `all-gather-start` | 159.46 MiB | `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1547` | `(c128[9,160,2419]{2,0,1}, c128[9,320,2419]{2,0,1})` |
| `module_0931.jit__kernel` | `all-gather-start` | 159.46 MiB | `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1546` | `(c128[9,160,2419]{2,0,1}, c128[9,320,2419]{2,0,1})` |
| `module_0931.jit__kernel` | `all-gather-start` | 159.46 MiB | `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1547` | `(c128[9,160,2419]{2,0,1}, c128[9,320,2419]{2,0,1})` |
| `module_1078.jit__kernel` | `all-gather-start` | 159.46 MiB | `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1546` | `(c128[9,160,2419]{2,0,1}, c128[9,320,2419]{2,0,1})` |
| `module_1078.jit__kernel` | `all-gather-start` | 159.46 MiB | `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1547` | `(c128[9,160,2419]{2,0,1}, c128[9,320,2419]{2,0,1})` |
| `module_0645.jit__kernel` | `all-gather-start` | 91.92 MiB | `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:688` | `(c128[9,334,668]{2,0,1}, c128[9,668,668]{2,0,1})` |
| `module_0647.jit__kernel` | `all-gather-start` | 91.92 MiB | `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:688` | `(c128[9,334,668]{2,0,1}, c128[9,668,668]{2,0,1})` |
| `module_0769.jit__kernel` | `all-gather-start` | 91.92 MiB | `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:688` | `(c128[9,334,668]{2,0,1}, c128[9,668,668]{2,0,1})` |
| `module_0771.jit__kernel` | `all-gather-start` | 91.92 MiB | `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:688` | `(c128[9,334,668]{2,0,1}, c128[9,668,668]{2,0,1})` |
| `module_0819.jit__kernel` | `all-gather-start` | 91.92 MiB | `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:688` | `(c128[9,334,668]{2,0,1}, c128[9,668,668]{2,0,1})` |
| `module_0893.jit__kernel` | `all-gather-start` | 91.92 MiB | `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:688` | `(c128[9,334,668]{2,0,1}, c128[9,668,668]{2,0,1})` |
| `module_0895.jit__kernel` | `all-gather-start` | 91.92 MiB | `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:688` | `(c128[9,334,668]{2,0,1}, c128[9,668,668]{2,0,1})` |
| `module_0925.jit__kernel` | `all-gather-start` | 91.92 MiB | `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:688` | `(c128[9,334,668]{2,0,1}, c128[9,668,668]{2,0,1})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 194 | 450.00 MiB | 2.59 GiB |
| `jit_convert_element_type` | 78 | 23.48 KiB | 168.15 KiB |
| `jit__per_rank` | 63 | 1012.50 MiB | 8.17 GiB |
| `jit_gather` | 55 | 1.76 GiB | 6.45 GiB |
| `jit_true_divide` | 54 | 11.01 MiB | 289.88 MiB |
| `jit_multiply` | 50 | 981.52 KiB | 17.04 MiB |
| `jit__psum` | 47 | 4.75 MiB | 11.39 MiB |
| `jit__take` | 45 | 3.16 MiB | 33.34 MiB |
| `jit_concatenate` | 43 | 2.11 MiB | 11.54 MiB |
| `jit_add` | 42 | 92.02 KiB | 2.43 MiB |
| `jit_reshape` | 37 | 30.64 MiB | 738.60 MiB |
| `jit_dynamic_slice` | 36 | 1.44 MiB | 23.24 MiB |
| `jit__moveaxis` | 20 | 30.64 MiB | 597.71 MiB |
| `jit_squeeze` | 20 | 61.34 KiB | 911.88 KiB |
| `jit_iota` | 19 | 640.00 B | 4.20 KiB |
| `jit_transpose` | 18 | 91.74 MiB | 995.82 MiB |
| `jit__einsum` | 18 | 3.16 MiB | 43.28 MiB |
| `jit__identity_fn` | 17 | 11.74 MiB | 127.38 MiB |
| `jit__kernel` | 15 | 2.99 GiB | 14.50 GiB |
| `jit__multi_slice` | 15 | 17.61 MiB | 206.93 MiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 115 |
| `__cublas$triangularSolve` | 34 |
| `lorrax_phdf5_write` | 24 |
| `xla_python_gpu_callback` | 12 |
| `__cusolver$cholesky` | 10 |
| `lorrax_phdf5_read_kchunk_union` | 9 |
| `lorrax_phdf5_read` | 3 |

