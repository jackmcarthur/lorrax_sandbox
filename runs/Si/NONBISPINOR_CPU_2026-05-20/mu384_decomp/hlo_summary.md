# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/NONBISPINOR_CPU_2026-05-20/mu384_decomp/xla_dump`
**Modules dumped:** 397
**Sum of per-module peak live HBM:** 103.94 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0243.jit_fn` | 24.28 GiB | 22.81 GiB — preallocated-temp: |
| `module_0245.jit_fn` | 24.28 GiB | 22.81 GiB — preallocated-temp: |
| `module_0342.jit_fn` | 24.28 GiB | 22.81 GiB — preallocated-temp: |
| `module_0069.jit__kernel` | 7.66 GiB | 7.62 GiB — preallocated-temp: |
| `module_0071.jit__kernel` | 7.66 GiB | 7.62 GiB — preallocated-temp: |
| `module_0255.jit_gather` | 1.60 GiB | 1.42 GiB — parameter 0, shape \|c128[64,216,6912]\| at ShapeIndex {}: |
| `module_0257.jit_gather` | 1.60 GiB | 1.42 GiB — parameter 0, shape \|c128[64,216,6912]\| at ShapeIndex {}: |
| `module_0021.jit__local_fft` | 1.58 GiB | 810.00 MiB — preallocated-temp: |
| `module_0023.jit__local_fft` | 1.58 GiB | 810.00 MiB — preallocated-temp: |
| `module_0269.jit__identity_fn` | 729.00 MiB | 364.50 MiB — preallocated-temp: |
| `module_0271.jit__identity_fn` | 729.00 MiB | 364.50 MiB — preallocated-temp: |
| `module_0157.jit_fn` | 693.56 MiB | 546.75 MiB — preallocated-temp: |
| `module_0159.jit_fn` | 693.56 MiB | 546.75 MiB — preallocated-temp: |
| `module_0273.jit__identity_fn` | 637.88 MiB | 273.38 MiB — preallocated-temp: |
| `module_0267.jit__solve_all_at_once` | 598.01 MiB | 227.81 MiB — preallocated-temp: |
| `module_0269.jit__solve_all_at_once` | 598.01 MiB | 227.81 MiB — preallocated-temp: |
| `module_0265.jit__reshard_z` | 546.75 MiB | 364.50 MiB — preallocated-temp: |
| `module_0267.jit__reshard_z` | 546.75 MiB | 364.50 MiB — preallocated-temp: |
| `module_0291.jit__kernel` | 430.24 MiB | 240.21 MiB — preallocated-temp: |
| `module_0293.jit__kernel` | 430.24 MiB | 240.21 MiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0243.jit_fn` | `all-gather` | 1.58 GiB | `` | `c128[64,60,2,13824]{3,2,0,1}` |
| `module_0245.jit_fn` | `all-gather` | 1.58 GiB | `` | `c128[64,60,2,13824]{3,2,0,1}` |
| `module_0342.jit_fn` | `all-gather` | 1.58 GiB | `` | `c128[64,60,2,13824]{3,2,0,1}` |
| `module_0031.jit__identity_fn` | `all-gather` | 275.62 MiB | `` | `c128[256,60,2,588]{3,2,1,0}` |
| `module_0033.jit__identity_fn` | `all-gather` | 275.62 MiB | `` | `c128[256,60,2,588]{3,2,1,0}` |
| `module_0034.jit__identity_fn` | `all-gather` | 275.62 MiB | `` | `c128[256,60,2,588]{3,2,1,0}` |
| `module_0213.jit__identity_fn` | `all-gather` | 275.62 MiB | `` | `c128[256,60,2,588]{3,2,1,0}` |
| `module_0298.jit__identity_fn` | `all-gather` | 275.62 MiB | `` | `c128[256,60,2,588]{3,2,1,0}` |
| `module_0265.jit__reshard_z` | `all-to-all` | 182.25 MiB | `` | `(c128[1,4,216,6912]{3,2,1,0}, c128[1,4,216,6912]{3,2,1,0})` |
| `module_0265.jit__reshard_z` | `all-to-all` | 182.25 MiB | `` | `(c128[4,432,1,1,3456]{4,3,2,1,0}, c128[4,432,1,1,3456]{4,3,2` |
| `module_0265.jit__reshard_z` | `collective-permute` | 182.25 MiB | `` | `c128[8,432,1,3456]{3,2,1,0}` |
| `module_0267.jit__reshard_z` | `all-to-all` | 182.25 MiB | `` | `(c128[1,4,216,6912]{3,2,1,0}, c128[1,4,216,6912]{3,2,1,0})` |
| `module_0267.jit__reshard_z` | `all-to-all` | 182.25 MiB | `` | `(c128[4,432,1,1,3456]{4,3,2,1,0}, c128[4,432,1,1,3456]{4,3,2` |
| `module_0267.jit__reshard_z` | `collective-permute` | 182.25 MiB | `` | `c128[8,432,1,3456]{3,2,1,0}` |
| `module_0269.jit__identity_fn` | `all-to-all` | 182.25 MiB | `` | `(c128[8,1,216,1,3456]{4,3,2,1,0}, c128[8,1,216,1,3456]{4,3,2` |
| `module_0269.jit__identity_fn` | `collective-permute` | 182.25 MiB | `` | `c128[8,216,1,6912]{3,2,1,0}` |
| `module_0271.jit__identity_fn` | `all-to-all` | 182.25 MiB | `` | `(c128[8,1,216,1,3456]{4,3,2,1,0}, c128[8,1,216,1,3456]{4,3,2` |
| `module_0271.jit__identity_fn` | `collective-permute` | 182.25 MiB | `` | `c128[8,216,1,6912]{3,2,1,0}` |
| `module_0273.jit__identity_fn` | `all-to-all` | 182.25 MiB | `` | `(c128[8,1,1,108,6912]{4,3,2,1,0}, c128[8,1,1,108,6912]{4,3,2` |
| `module_0366.jit__reshard_z` | `all-to-all` | 182.25 MiB | `` | `(c128[1,4,216,6912]{3,2,1,0}, c128[1,4,216,6912]{3,2,1,0})` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 80 | 364.50 MiB | 768.99 MiB |
| `jit__identity_fn` | 34 | 729.00 MiB | 3.43 GiB |
| `jit_convert_element_type` | 29 | 512.00 B | 1.84 KiB |
| `jit_less` | 18 | 6.92 KiB | 18.65 KiB |
| `jit_iota` | 18 | 2.30 KiB | 5.41 KiB |
| `jit_true_divide` | 15 | 50.63 MiB | 303.75 MiB |
| `jit_dynamic_slice` | 15 | 6.77 KiB | 19.02 KiB |
| `jit_select_n` | 15 | 5.48 KiB | 12.35 KiB |
| `jit_add` | 15 | 3.38 KiB | 7.68 KiB |
| `jit_squeeze` | 15 | 3.38 KiB | 9.31 KiB |
| `jit__broadcast_arrays` | 15 | 3.38 KiB | 7.78 KiB |
| `jit_gather` | 12 | 1.60 GiB | 3.30 GiB |
| `jit_multiply` | 12 | 25.19 KiB | 60.44 KiB |
| `jit_reshape` | 9 | 11.39 MiB | 45.56 MiB |
| `jit_concatenate` | 9 | 512.00 B | 1.38 KiB |
| `jit_fn` | 6 | 24.28 GiB | 74.20 GiB |
| `jit__kernel` | 6 | 7.66 GiB | 16.16 GiB |
| `jit__where` | 6 | 23.26 MiB | 69.30 MiB |
| `jit__moveaxis` | 6 | 11.39 MiB | 45.56 MiB |
| `jit_exp` | 6 | 48.00 KiB | 108.00 KiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `lapack_ztrsm_ffi` | 9 |
| `lapack_zpotrf_ffi` | 3 |
| `xla_ffi_python_cpu_callback` | 3 |

