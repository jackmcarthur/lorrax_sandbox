# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/profile_bse2/xla_dump`
**Modules dumped:** 64
**Sum of per-module peak live HBM:** 4.11 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0077.jit_scan` | 2.00 GiB | 1.76 GiB — preallocated-temp: |
| `module_0027.jit_dynamic_slice` | 450.00 MiB | 225.00 MiB — output shape is \|c128[1,1,1,4,4,4,480,480]\|, maybe-live-out: |
| `module_0029.jit_squeeze` | 450.00 MiB | 225.00 MiB — output shape is \|c128[4,4,4,480,480]\|, maybe-live-out: |
| `module_0031.jit_transpose` | 450.00 MiB | 225.00 MiB — output shape is \|c128[480,480,4,4,4]\|, maybe-live-out: |
| `module_0043.jit__lambda_` | 450.00 MiB | 225.00 MiB — output shape is \|c128[480,480,4,4,4]\|, maybe-live-out: |
| `module_0023.jit_dynamic_slice` | 228.52 MiB | 225.00 MiB — parameter 0, shape \|c128[1,1,1,4,4,4,480,480]\| at ShapeIndex {}: |
| `module_0015.jit_gather` | 60.00 MiB | 56.25 MiB — parameter 0, shape \|c128[64,60,2,480]\| at ShapeIndex {}: |
| `module_0045.jit_compute_pair_amplitude` | 30.00 MiB | 15.00 MiB — preallocated-temp: |
| `module_0017.jit_broadcast_in_dim` | 7.50 MiB | 3.75 MiB — output shape is \|c128[64,4,2,480]\|, maybe-live-out: |
| `module_0025.jit_squeeze` | 7.03 MiB | 3.52 MiB — output shape is \|c128[480,480]\|, maybe-live-out: |
| `module_0133.jit_matmul` | 6.72 MiB | 3.25 MiB — preallocated-temp: |
| `module_0071.jit_scatter` | 6.28 MiB | 3.12 MiB — output shape is \|c128[1024,200]\|, maybe-live-out: |
| `module_0063.jit_broadcast_in_dim` | 3.13 MiB | 0.00 B —  |
| `module_0107.jit_eigh` | 939.21 KiB | 312.60 KiB — preallocated-temp: |
| `module_0103.jit_add` | 937.50 KiB | 312.50 KiB — output shape is \|f64[200,200]\|, maybe-live-out: |
| `module_0141.jit_true_divide` | 640.16 KiB | 320.00 KiB — output shape is \|c128[20,1024]\|, maybe-live-out: |
| `module_0135.jit_transpose` | 640.00 KiB | 320.00 KiB — output shape is \|c128[20,1024]\|, maybe-live-out: |
| `module_0143.jit_reshape` | 640.00 KiB | 320.00 KiB — output shape is \|c128[20,4,4,64]\|, maybe-live-out: |
| `module_0129.jit_gather` | 343.83 KiB | 312.50 KiB — parameter 0, shape \|f64[200,200]\| at ShapeIndex {}: |
| `module_0137.jit_norm` | 320.18 KiB | 0.00 B —  |

## Sharding — collectives (largest by output bytes)

_No collective ops found (single-device or pure-SPMD-free)._

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 9 | 7.50 MiB | 10.69 MiB |
| `jit_convert_element_type` | 7 | 240.00 B | 424.00 B |
| `jit_add` | 5 | 937.50 KiB | 1017.89 KiB |
| `jit_dynamic_slice` | 4 | 450.00 MiB | 678.52 MiB |
| `jit_gather` | 4 | 60.00 MiB | 60.37 MiB |
| `jit__diag` | 3 | 314.06 KiB | 942.17 KiB |
| `jit_multiply` | 3 | 24.02 KiB | 48.35 KiB |
| `jit_squeeze` | 2 | 450.00 MiB | 457.03 MiB |
| `jit_transpose` | 2 | 450.00 MiB | 450.62 MiB |
| `jit_true_divide` | 2 | 640.16 KiB | 672.16 KiB |
| `jit_norm` | 2 | 320.18 KiB | 336.21 KiB |
| `jit__normal` | 2 | 8.01 KiB | 16.02 KiB |
| `jit_select_n` | 2 | 500.00 B | 600.00 B |
| `jit__broadcast_arrays` | 2 | 320.00 B | 384.00 B |
| `jit_less` | 2 | 188.00 B | 232.00 B |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 7 |
| `cusolver_syevd_ffi` | 1 |

