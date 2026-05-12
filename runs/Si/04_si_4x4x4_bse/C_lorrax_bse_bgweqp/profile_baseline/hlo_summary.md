# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_baseline/xla_dump`
**Modules dumped:** 72
**Sum of per-module peak live HBM:** 4.12 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0093.jit_scan` | 2.02 GiB | 1.78 GiB — preallocated-temp: |
| `module_0019.jit_scatter-add` | 453.52 MiB | 225.00 MiB — output shape is \|c128[1,1,1,4,4,4,480,480]\|, maybe-live-out: |
| `module_0053.jit_dynamic_slice` | 450.00 MiB | 225.00 MiB — output shape is \|c128[1,1,1,4,4,4,480,480]\|, maybe-live-out: |
| `module_0055.jit_squeeze` | 450.00 MiB | 225.00 MiB — output shape is \|c128[4,4,4,480,480]\|, maybe-live-out: |
| `module_0057.jit_transpose` | 450.00 MiB | 225.00 MiB — output shape is \|c128[480,480,4,4,4]\|, maybe-live-out: |
| `module_0049.jit_dynamic_slice` | 228.52 MiB | 225.00 MiB — parameter 0, shape \|c128[1,1,1,4,4,4,480,480]\| at ShapeIndex {}: |
| `module_0041.jit_gather` | 60.00 MiB | 56.25 MiB — parameter 0, shape \|c128[64,60,2,480]\| at ShapeIndex {}: |
| `module_0043.jit_broadcast_in_dim` | 7.50 MiB | 3.75 MiB — output shape is \|c128[64,4,2,480]\|, maybe-live-out: |
| `module_0007.jit_multiply` | 7.03 MiB | 3.52 MiB — output shape is \|c128[480,480]\|, maybe-live-out: |
| `module_0015.jit_broadcast_in_dim` | 7.03 MiB | 3.52 MiB — output shape is \|c128[1,1,1,480,480]\|, maybe-live-out: |
| `module_0017.jit__squeeze` | 7.03 MiB | 3.52 MiB — output shape is \|c128[1,1,1,480,480]\|, maybe-live-out: |
| `module_0051.jit_squeeze` | 7.03 MiB | 3.52 MiB — output shape is \|c128[480,480]\|, maybe-live-out: |
| `module_0149.jit_matmul` | 6.72 MiB | 3.25 MiB — preallocated-temp: |
| `module_0087.jit_scatter` | 6.28 MiB | 3.12 MiB — output shape is \|c128[1024,200]\|, maybe-live-out: |
| `module_0003.jit__einsum` | 3.53 MiB | 3.52 MiB — output shape is \|c128[480,480]\|, maybe-live-out: |
| `module_0083.jit_broadcast_in_dim` | 3.13 MiB | 0.00 B —  |
| `module_0123.jit_eigh` | 939.21 KiB | 312.60 KiB — preallocated-temp: |
| `module_0119.jit_add` | 937.50 KiB | 312.50 KiB — output shape is \|f64[200,200]\|, maybe-live-out: |
| `module_0157.jit_true_divide` | 640.16 KiB | 320.00 KiB — output shape is \|c128[20,1024]\|, maybe-live-out: |
| `module_0151.jit_transpose` | 640.00 KiB | 320.00 KiB — output shape is \|c128[20,1024]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

_No collective ops found (single-device or pure-SPMD-free)._

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 10 | 7.50 MiB | 17.72 MiB |
| `jit_convert_element_type` | 7 | 240.00 B | 424.00 B |
| `jit_dynamic_slice` | 5 | 450.00 MiB | 678.52 MiB |
| `jit_multiply` | 5 | 7.03 MiB | 7.08 MiB |
| `jit_add` | 5 | 937.50 KiB | 1017.89 KiB |
| `jit_gather` | 4 | 60.00 MiB | 60.37 MiB |
| `jit_squeeze` | 3 | 450.00 MiB | 457.03 MiB |
| `jit__diag` | 3 | 314.06 KiB | 942.17 KiB |
| `jit_transpose` | 2 | 450.00 MiB | 450.62 MiB |
| `jit__squeeze` | 2 | 7.03 MiB | 7.06 MiB |
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

