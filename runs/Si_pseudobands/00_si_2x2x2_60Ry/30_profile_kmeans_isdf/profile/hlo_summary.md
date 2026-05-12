# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/30_profile_kmeans_isdf/profile/xla_dump`
**Modules dumped:** 63
**Sum of per-module peak live HBM:** 1022.08 MiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0123.jit_pbc_distance_sq_batch` | 498.88 MiB | 427.15 MiB — preallocated-temp: |
| `module_0111.jit_kmeans_update_step` | 432.76 MiB | 431.68 MiB — preallocated-temp: |
| `module_0125.jit__argmin` | 71.73 MiB | 71.19 MiB — parameter 0, shape \|f32[46656,400]\| at ShapeIndex {}: |
| `module_0103.jit_pbc_distance_sq_single` | 2.31 MiB | 1.60 MiB — preallocated-temp: |
| `module_0035.jit_scatter` | 1.48 MiB | 729.00 KiB — output shape is \|c128[36,36,36]\|, maybe-live-out: |
| `module_0031.jit_convert_element_type` | 1.07 MiB | 729.00 KiB — output shape is \|c128[36,36,36]\|, maybe-live-out: |
| `module_0037.jit_convert_element_type` | 1.07 MiB | 729.00 KiB — parameter 0, shape \|c128[36,36,36]\| at ShapeIndex {}: |
| `module_0043.jit_multiply` | 1.07 MiB | 364.50 KiB — output shape is \|c64[36,36,36]\|, maybe-live-out: |
| `module_0071.jit_concatenate` | 1.07 MiB | 546.80 KiB — output shape is \|f32[36,36,36,3]\|, maybe-live-out: |
| `module_0073.jit_reshape` | 1.07 MiB | 546.80 KiB — output shape is \|f32[46656,3]\|, maybe-live-out: |
| `module_0047.jit_add` | 911.25 KiB | 364.50 KiB — output shape is \|f64[36,36,36]\|, maybe-live-out: |
| `module_0049.jit_multiply` | 729.01 KiB | 364.50 KiB — output shape is \|f64[36,36,36]\|, maybe-live-out: |
| `module_0039.jit_fft` | 729.00 KiB | 364.50 KiB — output shape is \|c64[36,36,36]\|, maybe-live-out: |
| `module_0041.jit_conjugate` | 729.00 KiB | 364.50 KiB — output shape is \|c64[36,36,36]\|, maybe-live-out: |
| `module_0119.jit_scatter` | 628.20 KiB | 312.50 KiB — output shape is \|f32[400,200]\|, maybe-live-out: |
| `module_0085.jit_dynamic_slice` | 546.78 KiB | 0.00 B —  |
| `module_0045.jit_real` | 546.75 KiB | 364.50 KiB — parameter 0, shape \|c64[36,36,36]\| at ShapeIndex {}: |
| `module_0105.jit_minimum` | 546.75 KiB | 182.20 KiB — output shape is \|f32[46656]\|, maybe-live-out: |
| `module_0107.jit_multiply` | 546.75 KiB | 182.20 KiB — output shape is \|f32[46656]\|, maybe-live-out: |
| `module_0005.jit_broadcast_in_dim` | 364.51 KiB | 0.00 B —  |

## Sharding — collectives (largest by output bytes)

_No collective ops found (single-device or pure-SPMD-free)._

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 11 | 364.51 KiB | 2.11 MiB |
| `jit_convert_element_type` | 9 | 1.07 MiB | 2.49 MiB |
| `jit_dynamic_slice` | 5 | 546.78 KiB | 689.60 KiB |
| `jit_squeeze` | 4 | 65.91 KiB | 85.53 KiB |
| `jit_scatter` | 3 | 1.48 MiB | 2.10 MiB |
| `jit_multiply` | 3 | 1.07 MiB | 2.31 MiB |
| `jit__squeeze` | 3 | 65.91 KiB | 69.05 KiB |
| `jit_concatenate` | 2 | 1.07 MiB | 1.12 MiB |
| `jit_reshape` | 2 | 1.07 MiB | 1.42 MiB |
| `jit_add` | 2 | 911.25 KiB | 927.73 KiB |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 4 |

