# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_low_mem_test/profile/xla_dump`
**Modules dumped:** 30
**Sum of per-module peak live HBM:** 998.54 MiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0039.jit__fft_gather_reshard` | 773.48 MiB | 506.29 MiB — preallocated-temp: |
| `module_0073.jit__compute_P_traced` | 28.12 MiB | 14.06 MiB — output shape is \|c128[9,320,320]\|, maybe-live-out: |
| `module_0009.jit__multi_slice` | 21.09 MiB | 14.06 MiB — parameter 0, shape \|c128[9,80,2,640]\| at ShapeIndex {}: |
| `module_0013.jit__multi_slice` | 21.09 MiB | 14.06 MiB — parameter 0, shape \|c128[9,640,80,2]\| at ShapeIndex {}: |
| `module_0057.jit_scatter` | 21.09 MiB | 7.03 MiB — output shape is \|c128[9,80,2,320]\|, maybe-live-out: |
| `module_0061.jit_scatter` | 21.09 MiB | 7.03 MiB — output shape is \|c128[9,320,80,2]\|, maybe-live-out: |
| `module_0067.jit_true_divide` | 14.06 MiB | 7.03 MiB — output shape is \|c128[9,80,2,320]\|, maybe-live-out: |
| `module_0071.jit_true_divide` | 14.06 MiB | 7.03 MiB — output shape is \|c128[9,320,80,2]\|, maybe-live-out: |
| `module_0003.jit_broadcast_in_dim` | 14.06 MiB | 0.00 B —  |
| `module_0005.jit_broadcast_in_dim` | 14.06 MiB | 0.00 B —  |
| `module_0007.jit__identity_fn` | 14.06 MiB | 7.03 MiB — output shape is \|c128[9,80,2,320]\|, maybe-live-out: |
| `module_0011.jit__identity_fn` | 14.06 MiB | 7.03 MiB — output shape is \|c128[9,320,80,2]\|, maybe-live-out: |
| `module_0055.jit__squeeze` | 14.06 MiB | 7.03 MiB — output shape is \|c128[9,80,2,320]\|, maybe-live-out: |
| `module_0059.jit__squeeze` | 14.06 MiB | 7.03 MiB — output shape is \|c128[9,320,80,2]\|, maybe-live-out: |
| `module_0031.jit_dynamic_slice` | 20.02 KiB | 15.00 KiB — parameter 0, shape \|s64[640,3]\| at ShapeIndex {}: |
| `module_0037.jit_add` | 15.00 KiB | 5.00 KiB — output shape is \|s64[640]\|, maybe-live-out: |
| `module_0035.jit_multiply` | 10.01 KiB | 5.00 KiB — output shape is \|s64[640]\|, maybe-live-out: |
| `module_0033.jit_squeeze` | 10.00 KiB | 5.00 KiB — output shape is \|s64[640]\|, maybe-live-out: |
| `module_0029.jit_true_divide` | 1.26 KiB | 0.00 B —  |
| `module_0063.jit_maximum` | 1.26 KiB | 0.00 B —  |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0039.jit__fft_gather_reshard` | `all-gather` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:717` | `c128[9,40,2,640]{3,2,0,1}` |
| `module_0039.jit__fft_gather_reshard` | `all-to-all` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:724` | `(c128[9,40,2,1,320]{4,3,2,1,0}, c128[9,40,2,1,320]{4,3,2,1,0` |
| `module_0039.jit__fft_gather_reshard` | `collective-permute` | 7.03 MiB | `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:727` | `c128[9,320,80,2]{3,2,1,0}` |
| `module_0039.jit__fft_gather_reshard` | `collective-permute` | 3.52 MiB | `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:717` | `c128[9,20,2,640]{3,2,0,1}` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 7 | 14.06 MiB | 28.13 MiB |
| `jit_true_divide` | 5 | 14.06 MiB | 28.13 MiB |
| `jit__multi_slice` | 2 | 21.09 MiB | 42.19 MiB |
| `jit_scatter` | 2 | 21.09 MiB | 42.19 MiB |
| `jit__identity_fn` | 2 | 14.06 MiB | 28.12 MiB |
| `jit__squeeze` | 2 | 14.06 MiB | 28.12 MiB |
| `jit_iota` | 2 | 640.00 B | 832.00 B |
| `jit__fft_gather_reshard` | 1 | 773.48 MiB | 773.48 MiB |
| `jit__compute_P_traced` | 1 | 28.12 MiB | 28.12 MiB |
| `jit_dynamic_slice` | 1 | 20.02 KiB | 20.02 KiB |
| `jit_add` | 1 | 15.00 KiB | 15.00 KiB |
| `jit_multiply` | 1 | 10.01 KiB | 10.01 KiB |
| `jit_squeeze` | 1 | 10.00 KiB | 10.00 KiB |
| `jit_maximum` | 1 | 1.26 KiB | 1.26 KiB |
| `jit_convert_element_type` | 1 | 32.00 B | 32.00 B |

