# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke/xla_dump`
**Modules dumped:** 3
**Sum of per-module peak live HBM:** 2.82 KiB (upper bound; peaks happen at different times)

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0005.jit_kernel` | 2.31 KiB | 1.30 KiB — preallocated-temp: |
| `module_0003.jit_broadcast_in_dim` | 516.00 B | 0.00 B —  |
| `module_0001.jit_convert_element_type` | 8.00 B | 0.00 B —  |

## Compute — aggregate op counts

| Op | Count |
|---|---:|
| `fusion` | 2 |
| `copy` | 1 |
| `reduce` | 1 |

### Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 1 |

## Sharding — collectives (largest by output bytes)

_No collective ops found (single-device or pure-SPMD-free)._

## Rematerialization warnings

_None._ 🎉

## Per-module file index

_(for deeper inspection — each module writes up to 10 files; the memory-usage-report is the most agent-readable)_

| Module | Files |
|---|---|
| `module_0005.jit_kernel` | module_0005.jit_kernel.autotune_results.pbtxt, module_0005.jit_kernel.before_optimizations.txt, module_0005.jit_kernel.gpu_target_config.pbtxt, … (+7) |
| `module_0003.jit_broadcast_in_dim` | module_0003.jit_broadcast_in_dim.autotune_results.pbtxt, module_0003.jit_broadcast_in_dim.before_optimizations.txt, module_0003.jit_broadcast_in_dim.gpu_target_config.pbtxt, … (+7) |
| `module_0001.jit_convert_element_type` | module_0001.jit_convert_element_type.autotune_results.pbtxt, module_0001.jit_convert_element_type.before_optimizations.txt, module_0001.jit_convert_element_type.gpu_target_config.pbtxt, … (+7) |
