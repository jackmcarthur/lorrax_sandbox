# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile_v3/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 680 | 0.671 | 0.055 |
| jaxpr→MLIR | 234 | 0.808 | 0.123 |
| XLA compile | 237 | 7.768 | 0.698 |

## Top 5 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `sigma_sx` | 3 | 1.034 | 0.361 |
| `sigma_coh` | 2 | 0.723 | 0.364 |
| `_kernel` | 1 | 0.698 | 0.698 |
| `broadcast_in_dim` | 31 | 0.484 | 0.166 |
| `_compute_CCT_LR` | 2 | 0.426 | 0.214 |

## Top 5 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 1 | 0.055 | 0.055 |
| `_fft_and_rslice` | 2 | 0.042 | 0.021 |
| `_solve_w` | 1 | 0.034 | 0.034 |
| `minimax_tau_integrate_chi` | 1 | 0.028 | 0.028 |
| `_compute_S_omega_jit` | 1 | 0.026 | 0.026 |

## Tracing cache misses

Total: **237** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:222:22` | 7 | never seen function: fft id=140167772357632 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1805:15` | 7 | never seen function: broadcast_in_dim id=139693544301216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:214:31` | 6 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[4,4,480,480,4] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1805:45` | 6 | never seen function: broadcast_in_dim id=139693544305856 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:385:18` | 5 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=139693547937184 defined at /global/homes/j/jackm/software/lorrax_B/src/fil |

## Persistent cache misses

_None._
