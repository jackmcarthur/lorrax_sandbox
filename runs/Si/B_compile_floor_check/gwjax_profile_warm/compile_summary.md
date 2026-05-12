# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile_warm/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 692 | 0.527 | 0.054 |
| jaxpr→MLIR | 333 | 0.982 | 0.154 |
| XLA compile | 369 | 10.157 | 0.694 |

## Top 5 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 75 | 0.931 | 0.149 |
| `_kernel` | 1 | 0.694 | 0.694 |
| `convert_element_type` | 37 | 0.691 | 0.281 |
| `sigma_sx` | 2 | 0.666 | 0.336 |
| `gather` | 26 | 0.627 | 0.039 |

## Top 5 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 1 | 0.054 | 0.054 |
| `_fft_and_rslice` | 2 | 0.042 | 0.021 |
| `_solve_w` | 1 | 0.034 | 0.034 |
| `get_sqrt_v_and_phase` | 1 | 0.026 | 0.026 |
| `broadcast_in_dim` | 88 | 0.022 | 0.002 |

## Tracing cache misses

Total: **321** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:43:12` | 17 | never seen function: add id=140358659771744 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/head_correction.py:424:10` | 15 | never seen function: convert_element_type id=140356380450528 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:24:20` | 9 | never seen function: broadcast_in_dim id=140360274556256 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:24:10` | 8 | never seen function: broadcast_in_dim id=140360274556576 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:30:24` | 8 | never seen function: add id=140112169585056 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
