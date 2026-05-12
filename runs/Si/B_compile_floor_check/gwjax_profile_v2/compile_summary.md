# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile_v2/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 632 | 0.600 | 0.055 |
| jaxpr→MLIR | 238 | 0.800 | 0.128 |
| XLA compile | 248 | 8.293 | 0.694 |

## Top 5 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `sigma_sx` | 3 | 1.033 | 0.361 |
| `sigma_coh` | 2 | 0.726 | 0.367 |
| `_kernel` | 1 | 0.694 | 0.694 |
| `broadcast_in_dim` | 37 | 0.547 | 0.187 |
| `true_divide` | 12 | 0.475 | 0.090 |

## Top 5 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 1 | 0.055 | 0.055 |
| `_fft_and_rslice` | 2 | 0.040 | 0.020 |
| `_solve_w` | 1 | 0.034 | 0.034 |
| `_compute_S_omega_jit` | 1 | 0.026 | 0.026 |
| `get_sqrt_v_and_phase` | 1 | 0.025 | 0.025 |

## Tracing cache misses

Total: **245** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/gw/wavefunction_bundle.py:150:15` | 9 | never seen function: gather id=140618976665632 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/wavefunction_bundle.py:144:15` | 8 | never seen function: gather id=140618976660032 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:214:31` | 7 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[4,4,480,480,4] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1805:15` | 6 | never seen function: broadcast_in_dim id=140620452958016 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1805:45` | 6 | never seen function: broadcast_in_dim id=140620452962656 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
