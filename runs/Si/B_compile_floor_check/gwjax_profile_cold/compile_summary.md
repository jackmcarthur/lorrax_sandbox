# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile_cold/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 725 | 0.876 | 0.120 |
| jaxpr→MLIR | 213 | 0.854 | 0.130 |
| XLA compile | 217 | 11.871 | 0.690 |

## Top 5 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `sigma_sx` | 4 | 1.416 | 0.376 |
| `_kernel` | 2 | 0.920 | 0.690 |
| `sigma_coh` | 2 | 0.716 | 0.360 |
| `broadcast_in_dim` | 28 | 0.605 | 0.194 |
| `true_divide` | 9 | 0.493 | 0.103 |

## Top 5 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_batched_chol` | 1 | 0.120 | 0.120 |
| `_kernel` | 2 | 0.069 | 0.055 |
| `fft_impl` | 43 | 0.064 | 0.004 |
| `_solve_w` | 1 | 0.034 | 0.034 |
| `minimax_tau_integrate_chi` | 1 | 0.027 | 0.027 |

## Tracing cache misses

Total: **263** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/gw/cohsex_sigma.py:199:18` | 11 | never seen function: _make_cohsex_kernels.<locals>.sigma_sx id=140567370227584 defined at /global/homes/j/jackm/software/lorrax_B/src/gw/coh |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:214:31` | 10 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[4,4,480,480,4] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:808:11` | 10 | never seen function: _make_fit_one_rchunk_kernel.<locals>._kernel id=140570591367936 defined at /global/homes/j/jackm/software/lorrax_B/src/ |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:500:6` | 9 | never seen function: _make_jittable_local_fft.<locals>._make_axis_wrapper.<locals>.fft_impl id=140569116314656 defined at /global/homes/j/ja |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:222:22` | 8 | never seen function: fft id=139726367066528 defined at /opt/jax/jax/_src/lax/fft.py:68 |

## Persistent cache misses

_None._
