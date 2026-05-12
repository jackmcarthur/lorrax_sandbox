# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile_restart/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 345 | 0.223 | 0.028 |
| jaxpr→MLIR | 200 | 0.571 | 0.080 |
| XLA compile | 223 | 7.906 | 0.364 |

## Top 10 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `gather` | 18 | 0.687 | 0.048 |
| `sigma_sx` | 2 | 0.684 | 0.345 |
| `broadcast_in_dim` | 34 | 0.581 | 0.165 |
| `convert_element_type` | 19 | 0.543 | 0.240 |
| `transpose` | 11 | 0.532 | 0.075 |
| `multiply` | 15 | 0.425 | 0.038 |
| `subtract` | 11 | 0.398 | 0.044 |
| `add` | 15 | 0.385 | 0.048 |
| `sigma_coh` | 1 | 0.364 | 0.364 |
| `minimax_tau_integrate_chi` | 1 | 0.333 | 0.333 |

## Top 10 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `minimax_tau_integrate_chi` | 1 | 0.028 | 0.028 |
| `inv` | 1 | 0.019 | 0.019 |
| `solve` | 1 | 0.018 | 0.018 |
| `broadcast_in_dim` | 51 | 0.013 | 0.001 |
| `fft_impl` | 8 | 0.011 | 0.003 |
| `_build_Gv_Gc` | 1 | 0.010 | 0.010 |
| `_einsum` | 11 | 0.008 | 0.001 |
| `_reduce_sum` | 15 | 0.007 | 0.001 |
| `multiply` | 20 | 0.007 | 0.001 |
| `_mean` | 6 | 0.007 | 0.001 |

## Tracing cache misses

Total: **161** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/gw/head_correction.py:424:10` | 12 | never seen function: convert_element_type id=139988359441568 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:43:12` | 9 | never seen function: add id=140102576828992 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:500:6` | 7 | never seen function: _make_jittable_local_fft.<locals>._make_axis_wrapper.<locals>.fft_impl id=140363763378400 defined at /global/homes/j/ja |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:375:11` | 6 | never seen function: convert_element_type id=139940108772000 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:45:10` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[8],  args[1]: i64[] closest seen input typ |
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:46:10` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[52],  args[1]: i64[] closest seen input ty |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:30:24` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[262144],  args[1]: i64[] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:195:12` | 5 | never seen function: _psum id=139944892010624 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:222:22` | 5 | never seen function: fft id=140368764486656 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:272:84` | 5 | for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566 tracing context doesn't match, e.g. due to config or context manager closest s |

## Persistent cache misses

_None._
