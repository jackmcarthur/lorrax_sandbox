# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/00_davidson_only/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 510 | 0.940 | 0.209 |
| jaxpr→MLIR | 215 | 0.862 | 0.146 |
| XLA compile | 215 | 14.665 | 0.774 |

## Top 15 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_ritz_and_residuals` | 4 | 2.080 | 0.774 |
| `multiply` | 31 | 1.454 | 0.058 |
| `_apply_H_sparse` | 4 | 1.167 | 0.304 |
| `precond_fn` | 8 | 1.135 | 0.145 |
| `broadcast_in_dim` | 23 | 0.870 | 0.048 |
| `add` | 16 | 0.740 | 0.049 |
| `compute_V_H_and_V_xc` | 1 | 0.670 | 0.670 |
| `species_structure_factors` | 1 | 0.565 | 0.565 |
| `_reduce_sum` | 8 | 0.439 | 0.072 |
| `convert_element_type` | 15 | 0.383 | 0.043 |
| `concatenate` | 8 | 0.371 | 0.048 |
| `transpose` | 5 | 0.368 | 0.085 |
| `integer_pow` | 7 | 0.317 | 0.047 |
| `exp` | 4 | 0.314 | 0.089 |
| `sqrt` | 5 | 0.310 | 0.114 |

## Top 15 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `compute_V_H_and_V_xc` | 1 | 0.209 | 0.209 |
| `_ritz_and_residuals` | 4 | 0.117 | 0.031 |
| `_apply_H_sparse` | 4 | 0.072 | 0.019 |
| `inv` | 4 | 0.063 | 0.016 |
| `solve` | 4 | 0.059 | 0.015 |
| `apply_H_k` | 4 | 0.053 | 0.014 |
| `broadcast_in_dim` | 23 | 0.045 | 0.040 |
| `_table_interp` | 3 | 0.033 | 0.011 |
| `true_divide` | 18 | 0.032 | 0.025 |
| `multiply` | 80 | 0.029 | 0.001 |
| `precond_fn` | 8 | 0.025 | 0.009 |
| `accumulate_species_on_G` | 1 | 0.023 | 0.023 |
| `interp_uniform_jax` | 1 | 0.020 | 0.020 |
| `_lu_solve` | 8 | 0.017 | 0.003 |
| `add` | 46 | 0.014 | 0.001 |

## Tracing cache misses

Total: **163** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:41:12` | 16 | never seen function: _lu_solve id=140046684468576 defined at /opt/jax/jax/_src/lax/linalg.py:1566 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:324:10` | 9 | never seen function: add id=140043585879104 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:156:12` | 8 | never seen function: make_dft_preconditioner.<locals>.precond_fn id=140043585498112 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/l |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/solid_harmonics.py:72:8` | 6 | never seen function: dynamic_slice id=140043589526496 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/ionic_gspace.py:225:21` | 5 | never seen function: convert_element_type id=140043590899392 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:89:12` | 5 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[] closest seen input typ |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:40:8` | 4 | never seen function: cholesky id=140046689384096 defined at /opt/jax/jax/_src/numpy/linalg.py:74 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:44:21` | 4 | never seen function: eigh id=140046689527712 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:91:8` | 4 | never seen function: _ritz_and_residuals id=140044193099712 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davids |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/solid_harmonics.py:79:8` | 4 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/solid_harmonics.py:82:8` | 4 | never seen function: concatenate id=140043585868064 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:325:10` | 4 | never seen function: concatenate id=140043586251616 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:553:11` | 4 | never seen function: convert_element_type id=140043586451264 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:617:12` | 4 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[12,2,36,36,36] closest seen input type signature |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:618:16` | 4 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |

## Persistent cache misses

_None._
