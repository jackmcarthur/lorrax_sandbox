# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/02_davidson_4gpu/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 514 | 1.588 | 0.319 |
| jaxpr→MLIR | 205 | 1.109 | 0.163 |
| XLA compile | 208 | 14.143 | 0.719 |

## Top 10 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_ritz_and_residuals` | 4 | 2.151 | 0.719 |
| `multiply` | 28 | 1.330 | 0.051 |
| `_apply_H_sparse` | 4 | 1.235 | 0.327 |
| `broadcast_in_dim` | 20 | 0.749 | 0.046 |
| `add` | 16 | 0.732 | 0.048 |
| `compute_V_H_and_V_xc` | 1 | 0.681 | 0.681 |
| `concatenate` | 12 | 0.565 | 0.050 |
| `species_structure_factors` | 1 | 0.510 | 0.510 |
| `transpose` | 6 | 0.456 | 0.086 |
| `_reduce_sum` | 8 | 0.448 | 0.074 |

## Top 10 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `compute_V_H_and_V_xc` | 1 | 0.319 | 0.319 |
| `_ritz_and_residuals` | 4 | 0.227 | 0.059 |
| `_apply_H_sparse` | 4 | 0.133 | 0.035 |
| `inv` | 4 | 0.126 | 0.033 |
| `solve` | 4 | 0.118 | 0.031 |
| `apply_H_k` | 4 | 0.100 | 0.025 |
| `multiply` | 77 | 0.088 | 0.039 |
| `_table_interp` | 3 | 0.062 | 0.022 |
| `accumulate_species_on_G` | 1 | 0.038 | 0.038 |
| `precond_fn` | 3 | 0.038 | 0.017 |

## Tracing cache misses

Total: **163** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:41:12` | 16 | never seen function: _lu_solve id=139945314692736 defined at /opt/jax/jax/_src/lax/linalg.py:1566 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:184:12` | 8 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[12,2,2120],  args[1]: c128[12,2,2 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:324:10` | 7 | never seen function: add id=139941176912576 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/solid_harmonics.py:72:8` | 6 | never seen function: dynamic_slice id=139941179890144 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/ionic_gspace.py:225:21` | 5 | never seen function: convert_element_type id=139941715018752 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:89:12` | 5 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[] closest seen input typ |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:40:8` | 4 | never seen function: cholesky id=139945319575488 defined at /opt/jax/jax/_src/numpy/linalg.py:74 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:44:21` | 4 | never seen function: eigh id=139945319719104 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:91:8` | 4 | never seen function: _ritz_and_residuals id=139941716288480 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davids |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:264:9` | 4 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[2109] closest seen input  |

## Persistent cache misses

_None._
