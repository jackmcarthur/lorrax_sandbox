# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/02_davidson_4gpu/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 517 | 0.887 | 0.174 |
| jaxpr→MLIR | 203 | 0.858 | 0.132 |
| XLA compile | 209 | 14.214 | 0.713 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_ritz_and_residuals` | 4 | 2.138 | 0.713 |
| `multiply` | 28 | 1.324 | 0.055 |
| `_apply_H_sparse` | 4 | 1.229 | 0.331 |
| `broadcast_in_dim` | 20 | 0.756 | 0.050 |
| `add` | 16 | 0.737 | 0.049 |
| `compute_V_H_and_V_xc` | 1 | 0.681 | 0.681 |
| `concatenate` | 12 | 0.567 | 0.051 |
| `species_structure_factors` | 1 | 0.512 | 0.512 |
| `_reduce_sum` | 8 | 0.448 | 0.072 |
| `transpose` | 6 | 0.448 | 0.087 |
| `precond_fn` | 3 | 0.445 | 0.152 |
| `convert_element_type` | 16 | 0.409 | 0.044 |
| `exp` | 4 | 0.312 | 0.088 |
| `sqrt` | 5 | 0.307 | 0.110 |
| `dynamic_slice` | 6 | 0.299 | 0.052 |
| `eigh` | 2 | 0.288 | 0.146 |
| `conjugate` | 6 | 0.288 | 0.049 |
| `_pad` | 6 | 0.286 | 0.050 |
| `_table_interp` | 3 | 0.282 | 0.097 |
| `integer_pow` | 5 | 0.223 | 0.046 |
| `true_divide` | 4 | 0.212 | 0.061 |
| `scatter` | 3 | 0.193 | 0.069 |
| `gather` | 4 | 0.193 | 0.050 |
| `real` | 4 | 0.186 | 0.048 |
| `matmul` | 7 | 0.177 | 0.028 |
| `_identity_fn` | 4 | 0.133 | 0.038 |
| `_power` | 1 | 0.122 | 0.122 |
| `_psum` | 2 | 0.107 | 0.056 |
| `accumulate_species_on_G` | 1 | 0.106 | 0.106 |
| `subtract` | 2 | 0.092 | 0.048 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `compute_V_H_and_V_xc` | 1 | 0.174 | 0.174 |
| `_ritz_and_residuals` | 4 | 0.121 | 0.032 |
| `_apply_H_sparse` | 4 | 0.074 | 0.019 |
| `inv` | 4 | 0.064 | 0.017 |
| `solve` | 4 | 0.060 | 0.016 |
| `apply_H_k` | 4 | 0.056 | 0.014 |
| `_table_interp` | 3 | 0.035 | 0.013 |
| `multiply` | 77 | 0.029 | 0.001 |
| `sqrt` | 7 | 0.025 | 0.023 |
| `accumulate_species_on_G` | 1 | 0.024 | 0.024 |
| `precond_fn` | 3 | 0.022 | 0.010 |
| `interp_uniform_jax` | 1 | 0.021 | 0.021 |
| `_lu_solve` | 8 | 0.017 | 0.003 |
| `add` | 46 | 0.014 | 0.000 |
| `clip` | 8 | 0.013 | 0.002 |
| `_einsum` | 22 | 0.012 | 0.001 |
| `true_divide` | 20 | 0.009 | 0.001 |
| `species_structure_factors` | 1 | 0.009 | 0.009 |
| `cholesky` | 4 | 0.008 | 0.002 |
| `subtract` | 17 | 0.007 | 0.001 |
| `_where` | 9 | 0.007 | 0.001 |
| `matmul` | 12 | 0.006 | 0.001 |
| `_reduce_sum` | 11 | 0.005 | 0.001 |
| `less` | 14 | 0.005 | 0.001 |
| `broadcast_in_dim` | 20 | 0.005 | 0.000 |
| `_broadcast_arrays` | 19 | 0.004 | 0.001 |
| `maximum` | 11 | 0.004 | 0.001 |
| `conjugate` | 11 | 0.004 | 0.001 |
| `_pad` | 6 | 0.003 | 0.001 |
| `convert_element_type` | 16 | 0.003 | 0.000 |

## Tracing cache misses

Total: **163** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:41:12` | 16 | never seen function: _lu_solve id=140527818421600 defined at /opt/jax/jax/_src/lax/linalg.py:1566 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:184:12` | 8 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[12,2,2120],  args[1]: c128[12,2,2 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:324:10` | 7 | never seen function: add id=140523680644160 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/solid_harmonics.py:72:8` | 6 | never seen function: dynamic_slice id=140523684228256 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/ionic_gspace.py:225:21` | 5 | never seen function: convert_element_type id=140523685697216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:89:12` | 5 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[] closest seen input typ |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:40:8` | 4 | never seen function: cholesky id=140527823353504 defined at /opt/jax/jax/_src/numpy/linalg.py:74 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:44:21` | 4 | never seen function: eigh id=140527823497120 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:91:8` | 4 | never seen function: _ritz_and_residuals id=140524220182464 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davids |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:264:9` | 4 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[2109] closest seen input  |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:553:11` | 4 | never seen function: convert_element_type id=140523681216320 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:617:12` | 4 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[12,2,36,36,36] closest seen input type signature |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:618:16` | 4 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/h_dft.py:26:11` | 4 | never seen function: apply_H_k id=140524220171904 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:587 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/h_dft.py:39:15` | 4 | never seen function: _apply_H_sparse id=140524220179264 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/h_dft.py:20 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:254:10` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: i32[2109],  min: i64[],  max: i64[] clos |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:256:8` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[2109],  min: f64[],  max: f64[] clos |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:317:12` | 3 | never seen function: _table_interp id=140524220243840 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:242 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/solid_harmonics.py:79:8` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/solid_harmonics.py:82:8` | 3 | never seen function: concatenate id=140523680633120 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:325:10` | 3 | never seen function: concatenate id=140523680983904 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/solvers/davidson.py:156:12` | 3 | never seen function: make_dft_preconditioner.<locals>.precond_fn id=140631056493728 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/l |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/radial_jax.py:155:11` | 2 | never seen function: _where id=140527825416128 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:311:13` | 2 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[2109,3] closest seen inpu |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:314:25` | 2 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[2109,3] closest seen input type si |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/solid_harmonics.py:82:23` | 2 | never seen function: broadcast_in_dim id=140523680631680 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/radial/solid_harmonics.py:93:43` | 2 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[2109] closest seen input type sign |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/vnl_ops.py:326:14` | 2 | for transpose defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[2109,68] closest seen input type si |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:448:15` | 2 | never seen function: dynamic_slice id=140523681210720 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/psp/dft_operators.py:558:17` | 2 | never seen function: _pad id=140527823067296 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 |

## Persistent cache misses

_None._
