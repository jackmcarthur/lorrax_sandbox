# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_davidson/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 250 | 0.799 | 0.078 |
| jaxpr→MLIR | 41 | 0.361 | 0.082 |
| XLA compile | 62 | 6.278 | 0.682 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_ritz_and_residuals` | 4 | 2.263 | 0.682 |
| `_ortho_expand` | 4 | 0.597 | 0.154 |
| `multiply` | 8 | 0.468 | 0.064 |
| `matvec_scan` | 1 | 0.424 | 0.424 |
| `concatenate` | 7 | 0.406 | 0.061 |
| `_multi_slice` | 5 | 0.401 | 0.152 |
| `broadcast_in_dim` | 8 | 0.344 | 0.060 |
| `_identity_fn` | 4 | 0.263 | 0.078 |
| `_psum` | 5 | 0.183 | 0.041 |
| `_impl` | 1 | 0.162 | 0.162 |
| `_orthonormalise_batch` | 1 | 0.134 | 0.134 |
| `conjugate` | 2 | 0.121 | 0.061 |
| `add` | 2 | 0.120 | 0.063 |
| `convert_element_type` | 4 | 0.113 | 0.029 |
| `scatter-add` | 1 | 0.107 | 0.107 |
| `sqrt` | 1 | 0.058 | 0.058 |
| `dynamic_slice` | 2 | 0.056 | 0.029 |
| `_squeeze` | 1 | 0.030 | 0.030 |
| `squeeze` | 1 | 0.027 | 0.027 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_ritz_and_residuals` | 4 | 0.231 | 0.078 |
| `solve` | 10 | 0.201 | 0.045 |
| `inv` | 5 | 0.095 | 0.020 |
| `matvec_scan` | 2 | 0.062 | 0.032 |
| `_matvec` | 2 | 0.058 | 0.030 |
| `_lu_solve` | 16 | 0.038 | 0.004 |
| `_ortho_expand` | 4 | 0.020 | 0.005 |
| `_einsum` | 20 | 0.011 | 0.001 |
| `_impl` | 1 | 0.008 | 0.008 |
| `cholesky` | 4 | 0.008 | 0.002 |
| `multiply` | 17 | 0.006 | 0.001 |
| `add` | 17 | 0.005 | 0.001 |
| `_psum` | 5 | 0.005 | 0.001 |
| `_local_ifftn` | 1 | 0.005 | 0.005 |
| `less` | 10 | 0.004 | 0.001 |
| `true_divide` | 9 | 0.004 | 0.001 |
| `conjugate` | 14 | 0.003 | 0.000 |
| `_reduce_sum` | 7 | 0.003 | 0.001 |
| `_orthonormalise_batch` | 1 | 0.003 | 0.003 |
| `dynamic_slice` | 2 | 0.003 | 0.002 |
| `negative` | 14 | 0.003 | 0.001 |
| `_broadcast_arrays` | 9 | 0.002 | 0.001 |
| `broadcast_in_dim` | 7 | 0.002 | 0.000 |
| `matmul` | 6 | 0.002 | 0.000 |
| `_squeeze` | 10 | 0.002 | 0.000 |
| `subtract` | 5 | 0.002 | 0.001 |
| `eigh` | 3 | 0.002 | 0.001 |
| `_multi_slice` | 5 | 0.001 | 0.000 |
| `concatenate` | 5 | 0.001 | 0.000 |
| `convert_element_type` | 6 | 0.001 | 0.000 |

## Tracing cache misses

Total: **89** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:79:12` | 17 | never seen function: _lu_solve id=139697737905184 defined at /opt/jax/jax/_src/lax/linalg.py:1566 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:83:19` | 13 | for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566 never seen input type signature: lu: c128[20,20],  permutation: i32[20],  b: c |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:217:16` | 8 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: c128[4,20,4,4,64] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:627:14` | 4 | never seen function: |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:215:12` | 4 | never seen function: broadcast_in_dim id=139694823420320 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:78:8` | 4 | never seen function: cholesky id=139697739101536 defined at /opt/jax/jax/_src/numpy/linalg.py:74 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:219:8` | 4 | never seen function: _ritz_and_residuals id=139697708537856 defined at /global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:87 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:338:12` | 4 | never seen function: _ortho_expand id=139697708540896 defined at /global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:150 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:340:12` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:82:21` | 3 | never seen function: eigh id=139697739261536 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:410:19` | 2 | never seen function: _psum id=140194110522048 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:552:50` | 2 | never seen function: dynamic_slice id=140474226055328 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:69:23` | 2 | never seen function: convert_element_type id=139694823328640 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:325:15` | 2 | never seen function: fft id=140057445237760 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/davidson_absorption.py:145:17` | 2 | never seen function: build_bse_simple_matvec.<locals>._matvec id=139694823334400 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/ |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/davidson_absorption.py:152:15` | 2 | never seen function: main.<locals>.matvec_scan id=139694823339040 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/davidson_absorp |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/davidson_absorption.py:48:8` | 2 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: c128[20,4,4,64] closest seen |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:393:18` | 1 | never seen function: _identity_fn id=139697747592672 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:618:11` | 1 | never seen function: broadcast_in_dim id=140474225933536 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:618:37` | 1 | never seen function: broadcast_in_dim id=140474226049248 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:547:16` | 1 | never seen function: convert_element_type id=140474226051648 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_davidson_helpers.py:53:15` | 1 | never seen function: _identity_fn id=139697708532896 defined at /opt/jax/jax/experimental/multihost_utils.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:292:8` | 1 | never seen function: _orthonormalise_batch id=139697708539456 defined at /global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:135 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:343:15` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_davidson_helpers.py:192:21` | 1 | never seen function: _where id=139697744883328 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_davidson_helpers.py:201:15` | 1 | never seen function: bse_diagonal_precond.<locals>._impl id=139694823340960 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/bse_d |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:59:15` | 1 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: f64[20] closest seen input t |

## Persistent cache misses

_None._
