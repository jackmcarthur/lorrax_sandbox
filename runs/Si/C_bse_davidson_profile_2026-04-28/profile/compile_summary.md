# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/C_bse_davidson_profile_2026-04-28/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 253 | 1.164 | 0.164 |
| jaxpr→MLIR | 40 | 0.366 | 0.119 |
| XLA compile | 47 | 4.900 | 0.558 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_ritz_and_residuals` | 4 | 2.012 | 0.558 |
| `<unnamed wrapped function>` | 10 | 0.595 | 0.134 |
| `_matvec_impl` | 1 | 0.531 | 0.531 |
| `concatenate` | 6 | 0.348 | 0.064 |
| `multiply` | 4 | 0.241 | 0.064 |
| `_multi_slice` | 1 | 0.174 | 0.174 |
| `_impl` | 1 | 0.164 | 0.164 |
| `conjugate` | 2 | 0.121 | 0.063 |
| `_identity_fn` | 4 | 0.120 | 0.033 |
| `add` | 2 | 0.117 | 0.059 |
| `reshape` | 3 | 0.105 | 0.040 |
| `scatter-add` | 1 | 0.104 | 0.104 |
| `_squeeze` | 2 | 0.061 | 0.031 |
| `gather` | 1 | 0.058 | 0.058 |
| `convert_element_type` | 2 | 0.058 | 0.029 |
| `_psum` | 1 | 0.032 | 0.032 |
| `dynamic_slice` | 1 | 0.031 | 0.031 |
| `broadcast_in_dim` | 1 | 0.030 | 0.030 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `solve` | 16 | 0.324 | 0.048 |
| `_ritz_and_residuals` | 4 | 0.227 | 0.079 |
| `inv` | 7 | 0.190 | 0.049 |
| `_matvec_impl` | 1 | 0.164 | 0.164 |
| `_lu_solve` | 16 | 0.099 | 0.033 |
| `_broadcast_arrays` | 12 | 0.062 | 0.031 |
| `_apply_W_from_T` | 1 | 0.017 | 0.017 |
| `_einsum` | 21 | 0.010 | 0.001 |
| `cholesky` | 4 | 0.008 | 0.002 |
| `multiply` | 19 | 0.008 | 0.001 |
| `_impl` | 1 | 0.008 | 0.008 |
| `add` | 26 | 0.007 | 0.000 |
| `conjugate` | 19 | 0.005 | 0.000 |
| `true_divide` | 12 | 0.005 | 0.001 |
| `less` | 12 | 0.004 | 0.001 |
| `negative` | 14 | 0.003 | 0.000 |
| `remainder` | 1 | 0.003 | 0.003 |
| `_apply_D_term` | 1 | 0.002 | 0.002 |
| `_reduce_sum` | 4 | 0.002 | 0.001 |
| `_psum` | 1 | 0.001 | 0.001 |
| `subtract` | 5 | 0.001 | 0.000 |
| `equal` | 5 | 0.001 | 0.000 |
| `_where` | 2 | 0.001 | 0.001 |
| `broadcast_in_dim` | 4 | 0.001 | 0.000 |
| `_squeeze` | 6 | 0.001 | 0.000 |
| `convert_element_type` | 4 | 0.001 | 0.000 |
| `sqrt` | 4 | 0.001 | 0.000 |
| `reshape` | 2 | 0.001 | 0.000 |
| `concatenate` | 4 | 0.001 | 0.000 |
| `abs` | 2 | 0.001 | 0.000 |

## Tracing cache misses

Total: **82** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:79:12` | 17 | never seen function: _lu_solve id=140661748839616 defined at /opt/jax/jax/_src/lax/linalg.py:1566 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:83:19` | 15 | for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566 never seen input type signature: lu: c128[10,10],  permutation: i32[10],  b: c |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:325:15` | 10 | never seen function: <jax._src.util.HashablePartial object at 0x7f2e98757e00> id=139838103059968 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:282:12` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:627:14` | 3 | never seen function: convert_element_type id=140659242395072 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:196:14` | 3 | never seen function: <jax._src.util.HashablePartial object at 0x7fedc825e420> id=140659241903136 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:78:8` | 3 | never seen function: cholesky id=140661753771520 defined at /opt/jax/jax/_src/numpy/linalg.py:74 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:250:32` | 3 | for _ritz_and_residuals defined at /global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:87 never seen input type signature: V: c1 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:217:23` | 3 | never seen function: reshape id=140659243055232 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:420:19` | 2 | never seen function: _psum id=139842468666784 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:618:11` | 2 | never seen function: broadcast_in_dim id=140242340438048 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:547:16` | 2 | never seen function: convert_element_type id=140659242204544 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:552:50` | 2 | never seen function: dynamic_slice id=139838103407008 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:264:17` | 2 | never seen function: _where id=140661755817760 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:403:18` | 1 | never seen function: _identity_fn id=140242490508928 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:404:18` | 1 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[64,8,2,480] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:618:37` | 1 | never seen function: broadcast_in_dim id=139838103253408 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_davidson_helpers.py:53:15` | 1 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: f64[4,64,2] closest seen inp |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:444:17` | 1 | never seen function: build_bse_ring_matvec.<locals>._apply_D_term id=140659241539104 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:343:15` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:449:17` | 1 | never seen function: build_bse_ring_matvec.<locals>._apply_W_from_T id=140659241537344 defined at /global/homes/j/jackm/software/lorrax_C/sr |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_davidson_helpers.py:192:21` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_davidson_helpers.py:201:15` | 1 | never seen function: bse_diagonal_precond.<locals>._impl id=140659241543264 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/bse_d |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:59:15` | 1 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: f64[10] closest seen input t |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:270:34` | 1 | never seen function: gather id=140661118619456 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
