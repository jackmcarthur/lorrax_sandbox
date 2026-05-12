# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/C_bse_davidson_profile_2026-04-28/profile_warmup/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 267 | 0.815 | 0.154 |
| jaxpr→MLIR | 52 | 0.368 | 0.102 |
| XLA compile | 61 | 5.494 | 0.576 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_ritz_and_residuals` | 4 | 2.018 | 0.576 |
| `<unnamed wrapped function>` | 12 | 0.566 | 0.137 |
| `_matvec_impl` | 1 | 0.529 | 0.529 |
| `concatenate` | 6 | 0.411 | 0.118 |
| `_identity_fn` | 5 | 0.355 | 0.231 |
| `broadcast_in_dim` | 7 | 0.315 | 0.058 |
| `_multi_slice` | 4 | 0.224 | 0.058 |
| `_psum` | 5 | 0.181 | 0.042 |
| `multiply` | 3 | 0.175 | 0.064 |
| `_impl` | 1 | 0.159 | 0.159 |
| `scatter-add` | 1 | 0.106 | 0.106 |
| `reshape` | 3 | 0.103 | 0.035 |
| `gather` | 1 | 0.062 | 0.062 |
| `_squeeze` | 2 | 0.061 | 0.030 |
| `add` | 1 | 0.060 | 0.060 |
| `conjugate` | 1 | 0.058 | 0.058 |
| `convert_element_type` | 2 | 0.056 | 0.029 |
| `dynamic_slice` | 1 | 0.029 | 0.029 |
| `squeeze` | 1 | 0.027 | 0.027 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_ritz_and_residuals` | 4 | 0.228 | 0.079 |
| `solve` | 9 | 0.179 | 0.047 |
| `_matvec_impl` | 1 | 0.154 | 0.154 |
| `inv` | 4 | 0.105 | 0.048 |
| `_lu_solve` | 16 | 0.037 | 0.003 |
| `_apply_W_from_T` | 1 | 0.017 | 0.017 |
| `_einsum` | 16 | 0.008 | 0.001 |
| `cholesky` | 4 | 0.008 | 0.002 |
| `_impl` | 1 | 0.008 | 0.008 |
| `multiply` | 19 | 0.007 | 0.001 |
| `add` | 21 | 0.006 | 0.000 |
| `_psum` | 5 | 0.005 | 0.001 |
| `_reduce_sum` | 9 | 0.004 | 0.001 |
| `true_divide` | 10 | 0.004 | 0.001 |
| `conjugate` | 17 | 0.004 | 0.000 |
| `less` | 11 | 0.004 | 0.001 |
| `broadcast_in_dim` | 7 | 0.003 | 0.002 |
| `negative` | 18 | 0.003 | 0.000 |
| `_squeeze` | 13 | 0.003 | 0.001 |
| `remainder` | 1 | 0.002 | 0.002 |
| `eigh` | 4 | 0.002 | 0.001 |
| `matmul` | 7 | 0.002 | 0.000 |
| `_apply_D_term` | 1 | 0.002 | 0.002 |
| `_broadcast_arrays` | 10 | 0.002 | 0.001 |
| `subtract` | 5 | 0.001 | 0.000 |
| `concatenate` | 5 | 0.001 | 0.000 |
| `_where` | 2 | 0.001 | 0.001 |
| `equal` | 5 | 0.001 | 0.000 |
| `sqrt` | 6 | 0.001 | 0.000 |
| `_multi_slice` | 5 | 0.001 | 0.000 |

## Tracing cache misses

Total: **96** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:79:12` | 16 | never seen function: _lu_solve id=140169471800512 defined at /opt/jax/jax/_src/lax/linalg.py:1566 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:83:19` | 12 | for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566 never seen input type signature: lu: c128[10,10],  permutation: i32[10],  b: c |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:325:15` | 10 | never seen function: <jax._src.util.HashablePartial object at 0x7f7b681f3980> id=140168004581760 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:171:16` | 8 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: c128[4,10,8,2,64] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:627:14` | 4 | never seen function: convert_element_type id=140166394275904 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:169:12` | 4 | never seen function: broadcast_in_dim id=140168003880096 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:78:8` | 4 | never seen function: cholesky id=140169476732416 defined at /opt/jax/jax/_src/numpy/linalg.py:74 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:82:21` | 4 | never seen function: eigh id=140169476892416 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:173:8` | 4 | never seen function: _ritz_and_residuals id=140169416198880 defined at /global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:87 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:282:12` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:230:23` | 3 | never seen function: reshape id=140165794906176 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:420:19` | 2 | never seen function: _psum id=140637980679904 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:552:50` | 2 | never seen function: dynamic_slice id=140166394268384 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:196:14` | 2 | never seen function: <jax._src.util.HashablePartial object at 0x7f7bbc4d7c50> id=140169416899664 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_davidson_helpers.py:53:15` | 2 | never seen function: _identity_fn id=140169417679040 defined at /opt/jax/jax/experimental/multihost_utils.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:264:17` | 2 | never seen function: _where id=140169478778656 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:403:18` | 1 | never seen function: _identity_fn id=140169481504384 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:404:18` | 1 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[64,8,2,480] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:618:11` | 1 | never seen function: broadcast_in_dim id=140169415598752 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:618:37` | 1 | never seen function: broadcast_in_dim id=140166394868608 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:547:16` | 1 | never seen function: convert_element_type id=140166394871648 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:444:17` | 1 | never seen function: build_bse_ring_matvec.<locals>._apply_D_term id=140166394075296 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:343:15` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:449:17` | 1 | never seen function: build_bse_ring_matvec.<locals>._apply_W_from_T id=140166394073536 defined at /global/homes/j/jackm/software/lorrax_C/sr |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:200:19` | 1 | never seen function: build_bse_ring_matvec.<locals>._matvec_impl id=140166394076736 defined at /global/homes/j/jackm/software/lorrax_C/src/b |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_davidson_helpers.py:192:21` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_davidson_helpers.py:201:15` | 1 | never seen function: bse_diagonal_precond.<locals>._impl id=140166394079456 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/bse_d |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:59:15` | 1 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: f64[10] closest seen input t |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/davidson.py:270:34` | 1 | never seen function: gather id=140165796258464 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
