# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_sharded_v3/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 215 | 0.431 | 0.143 |
| jaxpr→MLIR | 20 | 0.275 | 0.130 |
| XLA compile | 25 | 3.628 | 2.059 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_full_run` | 1 | 2.059 | 2.059 |
| `multiply` | 8 | 0.482 | 0.068 |
| `_identity_fn` | 1 | 0.256 | 0.256 |
| `_multi_slice` | 1 | 0.172 | 0.172 |
| `add` | 2 | 0.129 | 0.067 |
| `concatenate` | 2 | 0.111 | 0.056 |
| `scatter-add` | 1 | 0.105 | 0.105 |
| `reshape` | 2 | 0.068 | 0.035 |
| `conjugate` | 1 | 0.062 | 0.062 |
| `_squeeze` | 2 | 0.062 | 0.032 |
| `broadcast_in_dim` | 1 | 0.033 | 0.033 |
| `_psum` | 1 | 0.031 | 0.031 |
| `dynamic_slice` | 1 | 0.030 | 0.030 |
| `convert_element_type` | 1 | 0.029 | 0.029 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_full_run` | 1 | 0.143 | 0.143 |
| `_matvec_impl` | 1 | 0.087 | 0.087 |
| `fft_impl` | 10 | 0.045 | 0.033 |
| `negative` | 8 | 0.034 | 0.033 |
| `true_divide` | 11 | 0.033 | 0.031 |
| `_apply_W_from_T` | 1 | 0.011 | 0.011 |
| `norm` | 3 | 0.008 | 0.003 |
| `_diag` | 3 | 0.005 | 0.004 |
| `add` | 20 | 0.005 | 0.000 |
| `_uniform` | 2 | 0.005 | 0.003 |
| `_moveaxis` | 28 | 0.005 | 0.001 |
| `_normal` | 1 | 0.004 | 0.004 |
| `_normal_real` | 1 | 0.004 | 0.004 |
| `_where` | 5 | 0.004 | 0.001 |
| `multiply` | 12 | 0.004 | 0.001 |
| `conjugate` | 15 | 0.003 | 0.000 |
| `broadcast_in_dim` | 5 | 0.003 | 0.002 |
| `subtract` | 11 | 0.002 | 0.000 |
| `_reduce_sum` | 5 | 0.002 | 0.001 |
| `remainder` | 1 | 0.002 | 0.002 |
| `_einsum` | 6 | 0.002 | 0.001 |
| `vdot` | 1 | 0.002 | 0.002 |
| `sqrt` | 6 | 0.001 | 0.000 |
| `fft` | 7 | 0.001 | 0.000 |
| `_broadcast_arrays` | 6 | 0.001 | 0.001 |
| `eigh` | 1 | 0.001 | 0.001 |
| `_psum` | 1 | 0.001 | 0.001 |
| `less` | 4 | 0.001 | 0.000 |
| `real` | 6 | 0.001 | 0.000 |
| `equal` | 3 | 0.001 | 0.000 |

## Tracing cache misses

Total: **46** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:186:32` | 7 | never seen function: solve_bse_sharded.<locals>._full_run id=139979166201984 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/bse_ |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 5 | never seen function: fft id=140661796488192 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:247:9` | 4 | never seen function: _uniform id=140661788597664 defined at /opt/j2026-04-27 01:54:06,003 ja2026-04-27 01:54:06,003 jax._src.dispatch WARNIN |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:656:54` | 3 | never seen function: dynamic_slice id=140658707158720 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:659:14` | 3 | never seen function: convert_element_type id=140658707164160 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:404:19` | 2 | never seen function: _psum id=139982998101888 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:648:11` | 2 | never seen function: broadcast_in_dim id=140564684969824 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:651:19` | 2 | never seen function: convert_element_type id=140658707153440 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:264:17` | 2 | never seen function: _where id=140661794926208 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:288:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[200,200],  x: f64[200],  y: f64[2 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:387:18` | 1 | never seen function: _identity_fn id=140661797635552 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:648:37` | 1 | never seen function: broadcast_in_dim id=139979165836416 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:249:14` | 1 | never seen function: norm id=139983138340544 defined at /opt/jax/jax/_src/numpy/linalg.py:1076 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_ring_comm.py:449:17` | 1 | never seen function: build_bse_ring_matvec.<locals>._apply_W_from_T id=139979166193184 defined at /global/homes/j/jackm/software/lorrax_C/sr |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:177:17` | 1 | never seen function: build_bse_ring_matvec.<locals>._matvec_impl id=139979166200544 defined at /global/homes/j/jackm/software/lorrax_C/src/b |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:265:20` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:271:19` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: c128[],  y: c128[] closest |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:277:17` | 1 | for norm defined at /opt/jax/jax/_src/numpy/linalg.py:1076 never seen input type signature: x: c128[1024] closest seen input type signature  |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:290:12` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 never seen input type signature: v: f64[199] closest seen input type signatur |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:290:36` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 explanation unavailable! please open an issue at https://github.com/jax-ml/ja |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:292:22` | 1 | never seen function: eigh id=139983138334304 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:293:10` | 1 | never seen function: argsort id=139983143451712 defined at /opt/jax/jax/_src/numpy/sorting.py:92 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:297:12` | 1 | for norm defined at /opt/jax/jax/_src/numpy/linalg.py:1076 never seen input type signature: x: c128[20,1024] closest seen input type signatu |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:192:19` | 1 | never seen function: reshape id=140658709630464 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
