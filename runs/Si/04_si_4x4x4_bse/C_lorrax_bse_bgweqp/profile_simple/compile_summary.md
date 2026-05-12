# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_simple/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 180 | 0.216 | 0.094 |
| jaxpr→MLIR | 20 | 0.291 | 0.121 |
| XLA compile | 23 | 3.384 | 1.883 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_full_run` | 1 | 1.883 | 1.883 |
| `multiply` | 5 | 0.297 | 0.065 |
| `_identity_fn` | 1 | 0.290 | 0.290 |
| `_multi_slice` | 1 | 0.171 | 0.171 |
| `add` | 2 | 0.121 | 0.061 |
| `sqrt` | 2 | 0.121 | 0.064 |
| `conjugate` | 2 | 0.120 | 0.060 |
| `scatter-add` | 1 | 0.106 | 0.106 |
| `_squeeze` | 2 | 0.060 | 0.030 |
| `concatenate` | 1 | 0.056 | 0.056 |
| `_psum` | 1 | 0.043 | 0.043 |
| `broadcast_in_dim` | 1 | 0.031 | 0.031 |
| `dynamic_slice` | 1 | 0.030 | 0.030 |
| `reshape` | 1 | 0.028 | 0.028 |
| `convert_element_type` | 1 | 0.028 | 0.028 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_full_run` | 1 | 0.094 | 0.094 |
| `_matvec` | 1 | 0.032 | 0.032 |
| `fft_impl` | 7 | 0.011 | 0.003 |
| `norm` | 3 | 0.008 | 0.003 |
| `_diag` | 3 | 0.006 | 0.004 |
| `_normal` | 1 | 0.005 | 0.005 |
| `_einsum` | 8 | 0.004 | 0.001 |
| `multiply` | 12 | 0.004 | 0.001 |
| `_normal_real` | 1 | 0.004 | 0.004 |
| `_moveaxis` | 24 | 0.004 | 0.000 |
| `add` | 11 | 0.004 | 0.001 |
| `conjugate` | 13 | 0.003 | 0.000 |
| `true_divide` | 10 | 0.003 | 0.000 |
| `broadcast_in_dim` | 4 | 0.003 | 0.002 |
| `_where` | 3 | 0.003 | 0.001 |
| `_uniform` | 1 | 0.003 | 0.003 |
| `_reduce_sum` | 4 | 0.002 | 0.001 |
| `vdot` | 1 | 0.002 | 0.002 |
| `subtract` | 7 | 0.002 | 0.000 |
| `sqrt` | 7 | 0.002 | 0.000 |
| `negative` | 8 | 0.001 | 0.000 |
| `eigh` | 1 | 0.001 | 0.001 |
| `reshape` | 2 | 0.001 | 0.001 |
| `_psum` | 1 | 0.001 | 0.001 |
| `fft` | 6 | 0.001 | 0.000 |
| `_broadcast_arrays` | 4 | 0.001 | 0.000 |
| `less` | 3 | 0.001 | 0.000 |
| `convert_element_type` | 4 | 0.001 | 0.000 |
| `scatter-add` | 2 | 0.001 | 0.000 |
| `maximum` | 3 | 0.001 | 0.000 |

## Tracing cache misses

Total: **48** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:249:45` | 8 | never seen function: solve_bse_sharded.<locals>._full_run id=140520331032064 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/bse_ |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 6 | never seen function: fft id=139666653168800 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:656:54` | 3 | never seen function: dynamic_slice id=139664351172768 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:659:14` | 3 | never seen function: convert_element_type id=139664351178208 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:247:9` | 3 | never seen function: _uniform id=140524081323584 defined at /opt/jax/jax/_src/random.py:407 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:255:19` | 3 | never seen function: reshape id=139666030013440 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:648:11` | 2 | never seen function: broadcast_in_dim id=139666573203776 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:651:19` | 2 | never seen function: convert_element_type id=139664351167488 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:288:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[200,200],  x: f64[200],  y: f64[2 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:387:18` | 1 | never seen function: _identity_fn id=140099439511168 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:404:19` | 1 | never seen function: _psum id=140524020113120 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/soft2026-04-27` | 1 | never seen function: _multi_slice id=140209038733888 defined at /opt/jax/jax/_src/numpy/array_methods.py:616 but seen another function defi2 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:648:37` | 1 | never seen function: broadcast_in_dim id=140520330618464 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/s2026-04-27` | 1 | never seen function: concatenate id=139664351712800 defined at /opt/jax/jax/_src/dispatch.py:96 but seen another function defined on t2026-0 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:73:23` | 1 | never seen function: convert_element_type id=139664351233344 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:249:14` | 1 | never seen function: norm id=140524087770976 defined at /opt/jax/jax/_src/numpy/linalg.py:1076 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:199:21` | 1 | never seen function: build_bse_simple_matvec.<locals>._matvec id=140520331027744 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/ |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:265:20` | 1 | never seen function: _where id=140524089650976 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:271:19` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: c128[],  y: c128[] closest |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:277:17` | 1 | for norm defined at /opt/jax/jax/_src/numpy/linalg.py:1076 never seen input type signature: x: c128[1024] closest seen input type signature  |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:290:12` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 never seen input type signature: v: f64[199] closest seen input type signatur |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:290:36` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 explanation unavailable! please open an issue at https://github.com/jax-ml/ja |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:292:22` | 1 | never seen function: eigh id=140524087764736 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:293:10` | 1 | never seen function: argsort id=140524089162976 defined at /opt/jax/jax/_src/numpy/sorting.py:92 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:297:12` | 1 | for norm defined at /opt/jax/jax/_src/numpy/linalg.py:1076 never seen input type signature: x: c128[5,1024] closest seen input type signatur |

## Persistent cache misses

_None._
