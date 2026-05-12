# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_simple_fused/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 159 | 0.191 | 0.087 |
| jaxpr→MLIR | 21 | 0.268 | 0.122 |
| XLA compile | 26 | 3.362 | 1.802 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_full_run` | 1 | 1.802 | 1.802 |
| `multiply` | 6 | 0.358 | 0.063 |
| `_identity_fn` | 1 | 0.253 | 0.253 |
| `_multi_slice` | 1 | 0.193 | 0.193 |
| `conjugate` | 2 | 0.120 | 0.060 |
| `sqrt` | 2 | 0.113 | 0.058 |
| `scatter-add` | 1 | 0.101 | 0.101 |
| `concatenate` | 1 | 0.061 | 0.061 |
| `_squeeze` | 2 | 0.060 | 0.031 |
| `add` | 1 | 0.060 | 0.060 |
| `broadcast_in_dim` | 2 | 0.058 | 0.029 |
| `convert_element_type` | 2 | 0.057 | 0.029 |
| `reshape` | 1 | 0.033 | 0.033 |
| `_psum` | 1 | 0.032 | 0.032 |
| `dynamic_slice` | 1 | 0.031 | 0.031 |
| `squeeze` | 1 | 0.030 | 0.030 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_full_run` | 1 | 0.087 | 0.087 |
| `_matvec` | 1 | 0.025 | 0.025 |
| `norm` | 2 | 0.006 | 0.003 |
| `_einsum` | 12 | 0.006 | 0.001 |
| `_diag` | 3 | 0.006 | 0.004 |
| `_uniform` | 2 | 0.005 | 0.003 |
| `_normal` | 1 | 0.005 | 0.005 |
| `_normal_real` | 1 | 0.004 | 0.004 |
| `multiply` | 13 | 0.004 | 0.001 |
| `add` | 13 | 0.004 | 0.001 |
| `conjugate` | 11 | 0.003 | 0.000 |
| `_where` | 3 | 0.003 | 0.001 |
| `broadcast_in_dim` | 4 | 0.003 | 0.002 |
| `fft_impl` | 2 | 0.002 | 0.001 |
| `true_divide` | 6 | 0.002 | 0.000 |
| `_moveaxis` | 12 | 0.002 | 0.000 |
| `vdot` | 1 | 0.002 | 0.002 |
| `_reduce_sum` | 3 | 0.002 | 0.001 |
| `subtract` | 7 | 0.002 | 0.000 |
| `sqrt` | 5 | 0.002 | 0.001 |
| `eigh` | 1 | 0.001 | 0.001 |
| `negative` | 6 | 0.001 | 0.000 |
| `_squeeze` | 3 | 0.001 | 0.001 |
| `_psum` | 1 | 0.001 | 0.001 |
| `fft` | 6 | 0.001 | 0.000 |
| `_broadcast_arrays` | 4 | 0.001 | 0.000 |
| `less` | 3 | 0.001 | 0.000 |
| `convert_element_type` | 4 | 0.001 | 0.000 |
| `maximum` | 3 | 0.001 | 0.000 |
| `real` | 4 | 0.001 | 0.000 |

## Tracing cache misses

Total: **40** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:659:14` | 4 | never seen function: convert_element_type id=140609514893088 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:247:9` | 4 | never seen function: _uniform id=140612291731008 defined at /opt/j2026-04-27 02:52:23,917 ja2026-04-27 02:52:23,918 jax._src.dispatch WARNIN |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 3 | never seen function: fft id=140612301653152 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:249:45` | 3 | never seen function: solve_bse_sharded.<locals>._full_run id=140036067207552 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/bse_ |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:404:19` | 2 | never seen function: _psum id=140055596064800 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:648:11` | 2 | never seen function: broadcast_in_dim id=140612150631040 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:651:19` | 2 | never seen function: convert_element_type id=140609514882368 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:656:54` | 2 | never seen function: dynamic_slice id=140609514887648 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:90:14` | 2 | for fft defined at /opt/jax/jax/_src2026-04-27 02:52:23,940 jax._src.dispatch WARNING: Finished tracing + transforming _einsum for pjit in 0 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:288:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[200,200],  x: f64[200],  y: f64[2 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:387:18` | 1 | never seen function: _identity_fn id=140612302800512 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:648:37` | 1 | never seen function: broadcast_in_dim id=140036067747744 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:77:23` | 1 | never seen function: convert_element_type id=140609515145152 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:92:15` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:199:21` | 1 | never seen function: build_bse_simple_matvec.<locals>._matvec id=140036067206432 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/ |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:265:20` | 1 | never seen function: _where id=140039365144352 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:271:19` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: c128[],  y: c128[] closest |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:277:17` | 1 | for norm defined at /opt/jax/jax/_src/numpy/linalg.py:1076 never seen input type signature: x: c128[1024] closest seen input type signature  |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:290:12` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 never seen input type signature: v: f64[199] closest seen input type signatur |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:290:36` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 explanation unavailable! please open an issue at https://github.com/jax-ml/ja |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:292:22` | 1 | never seen function: eigh id=140039363225344 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:293:10` | 1 | never seen function: argsort id=140039362575584 defined at /opt/jax/jax/_src/numpy/sorting.py:92 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:297:12` | 1 | for norm defined at /opt/jax/jax/_src/numpy/linalg.py:1076 never seen input type signature: x: c128[5,1024] closest seen input type signatur |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:255:19` | 1 | never seen function: reshape id=140609515157632 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
