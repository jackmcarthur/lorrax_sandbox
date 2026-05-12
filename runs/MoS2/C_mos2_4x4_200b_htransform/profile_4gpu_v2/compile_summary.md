# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v2/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 353 | 0.239 | 0.034 |
| jaxpr→MLIR | 225 | 0.774 | 0.103 |
| XLA compile | 224 | 14.408 | 0.426 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `multiply` | 17 | 1.013 | 0.069 |
| `broadcast_in_dim` | 24 | 0.943 | 0.073 |
| `add` | 14 | 0.888 | 0.102 |
| `subtract` | 11 | 0.660 | 0.083 |
| `transpose` | 8 | 0.579 | 0.082 |
| `sort` | 4 | 0.519 | 0.231 |
| `_reduce_max` | 5 | 0.510 | 0.154 |
| `conjugate` | 8 | 0.509 | 0.071 |
| `gather` | 8 | 0.464 | 0.068 |
| `iota` | 8 | 0.432 | 0.057 |
| `_fft_gather_reshard` | 1 | 0.426 | 0.426 |
| `reshape` | 10 | 0.380 | 0.103 |
| `_where` | 6 | 0.334 | 0.057 |
| `matmul` | 6 | 0.323 | 0.132 |
| `exp` | 4 | 0.318 | 0.100 |
| `true_divide` | 5 | 0.305 | 0.066 |
| `dynamic_slice` | 5 | 0.304 | 0.072 |
| `cholesky` | 2 | 0.301 | 0.177 |
| `_fft_and_rslice` | 1 | 0.288 | 0.288 |
| `_reduce_min` | 3 | 0.286 | 0.148 |
| `abs` | 4 | 0.282 | 0.083 |
| `scan` | 2 | 0.261 | 0.134 |
| `eigvalsh` | 2 | 0.250 | 0.147 |
| `_solve_triangular` | 2 | 0.237 | 0.118 |
| `swapaxes` | 3 | 0.235 | 0.084 |
| `convert_element_type` | 6 | 0.222 | 0.059 |
| `_multi_slice` | 3 | 0.216 | 0.095 |
| `concatenate` | 3 | 0.189 | 0.068 |
| `greater_equal` | 3 | 0.172 | 0.057 |
| `negative` | 3 | 0.169 | 0.061 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `dynamic_slice` | 5 | 0.035 | 0.034 |
| `_fft_gather_reshard` | 1 | 0.018 | 0.018 |
| `_fft_and_rslice` | 1 | 0.017 | 0.017 |
| `multiply` | 29 | 0.011 | 0.001 |
| `broadcast_in_dim` | 23 | 0.010 | 0.002 |
| `fft_impl` | 6 | 0.010 | 0.003 |
| `add` | 23 | 0.009 | 0.001 |
| `_where` | 10 | 0.008 | 0.001 |
| `_diag` | 1 | 0.008 | 0.008 |
| `diagonal` | 1 | 0.008 | 0.008 |
| `subtract` | 14 | 0.007 | 0.002 |
| `norm` | 2 | 0.005 | 0.003 |
| `true_divide` | 12 | 0.005 | 0.001 |
| `matmul` | 8 | 0.004 | 0.001 |
| `cholesky` | 2 | 0.004 | 0.002 |
| `clip` | 3 | 0.004 | 0.002 |
| `negative` | 15 | 0.004 | 0.000 |
| `_accum` | 1 | 0.004 | 0.004 |
| `conjugate` | 13 | 0.003 | 0.000 |
| `_take` | 1 | 0.003 | 0.003 |
| `less` | 7 | 0.003 | 0.001 |
| `abs` | 5 | 0.003 | 0.001 |
| `reshape` | 10 | 0.003 | 0.000 |
| `_moveaxis` | 13 | 0.003 | 0.000 |
| `_svd_replicated` | 1 | 0.002 | 0.002 |
| `_broadcast_arrays` | 11 | 0.002 | 0.000 |
| `transpose` | 9 | 0.002 | 0.000 |
| `_reduce_max` | 5 | 0.002 | 0.001 |
| `_solve_triangular` | 2 | 0.002 | 0.002 |
| `gather` | 8 | 0.002 | 0.000 |

## Tracing cache misses

Total: **136** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:641:37` | 6 | for sort defined at /opt/jax/jax/_src/numpy/sorting.py:31 never seen input type signature: a: f64[181,960] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 4 | never seen function: fft id=140513203553280 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140511199231520 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:415:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140510331728928 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:188:8` | 4 | never seen function: convert_element_type id=140348796547488 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:192:10` | 4 | never seen function: convert_element_type id=140348796853664 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:193:44` | 3 | never seen function: iota id=140511200005632 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:596:23` | 3 | never seen function: broadcast_in_dim id=140348061196320 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:596:13` | 3 | never seen function: broadcast_in_dim id=140348061198880 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:639:19` | 3 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 6 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140351210445920 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140513204700640 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[16,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140513147706464 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=140511199220800 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140513147879104 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=140513201991296 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:155:8` | 2 | never seen function: _psum id=140513149321600 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:258:14` | 2 | never seen function: dynamic_slice id=140526433920928 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:271:15` | 2 | never seen function: convert_element_type id=140525357840992 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:280:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:281:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:282:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:571:45` | 2 | never seen function: gather id=140511198249664 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:571:25` | 2 | never seen function: eigh id=140351216903776 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:572:28` | 2 | never seen function: dynamic_slice id=140348063591104 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:573:39` | 2 | never seen function: gather id=140348063594304 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:576:41` | 2 | never seen function: gather id=140348062912480 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:581:39` | 2 | never seen function: diagonal id=140351216588160 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:589:14` | 2 | never seen function: reshape id=140523686040640 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
