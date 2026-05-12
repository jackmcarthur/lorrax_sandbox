# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v3/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 362 | 0.263 | 0.037 |
| jaxpr→MLIR | 190 | 0.692 | 0.100 |
| XLA compile | 190 | 13.105 | 0.442 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 23 | 0.900 | 0.065 |
| `transpose` | 6 | 0.792 | 0.442 |
| `multiply` | 12 | 0.700 | 0.068 |
| `subtract` | 11 | 0.666 | 0.072 |
| `add` | 11 | 0.646 | 0.067 |
| `sort` | 4 | 0.531 | 0.232 |
| `_reduce_max` | 5 | 0.513 | 0.155 |
| `gather` | 8 | 0.499 | 0.100 |
| `_fft_gather_reshard` | 1 | 0.438 | 0.438 |
| `iota` | 8 | 0.436 | 0.057 |
| `reshape` | 7 | 0.349 | 0.104 |
| `true_divide` | 5 | 0.321 | 0.068 |
| `cholesky` | 2 | 0.310 | 0.187 |
| `_kpath_batch` | 1 | 0.297 | 0.297 |
| `_reduce_min` | 3 | 0.296 | 0.149 |
| `_fft_and_rslice` | 1 | 0.293 | 0.293 |
| `dynamic_slice` | 5 | 0.287 | 0.061 |
| `_where` | 5 | 0.284 | 0.062 |
| `abs` | 4 | 0.282 | 0.086 |
| `scan` | 2 | 0.265 | 0.134 |
| `conjugate` | 4 | 0.258 | 0.073 |
| `matmul` | 4 | 0.255 | 0.132 |
| `convert_element_type` | 7 | 0.253 | 0.057 |
| `exp` | 3 | 0.224 | 0.098 |
| `_build_fH` | 1 | 0.218 | 0.218 |
| `concatenate` | 3 | 0.192 | 0.074 |
| `_multi_slice` | 3 | 0.178 | 0.062 |
| `greater_equal` | 3 | 0.169 | 0.059 |
| `_argmin` | 2 | 0.155 | 0.079 |
| `eigvalsh` | 1 | 0.144 | 0.144 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `sort` | 4 | 0.038 | 0.037 |
| `_fft_gather_reshard` | 1 | 0.018 | 0.018 |
| `fft_impl` | 9 | 0.016 | 0.003 |
| `_build_fH` | 1 | 0.016 | 0.016 |
| `_kpath_batch` | 1 | 0.012 | 0.012 |
| `multiply` | 29 | 0.010 | 0.001 |
| `_fft_and_rslice` | 1 | 0.010 | 0.010 |
| `_diag` | 1 | 0.009 | 0.009 |
| `_where` | 10 | 0.008 | 0.001 |
| `diagonal` | 1 | 0.008 | 0.008 |
| `broadcast_in_dim` | 22 | 0.008 | 0.002 |
| `add` | 23 | 0.008 | 0.001 |
| `subtract` | 14 | 0.005 | 0.000 |
| `matmul` | 8 | 0.005 | 0.002 |
| `norm` | 2 | 0.005 | 0.003 |
| `true_divide` | 13 | 0.005 | 0.001 |
| `cholesky` | 2 | 0.005 | 0.002 |
| `_moveaxis` | 23 | 0.004 | 0.000 |
| `_accum` | 1 | 0.004 | 0.004 |
| `conjugate` | 15 | 0.003 | 0.000 |
| `negative` | 16 | 0.003 | 0.000 |
| `_take` | 1 | 0.003 | 0.003 |
| `abs` | 5 | 0.003 | 0.001 |
| `less` | 7 | 0.003 | 0.001 |
| `_svd_replicated` | 1 | 0.002 | 0.002 |
| `_reduce_max` | 5 | 0.002 | 0.001 |
| `gather` | 8 | 0.002 | 0.000 |
| `clip` | 2 | 0.002 | 0.001 |
| `_broadcast_arrays` | 11 | 0.002 | 0.000 |
| `_reduce_sum` | 4 | 0.002 | 0.001 |

## Tracing cache misses

Total: **134** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 8 | never seen function: fft id=140056964072448 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:655:37` | 6 | for sort defined at /opt/jax/jax/_src/numpy/sorting.py:31 never seen input type signature: a: f64[181,960] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140053517483552 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:415:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140254309512224 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:189:8` | 4 | never seen function: convert_element_type id=140053518053792 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:193:10` | 4 | never seen function: convert_element_type id=140053517852064 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:574:17` | 4 | never seen function: h_transform.<locals>._build_fH id=140052985386208 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/ |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:194:44` | 3 | never seen function: iota id=139735687925248 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:599:8` | 3 | never seen function: broadcast_in_dim id=140254307894432 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:653:19` | 3 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 6 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140257156717664 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140257165755872 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[16,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140056872229984 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=140254312010816 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=139739312389824 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=140056962494080 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:156:8` | 2 | never seen function: _psum id=140257127514496 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:259:14` | 2 | never seen function: dynamic_slice id=140256862300064 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:272:15` | 2 | never seen function: convert_element_type id=140255110625888 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:281:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:282:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:283:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:584:45` | 2 | never seen function: gather id=139735155791552 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:584:25` | 2 | never seen function: eigh id=139739356358240 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:585:28` | 2 | never seen function: dynamic_slice id=140254311710080 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:586:39` | 2 | never seen function: gather id=139735155496480 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:589:41` | 2 | never seen function: gather id=140256858041024 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:595:39` | 2 | never seen function: diagonal id=140257157076352 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:390:10` | 2 | never seen function: iota id=140254311229664 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
