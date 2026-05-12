# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v4_rebased/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 362 | 0.258 | 0.035 |
| jaxpr→MLIR | 190 | 0.697 | 0.102 |
| XLA compile | 190 | 12.912 | 0.433 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 23 | 0.932 | 0.103 |
| `transpose` | 6 | 0.776 | 0.433 |
| `multiply` | 12 | 0.693 | 0.072 |
| `add` | 11 | 0.645 | 0.066 |
| `subtract` | 11 | 0.622 | 0.069 |
| `sort` | 4 | 0.530 | 0.228 |
| `_reduce_max` | 5 | 0.497 | 0.153 |
| `gather` | 8 | 0.461 | 0.067 |
| `iota` | 8 | 0.431 | 0.064 |
| `_fft_gather_reshard` | 1 | 0.418 | 0.418 |
| `reshape` | 7 | 0.351 | 0.104 |
| `true_divide` | 5 | 0.342 | 0.080 |
| `dynamic_slice` | 5 | 0.295 | 0.072 |
| `_kpath_batch` | 1 | 0.295 | 0.295 |
| `cholesky` | 2 | 0.293 | 0.174 |
| `_reduce_min` | 3 | 0.286 | 0.155 |
| `_where` | 5 | 0.280 | 0.060 |
| `_fft_and_rslice` | 1 | 0.280 | 0.280 |
| `abs` | 4 | 0.272 | 0.084 |
| `scan` | 2 | 0.257 | 0.133 |
| `matmul` | 4 | 0.255 | 0.137 |
| `convert_element_type` | 7 | 0.250 | 0.060 |
| `conjugate` | 4 | 0.239 | 0.064 |
| `_build_fH` | 1 | 0.216 | 0.216 |
| `exp` | 3 | 0.215 | 0.099 |
| `concatenate` | 3 | 0.199 | 0.084 |
| `_multi_slice` | 3 | 0.180 | 0.061 |
| `greater_equal` | 3 | 0.163 | 0.055 |
| `_argmin` | 2 | 0.156 | 0.079 |
| `eigvalsh` | 1 | 0.147 | 0.147 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `sort` | 4 | 0.035 | 0.035 |
| `_fft_gather_reshard` | 1 | 0.017 | 0.017 |
| `fft_impl` | 9 | 0.016 | 0.004 |
| `_build_fH` | 1 | 0.015 | 0.015 |
| `_kpath_batch` | 1 | 0.013 | 0.013 |
| `multiply` | 29 | 0.012 | 0.002 |
| `_fft_and_rslice` | 1 | 0.011 | 0.011 |
| `_where` | 10 | 0.008 | 0.001 |
| `broadcast_in_dim` | 22 | 0.008 | 0.002 |
| `add` | 23 | 0.008 | 0.001 |
| `_diag` | 1 | 0.007 | 0.007 |
| `diagonal` | 1 | 0.006 | 0.006 |
| `subtract` | 14 | 0.006 | 0.001 |
| `true_divide` | 13 | 0.005 | 0.001 |
| `norm` | 2 | 0.005 | 0.003 |
| `_moveaxis` | 23 | 0.005 | 0.000 |
| `matmul` | 8 | 0.004 | 0.001 |
| `negative` | 16 | 0.004 | 0.001 |
| `cholesky` | 2 | 0.004 | 0.002 |
| `_accum` | 1 | 0.003 | 0.003 |
| `conjugate` | 15 | 0.003 | 0.000 |
| `_take` | 1 | 0.003 | 0.003 |
| `abs` | 5 | 0.003 | 0.001 |
| `_svd_replicated` | 1 | 0.003 | 0.003 |
| `_reduce_max` | 5 | 0.003 | 0.001 |
| `_broadcast_arrays` | 11 | 0.002 | 0.000 |
| `less` | 7 | 0.002 | 0.001 |
| `clip` | 2 | 0.002 | 0.002 |
| `svd` | 1 | 0.002 | 0.002 |
| `_reduce_sum` | 4 | 0.002 | 0.001 |

## Tracing cache misses

Total: **134** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 8 | never seen function: fft id=140085090173952 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:655:37` | 6 | for sort defined at /opt/jax/jax/_src/numpy/sorting.py:31 never seen input type signature: a: f64[181,960] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140082976228672 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:415:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140557775697248 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:189:8` | 4 | never seen function: convert_element_type id=140081639633632 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:193:10` | 4 | never seen function: convert_element_type id=140084453105376 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:574:17` | 4 | never seen function: h_transform.<locals>._build_fH id=140557777278496 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/ |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:194:44` | 3 | never seen function: iota id=140084452609856 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:599:8` | 3 | never seen function: broadcast_in_dim id=140557774079456 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:653:19` | 3 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 6 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140241555053664 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140241564059104 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[16,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140241487961504 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=140082976396672 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140561597464640 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=140085088595584 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:156:8` | 2 | never seen function: _psum id=140085061567168 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:259:14` | 2 | never seen function: dynamic_slice id=140085056858336 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:272:15` | 2 | never seen function: convert_element_type id=140081640490624 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:281:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:282:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:283:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:584:45` | 2 | never seen function: gather id=140557776787456 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:584:25` | 2 | never seen function: eigh id=140085082973792 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:585:28` | 2 | never seen function: dynamic_slice id=140081641667968 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:586:39` | 2 | never seen function: gather id=140084929697312 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:589:41` | 2 | never seen function: gather id=140561597475840 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:595:39` | 2 | never seen function: diagonal id=140561900475776 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:390:10` | 2 | never seen function: iota id=140557776316960 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
