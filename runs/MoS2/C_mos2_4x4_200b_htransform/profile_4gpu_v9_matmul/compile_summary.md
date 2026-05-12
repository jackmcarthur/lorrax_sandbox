# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v9_matmul/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 319 | 0.388 | 0.048 |
| jaxpr→MLIR | 90 | 0.498 | 0.125 |
| XLA compile | 90 | 7.535 | 0.607 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_diag_stats` | 1 | 0.607 | 0.607 |
| `transpose` | 3 | 0.578 | 0.449 |
| `broadcast_in_dim` | 12 | 0.452 | 0.062 |
| `_fft_gather_reshard` | 1 | 0.428 | 0.428 |
| `subtract` | 5 | 0.295 | 0.072 |
| `_fft_and_rslice` | 1 | 0.279 | 0.279 |
| `_kpath_batch` | 1 | 0.275 | 0.275 |
| `dynamic_slice` | 5 | 0.274 | 0.059 |
| `true_divide` | 4 | 0.244 | 0.065 |
| `_post_kpath` | 1 | 0.223 | 0.223 |
| `_build_fH` | 1 | 0.222 | 0.222 |
| `add` | 4 | 0.220 | 0.057 |
| `_gamma_rt` | 1 | 0.218 | 0.218 |
| `iota` | 4 | 0.217 | 0.057 |
| `_build_S_chol` | 1 | 0.208 | 0.208 |
| `_finalize` | 1 | 0.200 | 0.200 |
| `_reduce_max` | 2 | 0.180 | 0.118 |
| `_multi_slice` | 3 | 0.176 | 0.060 |
| `cholesky` | 1 | 0.168 | 0.168 |
| `reshape` | 3 | 0.164 | 0.103 |
| `convert_element_type` | 5 | 0.162 | 0.057 |
| `_identity_fn` | 3 | 0.161 | 0.103 |
| `_argmin` | 2 | 0.154 | 0.080 |
| `_accum` | 1 | 0.131 | 0.131 |
| `_svd_replicated` | 1 | 0.131 | 0.131 |
| `_reduce_min` | 2 | 0.124 | 0.063 |
| `multiply` | 2 | 0.114 | 0.060 |
| `gather` | 2 | 0.111 | 0.056 |
| `squeeze` | 4 | 0.104 | 0.029 |
| `_fun_jit` | 1 | 0.089 | 0.089 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_post_kpath` | 1 | 0.048 | 0.048 |
| `_reduce_min` | 3 | 0.037 | 0.036 |
| `_reduce_max` | 5 | 0.034 | 0.032 |
| `_fun_jit` | 2 | 0.032 | 0.016 |
| `_fft_gather_reshard` | 1 | 0.018 | 0.018 |
| `fft_impl` | 9 | 0.018 | 0.004 |
| `_build_fH` | 1 | 0.016 | 0.016 |
| `_build_S_chol` | 1 | 0.015 | 0.015 |
| `_kpath_batch` | 1 | 0.012 | 0.012 |
| `_fft_and_rslice` | 1 | 0.011 | 0.011 |
| `multiply` | 28 | 0.009 | 0.001 |
| `_diag_stats` | 1 | 0.008 | 0.008 |
| `add` | 22 | 0.008 | 0.001 |
| `_diag` | 1 | 0.008 | 0.008 |
| `true_divide` | 13 | 0.007 | 0.002 |
| `diagonal` | 1 | 0.007 | 0.007 |
| `_finalize` | 1 | 0.007 | 0.007 |
| `_gamma_rt` | 1 | 0.006 | 0.006 |
| `_where` | 8 | 0.006 | 0.001 |
| `broadcast_in_dim` | 12 | 0.005 | 0.002 |
| `_moveaxis` | 23 | 0.004 | 0.000 |
| `subtract` | 12 | 0.004 | 0.000 |
| `cholesky` | 2 | 0.004 | 0.002 |
| `matmul` | 7 | 0.003 | 0.001 |
| `_accum` | 1 | 0.003 | 0.003 |
| `_take` | 1 | 0.003 | 0.003 |
| `_dfun_jit` | 1 | 0.003 | 0.003 |
| `negative` | 16 | 0.003 | 0.000 |
| `conjugate` | 14 | 0.003 | 0.000 |
| `abs` | 5 | 0.003 | 0.001 |

## Tracing cache misses

Total: **105** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 8 | never seen function: fft id=140712843133952 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140711115346528 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:415:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140173577371168 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:198:8` | 4 | never seen function: convert_element_type id=140711786564000 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:591:17` | 4 | never seen function: h_transform.<locals>._build_fH id=140710383710144 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/ |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:612:41` | 4 | never seen function: convert_element_type id=140710383965888 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:726:41` | 4 | never seen function: dynamic_slice id=139708041126400 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140712833195104 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140712844264928 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[16,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140712690805440 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=140711116169888 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140712690797920 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=140712841555584 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:156:8` | 2 | never seen function: _psum id=140712797340672 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:263:14` | 2 | never seen function: dynamic_slice id=140712795060992 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:284:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:285:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:286:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:292:11` | 2 | never seen function: _fun_jit id=140712795053952 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:273 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:602:25` | 2 | never seen function: eigh id=140712839652960 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:622:43` | 2 | never seen function: diagonal id=140712839320960 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:668:9` | 2 | never seen function: convert_element_type id=139708040129376 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:343:24` | 1 | never seen function: transpose id=139710724918784 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:846:19` | 1 | never seen function: broadcast_in_dim id=140712691526496 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:686:20` | 1 | never seen function: broadcast_in_dim id=140711116168128 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:879:26` | 1 | never seen function: scatter id=139709381269184 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:880:27` | 1 | for scatter defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[16,480,60,2],  args[1]: i32[0],  args |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:98:8` | 1 | never seen function: reshape id=140710381822624 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:104:17` | 1 | never seen function: svd id=140176428891840 defined at /opt/jax/jax/_src/numpy/linalg.py:199 |

## Persistent cache misses

_None._
