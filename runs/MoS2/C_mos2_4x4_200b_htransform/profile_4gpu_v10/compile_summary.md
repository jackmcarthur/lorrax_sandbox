# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v10/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 330 | 0.322 | 0.046 |
| jaxpr→MLIR | 93 | 0.513 | 0.103 |
| XLA compile | 93 | 7.705 | 0.432 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 13 | 0.503 | 0.060 |
| `_post_kpath` | 2 | 0.481 | 0.251 |
| `_fft_gather_reshard` | 1 | 0.432 | 0.432 |
| `_diag_eig_at_gamma` | 1 | 0.350 | 0.350 |
| `subtract` | 5 | 0.298 | 0.072 |
| `_fft_and_rslice` | 1 | 0.283 | 0.283 |
| `_kpath_batch` | 1 | 0.274 | 0.274 |
| `_diag_stats_fast` | 1 | 0.271 | 0.271 |
| `_identity_fn` | 4 | 0.270 | 0.107 |
| `true_divide` | 4 | 0.259 | 0.074 |
| `_build_fH` | 1 | 0.234 | 0.234 |
| `iota` | 4 | 0.228 | 0.063 |
| `_gamma_rt` | 1 | 0.223 | 0.223 |
| `dynamic_slice` | 4 | 0.221 | 0.057 |
| `add` | 4 | 0.220 | 0.057 |
| `_build_S_chol` | 1 | 0.210 | 0.210 |
| `_finalize` | 1 | 0.205 | 0.205 |
| `gather` | 3 | 0.192 | 0.072 |
| `_multi_slice` | 3 | 0.188 | 0.070 |
| `_reduce_max` | 2 | 0.186 | 0.123 |
| `cholesky` | 1 | 0.178 | 0.178 |
| `convert_element_type` | 5 | 0.172 | 0.056 |
| `reshape` | 3 | 0.164 | 0.101 |
| `_argmin` | 2 | 0.156 | 0.078 |
| `transpose` | 2 | 0.137 | 0.072 |
| `_accum` | 1 | 0.134 | 0.134 |
| `_reduce_min` | 2 | 0.130 | 0.069 |
| `norm` | 2 | 0.125 | 0.064 |
| `_svd_replicated` | 1 | 0.122 | 0.122 |
| `multiply` | 2 | 0.121 | 0.065 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_post_kpath` | 1 | 0.046 | 0.046 |
| `_fun_jit` | 2 | 0.032 | 0.016 |
| `_fft_gather_reshard` | 1 | 0.017 | 0.017 |
| `fft_impl` | 9 | 0.017 | 0.004 |
| `_build_S_chol` | 1 | 0.016 | 0.016 |
| `_build_fH` | 1 | 0.015 | 0.015 |
| `_kpath_batch` | 1 | 0.012 | 0.012 |
| `_fft_and_rslice` | 1 | 0.012 | 0.012 |
| `multiply` | 29 | 0.009 | 0.001 |
| `_diag` | 1 | 0.008 | 0.008 |
| `diagonal` | 1 | 0.007 | 0.007 |
| `add` | 23 | 0.007 | 0.001 |
| `broadcast_in_dim` | 15 | 0.007 | 0.002 |
| `true_divide` | 13 | 0.006 | 0.002 |
| `_finalize` | 1 | 0.006 | 0.006 |
| `_where` | 8 | 0.006 | 0.001 |
| `norm` | 2 | 0.005 | 0.003 |
| `_gamma_rt` | 1 | 0.005 | 0.005 |
| `_diag_eig_at_gamma` | 1 | 0.005 | 0.005 |
| `_moveaxis` | 23 | 0.004 | 0.000 |
| `cholesky` | 2 | 0.004 | 0.002 |
| `subtract` | 12 | 0.004 | 0.000 |
| `negative` | 16 | 0.004 | 0.001 |
| `_accum` | 1 | 0.003 | 0.003 |
| `_dfun_jit` | 1 | 0.003 | 0.003 |
| `_take` | 1 | 0.003 | 0.003 |
| `_diag_stats_fast` | 1 | 0.003 | 0.003 |
| `conjugate` | 15 | 0.003 | 0.000 |
| `abs` | 5 | 0.003 | 0.001 |
| `matmul` | 6 | 0.003 | 0.001 |

## Tracing cache misses

Total: **107** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 8 | never seen function: fft id=139639071785984 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=139637037322528 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:415:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140669445648640 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:198:8` | 4 | never seen function: convert_element_type id=139637038631008 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:587:17` | 4 | never seen function: h_transform.<locals>._build_fH id=139635697774208 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/ |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:612:31` | 4 | never seen function: convert_element_type id=140143380138560 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=1405465923002026-04-25 20:50:48,497 jax._src.interpreters.px2026-04-25 20:50:48,497 jax._src.in |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=139639072916960 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[16,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140146527562112 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=139637038211424 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140672595348512 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=139639070224000 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:156:8` | 2 | never seen function: _psum id=140672599573184 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:263:14` | 2 | never seen function: dynamic_slice id=139638931204320 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:284:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:285:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:286:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:292:11` | 2 | never seen function: _fun_jit id=139638931037760 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:273 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:613:76` | 2 | never seen function: dynamic_slice id=139635698070400 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:604:25` | 2 | never seen function: eigh id=139639068304992 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:616:41` | 2 | never seen function: gather id=140669447311808 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:626:43` | 2 | never seen function: diagonal id=140672666408320 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:671:9` | 2 | never seen function: convert_element_type id=139635697355328 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:728:46` | 2 | never seen function: norm id=140672666762944 defined at /opt/jax/jax/_src/numpy/linalg.py:1076 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:343:24` | 1 | never seen function: transpose id=140672595348672 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:686:20` | 1 | never seen function: broadcast_in_dim id=140670045493888 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:879:26` | 1 | never seen function: scatter id=140143378175360 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:880:27` | 1 | for scatter defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[16,480,60,2],  args[1]: i32[0],  args |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:98:8` | 1 | never seen function: reshape id=139635695853920 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
