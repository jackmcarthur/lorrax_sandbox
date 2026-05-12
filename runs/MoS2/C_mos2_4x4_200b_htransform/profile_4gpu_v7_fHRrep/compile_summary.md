# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v7_fHRrep/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 319 | 0.353 | 0.050 |
| jaxpr→MLIR | 90 | 0.458 | 0.102 |
| XLA compile | 90 | 7.688 | 0.601 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_diag_stats` | 1 | 0.601 | 0.601 |
| `transpose` | 3 | 0.562 | 0.422 |
| `broadcast_in_dim` | 12 | 0.474 | 0.061 |
| `_fft_gather_reshard` | 1 | 0.455 | 0.455 |
| `subtract` | 5 | 0.305 | 0.070 |
| `dynamic_slice` | 5 | 0.292 | 0.068 |
| `_fft_and_rslice` | 1 | 0.279 | 0.279 |
| `_kpath_batch` | 1 | 0.274 | 0.274 |
| `_post_kpath` | 1 | 0.247 | 0.247 |
| `true_divide` | 4 | 0.244 | 0.063 |
| `iota` | 4 | 0.223 | 0.059 |
| `add` | 4 | 0.222 | 0.058 |
| `_build_fH` | 1 | 0.218 | 0.218 |
| `_gamma_rt` | 1 | 0.217 | 0.217 |
| `_build_S_chol` | 1 | 0.207 | 0.207 |
| `_finalize` | 1 | 0.207 | 0.207 |
| `_reduce_max` | 2 | 0.181 | 0.119 |
| `_multi_slice` | 3 | 0.176 | 0.061 |
| `cholesky` | 1 | 0.174 | 0.174 |
| `convert_element_type` | 5 | 0.172 | 0.055 |
| `reshape` | 3 | 0.171 | 0.110 |
| `_identity_fn` | 3 | 0.171 | 0.103 |
| `_argmin` | 2 | 0.157 | 0.079 |
| `_accum` | 1 | 0.139 | 0.139 |
| `_reduce_min` | 2 | 0.132 | 0.072 |
| `_svd_replicated` | 1 | 0.124 | 0.124 |
| `multiply` | 2 | 0.119 | 0.063 |
| `gather` | 2 | 0.119 | 0.060 |
| `squeeze` | 4 | 0.107 | 0.029 |
| `_fun_jit` | 1 | 0.088 | 0.088 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_post_kpath` | 1 | 0.050 | 0.050 |
| `_reduce_min` | 3 | 0.034 | 0.033 |
| `_fun_jit` | 2 | 0.033 | 0.017 |
| `fft_impl` | 9 | 0.018 | 0.004 |
| `_fft_gather_reshard` | 1 | 0.018 | 0.018 |
| `_build_S_chol` | 1 | 0.016 | 0.016 |
| `_build_fH` | 1 | 0.015 | 0.015 |
| `_kpath_batch` | 1 | 0.012 | 0.012 |
| `_fft_and_rslice` | 1 | 0.012 | 0.012 |
| `multiply` | 28 | 0.009 | 0.001 |
| `_diag_stats` | 1 | 0.008 | 0.008 |
| `_diag` | 1 | 0.008 | 0.008 |
| `diagonal` | 1 | 0.007 | 0.007 |
| `true_divide` | 13 | 0.007 | 0.001 |
| `add` | 22 | 0.007 | 0.001 |
| `_finalize` | 1 | 0.006 | 0.006 |
| `_where` | 8 | 0.006 | 0.001 |
| `_gamma_rt` | 1 | 0.005 | 0.005 |
| `broadcast_in_dim` | 12 | 0.005 | 0.002 |
| `_moveaxis` | 23 | 0.004 | 0.000 |
| `subtract` | 12 | 0.004 | 0.000 |
| `cholesky` | 2 | 0.004 | 0.002 |
| `_accum` | 1 | 0.004 | 0.004 |
| `_dfun_jit` | 1 | 0.003 | 0.003 |
| `_take` | 1 | 0.003 | 0.003 |
| `matmul` | 6 | 0.003 | 0.001 |
| `negative` | 16 | 0.003 | 0.000 |
| `abs` | 5 | 0.003 | 0.001 |
| `conjugate` | 14 | 0.003 | 0.000 |
| `clip` | 2 | 0.003 | 0.002 |

## Tracing cache misses

Total: **104** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 8 | never seen function: fft id=140207206925312 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140203908354656 defined at /2026-04-25 20:42:44,148 jax._src |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:415:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140224915894848 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:198:8` | 4 | never seen function: convert_element_type id=139748573383072 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:586:17` | 4 | never seen function: h_transform.<locals>._build_fH id=140225918185408 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/ |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:607:41` | 4 | never seen function: convert_element_type id=140203909142208 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:721:41` | 4 | never seen function: dynamic_slice id=140203906932928 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140207196953696 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[16,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140206592092864 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=140224918315680 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140206592085344 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=140207205346944 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:156:8` | 2 | never seen function: _psum id=140207127512064 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:263:14` | 2 | never seen function: dynamic_slice id=139749231819520 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:284:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:285:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:286:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:292:11` | 2 | never seen function: _fun_jit id=140206591123328 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:273 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:597:25` | 2 | never seen function: eigh id=140207203444320 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:617:43` | 2 | never seen function: diagonal id=140207203112320 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:663:9` | 2 | never seen function: convert_element_type id=140203910032224 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:343:24` | 1 | never seen function: transpose id=140206592085504 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:846:19` | 1 | never seen function: broadcast_in_dim id=140224919087968 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 1 | never seen function: _multi_slice id=140228283232672 defined at /opt/jax/jax/_src/numpy/array_methods.py:616 but seen another function defin |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:686:20` | 1 | never seen function: broadcast_in_dim id=140203908144064 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:879:26` | 1 | never seen function: scatter id=140203911730880 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:880:27` | 1 | for scatter defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[16,480,60,2],  args[1]: i32[0],  args |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:98:8` | 1 | never seen function: reshape id=139746562400928 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:104:17` | 1 | never seen function: svd id=140228282678976 defined at /opt/jax/jax/_src/numpy/linalg.py:199 |

## Persistent cache misses

_None._
