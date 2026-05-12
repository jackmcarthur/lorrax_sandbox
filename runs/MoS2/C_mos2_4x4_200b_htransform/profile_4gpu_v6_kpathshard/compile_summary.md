# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v6_kpathshard/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 316 | 0.351 | 0.047 |
| jaxpr→MLIR | 89 | 0.444 | 0.099 |
| XLA compile | 88 | 7.325 | 0.631 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_diag_stats` | 1 | 0.631 | 0.631 |
| `broadcast_in_dim` | 12 | 0.498 | 0.065 |
| `_fft_gather_reshard` | 1 | 0.443 | 0.443 |
| `_kpath_batch` | 1 | 0.325 | 0.325 |
| `subtract` | 5 | 0.297 | 0.066 |
| `dynamic_slice` | 5 | 0.283 | 0.060 |
| `_fft_and_rslice` | 1 | 0.281 | 0.281 |
| `true_divide` | 4 | 0.252 | 0.067 |
| `_post_kpath` | 1 | 0.235 | 0.235 |
| `iota` | 4 | 0.235 | 0.062 |
| `_gamma_rt` | 1 | 0.229 | 0.229 |
| `_build_fH` | 1 | 0.227 | 0.227 |
| `add` | 4 | 0.226 | 0.060 |
| `_build_S_chol` | 1 | 0.222 | 0.222 |
| `_finalize` | 1 | 0.212 | 0.212 |
| `_reduce_max` | 2 | 0.182 | 0.120 |
| `_multi_slice` | 3 | 0.177 | 0.062 |
| `convert_element_type` | 5 | 0.175 | 0.060 |
| `cholesky` | 1 | 0.174 | 0.174 |
| `reshape` | 3 | 0.165 | 0.102 |
| `_argmin` | 2 | 0.155 | 0.078 |
| `transpose` | 2 | 0.143 | 0.076 |
| `_accum` | 1 | 0.138 | 0.138 |
| `_reduce_min` | 2 | 0.135 | 0.072 |
| `gather` | 2 | 0.122 | 0.062 |
| `multiply` | 2 | 0.121 | 0.064 |
| `_svd_replicated` | 1 | 0.121 | 0.121 |
| `squeeze` | 4 | 0.111 | 0.028 |
| `_fun_jit` | 1 | 0.091 | 0.091 |
| `abs` | 1 | 0.085 | 0.085 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_post_kpath` | 1 | 0.047 | 0.047 |
| `add` | 22 | 0.042 | 0.036 |
| `_fun_jit` | 2 | 0.033 | 0.017 |
| `_fft_gather_reshard` | 1 | 0.018 | 0.018 |
| `fft_impl` | 9 | 0.018 | 0.004 |
| `_build_S_chol` | 1 | 0.016 | 0.016 |
| `_build_fH` | 1 | 0.015 | 0.015 |
| `_kpath_batch` | 1 | 0.012 | 0.012 |
| `_fft_and_rslice` | 1 | 0.011 | 0.011 |
| `multiply` | 28 | 0.009 | 0.001 |
| `_diag_stats` | 1 | 0.008 | 0.008 |
| `_diag` | 1 | 0.008 | 0.008 |
| `diagonal` | 1 | 0.007 | 0.007 |
| `true_divide` | 13 | 0.007 | 0.002 |
| `_where` | 8 | 0.006 | 0.001 |
| `_finalize` | 1 | 0.006 | 0.006 |
| `_gamma_rt` | 1 | 0.005 | 0.005 |
| `subtract` | 12 | 0.005 | 0.002 |
| `broadcast_in_dim` | 12 | 0.005 | 0.002 |
| `_moveaxis` | 23 | 0.004 | 0.000 |
| `cholesky` | 2 | 0.004 | 0.002 |
| `_accum` | 1 | 0.004 | 0.004 |
| `_dfun_jit` | 1 | 0.003 | 0.003 |
| `_take` | 1 | 0.003 | 0.003 |
| `conjugate` | 14 | 0.003 | 0.000 |
| `abs` | 5 | 0.003 | 0.001 |
| `negative` | 16 | 0.003 | 0.000 |
| `matmul` | 6 | 0.003 | 0.001 |
| `norm` | 1 | 0.002 | 0.002 |
| `clip` | 2 | 0.002 | 0.002 |

## Tracing cache misses

Total: **104** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 8 | never seen function: fft id=140599310500864 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140596164563872 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:415:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140677698286464 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:198:8` | 4 | never seen function: convert_element_type id=140596694944480 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:586:17` | 4 | never seen function: h_transform.<locals>._build_fH id=140677701693696 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/ |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:607:41` | 4 | never seen function: convert_element_type id=140677698787328 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:724:41` | 4 | never seen function: dynamic_slice id=140596160355744 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=1403863913012026-04-25 20:39:39,419 jax._src.interpreters.px2026-04-25 20:39:39,419 jax._src.in |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140158701340128 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[16,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140598849661952 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=140596165338080 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140598849654432 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=140599308922496 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:156:8` | 2 | never seen function: _psum id=140680581989696 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:263:14` | 2 | never seen function: dynamic_slice id=140680313650240 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:284:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:285:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:286:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:292:11` | 2 | never seen function: _fun_jit id=140680313643200 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:273 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:597:25` | 2 | never seen function: eigh id=140599306987104 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:617:43` | 2 | never seen function: diagonal id=140599306687872 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:666:9` | 2 | never seen function: convert_element_type id=140155325257568 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:343:24` | 1 | never seen function: transpose id=140385644138304 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:846:19` | 1 | never seen function: broadcast_in_dim id=140155857431712 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:686:20` | 1 | never seen function: broadcast_in_dim id=140677700721920 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:879:26` | 1 | never seen function: scatter id=140677697951744 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:880:27` | 1 | for scatter defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[16,480,60,2],  args[1]: i32[0],  args |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:98:8` | 1 | never seen function: reshape id=140677697954784 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:104:17` | 1 | never seen function: svd id=140680615608000 defined at /opt/jax/jax/_src/numpy/linalg.py:199 |

## Persistent cache misses

_None._
