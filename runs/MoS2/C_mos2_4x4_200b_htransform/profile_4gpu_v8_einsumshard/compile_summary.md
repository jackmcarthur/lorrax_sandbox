# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v8_einsumshard/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 318 | 0.314 | 0.046 |
| jaxpr→MLIR | 90 | 0.505 | 0.126 |
| XLA compile | 90 | 7.983 | 0.603 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_diag_stats` | 1 | 0.603 | 0.603 |
| `transpose` | 3 | 0.600 | 0.452 |
| `broadcast_in_dim` | 12 | 0.474 | 0.067 |
| `_fft_gather_reshard` | 1 | 0.428 | 0.428 |
| `dynamic_slice` | 5 | 0.407 | 0.170 |
| `subtract` | 5 | 0.320 | 0.076 |
| `_kpath_batch` | 1 | 0.290 | 0.290 |
| `_fft_and_rslice` | 1 | 0.287 | 0.287 |
| `_post_kpath` | 1 | 0.254 | 0.254 |
| `true_divide` | 4 | 0.243 | 0.063 |
| `add` | 4 | 0.236 | 0.063 |
| `iota` | 4 | 0.230 | 0.067 |
| `_build_fH` | 1 | 0.221 | 0.221 |
| `_build_S_chol` | 1 | 0.221 | 0.221 |
| `_gamma_rt` | 1 | 0.217 | 0.217 |
| `_finalize` | 1 | 0.198 | 0.198 |
| `cholesky` | 1 | 0.187 | 0.187 |
| `_reduce_max` | 2 | 0.184 | 0.120 |
| `_identity_fn` | 3 | 0.184 | 0.125 |
| `_argmin` | 2 | 0.173 | 0.086 |
| `_multi_slice` | 3 | 0.172 | 0.059 |
| `convert_element_type` | 5 | 0.167 | 0.060 |
| `reshape` | 3 | 0.165 | 0.101 |
| `_reduce_min` | 2 | 0.141 | 0.072 |
| `gather` | 2 | 0.139 | 0.075 |
| `_accum` | 1 | 0.136 | 0.136 |
| `_svd_replicated` | 1 | 0.124 | 0.124 |
| `multiply` | 2 | 0.118 | 0.062 |
| `squeeze` | 4 | 0.104 | 0.027 |
| `_fun_jit` | 1 | 0.098 | 0.098 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_post_kpath` | 1 | 0.046 | 0.046 |
| `_fun_jit` | 2 | 0.032 | 0.017 |
| `fft_impl` | 9 | 0.018 | 0.004 |
| `_fft_gather_reshard` | 1 | 0.017 | 0.017 |
| `_build_S_chol` | 1 | 0.016 | 0.016 |
| `_build_fH` | 1 | 0.016 | 0.016 |
| `_kpath_batch` | 1 | 0.012 | 0.012 |
| `_fft_and_rslice` | 1 | 0.010 | 0.010 |
| `multiply` | 28 | 0.009 | 0.001 |
| `_diag_stats` | 1 | 0.008 | 0.008 |
| `_diag` | 1 | 0.008 | 0.008 |
| `diagonal` | 1 | 0.007 | 0.007 |
| `_finalize` | 1 | 0.007 | 0.007 |
| `add` | 22 | 0.007 | 0.001 |
| `true_divide` | 13 | 0.007 | 0.001 |
| `_where` | 8 | 0.006 | 0.001 |
| `_gamma_rt` | 1 | 0.005 | 0.005 |
| `broadcast_in_dim` | 12 | 0.005 | 0.002 |
| `_moveaxis` | 23 | 0.004 | 0.000 |
| `subtract` | 12 | 0.004 | 0.000 |
| `cholesky` | 2 | 0.004 | 0.002 |
| `_accum` | 1 | 0.003 | 0.003 |
| `_dfun_jit` | 1 | 0.003 | 0.003 |
| `matmul` | 6 | 0.003 | 0.001 |
| `_take` | 1 | 0.003 | 0.003 |
| `abs` | 5 | 0.003 | 0.001 |
| `negative` | 16 | 0.003 | 0.000 |
| `conjugate` | 14 | 0.003 | 0.000 |
| `clip` | 2 | 0.002 | 0.002 |
| `norm` | 1 | 0.002 | 0.002 |

## Tracing cache misses

Total: **105** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 8 | never seen function: fft id=140181233697792 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140177471673248 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:415:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=139777905587072 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:198:8` | 4 | never seen function: convert_element_type id=140177937746656 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:596:17` | 4 | never seen function: h_transform.<locals>._build_fH id=140177937395968 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/ |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:617:41` | 4 | never seen function: convert_element_type id=140177935538176 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:731:41` | 4 | never seen function: dynamic_slice id=140177470218752 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140181223726176 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140181234845152 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[16,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140180621847552 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=140177471431648 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140180621840032 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=140181232135808 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:156:8` | 2 | never seen function: _psum id=140340470695232 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:263:14` | 2 | never seen function: dynamic_slice id=140180620901440 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:284:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:285:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:286:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:292:11` | 2 | never seen function: _fun_jit id=140180620894400 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:273 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:607:25` | 2 | never seen function: eigh id=140181230184032 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:627:43` | 2 | never seen function: diagonal id=140181229884800 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:673:9` | 2 | never seen function: convert_element_type id=140177472236704 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:343:24` | 1 | never seen function: transpose id=140180621840192 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:846:19` | 1 | never seen function: broadcast_in_dim id=140180620995744 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:686:20` | 1 | never seen function: broadcast_in_dim id=140177471429888 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:879:26` | 1 | never seen function: scatter id=140337257682944 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:880:27` | 1 | for scatter defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[16,480,60,2],  args[1]: i32[0],  args |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:98:8` | 1 | never seen function: reshape id=139777905238528 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:104:17` | 1 | never seen function: svd id=140181230059200 defined at /opt/jax/jax/_src/numpy/linalg.py:199 |

## Persistent cache misses

_None._
