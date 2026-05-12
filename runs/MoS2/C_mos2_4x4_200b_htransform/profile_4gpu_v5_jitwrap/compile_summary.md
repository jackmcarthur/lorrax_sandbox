# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu_v5_jitwrap/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 323 | 0.319 | 0.050 |
| jaxpr→MLIR | 100 | 0.461 | 0.057 |
| XLA compile | 102 | 8.176 | 0.604 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_diag_stats` | 1 | 0.604 | 0.604 |
| `transpose` | 4 | 0.604 | 0.411 |
| `broadcast_in_dim` | 13 | 0.506 | 0.056 |
| `_fft_gather_reshard` | 1 | 0.423 | 0.423 |
| `_kpath_batch` | 1 | 0.296 | 0.296 |
| `reshape` | 5 | 0.292 | 0.100 |
| `subtract` | 5 | 0.288 | 0.068 |
| `add` | 5 | 0.285 | 0.064 |
| `_fft_and_rslice` | 1 | 0.283 | 0.283 |
| `dynamic_slice` | 5 | 0.277 | 0.062 |
| `true_divide` | 4 | 0.247 | 0.066 |
| `multiply` | 4 | 0.244 | 0.067 |
| `iota` | 4 | 0.223 | 0.061 |
| `_build_fH` | 1 | 0.218 | 0.218 |
| `_finalize` | 1 | 0.214 | 0.214 |
| `_build_S_chol` | 1 | 0.210 | 0.210 |
| `_post_kpath` | 1 | 0.199 | 0.199 |
| `matmul` | 2 | 0.189 | 0.130 |
| `_reduce_max` | 2 | 0.181 | 0.119 |
| `_multi_slice` | 3 | 0.174 | 0.061 |
| `cholesky` | 1 | 0.171 | 0.171 |
| `convert_element_type` | 5 | 0.170 | 0.055 |
| `gather` | 3 | 0.160 | 0.065 |
| `_argmin` | 2 | 0.150 | 0.076 |
| `_accum` | 1 | 0.135 | 0.135 |
| `_reduce_min` | 2 | 0.131 | 0.067 |
| `norm` | 2 | 0.126 | 0.063 |
| `_svd_replicated` | 1 | 0.125 | 0.125 |
| `conjugate` | 2 | 0.124 | 0.062 |
| `squeeze` | 4 | 0.105 | 0.028 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_post_kpath` | 1 | 0.050 | 0.050 |
| `_fun_jit` | 2 | 0.034 | 0.017 |
| `_fft_gather_reshard` | 1 | 0.018 | 0.018 |
| `fft_impl` | 9 | 0.018 | 0.004 |
| `_build_S_chol` | 1 | 0.015 | 0.015 |
| `_build_fH` | 1 | 0.015 | 0.015 |
| `_kpath_batch` | 1 | 0.012 | 0.012 |
| `_fft_and_rslice` | 1 | 0.011 | 0.011 |
| `multiply` | 28 | 0.010 | 0.001 |
| `_diag_stats` | 1 | 0.008 | 0.008 |
| `_diag` | 1 | 0.007 | 0.007 |
| `diagonal` | 1 | 0.007 | 0.007 |
| `add` | 22 | 0.007 | 0.001 |
| `true_divide` | 13 | 0.006 | 0.001 |
| `_where` | 8 | 0.006 | 0.001 |
| `_finalize` | 1 | 0.006 | 0.006 |
| `broadcast_in_dim` | 14 | 0.005 | 0.002 |
| `_moveaxis` | 23 | 0.004 | 0.000 |
| `subtract` | 12 | 0.004 | 0.000 |
| `cholesky` | 2 | 0.004 | 0.002 |
| `matmul` | 8 | 0.004 | 0.001 |
| `_accum` | 1 | 0.004 | 0.004 |
| `_dfun_jit` | 1 | 0.003 | 0.003 |
| `_take` | 1 | 0.003 | 0.003 |
| `abs` | 5 | 0.003 | 0.001 |
| `negative` | 16 | 0.003 | 0.000 |
| `conjugate` | 14 | 0.003 | 0.000 |
| `norm` | 1 | 0.003 | 0.003 |
| `_svd_replicated` | 1 | 0.002 | 0.002 |
| `clip` | 2 | 0.002 | 0.002 |

## Tracing cache misses

Total: **107** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 8 | never seen function: fft id=139687920010240 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=139684620538464 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:415:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140339271892512 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:198:8` | 4 | never seen function: convert_element_type id=140340070648224 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:586:17` | 4 | never seen function: h_transform.<locals>._build_fH id=140340070297536 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/ |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:607:41` | 4 | never seen function: convert_element_type id=140339269818048 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:701:41` | 4 | never seen function: dynamic_slice id=140339268093728 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=1401299958072026-04-25 20:28:56,951 jax._src.interpreters.px2026-04-25 20:28:56,951 jax._src.in |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=139687921157600 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[16,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=139687302916800 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=139684623835808 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=139817628454240 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=139687918431872 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:156:8` | 2 | never seen function: _psum id=139687841072128 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:263:14` | 2 | never seen function: dynamic_slice id=140341885745920 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:284:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:285:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:286:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:292:11` | 2 | never seen function: _fun_jit id=140341885738880 defined at /global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:273 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:597:25` | 2 | never seen function: eigh id=140342197102176 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:617:43` | 2 | never seen function: diagonal id=140342196786560 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:648:9` | 2 | never seen function: convert_element_type id=140339270689920 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:651:18` | 2 | never seen function: gather id=139687303399520 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:343:24` | 1 | never seen function: transpose id=139687302909440 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:686:20` | 1 | never seen function: broadcast_in_dim id=140339268836288 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:879:26` | 1 | never seen function: scatter id=140339272472256 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:880:27` | 1 | for scatter defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[16,480,60,2],  args[1]: i32[0],  args |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:98:8` | 1 | never seen function: reshape id=140339272475296 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:104:17` | 1 | never seen function: svd id=140342196960960 defined at /opt/jax/jax/_src/numpy/linalg.py:199 |

## Persistent cache misses

_None._
