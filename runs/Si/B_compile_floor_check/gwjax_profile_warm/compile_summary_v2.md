# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile_warm/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 692 | 0.527 | 0.054 |
| jaxpr→MLIR | 333 | 0.982 | 0.154 |
| XLA compile | 369 | 10.157 | 0.694 |

## Top 60 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 75 | 0.931 | 0.149 |
| `_kernel` | 1 | 0.694 | 0.694 |
| `convert_element_type` | 37 | 0.691 | 0.281 |
| `sigma_sx` | 2 | 0.666 | 0.336 |
| `gather` | 26 | 0.627 | 0.039 |
| `true_divide` | 12 | 0.583 | 0.089 |
| `add` | 22 | 0.534 | 0.039 |
| `sigma_coh` | 1 | 0.364 | 0.364 |
| `concatenate` | 16 | 0.364 | 0.068 |
| `multiply` | 13 | 0.361 | 0.040 |
| `minimax_tau_integrate_chi` | 1 | 0.341 | 0.341 |
| `subtract` | 9 | 0.328 | 0.041 |
| `iota` | 12 | 0.249 | 0.037 |
| `select_n` | 9 | 0.245 | 0.037 |
| `eigh` | 3 | 0.238 | 0.118 |
| `_local_fft` | 2 | 0.235 | 0.121 |
| `less` | 8 | 0.212 | 0.041 |
| `_compute_CCT_LR` | 1 | 0.211 | 0.211 |
| `transpose` | 11 | 0.202 | 0.036 |
| `get_sqrt_v_and_phase` | 2 | 0.189 | 0.177 |
| `_mean` | 3 | 0.187 | 0.086 |
| `_einsum` | 5 | 0.156 | 0.057 |
| `matmul` | 6 | 0.126 | 0.034 |
| `_argmin` | 2 | 0.124 | 0.063 |
| `_multi_slice` | 3 | 0.109 | 0.039 |
| `norm` | 2 | 0.095 | 0.048 |
| `_where` | 2 | 0.084 | 0.047 |
| `_per_rank` | 7 | 0.081 | 0.017 |
| `_diag` | 2 | 0.076 | 0.038 |
| `conjugate` | 4 | 0.076 | 0.037 |
| `equal` | 4 | 0.072 | 0.035 |
| `_broadcast_arrays` | 6 | 0.068 | 0.035 |
| `swapaxes` | 2 | 0.064 | 0.061 |
| `inv` | 1 | 0.063 | 0.063 |
| `_psum` | 14 | 0.058 | 0.010 |
| `dynamic_slice` | 3 | 0.048 | 0.036 |
| `abs` | 1 | 0.046 | 0.046 |
| `_reduce_sum` | 2 | 0.045 | 0.043 |
| `reshape` | 8 | 0.045 | 0.009 |
| `maximum` | 2 | 0.038 | 0.035 |
| `greater` | 2 | 0.036 | 0.034 |
| `clip` | 2 | 0.034 | 0.032 |
| `greater_equal` | 1 | 0.033 | 0.033 |
| `sum` | 3 | 0.028 | 0.010 |
| `_identity_fn` | 4 | 0.022 | 0.008 |
| `_compute_P_traced` | 1 | 0.012 | 0.012 |
| `_moveaxis` | 2 | 0.012 | 0.008 |
| `scatter` | 2 | 0.011 | 0.008 |
| `squeeze` | 2 | 0.011 | 0.008 |
| `_solve_w` | 1 | 0.009 | 0.009 |
| `hartree` | 1 | 0.005 | 0.005 |
| `_squeeze` | 2 | 0.005 | 0.003 |
| `_take` | 1 | 0.004 | 0.004 |
| `trace` | 1 | 0.003 | 0.003 |
| `real` | 1 | 0.003 | 0.003 |
| `copy` | 1 | 0.002 | 0.002 |

## Top 60 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 1 | 0.054 | 0.054 |
| `_fft_and_rslice` | 2 | 0.042 | 0.021 |
| `_solve_w` | 1 | 0.034 | 0.034 |
| `get_sqrt_v_and_phase` | 1 | 0.026 | 0.026 |
| `broadcast_in_dim` | 88 | 0.022 | 0.002 |
| `multiply` | 47 | 0.019 | 0.002 |
| `fft_impl` | 12 | 0.019 | 0.004 |
| `solve` | 1 | 0.017 | 0.017 |
| `_per_rank` | 7 | 0.016 | 0.008 |
| `add` | 49 | 0.016 | 0.001 |
| `_psum` | 15 | 0.016 | 0.002 |
| `_reduce_sum` | 27 | 0.014 | 0.001 |
| `_where` | 17 | 0.014 | 0.002 |
| `_take` | 4 | 0.013 | 0.003 |
| `_compute_CCT_LR` | 1 | 0.012 | 0.012 |
| `_moveaxis` | 47 | 0.011 | 0.000 |
| `true_divide` | 29 | 0.010 | 0.001 |
| `_lu_solve` | 4 | 0.009 | 0.003 |
| `convert_element_type` | 37 | 0.009 | 0.001 |
| `_right_ifft_mul_fft` | 1 | 0.009 | 0.009 |
| `less` | 21 | 0.008 | 0.001 |
| `_broadcast_arrays` | 28 | 0.008 | 0.001 |
| `_einsum` | 13 | 0.008 | 0.001 |
| `_left_ifft_conj` | 1 | 0.007 | 0.007 |
| `eigh` | 3 | 0.006 | 0.002 |
| `_solve_all_at_once` | 1 | 0.005 | 0.005 |
| `hartree` | 1 | 0.005 | 0.005 |
| `norm` | 2 | 0.005 | 0.003 |
| `conjugate` | 22 | 0.005 | 0.001 |
| `concatenate` | 19 | 0.005 | 0.000 |
| `gather` | 18 | 0.004 | 0.000 |
| `remainder` | 2 | 0.004 | 0.003 |
| `trace` | 1 | 0.004 | 0.004 |
| `_local_fft` | 1 | 0.004 | 0.004 |
| `floor_divide` | 1 | 0.004 | 0.004 |
| `sigma_coh` | 1 | 0.003 | 0.003 |
| `matmul` | 6 | 0.003 | 0.001 |
| `lu_solve` | 1 | 0.003 | 0.003 |
| `equal` | 10 | 0.003 | 0.001 |
| `clip` | 2 | 0.003 | 0.001 |
| `subtract` | 9 | 0.003 | 0.000 |
| `iota` | 13 | 0.003 | 0.000 |
| `fft` | 11 | 0.003 | 0.000 |
| `negative` | 11 | 0.003 | 0.000 |
| `cholesky` | 1 | 0.003 | 0.003 |
| `sum` | 3 | 0.002 | 0.001 |
| `_accum` | 1 | 0.002 | 0.002 |
| `_mean` | 2 | 0.002 | 0.001 |
| `select_n` | 9 | 0.002 | 0.000 |
| `reshape` | 10 | 0.002 | 0.000 |
| `sqrt` | 6 | 0.002 | 0.001 |
| `exp` | 5 | 0.002 | 0.000 |
| `greater` | 4 | 0.002 | 0.001 |
| `transpose` | 7 | 0.001 | 0.000 |
| `_solve_triangular` | 3 | 0.001 | 0.001 |
| `_compute_P_traced` | 1 | 0.001 | 0.001 |
| `_squeeze` | 8 | 0.001 | 0.000 |
| `maximum` | 3 | 0.001 | 0.000 |
| `dynamic_slice` | 3 | 0.001 | 0.000 |
| `_multi_slice` | 4 | 0.001 | 0.000 |

## Tracing cache misses

Total: **321** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:43:12` | 17 | never seen function: add id=140358659771744 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/head_correction.py:424:10` | 15 | never seen function: convert_element_type id=140356380450528 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:24:20` | 9 | never seen function: broadcast_in_dim id=140360274556256 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:24:10` | 8 | never seen function: broadcast_in_dim id=140360274556576 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:30:24` | 8 | never seen function: add id=140112169585056 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/wavefunction_bundle.py:144:15` | 8 | never seen function: gather id=140358660059712 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/wavefunction_bundle.py:150:15` | 8 | never seen function: gather id=140358660163680 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1805:45` | 7 | never seen function: broadcast_in_dim id=140358662621120 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:46:10` | 7 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[52],  args[1]: i64[] closest seen input ty |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:214:31` | 6 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[4,4,480,480,4] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:212:8` | 6 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=140112170755680 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1805:15` | 6 | never seen function: broadcast_in_dim id=140358662042656 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:45:10` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[8],  args[1]: i64[] closest seen input typ |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:385:18` | 5 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=140360274367232 defined at /global/homes/j/jackm/software/lorrax_B/src/fil |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=140367433369216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:195:12` | 4 | never seen function: _psum id=140288629368256 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:83:19` | 4 | never seen function: _make_jittable_local_fft.<locals>._local_fft id=140111770806336 defined at /global/homes/j/jackm/software/lorrax_B/src/ |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:720:22` | 4 | never seen function: _where id=140117622779520 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:382:21` | 4 | never seen function: convert_element_type id=140360274370912 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/gw_init.py:677:40` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:375:11` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/head_correction.py:425:12` | 4 | never seen function: broadcast_in_dim id=140356381001504 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/head_correction.py:425:30` | 4 | never seen function: broadcast_in_dim id=140356381003264 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/head_correction.py:426:11` | 4 | never seen function: broadcast_in_dim id=140356381004864 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:195:27` | 3 | never seen function: convert_element_type id=140585088349216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:196:12` | 3 | never seen function: sum id=140365724057280 defined at /opt/jax/jax/_src/numpy/reductions.py:240 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:27:8` | 3 | never seen function: broadcast_in_dim id=140112169689440 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:365:25` | 3 | never seen function: round id=140365717318976 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3451 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:386:12` | 3 | never seen function: gather id=140358659906976 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:496:16` | 3 | never seen function: convert_element_type id=140358660177280 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:272:84` | 3 | for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566 tracing context doesn't match, e.g. due to config or context manager closest s |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:313:23` | 3 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/gw_jax.py:560:19` | 3 | never seen function: eigh id=140365717792352 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:222:22` | 2 | never seen function: fft id=140117624357888 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140360272026-04-25 17:24:34,760 jax._src.compiler WARNING: P2026-04-25 17:24:34,761 jax._src.in |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140117625488864 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[64,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140112173464000 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140112173849216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:970:40` | 2 | never seen function: broadcast_in_dim id=140360274014944 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/cholesky_2d.py:61:12` | 2 | never seen function: reshape id=140360275041696 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/cholesky_2d.py:65:19` | 2 | never seen function: broadcast_in_dim id=140112171970656 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/cholesky_2d.py:161:22` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: c128[1,1,240,240],  y: c12 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/phdf5_wfn_reader.py:154:26` | 2 | never seen function: _where id=140365723430528 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/gvec_fft_box.py:120:19` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:1202:36` | 2 | never seen function: convert_element_type id=140360274015424 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:415:25` | 2 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140360271627264 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:416:25` | 2 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._reshard_rchunk id=140360271627104 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:808:11` | 2 | never seen function: _make_jittable_local_fft.<locals>._make_axis_wrapper.<locals>.fft_impl id=140111771034592 defined at /global/homes/j/ja |
| `/global/homes/j/jackm/software/lorrax_B/src/common/gpu_utils.py:41:12` | 2 | never seen function: convert_element_type id=140360271196256 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:286:17` | 2 | never seen function: broadcast_in_dim id=140360274546976 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:23:8` | 2 | never seen function: convert_element_type id=140360274552576 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:27:33` | 2 | never seen function: broadcast_in_dim id=140112169691680 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:28:9` | 2 | never seen function: norm id=140117617147584 defined at /opt/jax/jax/_src/numpy/linalg.py:1076 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/gw_init.py:677:30` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[480,480],  x: c128[480,480],  y:  |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:315:16` | 2 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: c128[4,480] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:239:36` | 2 | never seen function: gather id=140358661773312 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:385:12` | 2 | never seen function: gather id=140358659902336 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:487:10` | 2 | never seen function: convert_element_type id=140358659911136 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:249:40` | 2 | never seen function: _lu_solve id=140365716436000 defined at /opt/jax/jax/_src/lax/linalg.py:1566 |

## Persistent cache misses

_None._
