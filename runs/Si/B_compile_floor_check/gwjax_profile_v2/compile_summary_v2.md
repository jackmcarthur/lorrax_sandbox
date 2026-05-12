# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile_v2/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 632 | 0.600 | 0.055 |
| jaxpr→MLIR | 238 | 0.800 | 0.128 |
| XLA compile | 248 | 8.293 | 0.694 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `sigma_sx` | 3 | 1.033 | 0.361 |
| `sigma_coh` | 2 | 0.726 | 0.367 |
| `_kernel` | 1 | 0.694 | 0.694 |
| `broadcast_in_dim` | 37 | 0.547 | 0.187 |
| `true_divide` | 12 | 0.475 | 0.090 |
| `_compute_CCT_LR` | 2 | 0.434 | 0.219 |
| `convert_element_type` | 24 | 0.426 | 0.271 |
| `minimax_tau_integrate_chi` | 1 | 0.353 | 0.353 |
| `gather` | 17 | 0.320 | 0.052 |
| `concatenate` | 14 | 0.307 | 0.070 |
| `_mean` | 8 | 0.269 | 0.088 |
| `eigh` | 3 | 0.243 | 0.120 |
| `get_sqrt_v_and_phase` | 2 | 0.197 | 0.183 |
| `hartree` | 2 | 0.190 | 0.183 |
| `subtract` | 6 | 0.171 | 0.036 |
| `add` | 7 | 0.162 | 0.039 |
| `_compute_S_omega_jit` | 1 | 0.152 | 0.152 |
| `multiply` | 6 | 0.149 | 0.037 |
| `_multi_slice` | 4 | 0.144 | 0.038 |
| `_expand_band_diagonal_to_kij_jit` | 4 | 0.133 | 0.046 |
| `_local_fft` | 1 | 0.127 | 0.127 |
| `wrap_points_to_voronoi` | 2 | 0.125 | 0.120 |
| `_einsum` | 5 | 0.110 | 0.058 |
| `_per_rank` | 8 | 0.097 | 0.018 |
| `_psum` | 15 | 0.092 | 0.026 |
| `dynamic_slice` | 4 | 0.080 | 0.037 |
| `iota` | 2 | 0.070 | 0.036 |
| `swapaxes` | 2 | 0.065 | 0.061 |
| `_where` | 1 | 0.062 | 0.062 |
| `maximum` | 2 | 0.043 | 0.038 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 1 | 0.055 | 0.055 |
| `_fft_and_rslice` | 2 | 0.040 | 0.020 |
| `_solve_w` | 1 | 0.034 | 0.034 |
| `_compute_S_omega_jit` | 1 | 0.026 | 0.026 |
| `get_sqrt_v_and_phase` | 1 | 0.025 | 0.025 |
| `inv` | 1 | 0.019 | 0.019 |
| `wrap_points_to_voronoi` | 2 | 0.018 | 0.010 |
| `fft_impl` | 12 | 0.018 | 0.004 |
| `multiply` | 50 | 0.018 | 0.001 |
| `solve` | 1 | 0.017 | 0.017 |
| `_reduce_sum` | 29 | 0.015 | 0.001 |
| `_psum` | 14 | 0.014 | 0.001 |
| `add` | 43 | 0.014 | 0.001 |
| `_per_rank` | 8 | 0.013 | 0.008 |
| `broadcast_in_dim` | 44 | 0.013 | 0.002 |
| `_einsum` | 19 | 0.012 | 0.001 |
| `_compute_CCT_LR` | 1 | 0.012 | 0.012 |
| `_mean` | 8 | 0.011 | 0.004 |
| `_moveaxis` | 48 | 0.011 | 0.000 |
| `_expand_band_diagonal_to_kij_jit` | 4 | 0.011 | 0.003 |
| `true_divide` | 31 | 0.011 | 0.001 |
| `_where` | 13 | 0.010 | 0.002 |
| `hartree` | 2 | 0.010 | 0.005 |
| `_take` | 3 | 0.010 | 0.003 |
| `_build_Gv_Gc` | 1 | 0.009 | 0.009 |
| `_right_ifft_mul_fft` | 1 | 0.009 | 0.009 |
| `sigma_coh` | 2 | 0.008 | 0.004 |
| `_left_ifft_conj` | 1 | 0.007 | 0.007 |
| `_lu_solve` | 3 | 0.007 | 0.003 |
| `norm` | 2 | 0.006 | 0.003 |

## Tracing cache misses

Total: **245** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/gw/wavefunction_bundle.py:150:15` | 9 | never seen function: gather id=140618976665632 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/wavefunction_bundle.py:144:15` | 8 | never seen function: gather id=140618976660032 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:214:31` | 7 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[4,4,480,480,4] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1805:15` | 6 | never seen function: broadcast_in_dim id=140620452958016 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1805:45` | 6 | never seen function: broadcast_in_dim id=140620452962656 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:385:18` | 5 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=140620455336256 defined at /global/homes/j/jackm/software/lorrax_B/src/fil |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=140618977681600 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:83:19` | 4 | never seen function: _make_jittable_local_fft.<locals>._local_fft id=140036404473472 defined at /global/homes/j/jackm/software/lorrax_B/src/ |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:808:11` | 4 | never seen function: _make_fit_one_rchunk_kernel.<locals>._kernel id=140036403728992 defined at /global/homes/j/jackm/software/lorrax_B/src/ |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:382:21` | 4 | never seen function: convert_element_type id=140620455339936 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/gw_init.py:677:40` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:375:11` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:281:84` | 4 | for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566 tracing context doesn't match, e.g. due to config or context manager closest s |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:322:23` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:324:22` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[] closest seen input type si |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/head_correction.py:438:11` | 4 | never seen function: _expand_band_diagonal_to_kij_jit id=140620457291552 defined at /global/homes/j/jackm/software/lorrax_B/src/gw/head_corr |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:195:27` | 3 | never seen function: convert_element_type id=140620457540192 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:195:12` | 3 | never seen function: _multi_slice id=140175297278368 defined at /opt/jax/jax/_src/numpy/array_methods.py:616 but seen another function defin |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:222:22` | 3 | never seen function: fft id=140041782559744 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:693:24` | 3 | never seen function: dynamic_slice id=140036404256960 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:212:8` | 3 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=140036408388768 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:365:25` | 3 | never seen function: round id=140625898106176 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3451 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:386:12` | 3 | never seen function: gather id=140618976327008 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/gw_jax.py:560:19` | 3 | never seen function: eigh id=140625898579552 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:196:12` | 2 | never seen function: sum id=139691925536448 defined at /opt/jax/jax/_src/numpy/reductions.py:240 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=1400364095550722026-04-25 18:56:02,938 jax._src.dispatch WARNING: Finished tracing + transformi |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140041783690720 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[64,480,60,2] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140036409425984 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:687:20` | 2 | never seen function: broadcast_in_dim id=140036403739872 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
