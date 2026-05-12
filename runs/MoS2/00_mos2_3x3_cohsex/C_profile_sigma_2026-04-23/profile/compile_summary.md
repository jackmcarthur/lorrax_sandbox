# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/C_profile_sigma_2026-04-23/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 684 | 0.758 | 0.056 |
| jaxpr→MLIR | 244 | 0.887 | 0.120 |
| XLA compile | 245 | 10.354 | 0.759 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_tau_kernel` | 2 | 1.289 | 0.648 |
| `_per_rank` | 17 | 0.866 | 0.156 |
| `sigma_sx` | 2 | 0.760 | 0.398 |
| `_kernel` | 1 | 0.759 | 0.759 |
| `broadcast_in_dim` | 38 | 0.730 | 0.060 |
| `_local_fft` | 1 | 0.543 | 0.543 |
| `gather` | 14 | 0.514 | 0.121 |
| `true_divide` | 8 | 0.389 | 0.097 |
| `_identity_fn` | 11 | 0.383 | 0.147 |
| `add` | 11 | 0.343 | 0.067 |
| `minimax_tau_integrate_chi` | 1 | 0.333 | 0.333 |
| `sigma_coh` | 1 | 0.330 | 0.330 |
| `_take` | 7 | 0.303 | 0.074 |
| `convert_element_type` | 28 | 0.297 | 0.031 |
| `_compute_CCT_LR` | 1 | 0.241 | 0.241 |
| `concatenate` | 8 | 0.229 | 0.139 |
| `get_sqrt_v_and_phase` | 2 | 0.227 | 0.202 |
| `multiply` | 12 | 0.220 | 0.061 |
| `_psum` | 15 | 0.172 | 0.063 |
| `eigh` | 2 | 0.171 | 0.165 |
| `iota` | 3 | 0.165 | 0.061 |
| `subtract` | 4 | 0.118 | 0.054 |
| `_multi_slice` | 2 | 0.118 | 0.062 |
| `reshape` | 7 | 0.105 | 0.031 |
| `swapaxes` | 2 | 0.097 | 0.093 |
| `abs` | 2 | 0.088 | 0.084 |
| `_where` | 4 | 0.078 | 0.066 |
| `conjugate` | 3 | 0.066 | 0.056 |
| `maximum` | 2 | 0.063 | 0.059 |
| `dynamic_slice` | 1 | 0.058 | 0.058 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 1 | 0.056 | 0.056 |
| `_tau_kernel` | 2 | 0.052 | 0.026 |
| `_moveaxis` | 85 | 0.050 | 0.027 |
| `_sigma_kij_kernel` | 2 | 0.047 | 0.024 |
| `fft_impl` | 30 | 0.045 | 0.004 |
| `minimax_tau_integrate_chi` | 2 | 0.040 | 0.031 |
| `_per_rank` | 17 | 0.034 | 0.012 |
| `_local_fft` | 1 | 0.033 | 0.033 |
| `sigma_sx` | 1 | 0.025 | 0.025 |
| `get_sqrt_v_and_phase` | 1 | 0.023 | 0.023 |
| `_fft_and_rslice` | 1 | 0.023 | 0.023 |
| `multiply` | 55 | 0.021 | 0.001 |
| `_take` | 10 | 0.021 | 0.003 |
| `_convolve` | 1 | 0.021 | 0.021 |
| `_where` | 22 | 0.016 | 0.001 |
| `_psum` | 15 | 0.016 | 0.001 |
| `_fft_gather_reshard` | 1 | 0.015 | 0.015 |
| `add` | 38 | 0.015 | 0.003 |
| `_compute_CCT_LR` | 1 | 0.014 | 0.014 |
| `_einsum` | 19 | 0.013 | 0.001 |
| `_prepare_sigma_state` | 1 | 0.013 | 0.013 |
| `true_divide` | 32 | 0.012 | 0.001 |
| `_reduce_sum` | 19 | 0.010 | 0.001 |
| `_build_Gv_Gc` | 1 | 0.010 | 0.010 |
| `broadcast_in_dim` | 38 | 0.009 | 0.000 |
| `conjugate` | 33 | 0.009 | 0.001 |
| `_right_ifft_mul_fft` | 1 | 0.008 | 0.008 |
| `_left_ifft_conj` | 1 | 0.008 | 0.008 |
| `fft` | 29 | 0.006 | 0.000 |
| `hartree` | 1 | 0.006 | 0.006 |

## Tracing cache misses

Total: **278** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 18 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[3,1,640,640,3] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:385:18` | 13 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=140165319461920 defined at /global/homes/j/jackm/software/lorrax_C/src/fil |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:222:22` | 11 | never seen function: fft id=140356989551616 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/cohsex_sigma.py:199:18` | 10 | never seen function: _make_cohsex_kernels.<locals>.sigma_sx id=140164319200896 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/coh |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:587:6` | 9 | never seen function: _make_jittable_local_fft.<locals>._make_axis_wrapper.<locals>.fft_impl id=140164315079168 defined at /global/homes/j/ja |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:382:21` | 8 | never seen function: convert_element_type id=140165319462560 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:212:8` | 7 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=140349607271360 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:375:11` | 6 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:144:15` | 6 | never seen function: gather id=140164917468096 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=140174245823968 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:150:15` | 5 | never seen function: gather id=140164917473696 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:188:12` | 5 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: f64[9,80] closest seen input |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:83:19` | 4 | never seen function: _make_jittable_local_fft.<locals>._local_fft id=140166930277184 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/common/phdf5_wfn_reader.py:422:14` | 4 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:808:11` | 4 | never seen function: _make_fit_one_rchunk_kernel.<locals>._kernel id=140349070528640 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1805:15` | 4 | never seen function: broadcast_in_dim id=140164918595392 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1805:45` | 4 | never seen function: broadcast_in_dim id=140164918600832 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_init.py:677:40` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:315:16` | 4 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: c128[4,640] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:385:12` | 4 | never seen function: gather id=140164917032128 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:386:12` | 4 | never seen function: gather id=140164917266208 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:151:16` | 3 | never seen function: _psum id=140172372566208 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/minimax_screening.py:324:20` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[10] closest seen input ty |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140349600292896 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140356990698976 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[9,640,80,2] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140353829266240 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=140349070539200 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140349070036320 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 2 | never seen function: _where id=140356987956864 defined at /opt/jax/jax/_src/numpy/util.py:287 |

## Persistent cache misses

_None._
