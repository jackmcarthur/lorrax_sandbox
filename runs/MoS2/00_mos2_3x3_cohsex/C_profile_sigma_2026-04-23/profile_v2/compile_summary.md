# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/C_profile_sigma_2026-04-23/profile_v2/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 721 | 0.609 | 0.031 |
| jaxpr→MLIR | 237 | 0.853 | 0.126 |
| XLA compile | 265 | 10.574 | 0.763 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `sigma_sx` | 2 | 0.778 | 0.393 |
| `gather` | 20 | 0.772 | 0.069 |
| `_kernel` | 1 | 0.763 | 0.763 |
| `broadcast_in_dim` | 33 | 0.737 | 0.188 |
| `_per_rank` | 15 | 0.729 | 0.153 |
| `true_divide` | 11 | 0.633 | 0.100 |
| `_tau_kernel` | 1 | 0.627 | 0.627 |
| `convert_element_type` | 28 | 0.497 | 0.247 |
| `add` | 12 | 0.373 | 0.065 |
| `minimax_tau_integrate_chi` | 1 | 0.328 | 0.328 |
| `eigh` | 3 | 0.327 | 0.164 |
| `sigma_coh` | 1 | 0.324 | 0.324 |
| `_take` | 7 | 0.291 | 0.076 |
| `iota` | 5 | 0.267 | 0.057 |
| `multiply` | 13 | 0.251 | 0.063 |
| `get_sqrt_v_and_phase` | 2 | 0.239 | 0.217 |
| `_compute_CCT_LR` | 1 | 0.238 | 0.238 |
| `_multi_slice` | 4 | 0.234 | 0.073 |
| `_psum` | 18 | 0.215 | 0.056 |
| `concatenate` | 8 | 0.214 | 0.136 |
| `_identity_fn` | 11 | 0.212 | 0.135 |
| `subtract` | 5 | 0.179 | 0.062 |
| `_local_fft` | 1 | 0.167 | 0.167 |
| `_where` | 5 | 0.142 | 0.068 |
| `dynamic_slice` | 2 | 0.114 | 0.059 |
| `maximum` | 3 | 0.113 | 0.056 |
| `greater_equal` | 2 | 0.107 | 0.055 |
| `sum` | 3 | 0.105 | 0.039 |
| `reshape` | 7 | 0.100 | 0.033 |
| `swapaxes` | 2 | 0.093 | 0.090 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `fft_impl` | 27 | 0.041 | 0.004 |
| `minimax_tau_integrate_chi` | 2 | 0.040 | 0.031 |
| `_tau_kernel` | 1 | 0.027 | 0.027 |
| `_per_rank` | 18 | 0.026 | 0.012 |
| `sigma_sx` | 1 | 0.026 | 0.026 |
| `_take` | 11 | 0.025 | 0.003 |
| `_sigma_kij_kernel` | 1 | 0.023 | 0.023 |
| `multiply` | 56 | 0.022 | 0.001 |
| `_moveaxis` | 87 | 0.022 | 0.000 |
| `_convolve` | 1 | 0.022 | 0.022 |
| `get_sqrt_v_and_phase` | 1 | 0.021 | 0.021 |
| `_fft_and_rslice` | 1 | 0.021 | 0.021 |
| `_psum` | 18 | 0.020 | 0.002 |
| `_fft_gather_reshard` | 1 | 0.016 | 0.016 |
| `_where` | 22 | 0.015 | 0.001 |
| `broadcast_in_dim` | 50 | 0.014 | 0.001 |
| `add` | 42 | 0.013 | 0.001 |
| `_reduce_sum` | 23 | 0.013 | 0.001 |
| `_einsum` | 19 | 0.013 | 0.001 |
| `_compute_CCT_LR` | 1 | 0.012 | 0.012 |
| `_prepare_sigma_state` | 1 | 0.012 | 0.012 |
| `true_divide` | 30 | 0.011 | 0.001 |
| `_build_Gv_Gc` | 1 | 0.010 | 0.010 |
| `_right_ifft_mul_fft` | 1 | 0.009 | 0.009 |
| `conjugate` | 34 | 0.009 | 0.000 |
| `_left_ifft_conj` | 1 | 0.008 | 0.008 |
| `fft` | 33 | 0.007 | 0.000 |
| `convert_element_type` | 30 | 0.007 | 0.001 |
| `eigh` | 3 | 0.007 | 0.002 |
| `_solve_all_at_once` | 1 | 0.006 | 0.006 |

## Tracing cache misses

Total: **289** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 19 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[3,1,640,640,3] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:385:18` | 14 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=139723809789312 defined at /global/homes/j/jackm/software/lorrax_C/src/fil |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:222:22` | 10 | never seen function: fft id=139931948778496 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/cohsex_sigma.py:199:18` | 10 | never seen function: _make_cohsex_kernels.<locals>.sigma_sx id=139722133834144 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/coh |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:600:6` | 9 | never seen function: _make_jittable_local_fft.<locals>._make_axis_wrapper.<locals>.fft_impl id=139722132776224 defined at /global/homes/j/ja |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:382:21` | 8 | never seen function: convert_element_type id=139723809789952 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:375:11` | 6 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:144:15` | 6 | never seen function: gather id=139722139541216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:212:8` | 5 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=139966140579200 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1805:45` | 5 | never seen function: broadcast_in_dim id=139722869501408 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=139731664963040 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:150:15` | 5 | never seen function: gather id=139722139546816 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:188:12` | 5 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: f64[9,80] closest seen input |
| `/global/homes/j/jackm/software/lorrax_C/src/runtime/__init__.py:195:12` | 4 | never seen function: _psum id=139770923770048 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:83:19` | 4 | never seen function: _make_jittable_local_fft.<locals>._local_fft id=139925874110560 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/common/phdf5_wfn_reader.py:422:14` | 4 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1805:15` | 4 | never seen function: broadcast_in_dim id=139722869495968 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_init.py:677:40` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:315:16` | 4 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: c128[4,640] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:385:12` | 4 | never seen function: gather id=139722139138016 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:386:12` | 4 | never seen function: gather id=139722139339328 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/runtime/__init__.py:195:27` | 3 | never seen function: convert_element_type id=139925875570336 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 3 | never seen function: iota id=139925875571776 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 3 | never seen function: iota id=139925274476224 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 3 | never seen function: dynamic_slice id=139925874892032 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:970:40` | 3 | never seen function: broadcast_in_dim id=139724348328512 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:151:16` | 3 | never seen function: _psum id=139729918353600 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/gvec_fft_box.py:120:19` | 3 | for _where defined at /opt/jax2026-04-23 13:05:14,405 jax._src.dispatch WARNING: Finished tracing + transforming multiply for pjit in 0.0007 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:487:10` | 3 | never seen function: convert_element_type id=139722139343488 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/minimax_screening.py:324:20` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[10] closest seen input ty |

## Persistent cache misses

_None._
