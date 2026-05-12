# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex/profile_gwjax_compile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 832 | 0.849 | 0.122 |
| jaxpr→MLIR | 340 | 1.120 | 0.150 |
| XLA compile | 347 | 14.692 | 0.687 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 65 | 1.186 | 0.145 |
| `sigma_sx` | 3 | 0.986 | 0.354 |
| `_kernel` | 2 | 0.887 | 0.687 |
| `convert_element_type` | 37 | 0.783 | 0.288 |
| `multiply` | 19 | 0.681 | 0.069 |
| `gather` | 15 | 0.676 | 0.091 |
| `true_divide` | 12 | 0.632 | 0.094 |
| `add` | 17 | 0.596 | 0.066 |
| `concatenate` | 10 | 0.437 | 0.120 |
| `eigh` | 3 | 0.418 | 0.141 |
| `transpose` | 8 | 0.392 | 0.076 |
| `_fft_gather_reshard` | 1 | 0.362 | 0.362 |
| `iota` | 12 | 0.349 | 0.040 |
| `_batched_chol` | 1 | 0.346 | 0.346 |
| `subtract` | 9 | 0.330 | 0.043 |
| `_per_rank` | 9 | 0.323 | 0.129 |
| `minimax_tau_integrate_chi` | 1 | 0.298 | 0.298 |
| `sigma_coh` | 1 | 0.284 | 0.284 |
| `_psum` | 15 | 0.272 | 0.038 |
| `_local_fft` | 2 | 0.265 | 0.134 |
| `_solve_w` | 1 | 0.259 | 0.259 |
| `_mean` | 4 | 0.232 | 0.090 |
| `less` | 5 | 0.226 | 0.081 |
| `select_n` | 5 | 0.226 | 0.083 |
| `_compute_CCT_LR` | 1 | 0.209 | 0.209 |
| `get_sqrt_v_and_phase` | 1 | 0.202 | 0.202 |
| `dynamic_slice` | 6 | 0.195 | 0.038 |
| `hartree` | 1 | 0.189 | 0.189 |
| `_take` | 4 | 0.187 | 0.049 |
| `_einsum` | 4 | 0.162 | 0.059 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_batched_chol` | 1 | 0.122 | 0.122 |
| `_kernel` | 2 | 0.071 | 0.055 |
| `fft_impl` | 42 | 0.067 | 0.004 |
| `_solve_w` | 1 | 0.036 | 0.036 |
| `minimax_tau_integrate_chi` | 1 | 0.029 | 0.029 |
| `_per_rank` | 9 | 0.027 | 0.012 |
| `_moveaxis` | 106 | 0.026 | 0.001 |
| `sigma_sx` | 1 | 0.024 | 0.024 |
| `_fft_and_rslice` | 1 | 0.021 | 0.021 |
| `_convolve` | 1 | 0.020 | 0.020 |
| `get_sqrt_v_and_phase` | 1 | 0.020 | 0.020 |
| `_take` | 8 | 0.019 | 0.003 |
| `inv` | 1 | 0.019 | 0.019 |
| `multiply` | 55 | 0.018 | 0.001 |
| `solve` | 1 | 0.017 | 0.017 |
| `add` | 52 | 0.017 | 0.001 |
| `broadcast_in_dim` | 65 | 0.016 | 0.000 |
| `_psum` | 15 | 0.015 | 0.002 |
| `_fft_gather_reshard` | 1 | 0.015 | 0.015 |
| `true_divide` | 38 | 0.015 | 0.001 |
| `_reduce_sum` | 26 | 0.014 | 0.001 |
| `_compute_CCT_LR` | 1 | 0.012 | 0.012 |
| `_where` | 16 | 0.011 | 0.001 |
| `_einsum` | 19 | 0.011 | 0.001 |
| `_build_Gv_Gc` | 1 | 0.010 | 0.010 |
| `conjugate` | 31 | 0.009 | 0.001 |
| `_right_ifft_mul_fft` | 1 | 0.009 | 0.009 |
| `less` | 21 | 0.008 | 0.001 |
| `convert_element_type` | 36 | 0.008 | 0.000 |
| `_left_ifft_conj` | 1 | 0.008 | 0.008 |

## Tracing cache misses

Total: **358** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 20 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[3,1,640,640,3] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:222:22` | 16 | never seen function: fft id=139755603969184 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:424:10` | 15 | never seen function: convert_element_type id=140659376677376 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:808:11` | 10 | never seen function: _make_fit_one_rchunk_kernel.<locals>._kernel id=140658843171424 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/cohsex_sigma.py:199:18` | 10 | never seen function: _make_cohsex_kernels.<locals>.sigma_sx id=140655616639840 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/coh |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:500:6` | 9 | never seen function: _make_jittable_local_fft.<locals>._make_axis_wrapper.<locals>.fft_impl id=140657234325120 defined at /global/homes/j/ja |
| `/global/homes/j/jackm/software/lorrax_C/src/common/chi_from_dipole.py:43:12` | 9 | never seen function: add id=140657229006624 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:212:8` | 7 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=140659376310528 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_C/src/common/chi_from_dipole.py:45:10` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[26],  args[1]: i64[] closest seen input ty |
| `/global/homes/j/jackm/software/lorrax_C/src/common/chi_from_dipole.py:46:10` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[54],  args[1]: i64[] closest seen input ty |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:30:24` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[262144],  args[1]: i64[] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:385:18` | 5 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=140657764309792 defined at /global/homes/j/jackm/software/lorrax_C/src/fil |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=140665930434176 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/gpu_utils.py:41:12` | 4 | never seen function: convert_element_type id=139768441354848 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/runtime/__init__.py:195:12` | 4 | never seen function: _psum id=139755177158400 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:83:19` | 4 | never seen function: _make_jittable_local_fft.<locals>._local_fft id=139750452624768 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:736:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140658843303776 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:382:21` | 4 | never seen function: convert_element_type id=140657764310272 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1845:32` | 4 | never seen function: _make_V_q_caseA_kernel.<locals>._kernel id=140657764602464 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/co |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_init.py:677:40` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:375:11` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:144:15` | 4 | never seen function: gather id=140657234516448 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:150:15` | 4 | never seen function: gather id=140657234522048 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:272:84` | 4 | for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566 tracing context doesn't match, e.g. due to config or context manager closest s |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:300:11` | 4 | never seen function: convert_element_type id=140655619450464 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:425:12` | 4 | never seen function: broadcast_in_dim id=140655615641056 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:425:30` | 4 | never seen function: broadcast_in_dim id=140655615642816 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:426:11` | 4 | never seen function: broadcast_in_dim id=140655615644416 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 3 | never2026-04-25 17:17:22,369 jax._src.dispatch WARNING: Finished XLA compilation of jit(_local_fft) in 0.134202957 sec |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:24:20` | 3 | never seen function: broadcast_in_dim id=140655619011616 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
