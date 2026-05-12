# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex/profile_gwjax_compile_v2/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 745 | 0.693 | 0.054 |
| jaxpr→MLIR | 242 | 0.851 | 0.120 |
| XLA compile | 255 | 8.892 | 0.712 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `sigma_sx` | 3 | 0.979 | 0.344 |
| `_kernel` | 2 | 0.907 | 0.712 |
| `sigma_coh` | 2 | 0.569 | 0.287 |
| `broadcast_in_dim` | 29 | 0.499 | 0.139 |
| `true_divide` | 12 | 0.454 | 0.091 |
| `_compute_CCT_LR` | 2 | 0.435 | 0.221 |
| `convert_element_type` | 26 | 0.396 | 0.233 |
| `_per_rank` | 9 | 0.323 | 0.125 |
| `multiply` | 10 | 0.316 | 0.038 |
| `minimax_tau_integrate_chi` | 1 | 0.303 | 0.303 |
| `eigh` | 3 | 0.278 | 0.138 |
| `get_sqrt_v_and_phase` | 1 | 0.258 | 0.258 |
| `_mean` | 4 | 0.229 | 0.091 |
| `hartree` | 2 | 0.194 | 0.188 |
| `_take` | 7 | 0.194 | 0.048 |
| `subtract` | 7 | 0.180 | 0.041 |
| `_local_fft` | 1 | 0.147 | 0.147 |
| `_multi_slice` | 4 | 0.146 | 0.039 |
| `add` | 7 | 0.145 | 0.035 |
| `dynamic_slice` | 7 | 0.124 | 0.039 |
| `wrap_points_to_voronoi` | 2 | 0.123 | 0.118 |
| `_expand_band_diagonal_to_kij_jit` | 4 | 0.122 | 0.041 |
| `_compute_S_omega_jit` | 1 | 0.119 | 0.119 |
| `yr` | 3 | 0.117 | 0.041 |
| `xn` | 3 | 0.116 | 0.041 |
| `_einsum` | 4 | 0.106 | 0.058 |
| `iota` | 3 | 0.102 | 0.035 |
| `transpose` | 9 | 0.094 | 0.037 |
| `concatenate` | 6 | 0.092 | 0.041 |
| `_psum` | 15 | 0.086 | 0.024 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 2 | 0.070 | 0.054 |
| `_solve_w` | 1 | 0.037 | 0.037 |
| `minimax_tau_integrate_chi` | 1 | 0.029 | 0.029 |
| `fft_impl` | 15 | 0.027 | 0.004 |
| `_take` | 12 | 0.027 | 0.003 |
| `_per_rank` | 9 | 0.027 | 0.012 |
| `_compute_S_omega_jit` | 1 | 0.025 | 0.025 |
| `get_sqrt_v_and_phase` | 1 | 0.022 | 0.022 |
| `multiply` | 59 | 0.021 | 0.001 |
| `_fft_and_rslice` | 1 | 0.020 | 0.020 |
| `_moveaxis` | 84 | 0.019 | 0.000 |
| `_build_Gv_Gc` | 2 | 0.019 | 0.010 |
| `inv` | 1 | 0.018 | 0.018 |
| `solve` | 1 | 0.017 | 0.017 |
| `_psum` | 15 | 0.016 | 0.001 |
| `_reduce_sum` | 31 | 0.015 | 0.001 |
| `_einsum` | 25 | 0.015 | 0.001 |
| `add` | 45 | 0.015 | 0.001 |
| `_where` | 18 | 0.014 | 0.001 |
| `true_divide` | 33 | 0.013 | 0.001 |
| `norm` | 5 | 0.013 | 0.003 |
| `_compute_CCT_LR` | 1 | 0.013 | 0.013 |
| `_expand_band_diagonal_to_kij_jit` | 4 | 0.012 | 0.004 |
| `hartree` | 2 | 0.010 | 0.005 |
| `wrap_points_to_voronoi` | 1 | 0.009 | 0.009 |
| `conjugate` | 35 | 0.009 | 0.001 |
| `_right_ifft_mul_fft` | 1 | 0.008 | 0.008 |
| `broadcast_in_dim` | 29 | 0.008 | 0.001 |
| `_left_ifft_conj` | 1 | 0.008 | 0.008 |
| `less` | 19 | 0.008 | 0.001 |

## Tracing cache misses

Total: **268** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:222:22` | 16 | never seen function: fft id=140282418562688 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:214:31` | 13 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[3,1,640,640,3] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:309:11` | 8 | never seen function: convert_element_type id=139824003316960 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:385:18` | 5 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=139827227227968 defined at /global/homes/j/jackm/software/lorrax_C/src/fil |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=139833933613152 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/gpu_utils.py:41:12` | 4 | never seen function: convert_element_type id=140615217682112 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/runtime/__init__.py:195:12` | 4 | never seen function: _psum id=139831479429504 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:83:19` | 4 | never seen function: _make_jittable_local_fft.<locals>._local_fft id=140277191449056 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:720:22` | 4 | never seen function: _where id=140282416967936 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/phdf5_wfn_reader.py:422:14` | 4 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:808:11` | 4 | never seen function: _make_fit_one_rchunk_kernel.<locals>._kernel id=140277191455456 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:382:21` | 4 | never seen function: convert_element_type id=139827227228448 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1852:32` | 4 | never seen function: _make_V_q_caseA_kernel.<locals>._kernel id=139825616810720 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/co |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_init.py:677:40` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:375:11` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:281:84` | 4 | for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566 tracing context doesn't match, e.g. due to config or context manager closest s |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:438:11` | 4 | never seen function: _expand_band_diagonal_to_kij_jit id=139827229520448 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/head_corr |
| `/global/homes/j/jackm/software/lorrax_C/src/runtime/__init__.py:196:12` | 3 | never seen function: sum id=140282417889664 defined at /opt/jax/jax/_src/numpy/reductions.py:240 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/cholesky_2d.py:148:22` | 3 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to conf2026-04-25 20:25:04,210 jax._src.di |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:37:9` | 3 | never seen function: norm id=139832217911680 defined at /opt/jax/jax/_src/numpy/linalg.py:1076 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:324:22` | 3 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[] closest seen input type si |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:560:19` | 3 | never seen function: eigh id=139832217823456 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/cohsex_sigma.py:199:18` | 3 | never seen function: _make_jittable_local_fft.<locals>._make_axis_wrapper.<locals>.fft_impl id=140402550593536 defined at /global/homes/j/ja |
| `/global/homes/j/jackm/software/lorrax_C/src/runtime/__init__.py:195:27` | 2 | never seen function: broadcast_in_dim id=140277193145728 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140277196476064 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140282419693664 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[9,640,80,2] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:685:20` | 2 | never seen function: iota id=140277191453216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:687:20` | 2 | never seen function: iota id=140277190829984 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:693:24` | 2 | never seen function: dynamic_slice id=140277191232224 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
