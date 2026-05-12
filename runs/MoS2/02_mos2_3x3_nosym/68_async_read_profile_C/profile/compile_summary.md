# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/68_async_read_profile_C/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 706 | 0.825 | 0.136 |
| jaxpr→MLIR | 248 | 0.950 | 0.121 |
| XLA compile | 247 | 16.425 | 0.682 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 37 | 1.439 | 0.190 |
| `convert_element_type` | 21 | 0.891 | 0.232 |
| `true_divide` | 11 | 0.787 | 0.111 |
| `sigma_sx` | 2 | 0.725 | 0.381 |
| `_identity_fn` | 15 | 0.716 | 0.156 |
| `_tau_kernel` | 1 | 0.682 | 0.682 |
| `gather` | 10 | 0.632 | 0.108 |
| `concatenate` | 9 | 0.468 | 0.056 |
| `add` | 8 | 0.440 | 0.060 |
| `_fft_gather_reshard` | 1 | 0.407 | 0.407 |
| `reshape` | 13 | 0.380 | 0.032 |
| `_per_rank` | 10 | 0.376 | 0.052 |
| `transpose` | 7 | 0.372 | 0.080 |
| `multiply` | 7 | 0.371 | 0.057 |
| `_batched_chol` | 1 | 0.363 | 0.363 |
| `iota` | 7 | 0.329 | 0.049 |
| `_project_tau_onto_omega` | 2 | 0.327 | 0.164 |
| `_multi_slice` | 6 | 0.324 | 0.055 |
| `_psum` | 8 | 0.323 | 0.059 |
| `sigma_coh` | 1 | 0.315 | 0.315 |
| `_tau_step` | 1 | 0.312 | 0.312 |
| `_fft_and_rslice` | 1 | 0.298 | 0.298 |
| `_solve_w` | 1 | 0.273 | 0.273 |
| `_prepare_sigma_state` | 1 | 0.259 | 0.259 |
| `dynamic_slice` | 5 | 0.251 | 0.055 |
| `_reduce_max` | 2 | 0.244 | 0.124 |
| `_where` | 4 | 0.244 | 0.066 |
| `subtract` | 4 | 0.231 | 0.066 |
| `_compute_CCT_LR` | 1 | 0.217 | 0.217 |
| `hartree` | 1 | 0.208 | 0.208 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_batched_chol` | 1 | 0.136 | 0.136 |
| `fft_impl` | 42 | 0.072 | 0.005 |
| `_solve_w` | 1 | 0.043 | 0.043 |
| `_tau_kernel` | 1 | 0.032 | 0.032 |
| `sigma_sx` | 1 | 0.030 | 0.030 |
| `_convolve` | 1 | 0.026 | 0.026 |
| `_project_tau_onto_omega` | 2 | 0.024 | 0.012 |
| `_moveaxis` | 97 | 0.024 | 0.000 |
| `_sigma_kij_kernel` | 1 | 0.023 | 0.023 |
| `multiply` | 61 | 0.022 | 0.001 |
| `_tau_step` | 1 | 0.021 | 0.021 |
| `_fft_gather_reshard` | 1 | 0.021 | 0.021 |
| `_fft_and_rslice` | 1 | 0.019 | 0.019 |
| `get_sqrt_v_and_phase` | 1 | 0.019 | 0.019 |
| `ifftn` | 2 | 0.019 | 0.010 |
| `true_divide` | 32 | 0.014 | 0.001 |
| `_prepare_sigma_state` | 1 | 0.013 | 0.013 |
| `_where` | 16 | 0.013 | 0.002 |
| `add` | 35 | 0.012 | 0.001 |
| `_compute_CCT_LR` | 1 | 0.012 | 0.012 |
| `_einsum` | 18 | 0.011 | 0.002 |
| `_single_chunk_proc` | 1 | 0.011 | 0.011 |
| `broadcast_in_dim` | 37 | 0.011 | 0.001 |
| `conjugate` | 30 | 0.010 | 0.002 |
| `_psum` | 8 | 0.010 | 0.002 |
| `_per_rank` | 11 | 0.010 | 0.001 |
| `fft` | 34 | 0.009 | 0.001 |
| `_left_ifft_conj` | 1 | 0.008 | 0.008 |
| `_build_tau_operands` | 1 | 0.008 | 0.008 |
| `_build_G` | 1 | 0.008 | 0.008 |

## Tracing cache misses

Total: **288** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:112:31` | 28 | never seen function: fft id=140145777919296 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:120:22` | 10 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:399:13` | 10 | never seen function: main.<locals>.sigma_sx id=139720677751552 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:369 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:1191:57` | 10 | never seen function: _get_sigma_tau_kernel.<locals>._tau_kernel id=140294570539456 defined at /global/homes/j/jackm/software/lorrax_C/src/gw |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:297:18` | 9 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=139727500242720 defined at /global/homes/j/jackm/software/lorrax_C/src/fil |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:243:8` | 7 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=140132236813440 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:126:24` | 7 | never seen function: _get_chi_minimax_kernel.<locals>._tau_step id=140132236476960 defined at /global/homes/j/jackm/software/lorrax_C/src/gw |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:294:21` | 5 | never seen function: convert_element_type id=139767497216608 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_init.py:423:40` | 5 | never seen function: convert_element_type id=139723339721888 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:190:29` | 5 | never seen function: convert_element_type id=140294610975648 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:173:12` | 5 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: f64[9,80] closest seen input |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:583:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140132237486304 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:363:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=139763144440256 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:343:12` | 4 | never seen function: compute_ZCT_from_left_right_zchunk.<locals>._left_ifft_conj id=139763144887168 defined at /global/homes/j/jackm/softwar |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:344:10` | 4 | never seen function: compute_ZCT_from_left_right_zchunk.<locals>._right_ifft_mul_fft id=140132872963552 defined at /global/homes/j/jackm/sof |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:304:11` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:111:15` | 4 | never seen function: gather id=139723902730464 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:117:15` | 4 | never seen function: gather id=140132236170784 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:1279:13` | 4 | never seen function: convert_element_type id=139722870875104 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:881:21` | 4 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: c128[4,21,9,80,80] closest seen inp |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:970:27` | 3 | never seen function: reshape id=139723340944928 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:128:30` | 3 | never seen function: dynamic_slice id=140132236482720 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:677:18` | 2 | never seen function: convert_element_type id=139737329491360 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:679:18` | 2 | never seen function: _identity_fn id=139776058754848 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:680:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[9,640,80,2] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:532:20` | 2 | never seen function: iota id=140142240029312 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:534:20` | 2 | never seen function: iota id=140132237260128 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:540:24` | 2 | never seen function: dynamic_slice id=139763145148672 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:567:22` | 2 | never seen function: _where id=140145776324544 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:128:8` | 2 | never seen function: compute_pair_density_spin_traced.<locals>._compute_P_traced id=140132236807520 defined at /global/homes/j/jackm/softwar |

## Persistent cache misses

_None._
