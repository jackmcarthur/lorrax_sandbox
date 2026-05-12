# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/25_lorrax_final_profile_C/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 765 | 0.824 | 0.126 |
| jaxpr→MLIR | 304 | 1.095 | 0.147 |
| XLA compile | 304 | 19.933 | 0.801 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 51 | 1.902 | 0.166 |
| `_identity_fn` | 24 | 1.213 | 0.147 |
| `true_divide` | 18 | 1.155 | 0.110 |
| `convert_element_type` | 20 | 0.924 | 0.281 |
| `_batched_chol` | 1 | 0.801 | 0.801 |
| `gather` | 13 | 0.793 | 0.110 |
| `add` | 14 | 0.752 | 0.062 |
| `sigma_sx` | 2 | 0.737 | 0.385 |
| `_tau_kernel` | 1 | 0.710 | 0.710 |
| `iota` | 13 | 0.630 | 0.054 |
| `multiply` | 9 | 0.476 | 0.058 |
| `_fft_gather_reshard` | 1 | 0.402 | 0.402 |
| `reshape` | 14 | 0.402 | 0.033 |
| `scatter` | 8 | 0.395 | 0.078 |
| `transpose` | 7 | 0.385 | 0.084 |
| `_where` | 6 | 0.337 | 0.060 |
| `_reduce_max` | 3 | 0.336 | 0.129 |
| `subtract` | 6 | 0.328 | 0.066 |
| `_project_tau_onto_omega` | 2 | 0.327 | 0.164 |
| `sigma_coh` | 1 | 0.321 | 0.321 |
| `_multi_slice` | 6 | 0.320 | 0.059 |
| `eigh` | 2 | 0.313 | 0.160 |
| `_tau_step` | 1 | 0.310 | 0.310 |
| `concatenate` | 6 | 0.307 | 0.057 |
| `_fft_and_rslice` | 1 | 0.283 | 0.283 |
| `_solve_w` | 1 | 0.271 | 0.271 |
| `_prepare_sigma_state` | 1 | 0.260 | 0.260 |
| `_squeeze` | 8 | 0.221 | 0.033 |
| `sqrt` | 4 | 0.219 | 0.061 |
| `_psum` | 5 | 0.211 | 0.054 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_batched_chol` | 1 | 0.126 | 0.126 |
| `fft_impl` | 42 | 0.071 | 0.005 |
| `_solve_w` | 1 | 0.039 | 0.039 |
| `_tau_kernel` | 1 | 0.033 | 0.033 |
| `sigma_sx` | 1 | 0.030 | 0.030 |
| `_convolve` | 1 | 0.025 | 0.025 |
| `_moveaxis` | 97 | 0.025 | 0.001 |
| `_project_tau_onto_omega` | 2 | 0.024 | 0.014 |
| `multiply` | 63 | 0.024 | 0.001 |
| `_sigma_kij_kernel` | 1 | 0.023 | 0.023 |
| `_tau_step` | 1 | 0.021 | 0.021 |
| `get_sqrt_v_and_phase` | 1 | 0.021 | 0.021 |
| `_fft_gather_reshard` | 1 | 0.020 | 0.020 |
| `_fft_and_rslice` | 1 | 0.019 | 0.019 |
| `ifftn` | 2 | 0.019 | 0.010 |
| `true_divide` | 40 | 0.019 | 0.002 |
| `add` | 42 | 0.016 | 0.002 |
| `broadcast_in_dim` | 51 | 0.015 | 0.002 |
| `_where` | 18 | 0.014 | 0.001 |
| `_prepare_sigma_state` | 1 | 0.013 | 0.013 |
| `_single_chunk_proc` | 1 | 0.013 | 0.013 |
| `_einsum` | 18 | 0.013 | 0.002 |
| `_compute_CCT_LR` | 1 | 0.012 | 0.012 |
| `conjugate` | 32 | 0.010 | 0.002 |
| `_build_tau_operands` | 1 | 0.009 | 0.009 |
| `fft` | 34 | 0.009 | 0.000 |
| `_build_G` | 1 | 0.008 | 0.008 |
| `_left_ifft_conj` | 1 | 0.008 | 0.008 |
| `_solve_all_at_once` | 1 | 0.008 | 0.008 |
| `_psum` | 5 | 0.007 | 0.002 |

## Tracing cache misses

Total: **313** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:112:31` | 28 | never seen function: fft id=140295748207776 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:120:22` | 10 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:378:13` | 10 | never seen function: main.<locals>.sigma_sx id=140477446079008 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:348 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:1203:35` | 10 | never seen function: _get_sigma_tau_kernel.<locals>._tau_kernel id=140477443683360 defined at /global/homes/j/jackm/software/lorrax_C/src/gw |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_allgather.py:48:15` | 9 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 tracing context doesn't match, e.g. due to config or context man |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:243:8` | 7 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=140289479087488 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:47:10` | 7 | never seen function: convert_element_type id=140289477900416 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:126:24` | 7 | never seen function: _get_chi_minimax_kernel.<locals>._tau_step id=140477448361984 defined at /global/homes/j/jackm/software/lorrax_C/src/gw |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_init.py:409:40` | 5 | never seen function: convert_element_type id=140289806210752 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:63:10` | 5 | never seen function: add id=140289479834208 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:565:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140289479825888 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:345:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140477715380672 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:343:12` | 4 | never seen function: compute_ZCT_from_left_right_zchunk.<locals>._left_ifft_conj id=140289479609376 defined at /global/homes/j/jackm/softwar |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:344:10` | 4 | never seen function: compute_ZCT_from_left_right_zchunk.<locals>._right_ifft_mul_fft id=140477717691456 defined at /global/homes/j/jackm/sof |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:49:10` | 4 | never seen function: broadcast_in_dim id=140081169272960 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:82:19` | 4 | never seen function: convert_element_type id=140289480618560 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:169:15` | 4 | never seen function: convert_element_type id=140289480174432 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:304:11` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:110:15` | 4 | never seen function: gather id=140289405490560 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:116:15` | 4 | never seen function: gather id=140289405498080 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:148:35` | 4 | never seen function: convert_element_type id=140080627913632 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:173:12` | 4 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: bool[9,80] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:892:21` | 4 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: c128[4,21,9,80,80] closest seen inp |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:190:29` | 3 | never seen function: concatenate id=140080631928352 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:128:30` | 3 | never seen function: dynamic_slice id=140477448365664 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:1293:13` | 3 | never seen function: iota id=140289142347296 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:659:18` | 2 | never seen function: convert_element_type id=140484745534720 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:661:18` | 2 | never seen function: _identity_fn id=140295749338752 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:662:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[9,640,80,2] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:514:20` | 2 | never seen function: iota id=140292361165440 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
