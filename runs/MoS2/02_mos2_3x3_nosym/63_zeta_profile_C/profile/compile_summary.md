# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/63_zeta_profile_C/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 698 | 0.797 | 0.123 |
| jaxpr→MLIR | 285 | 1.061 | 0.133 |
| XLA compile | 285 | 17.906 | 0.675 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `<unnamed wrapped function>` | 54 | 1.918 | 0.046 |
| `broadcast_in_dim` | 37 | 1.474 | 0.216 |
| `true_divide` | 11 | 0.777 | 0.113 |
| `convert_element_type` | 15 | 0.757 | 0.282 |
| `sigma_sx` | 2 | 0.730 | 0.379 |
| `_identity_fn` | 15 | 0.712 | 0.146 |
| `_tau_kernel` | 1 | 0.675 | 0.675 |
| `gather` | 10 | 0.640 | 0.106 |
| `concatenate` | 9 | 0.472 | 0.058 |
| `add` | 8 | 0.444 | 0.060 |
| `_fft_gather_reshard` | 1 | 0.393 | 0.393 |
| `reshape` | 13 | 0.387 | 0.034 |
| `_batched_chol` | 1 | 0.381 | 0.381 |
| `transpose` | 7 | 0.375 | 0.083 |
| `multiply` | 7 | 0.372 | 0.058 |
| `iota` | 7 | 0.344 | 0.056 |
| `_project_tau_onto_omega` | 2 | 0.333 | 0.168 |
| `_psum` | 8 | 0.321 | 0.057 |
| `_multi_slice` | 6 | 0.320 | 0.055 |
| `sigma_coh` | 1 | 0.317 | 0.317 |
| `_tau_step` | 1 | 0.310 | 0.310 |
| `_fft_and_rslice` | 1 | 0.297 | 0.297 |
| `_solve_w` | 1 | 0.269 | 0.269 |
| `_prepare_sigma_state` | 1 | 0.260 | 0.260 |
| `dynamic_slice` | 5 | 0.256 | 0.055 |
| `_reduce_max` | 2 | 0.247 | 0.129 |
| `_where` | 4 | 0.239 | 0.062 |
| `subtract` | 4 | 0.234 | 0.069 |
| `_compute_CCT_LR` | 1 | 0.212 | 0.212 |
| `hartree` | 1 | 0.206 | 0.206 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_batched_chol` | 1 | 0.123 | 0.123 |
| `fft_impl` | 42 | 0.072 | 0.004 |
| `_solve_w` | 1 | 0.040 | 0.040 |
| `_tau_kernel` | 1 | 0.032 | 0.032 |
| `sigma_sx` | 1 | 0.030 | 0.030 |
| `get_sqrt_v_and_phase` | 1 | 0.030 | 0.030 |
| `_project_tau_onto_omega` | 2 | 0.026 | 0.013 |
| `_convolve` | 1 | 0.025 | 0.025 |
| `_moveaxis` | 97 | 0.024 | 0.001 |
| `multiply` | 61 | 0.023 | 0.001 |
| `_sigma_kij_kernel` | 1 | 0.022 | 0.022 |
| `_tau_step` | 1 | 0.020 | 0.020 |
| `_fft_gather_reshard` | 1 | 0.020 | 0.020 |
| `_fft_and_rslice` | 1 | 0.019 | 0.019 |
| `ifftn` | 2 | 0.019 | 0.010 |
| `true_divide` | 32 | 0.014 | 0.001 |
| `_where` | 16 | 0.013 | 0.001 |
| `_prepare_sigma_state` | 1 | 0.013 | 0.013 |
| `add` | 35 | 0.012 | 0.001 |
| `_compute_CCT_LR` | 1 | 0.011 | 0.011 |
| `_einsum` | 18 | 0.011 | 0.002 |
| `broadcast_in_dim` | 37 | 0.010 | 0.000 |
| `_single_chunk_proc` | 1 | 0.009 | 0.009 |
| `conjugate` | 30 | 0.009 | 0.002 |
| `_psum` | 8 | 0.009 | 0.001 |
| `_solve_all_at_once` | 1 | 0.009 | 0.009 |
| `fft` | 34 | 0.008 | 0.001 |
| `_build_tau_operands` | 1 | 0.008 | 0.008 |
| `_build_G` | 1 | 0.008 | 0.008 |
| `_left_ifft_conj` | 1 | 0.008 | 0.008 |

## Tracing cache misses

Total: **326** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:112:31` | 28 | never seen function: fft id=140696109761856 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:294:18` | 24 | never seen function: <jax._src.util.HashablePartial object at 0x7fa6fe9aa8d0> id=140355212847312 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/ffi/phdf5/write.py:53:11` | 12 | never seen function: <jax._src.util.HashablePartial object at 0x7fa6d8232270> id=140354567479920 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:120:22` | 10 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:399:13` | 10 | never seen function: main.<locals>.sigma_sx id=140307408878976 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:369 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:1191:57` | 10 | never seen function: _get_sigma_tau_kernel.<locals>._tau_kernel id=140640358398720 defined at /global/homes/j/jackm/software/lorrax_C/src/gw |
| `/global/homes/j/jackm/software/lorrax_C/src/ffi/phdf5/read.py:42:11` | 9 | never seen function: <jax._src.util.HashablePartial object at 0x7f9720121310> id=140287054844688 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:356:17` | 9 | never seen function: <jax._src.util.HashablePartial object at 0x7fa609f58ef0> id=140351108386544 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:243:8` | 7 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=140640756785984 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:126:24` | 7 | never seen function: _get_chi_minimax_kernel.<locals>._tau_step id=140307409486848 defined at /global/homes/j/jackm/software/lorrax_C/src/gw |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_init.py:423:40` | 5 | never seen function: convert_element_type id=140640756391968 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:190:29` | 5 | never seen function: convert_element_type id=140364498143168 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:173:12` | 5 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: f64[9,80] closest seen input |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:583:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140640757540768 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:363:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140640756782784 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:343:12` | 4 | never seen function: compute_ZCT_from_left_right_zchunk.<locals>._left_ifft_conj id=140640757308352 defined at /global/homes/j/jackm/softwar |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:344:10` | 4 | never seen function: compute_ZCT_from_left_right_zchunk.<locals>._right_ifft_mul_fft id=140682326319616 defined at /global/homes/j/jackm/sof |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:304:11` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:111:15` | 4 | never seen function: gather id=140307409109216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:117:15` | 4 | never seen function: gather id=140354308446208 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:1279:13` | 4 | never seen function: convert_element_type id=140640357950752 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:881:21` | 4 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: c128[4,21,9,80,80] closest seen inp |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:970:27` | 3 | never seen function: reshape id=140302249147456 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:128:30` | 3 | never seen function: dynamic_slice id=140307409492608 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:677:18` | 2 | never seen function: convert_element_type id=140362773765536 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:679:18` | 2 | never seen function: _identity_fn id=140300531402528 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:680:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[9,640,80,2] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:532:20` | 2 | never seen function: iota id=140692730763584 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:534:20` | 2 | never seen function: iota id=140640757314592 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:540:24` | 2 | never seen function: dynamic_slice id=140640757532608 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
