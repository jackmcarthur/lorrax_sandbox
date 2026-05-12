# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/51_ffi_async_profile_C/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 698 | 0.813 | 0.126 |
| jaxpr→MLIR | 285 | 1.094 | 0.151 |
| XLA compile | 285 | 18.246 | 0.804 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `<unnamed wrapped function>` | 54 | 1.916 | 0.048 |
| `broadcast_in_dim` | 37 | 1.401 | 0.176 |
| `_batched_chol` | 1 | 0.804 | 0.804 |
| `convert_element_type` | 15 | 0.791 | 0.312 |
| `true_divide` | 11 | 0.774 | 0.110 |
| `sigma_sx` | 2 | 0.735 | 0.383 |
| `_identity_fn` | 15 | 0.713 | 0.148 |
| `_tau_kernel` | 1 | 0.675 | 0.675 |
| `gather` | 10 | 0.631 | 0.109 |
| `concatenate` | 9 | 0.460 | 0.057 |
| `add` | 8 | 0.430 | 0.059 |
| `_fft_gather_reshard` | 1 | 0.399 | 0.399 |
| `reshape` | 13 | 0.382 | 0.036 |
| `transpose` | 7 | 0.376 | 0.081 |
| `multiply` | 7 | 0.371 | 0.057 |
| `_project_tau_onto_omega` | 2 | 0.331 | 0.167 |
| `iota` | 7 | 0.330 | 0.051 |
| `_psum` | 8 | 0.320 | 0.060 |
| `_tau_step` | 1 | 0.320 | 0.320 |
| `_multi_slice` | 6 | 0.316 | 0.057 |
| `sigma_coh` | 1 | 0.316 | 0.316 |
| `_fft_and_rslice` | 1 | 0.296 | 0.296 |
| `_solve_w` | 1 | 0.272 | 0.272 |
| `_prepare_sigma_state` | 1 | 0.260 | 0.260 |
| `dynamic_slice` | 5 | 0.258 | 0.053 |
| `subtract` | 4 | 0.246 | 0.068 |
| `_reduce_max` | 2 | 0.245 | 0.127 |
| `_where` | 4 | 0.238 | 0.064 |
| `hartree` | 1 | 0.218 | 0.218 |
| `_compute_CCT_LR` | 1 | 0.212 | 0.212 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_batched_chol` | 1 | 0.126 | 0.126 |
| `fft_impl` | 42 | 0.072 | 0.004 |
| `_solve_w` | 1 | 0.040 | 0.040 |
| `_tau_kernel` | 1 | 0.032 | 0.032 |
| `sigma_sx` | 1 | 0.030 | 0.030 |
| `get_sqrt_v_and_phase` | 1 | 0.027 | 0.027 |
| `_convolve` | 1 | 0.025 | 0.025 |
| `_moveaxis` | 96 | 0.025 | 0.001 |
| `_project_tau_onto_omega` | 2 | 0.025 | 0.013 |
| `multiply` | 61 | 0.024 | 0.002 |
| `_sigma_kij_kernel` | 1 | 0.022 | 0.022 |
| `_fft_gather_reshard` | 1 | 0.021 | 0.021 |
| `_fft_and_rslice` | 1 | 0.020 | 0.020 |
| `_tau_step` | 1 | 0.020 | 0.020 |
| `ifftn` | 2 | 0.018 | 0.010 |
| `true_divide` | 33 | 0.015 | 0.001 |
| `_prepare_sigma_state` | 1 | 0.013 | 0.013 |
| `_compute_CCT_LR` | 1 | 0.013 | 0.013 |
| `add` | 35 | 0.012 | 0.001 |
| `_einsum` | 18 | 0.012 | 0.002 |
| `_where` | 16 | 0.011 | 0.001 |
| `_single_chunk_proc` | 1 | 0.011 | 0.011 |
| `broadcast_in_dim` | 37 | 0.010 | 0.001 |
| `fft` | 34 | 0.010 | 0.001 |
| `_psum` | 8 | 0.009 | 0.001 |
| `_build_tau_operands` | 1 | 0.009 | 0.009 |
| `conjugate` | 30 | 0.009 | 0.002 |
| `_solve_all_at_once` | 1 | 0.008 | 0.008 |
| `_build_G` | 1 | 0.008 | 0.008 |
| `_reduce_sum` | 12 | 0.007 | 0.001 |

## Tracing cache misses

Total: **326** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:112:31` | 28 | never seen function: fft id=140313439372608 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:295:18` | 24 | never seen function: <jax._src.util.HashablePartial object at 0x7fa7bb47fa70> id=140358378322544 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/ffi/phdf5/write.py:53:11` | 12 | never seen function: <jax._src.util.HashablePartial object at 0x7f5e9fee5070> id=140044386848880 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:120:22` | 10 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:387:13` | 10 | never seen function: main.<locals>.sigma_sx id=140312099782720 defined at /global/homes/j/jackm/software/lorrax_C/src/gw/gw_jax.py:357 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:1191:57` | 10 | never seen function: _get_sigma_tau_kernel.<locals>._tau_kernel id=140258401894080 defined at /global/homes/j/jackm/software/lorrax_C/src/gw |
| `/global/homes/j/jackm/software/lorrax_C/src/ffi/phdf5/read.py:42:11` | 9 | never seen function: <jax._src.util.HashablePartial object at 0x7f9b687a00e0> id=140305449484512 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:357:17` | 9 | never seen function: <jax._src.util.HashablePartial object at 0x7f9b687a1d60> id=140305449491808 defined at |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:243:8` | 7 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=140305845474848 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:126:24` | 7 | never seen function: _get_chi_minimax_kernel.<locals>._tau_step id=140259878208672 defined at /global/homes/j/jackm/software/lorrax_C/src/gw |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_init.py:423:40` | 5 | never seen function: convert_element_type id=140358057369632 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:190:29` | 5 | never seen function: convert_element_type id=140322357508064 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:173:12` | 5 | for _identity_fn defined at /opt/jax/jax/experimental/multihost_utils.py:96 never seen input type signature: x: f64[9,80] closest seen input |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:583:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140305846065792 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:363:25` | 4 | never seen function: get_sharded_wfns_rchunk_slice.<locals>._fft_and_rslice id=140044386172128 defined at /global/homes/j/jackm/software/lor |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:343:12` | 4 | never seen function: compute_ZCT_from_left_right_zchunk.<locals>._left_ifft_conj id=140044455782304 defined at /global/homes/j/jackm/softwar |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:344:10` | 4 | never seen function: compute_ZCT_from_left_right_zchunk.<locals>._right_ifft_mul_fft id=140044455987296 defined at /global/homes/j/jackm/sof |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:304:11` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:111:15` | 4 | never seen function: gather id=140312099980448 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:117:15` | 4 | never seen function: gather id=140312099986528 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:1279:13` | 4 | never seen function: convert_element_type id=140311348242016 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/ppm_sigma.py:881:21` | 4 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: c128[4,21,9,80,80] closest seen inp |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:970:27` | 3 | never seen function: reshape id=140305850185216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:128:30` | 3 | never seen function: dynamic_slice id=140259878214432 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:677:18` | 2 | never seen function: convert_element_type id=140364636708256 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:679:18` | 2 | never seen function: _identity_fn id=140313440503584 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:680:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[9,640,80,2] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:532:20` | 2 | never seen function: iota id=140309674121856 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:534:20` | 2 | never seen function: iota id=140305845823232 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:540:24` | 2 | never seen function: dynamic_slice id=140305845828192 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
