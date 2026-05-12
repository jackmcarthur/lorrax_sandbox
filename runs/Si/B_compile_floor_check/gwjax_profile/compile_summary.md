# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 747 | 0.603 | 0.035 |
| jaxpr→MLIR | 340 | 1.068 | 0.119 |
| XLA compile | 354 | 19.041 | 0.739 |

## Top 20 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 77 | 2.293 | 0.183 |
| `gather` | 15 | 0.999 | 0.122 |
| `_kernel` | 2 | 0.990 | 0.739 |
| `add` | 20 | 0.951 | 0.074 |
| `convert_element_type` | 33 | 0.860 | 0.247 |
| `sigma_sx` | 2 | 0.790 | 0.397 |
| `true_divide` | 11 | 0.783 | 0.112 |
| `multiply` | 13 | 0.592 | 0.061 |
| `concatenate` | 14 | 0.580 | 0.088 |
| `subtract` | 10 | 0.549 | 0.073 |
| `_psum` | 15 | 0.524 | 0.065 |
| `_batched_chol` | 1 | 0.509 | 0.509 |
| `iota` | 11 | 0.499 | 0.058 |
| `_compute_CCT_LR` | 2 | 0.489 | 0.249 |
| `eigh` | 3 | 0.452 | 0.158 |
| `transpose` | 10 | 0.444 | 0.087 |
| `_fft_gather_reshard` | 1 | 0.418 | 0.418 |
| `sigma_coh` | 1 | 0.410 | 0.410 |
| `minimax_tau_integrate_chi` | 1 | 0.373 | 0.373 |
| `_per_rank` | 7 | 0.357 | 0.145 |

## Top 20 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_fft_and_rslice` | 2 | 0.041 | 0.021 |
| `fft_impl` | 26 | 0.040 | 0.004 |
| `_solve_w` | 1 | 0.035 | 0.035 |
| `minimax_tau_integrate_chi` | 1 | 0.029 | 0.029 |
| `get_sqrt_v_and_phase` | 1 | 0.025 | 0.025 |
| `broadcast_in_dim` | 78 | 0.022 | 0.002 |
| `_moveaxis` | 81 | 0.020 | 0.001 |
| `inv` | 1 | 0.020 | 0.020 |
| `add` | 52 | 0.019 | 0.002 |
| `solve` | 1 | 0.018 | 0.018 |
| `multiply` | 49 | 0.016 | 0.001 |
| `_psum` | 15 | 0.016 | 0.001 |
| `_fft_gather_reshard` | 1 | 0.016 | 0.016 |
| `_kernel` | 1 | 0.016 | 0.016 |
| `_reduce_sum` | 26 | 0.014 | 0.001 |
| `_per_rank` | 7 | 0.013 | 0.009 |
| `_take` | 4 | 0.013 | 0.004 |
| `_compute_CCT_LR` | 1 | 0.012 | 0.012 |
| `true_divide` | 29 | 0.012 | 0.001 |
| `_einsum` | 17 | 0.011 | 0.001 |

## Tracing cache misses

Total: **332** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/gw/head_correction.py:424:10` | 15 | never seen function: convert_element_type id=139691072954080 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:30:24` | 13 | never seen function: add id=140617706506656 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:24:10` | 12 | never seen function: broadcast_in_dim id=139698849855136 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/vcoul.py:24:20` | 11 | never seen function: broadcast_in_dim id=139698849854816 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:808:11` | 9 | never seen function: _make_jittable_local_fft.<locals>._make_axis_wrapper.<locals>.fft_impl id=139698850988736 defined at /global/homes/j/ja |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:500:6` | 9 | never seen function: _make_jittable_local_fft.<locals>._make_axis_wrapper.<locals>.fft_impl id=139697235215840 defined at /global/homes/j/ja |
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:43:12` | 9 | never seen function: add id=139696704343392 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:222:22` | 8 | never seen function: fft id=140624371532800 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:214:31` | 8 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[4,4,480,480,4] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:45:10` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[8],  args[1]: i64[] closest seen input typ |
| `/global/homes/j/jackm/software/lorrax_B/src/common/chi_from_dipole.py:46:10` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[52],  args[1]: i64[] closest seen input ty |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:83:19` | 5 | never seen function: _make_jittable_local_fft.<locals>._local_f2026-04-25 17:23:09,217 jax._src.dispatch WARNING: Finished tracing + transfo |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:385:18` | 5 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=139698849616640 defined at /global/homes/j/jackm/software/lorrax_B/src/fil |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=139697236482048 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:382:21` | 4 | never seen function: convert_element_type id=139698849620320 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/compute_vcoul.py:1845:32` | 4 | never seen function: _make_V_q_caseA_kernel.<locals>._kernel id=139697237109280 defined at /global/homes/j/jackm/software/lorrax_B/src/gw/co |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/gw_init.py:677:40` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:375:11` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/wavefunction_bundle.py:144:15` | 4 | never seen function: gather id=139697235227200 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/wavefunction_bundle.py:150:15` | 4 | never seen function: gather id=139697235347552 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
