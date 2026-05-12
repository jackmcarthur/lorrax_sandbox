# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/B_bispinor_profile/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 1161 | 1.529 | 0.129 |
| jaxpr→MLIR | 473 | 1.844 | 0.136 |
| XLA compile | 511 | 23.475 | 0.646 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 5 | 2.639 | 0.646 |
| `broadcast_in_dim` | 98 | 2.551 | 0.177 |
| `gather` | 34 | 1.945 | 0.120 |
| `concatenate` | 24 | 1.208 | 0.077 |
| `multiply` | 23 | 1.129 | 0.063 |
| `_per_rank` | 28 | 1.082 | 0.176 |
| `true_divide` | 20 | 0.993 | 0.100 |
| `add` | 20 | 0.968 | 0.068 |
| `_batched_chol` | 2 | 0.945 | 0.560 |
| `_psum` | 27 | 0.859 | 0.073 |
| `dynamic_slice` | 15 | 0.774 | 0.065 |
| `convert_element_type` | 36 | 0.674 | 0.051 |
| `transpose` | 10 | 0.673 | 0.099 |
| `_einsum` | 7 | 0.601 | 0.119 |
| `negative` | 9 | 0.440 | 0.058 |
| `_take` | 20 | 0.419 | 0.075 |
| `subtract` | 8 | 0.390 | 0.062 |
| `_compute_CCT_LR` | 4 | 0.373 | 0.204 |
| `_multi_slice` | 6 | 0.346 | 0.062 |
| `reshape` | 15 | 0.312 | 0.040 |
| `get_sqrt_v_and_phase` | 1 | 0.296 | 0.296 |
| `maximum` | 11 | 0.295 | 0.059 |
| `trace` | 2 | 0.261 | 0.131 |
| `_identity_fn` | 7 | 0.257 | 0.069 |
| `_fft_gather_reshard` | 1 | 0.209 | 0.209 |
| `matmul` | 7 | 0.171 | 0.030 |
| `iota` | 6 | 0.168 | 0.058 |
| `squeeze` | 7 | 0.166 | 0.028 |
| `cumsum` | 2 | 0.164 | 0.089 |
| `_local_fft` | 1 | 0.160 | 0.160 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_per_rank` | 30 | 0.259 | 0.039 |
| `_batched_chol` | 2 | 0.253 | 0.129 |
| `_kernel` | 5 | 0.229 | 0.055 |
| `_fft_and_rslice` | 4 | 0.076 | 0.020 |
| `_take` | 30 | 0.061 | 0.004 |
| `multiply` | 125 | 0.060 | 0.002 |
| `_compute_CCT_LR` | 4 | 0.059 | 0.016 |
| `add` | 109 | 0.047 | 0.002 |
| `_psum` | 27 | 0.033 | 0.002 |
| `_where` | 37 | 0.031 | 0.003 |
| `_right_ifft_mul_fft` | 4 | 0.028 | 0.007 |
| `broadcast_in_dim` | 102 | 0.025 | 0.000 |
| `true_divide` | 56 | 0.025 | 0.001 |
| `get_sqrt_v_and_phase` | 1 | 0.023 | 0.023 |
| `_solve_all_at_once` | 4 | 0.022 | 0.006 |
| `_left_ifft_conj` | 4 | 0.022 | 0.006 |
| `_reduce_sum` | 32 | 0.022 | 0.001 |
| `_fft_gather_reshard` | 1 | 0.016 | 0.016 |
| `floor_divide` | 4 | 0.016 | 0.004 |
| `_einsum` | 25 | 0.015 | 0.001 |
| `less` | 31 | 0.012 | 0.001 |
| `_moveaxis` | 49 | 0.011 | 0.002 |
| `_accum` | 4 | 0.010 | 0.003 |
| `subtract` | 22 | 0.010 | 0.001 |
| `_compute_P_vertex` | 6 | 0.009 | 0.002 |
| `_broadcast_arrays` | 41 | 0.009 | 0.000 |
| `cholesky` | 3 | 0.009 | 0.004 |
| `convert_element_type` | 38 | 0.008 | 0.000 |
| `remainder` | 4 | 0.008 | 0.003 |
| `negative` | 30 | 0.008 | 0.002 |

## Tracing cache misses

Total: **495** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/common/bispinor_init.py:48:16` | 28 | never seen function: broadcast_in_dim id=140399401396928 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:1145:22` | 16 | never seen function: convert_element_type id=140399934414080 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/bispinor_init.py:49:9` | 15 | never seen function: dynamic_slice id=140399401390208 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/gamma_matrices.py:45:11` | 15 | never seen function: cumsum id=140406375749312 defined at /opt/jax/jax/_src/numpy/reductions.py:2030 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:325:15` | 13 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:1147:22` | 12 | never seen function: gather id=140399400233024 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:1148:23` | 12 | never seen function: gather id=140399400237824 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:385:18` | 12 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=140399935064640 defined at /global/homes/j/jackm/software/lorrax_B/src/fil |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:1146:23` | 10 | never seen function: gather id=140399400228064 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:382:21` | 10 | never seen function: convert_element_type id=140399935067360 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:271:29` | 10 | never seen function: convert_element_type id=140399934028864 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:343:15` | 9 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/common/cholesky_2d.py:65:19` | 8 | never seen function: broadcast_in_dim id=140399400882624 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/phdf5_wfn_reader.py:375:12` | 8 | never seen function: _where id=140406377131488 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/phdf5_wfn_reader.py:458:14` | 8 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/homes/j/jackm/software/lorrax_B/src/common/gvec_fft_box.py:120:19` | 8 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[9,24,24,80],  x: i32[9,24,24,80], |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/gw_init.py:750:40` | 8 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_B/src/gw/w_isdf.py:375:11` | 8 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/common/gamma_matrices.py:46:17` | 7 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[4],  args[1]: i64[] closest seen input typ |
| `/global/homes/j/jackm/software/lorrax_B/src/common/bispinor_init.py:54:57` | 6 | never seen function: dynamic_slice id=140399401503936 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:1160:40` | 6 | never seen function: broadcast_in_dim id=140399400424832 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/cholesky_2d.py:148:22` | 6 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:209:8` | 6 | never seen function: compute_pair_density_with_vertex.<locals>._compute_P_vertex id=140399401391328 defined at /global/homes/j/jackm/softwar |
| `/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py:1161:42` | 5 | never seen function: broadcast_in_dim id=140399400428032 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/gpu_utils.py:41:12` | 4 | never seen function: convert_element_type id=139683210417536 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:195:12` | 4 | neve2026-05-03 03:33:24,427 jax._src.pjit WARNING: TRACING CACHE MISS at /global/homes/j/jackm/software/lorra2026-05-03 03:33:24,427 jax._sr |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:222:22` | 4 | never seen function: fft id=139688819655520 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:83:19` | 4 | never seen function: _make_jittable_local_fft.<locals>._local_fft id=139682740984928 defined at /global/homes/j/jackm/software/lorrax_B/src/ |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:848:18` | 4 | never seen function: convert_element_type id=139682742788288 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/common/load_wfns.py:850:18` | 4 | never seen function: _identity_fn id=140406379840832 defined at /opt/jax/jax/_src/pjit.py:2575 |

## Persistent cache misses

_None._
