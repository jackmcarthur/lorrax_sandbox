# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/C_heuristic_4gb_16gpu/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 661 | 2.302 | 0.569 |
| jaxpr→MLIR | 333 | 1.080 | 0.152 |
| XLA compile | 333 | 15.868 | 0.765 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `broadcast_in_dim` | 66 | 2.109 | 0.184 |
| `convert_element_type` | 31 | 1.134 | 0.322 |
| `true_divide` | 11 | 0.910 | 0.120 |
| `add` | 19 | 0.843 | 0.085 |
| `gather` | 20 | 0.828 | 0.075 |
| `multiply` | 16 | 0.819 | 0.071 |
| `_kernel` | 1 | 0.765 | 0.765 |
| `sigma_sx` | 2 | 0.752 | 0.376 |
| `iota` | 13 | 0.533 | 0.071 |
| `subtract` | 8 | 0.516 | 0.073 |
| `concatenate` | 12 | 0.357 | 0.066 |
| `less` | 7 | 0.351 | 0.094 |
| `dynamic_slice` | 9 | 0.344 | 0.061 |
| `_mean` | 3 | 0.339 | 0.127 |
| `sigma_coh` | 1 | 0.333 | 0.333 |
| `select_n` | 7 | 0.326 | 0.066 |
| `_compute_CCT_LR` | 1 | 0.287 | 0.287 |
| `get_sqrt_v_and_phase` | 2 | 0.266 | 0.235 |
| `transpose` | 8 | 0.249 | 0.066 |
| `scatter` | 4 | 0.236 | 0.140 |
| `_einsum` | 4 | 0.230 | 0.082 |
| `reshape` | 10 | 0.209 | 0.043 |
| `_per_rank` | 5 | 0.204 | 0.045 |
| `squeeze` | 6 | 0.186 | 0.059 |
| `_squeeze` | 4 | 0.185 | 0.090 |
| `eigh` | 2 | 0.173 | 0.163 |
| `_broadcast_arrays` | 6 | 0.167 | 0.060 |
| `matmul` | 4 | 0.157 | 0.063 |
| `_multi_slice` | 2 | 0.156 | 0.084 |
| `conjugate` | 4 | 0.141 | 0.070 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 1 | 0.569 | 0.569 |
| `get_sqrt_v_and_phase` | 2 | 0.347 | 0.321 |
| `_fft_and_rslice` | 2 | 0.304 | 0.237 |
| `_chi_scan` | 1 | 0.209 | 0.209 |
| `_left_ifft_conj` | 1 | 0.103 | 0.103 |
| `fft_impl` | 12 | 0.102 | 0.028 |
| `_build_G` | 1 | 0.072 | 0.072 |
| `trace` | 1 | 0.056 | 0.056 |
| `_where` | 11 | 0.043 | 0.015 |
| `_solve_all_at_once` | 1 | 0.041 | 0.041 |
| `_compute_CCT_LR` | 1 | 0.034 | 0.034 |
| `_fft_gather_reshard` | 1 | 0.031 | 0.031 |
| `_diag` | 1 | 0.027 | 0.027 |
| `_right_ifft_mul_fft` | 1 | 0.027 | 0.027 |
| `inv` | 1 | 0.025 | 0.025 |
| `solve` | 1 | 0.024 | 0.024 |
| `broadcast_in_dim` | 70 | 0.021 | 0.001 |
| `multiply` | 51 | 0.020 | 0.001 |
| `_moveaxis` | 62 | 0.016 | 0.001 |
| `add` | 42 | 0.016 | 0.001 |
| `_accum` | 1 | 0.015 | 0.015 |
| `true_divide` | 35 | 0.014 | 0.001 |
| `cholesky` | 1 | 0.012 | 0.012 |
| `_take` | 1 | 0.010 | 0.010 |
| `clip` | 2 | 0.009 | 0.008 |
| `_einsum` | 14 | 0.009 | 0.001 |
| `hartree` | 1 | 0.008 | 0.008 |
| `_reduce_sum` | 16 | 0.008 | 0.001 |
| `convert_element_type` | 30 | 0.007 | 0.001 |
| `norm` | 2 | 0.007 | 0.005 |

## Tracing cache misses

Total: **293** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:114:31` | 16 | never seen function: fft id=140649191343264 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:424:10` | 15 | never seen function: convert_element_type id=140661730717632 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/chi_from_dipole.py:43:12` | 13 | never seen function: add id=140661845498816 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:246:8` | 7 | never seen function: compute_CCT_from_left_right.<locals>._compute_CCT_LR id=140644214794784 defined at /global/homes/j/jackm/software/lorra |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:402:11` | 6 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 3 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:113:15` | 6 | never seen function: gather id=140663374425088 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/wavefunction_bundle.py:119:15` | 6 | never seen function: gather id=140663374428608 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/chi_from_dipole.py:45:10` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[26],  args[1]: i64[] closest seen input ty |
| `/global/homes/j/jackm/software/lorrax_C/src/common/chi_from_dipole.py:46:10` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[54],  args[1]: i64[] closest seen input ty |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:30:24` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[262144],  args[1]: i64[] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:385:18` | 5 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=140663562712576 defined at /global/homes/j/jackm/software/lorrax_C/src/fil |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=140661854327392 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:122:22` | 5 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/global/homes/j/jackm/software/lorrax_C/src/common/isdf_fitting.py:828:11` | 4 | never seen function: _make_fit_one_rchunk_kernel.<locals>._kernel id=139711409763840 defined at /global/homes/j/jackm/software/lorrax_C/src/ |
| `/global/homes/j/jackm/software/lorrax_C/src/file_io/_slab_io_ffi.py:382:21` | 4 | never seen function: convert_element_type id=140663562701376 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1026:16` | 4 | never seen function: broadcast_in_dim id=140663563735808 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/gw_init.py:523:40` | 4 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:412:12` | 4 | never seen function: gather id=140663374612256 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/w_isdf.py:413:12` | 4 | never seen function: gather id=140663374615936 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:272:84` | 4 | for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566 tracing context doesn't match, e.g. due to config or context manager closest s |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:300:11` | 4 | never seen function: convert_element_type id=139709774381280 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:425:12` | 4 | never seen function: broadcast_in_dim id=140661726140416 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:425:30` | 4 | never seen function: broadcast_in_dim id=140661726142176 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:426:11` | 4 | never seen function: broadcast_in_dim id=140661726143776 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/compute_vcoul.py:1018:27` | 3 | never seen function: reshape id=140663563741728 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:24:20` | 3 | never seen function: broadcast_in_dim id=140446525029120 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/vcoul.py:24:10` | 3 | never seen function: broadcast_in_dim id=139709778869632 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:845:18` | 2 | never seen function: convert_element_type id=140649181371648 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:847:18` | 2 | never seen function: _identity_fn id=140649192490624 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/load_wfns.py:848:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[9,640,80,2] closest seen input type sign |

## Persistent cache misses

_None._
