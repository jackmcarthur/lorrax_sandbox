# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_gflat_bispinor_unified_2026-05-12/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 1747 | 3.243 | 0.138 |
| jaxpr→MLIR | 593 | 3.190 | 0.148 |
| XLA compile | 601 | 40.472 | 1.561 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 8 | 12.006 | 1.561 |
| `fn` | 17 | 3.278 | 0.523 |
| `sigma_sx` | 8 | 2.177 | 0.281 |
| `_per_rank` | 73 | 1.920 | 0.099 |
| `broadcast_in_dim` | 70 | 1.771 | 0.169 |
| `convert_element_type` | 50 | 1.095 | 0.304 |
| `multiply` | 20 | 0.989 | 0.080 |
| `_take` | 24 | 0.966 | 0.092 |
| `_psum` | 45 | 0.858 | 0.064 |
| `true_divide` | 14 | 0.832 | 0.121 |
| `dynamic_slice` | 26 | 0.665 | 0.067 |
| `transpose` | 9 | 0.656 | 0.097 |
| `concatenate` | 16 | 0.566 | 0.070 |
| `gather` | 7 | 0.545 | 0.121 |
| `conjugate` | 6 | 0.527 | 0.175 |
| `eigh` | 3 | 0.527 | 0.177 |
| `hartree` | 2 | 0.511 | 0.258 |
| `_ifft_contract_fft` | 4 | 0.503 | 0.255 |
| `add` | 15 | 0.501 | 0.070 |
| `_einsum` | 7 | 0.500 | 0.118 |
| `swapaxes` | 4 | 0.470 | 0.124 |
| `sigma_coh` | 2 | 0.453 | 0.230 |
| `<lambda>` | 18 | 0.419 | 0.076 |
| `scatter` | 5 | 0.417 | 0.087 |
| `_batched_chol` | 1 | 0.383 | 0.383 |
| `subtract` | 9 | 0.364 | 0.068 |
| `minimax_tau_integrate_chi` | 1 | 0.359 | 0.359 |
| `_multi_slice` | 6 | 0.343 | 0.060 |
| `_mean` | 4 | 0.327 | 0.109 |
| `_identity_fn` | 7 | 0.314 | 0.130 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 7 | 0.709 | 0.138 |
| `fn` | 24 | 0.473 | 0.046 |
| `_solve_all_at_once` | 8 | 0.228 | 0.049 |
| `_per_rank` | 74 | 0.134 | 0.013 |
| `_right_ifft_contract_fft` | 8 | 0.129 | 0.021 |
| `_batched_chol` | 1 | 0.128 | 0.128 |
| `_take` | 69 | 0.122 | 0.004 |
| `solve` | 7 | 0.119 | 0.023 |
| `sigma_sx` | 4 | 0.109 | 0.029 |
| `_convolve` | 4 | 0.082 | 0.022 |
| `_ifft_contract_fft` | 4 | 0.081 | 0.023 |
| `multiply` | 201 | 0.068 | 0.001 |
| `_psum` | 46 | 0.051 | 0.002 |
| `_where` | 56 | 0.046 | 0.004 |
| `_left_ifft_conj` | 8 | 0.045 | 0.006 |
| `_reduce_sum` | 72 | 0.042 | 0.001 |
| `add` | 126 | 0.040 | 0.001 |
| `floor_divide` | 10 | 0.037 | 0.004 |
| `_solve_w` | 1 | 0.037 | 0.037 |
| `minimax_tau_integrate_chi` | 1 | 0.034 | 0.034 |
| `_ifft_conj` | 4 | 0.034 | 0.009 |
| `_lu_solve` | 9 | 0.030 | 0.004 |
| `_einsum` | 50 | 0.029 | 0.001 |
| `_compute_S_omega_jit` | 1 | 0.027 | 0.027 |
| `_accum` | 16 | 0.025 | 0.002 |
| `less` | 61 | 0.024 | 0.001 |
| `_moveaxis` | 104 | 0.023 | 0.001 |
| `remainder` | 10 | 0.020 | 0.003 |
| `inv` | 1 | 0.019 | 0.019 |
| `true_divide` | 49 | 0.018 | 0.001 |

## Tracing cache misses

Total: **756** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:587:21` | 30 | never seen function: read_kchunk_union_sharded.<locals>._per_rank id=139818299056480 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/ |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:122:10` | 28 | never seen function: _where id=139824369415680 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/fft_helpers.py:325:15` | 26 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/fft_helpers.py:343:15` | 22 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[320,320,3,3,1] closest seen input type signature |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/_slab_io_ffi.py:697:17` | 20 | never seen function: _FfiBackend.read_slab.<locals>._per_rank id=139816154238624 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorr |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/_slab_io_ffi.py:585:18` | 17 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=139818301193120 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lor |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:956:18` | 16 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:776:9` | 16 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[14692],  x: i32[14692],  y: i32[1 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:148:8` | 16 | never seen function: accum_pair_density.<locals>._accum id=139818837016416 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/s |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:883:9` | 15 | never seen function: dynamic_slice id=139818299822528 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:46:11` | 15 | never seen function: cumsum id=139824368082656 defined at /opt/jax/jax/_src/numpy/reductions.py:2030 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:99:28` | 15 | for dynamic_slice defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[4,4],  args[1]: i64[],  args[2] |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:99:11` | 14 | never seen function: dynamic_slice id=139818300922816 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:777:9` | 12 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:669:11` | 12 | never seen function: accumulate_rchunk_to_gflat.<locals>.fn id=139818299820448 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:919:15` | 12 | never seen function: _lu_solve id=139824362470304 defined at /opt/jax/jax/_src/lax/linalg.py:1566 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:598:18` | 10 | never seen function: _where id=139824369415680 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/_slab_io_ffi.py:131:11` | 10 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: i64[4,3] closest seen input type si |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:891:9` | 8 | never seen function: broadcast_in_dim id=139818299827008 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1408:15` | 8 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:781:15` | 8 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[9,24],  indices: i32[14692] closest se |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:783:15` | 8 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[9,80],  indices: i32[14692] closest se |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:394:11` | 8 | never seen function: to_rchunk.<locals>.fn id=139818837003616 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:453:17` | 8 | never seen function: z_q_from_pair.<locals>._left_ifft_conj id=139818299966784 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:454:10` | 8 | never seen function: z_q_from_pair.<locals>._right_ifft_contract_fft id=139818299966944 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sourc |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1016:16` | 8 | never seen function: solve_zeta.<locals>._reshard_z id=139818299830528 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/c |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1025:17` | 8 | never seen function: solve_zeta.<locals>._solve_all_at_once id=139818299821728 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:602:24` | 7 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: bool[9],  indices: i32[9] closest seen inpu |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:957:18` | 7 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[1,1,9,1],  x: c128[4,2,9,1963],   |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:108:15` | 7 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[16,4,17676],  indices: i32[9,24,24,80] |

## Persistent cache misses

_None._
