# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_wfn_rchunk_shardmap_2026-05-13/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 1329 | 1.467 | 0.038 |
| jaxpr→MLIR | 460 | 1.787 | 0.127 |
| XLA compile | 486 | 15.712 | 1.261 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 15 | 5.448 | 1.261 |
| `_per_rank` | 47 | 1.619 | 0.101 |
| `broadcast_in_dim` | 45 | 0.866 | 0.167 |
| `sigma_sx` | 7 | 0.804 | 0.268 |
| `convert_element_type` | 42 | 0.740 | 0.301 |
| `_take` | 22 | 0.568 | 0.083 |
| `true_divide` | 17 | 0.556 | 0.121 |
| `dynamic_slice` | 20 | 0.373 | 0.057 |
| `eigh` | 3 | 0.335 | 0.165 |
| `squeeze` | 23 | 0.315 | 0.032 |
| `add` | 9 | 0.295 | 0.063 |
| `transpose` | 14 | 0.259 | 0.087 |
| `_psum` | 44 | 0.253 | 0.041 |
| `multiply` | 16 | 0.237 | 0.065 |
| `cumsum` | 4 | 0.209 | 0.083 |
| `swapaxes` | 4 | 0.201 | 0.098 |
| `concatenate` | 8 | 0.194 | 0.064 |
| `_local_fft` | 1 | 0.176 | 0.176 |
| `_multi_slice` | 5 | 0.174 | 0.059 |
| `select_n` | 4 | 0.170 | 0.058 |
| `floor_divide` | 2 | 0.140 | 0.073 |
| `conjugate` | 6 | 0.136 | 0.062 |
| `_expand_band_diagonal_to_kij_jit` | 4 | 0.135 | 0.067 |
| `_einsum` | 7 | 0.135 | 0.111 |
| `_pair_density` | 4 | 0.117 | 0.106 |
| `less` | 3 | 0.112 | 0.054 |
| `_broadcast_arrays` | 6 | 0.091 | 0.029 |
| `gather` | 7 | 0.080 | 0.057 |
| `_potrf` | 1 | 0.080 | 0.080 |
| `_identity_fn` | 6 | 0.074 | 0.032 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 9 | 0.119 | 0.025 |
| `fn` | 4 | 0.117 | 0.033 |
| `_take` | 59 | 0.106 | 0.004 |
| `_per_rank` | 47 | 0.093 | 0.013 |
| `sigma_sx` | 3 | 0.087 | 0.031 |
| `_convolve` | 3 | 0.067 | 0.025 |
| `multiply` | 173 | 0.063 | 0.002 |
| `_ifft_contract_fft` | 3 | 0.058 | 0.021 |
| `_psum` | 45 | 0.051 | 0.002 |
| `_reduce_sum` | 64 | 0.039 | 0.001 |
| `_solve_w` | 1 | 0.038 | 0.038 |
| `convert_element_type` | 42 | 0.037 | 0.027 |
| `minimax_tau_integrate_chi` | 1 | 0.034 | 0.034 |
| `_where` | 38 | 0.033 | 0.002 |
| `add` | 90 | 0.031 | 0.002 |
| `_einsum` | 45 | 0.028 | 0.001 |
| `_compute_S_omega_jit` | 1 | 0.027 | 0.027 |
| `get_sqrt_v_and_phase` | 1 | 0.026 | 0.026 |
| `_ifft_conj` | 3 | 0.023 | 0.008 |
| `_local_fftn` | 6 | 0.022 | 0.005 |
| `floor_divide` | 5 | 0.019 | 0.004 |
| `inv` | 1 | 0.018 | 0.018 |
| `_right_ifft_contract_fft` | 1 | 0.018 | 0.018 |
| `true_divide` | 44 | 0.017 | 0.001 |
| `solve` | 1 | 0.017 | 0.017 |
| `_moveaxis` | 74 | 0.017 | 0.001 |
| `less` | 42 | 0.016 | 0.001 |
| `broadcast_in_dim` | 50 | 0.012 | 0.000 |
| `_left_ifft_conj` | 2 | 0.011 | 0.006 |
| `_accum` | 5 | 0.011 | 0.003 |

## Tracing cache misses

Total: **527** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/_slab_io_ffi.py:706:17` | 20 | never seen function: _get_read_sm.<locals>._per_rank id=140649378043904 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/ |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/_slab_io_ffi.py:621:18` | 17 | never seen function: _get_write_sm.<locals>._per_rank id=140650992106400 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:124:10` | 17 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[4],  x: i32[4],  y: i32[4] closes |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:101:28` | 16 | for dynamic_slice defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[4,4],  args[1]: i64[],  args[2] |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/fft_helpers.py:343:15` | 15 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[328,328,3,3,1] closest seen input type signature |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:48:11` | 14 | never seen function: cumsum id=140657891368672 defined at /opt/jax/jax/_src/numpy/reductions.py:2030 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:109:15` | 13 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[16,4,17676],  indices: i32[9,24,24,80] |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/fft_helpers.py:325:15` | 13 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:971:18` | 11 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:603:18` | 10 | never seen function: _where id=140657892718080 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:101:11` | 10 | never seen function: dynamic_slice id=140651525088864 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/_slab_io_ffi.py:132:11` | 10 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: i64[4,3] closest seen input type si |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1550:15` | 9 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:838:9` | 9 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[5760],  x: i32[5760],  y: i32[576 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:605:22` | 7 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[9,1963],  indices: i32[9] closest seen |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:49:17` | 7 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[4],  args[1]: i64[] closest seen input typ |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:471:23` | 7 | never seen function: fft id=140657894345600 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:1181:16` | 7 | never seen function: compute_V_q_tile.<locals>._init_V id=140649377664832 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/sr |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:607:24` | 6 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: bool[9],  indices: i32[9] closest seen inpu |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/load_wfns.py:784:54` | 6 | never seen function: convert_element_type id=140651526040256 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1727:42` | 6 | never seen function: broadcast_in_dim id=140034596911872 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:839:9` | 6 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:305:23` | 6 | never seen function: make_sharded_fftn_3d.<locals>._local_fftn id=140649382584672 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lor |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:578:22` | 5 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: i64[4,5,4] closest seen input type  |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:592:21` | 5 | never seen function: _read_kchunk_union_sharded_cached.<locals>._per_rank id=140651526033216 defined at /pscratch/sd/j/jackm/lorrax_sandbox/ |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:972:18` | 5 | for _where defined at /opt2026-05-13 00:28:05,745 jax._src.interpreters.pxla WARNING: Compiling _per_rank with global shapes and types [Shap |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:775:12` | 5 | never seen function: convert_element_type id=140034928116736 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:170:8` | 5 | never seen function: accum_pair_density.<locals>._accum id=140650991877824 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/s |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_bispinor.py:111:20` | 5 | never seen function: add id=140651526554144 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/tagged_arrays.py:294:29` | 5 | never seen function: convert_element_type id=140649374742176 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
