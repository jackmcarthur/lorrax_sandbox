# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_perf_before_2026-05-12/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 1558 | 2.348 | 0.139 |
| jaxpr→MLIR | 576 | 2.225 | 0.128 |
| XLA compile | 606 | 19.823 | 1.481 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 10 | 5.848 | 1.481 |
| `_per_rank` | 74 | 1.952 | 0.102 |
| `sigma_sx` | 8 | 1.125 | 0.293 |
| `broadcast_in_dim` | 58 | 0.990 | 0.167 |
| `_take` | 26 | 0.929 | 0.134 |
| `convert_element_type` | 51 | 0.757 | 0.297 |
| `true_divide` | 19 | 0.639 | 0.099 |
| `add` | 18 | 0.571 | 0.119 |
| `multiply` | 24 | 0.532 | 0.116 |
| `_ifft_contract_fft` | 5 | 0.485 | 0.233 |
| `concatenate` | 18 | 0.484 | 0.071 |
| `eigh` | 3 | 0.413 | 0.209 |
| `dynamic_slice` | 27 | 0.285 | 0.060 |
| `sigma_coh` | 2 | 0.276 | 0.269 |
| `hartree` | 2 | 0.265 | 0.259 |
| `_einsum` | 8 | 0.261 | 0.126 |
| `squeeze` | 29 | 0.248 | 0.030 |
| `swapaxes` | 4 | 0.247 | 0.122 |
| `_ifft_conj` | 5 | 0.237 | 0.113 |
| `transpose` | 13 | 0.212 | 0.103 |
| `cumsum` | 4 | 0.207 | 0.084 |
| `_psum` | 42 | 0.204 | 0.043 |
| `_multi_slice` | 5 | 0.176 | 0.060 |
| `_local_fft` | 1 | 0.161 | 0.161 |
| `_expand_band_diagonal_to_kij_jit` | 4 | 0.154 | 0.080 |
| `floor_divide` | 2 | 0.144 | 0.073 |
| `conjugate` | 5 | 0.139 | 0.128 |
| `gather` | 8 | 0.132 | 0.056 |
| `reshape` | 8 | 0.129 | 0.032 |
| `_pair_density` | 4 | 0.115 | 0.106 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 8 | 0.372 | 0.139 |
| `fn` | 5 | 0.188 | 0.044 |
| `_per_rank` | 75 | 0.144 | 0.013 |
| `_take` | 60 | 0.109 | 0.004 |
| `sigma_sx` | 4 | 0.108 | 0.029 |
| `_solve_w` | 1 | 0.093 | 0.093 |
| `_solve_all_at_once` | 2 | 0.093 | 0.046 |
| `_lu_solve` | 9 | 0.082 | 0.056 |
| `_convolve` | 4 | 0.081 | 0.022 |
| `_ifft_contract_fft` | 4 | 0.081 | 0.023 |
| `multiply` | 184 | 0.064 | 0.001 |
| `solve` | 3 | 0.060 | 0.022 |
| `_right_ifft_contract_fft` | 3 | 0.059 | 0.020 |
| `lu_solve` | 1 | 0.057 | 0.057 |
| `_psum` | 44 | 0.051 | 0.002 |
| `_reduce_sum` | 71 | 0.042 | 0.001 |
| `_where` | 49 | 0.041 | 0.002 |
| `convert_element_type` | 56 | 0.038 | 0.027 |
| `add` | 98 | 0.034 | 0.001 |
| `minimax_tau_integrate_chi` | 1 | 0.033 | 0.033 |
| `_einsum` | 51 | 0.029 | 0.001 |
| `_compute_S_omega_jit` | 1 | 0.028 | 0.028 |
| `_ifft_conj` | 3 | 0.026 | 0.009 |
| `get_sqrt_v_and_phase` | 1 | 0.025 | 0.025 |
| `_left_ifft_conj` | 4 | 0.024 | 0.006 |
| `_moveaxis` | 94 | 0.022 | 0.001 |
| `_local_fftn` | 6 | 0.022 | 0.004 |
| `broadcast_in_dim` | 66 | 0.018 | 0.001 |
| `less` | 46 | 0.018 | 0.001 |
| `inv` | 1 | 0.018 | 0.018 |

## Tracing cache misses

Total: **667** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:587:21` | 30 | never seen function: read_kchunk_union_sharded.<locals>._per_rank id=140014794031808 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/ |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/fft_helpers.py:325:15` | 23 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:122:10` | 22 | never seen function: _where id=140021088711872 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/fft_helpers.py:343:15` | 20 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[320,320,3,3,1] closest seen input type signature |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/_slab_io_ffi.py:697:17` | 20 | never seen function: _FfiBackend.read_slab.<locals>._per_rank id=140012648517760 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorr |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:46:11` | 18 | never seen function: cumsum id=140021085281696 defined at /opt/jax/jax/_src/numpy/reductions.py:2030 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:99:28` | 17 | for dynamic_slice defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[4,4],  args[1]: i64[],  args[2] |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/_slab_io_ffi.py:585:18` | 17 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=140014794457792 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lor |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:883:9` | 16 | never seen function: dynamic_slice id=139985873791040 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:956:18` | 15 | for _where defined at /opt/jax/jax/_src/numpy/ut2026-05-12 16:50:33,434 jax._src.compiler WARNING: Not writing persistent cache entry since  |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:99:11` | 14 | never seen function: dynamic_slice id=140014794044128 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:598:18` | 10 | never seen function: _where id=140021088711872 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/_slab_io_ffi.py:131:11` | 10 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: i64[4,3] closest seen input type si |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:108:15` | 9 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[16,4,17676],  indices: i32[9,24,24,80] |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:770:12` | 8 | never seen function: convert_element_type id=139985873388224 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:891:9` | 8 | never seen function: broadcast_in_dim id=139663683140832 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1461:15` | 8 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:732:9` | 8 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[5760],  x: i32[5760],  y: i32[576 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/wfn_transforms.py:733:9` | 8 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:148:8` | 8 | never seen function: accum_pair_density.<locals>._accum id=140014263400096 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/s |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:966:15` | 8 | never seen function: _lu_solve id=140021079669344 defined at /opt/jax/jax/_src/lax/linalg.py:1566 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:602:24` | 7 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: bool[9],  indices: i32[9] closest seen inpu |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:957:18` | 7 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[1,1,9,1],  x: c128[4,2,9,1963],   |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/gamma_matrices.py:47:17` | 7 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[4],  args[1]: i64[] closest seen input typ |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:1181:16` | 7 | never seen function: compute_V_q_tile.<locals>._init_V id=140012648189280 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/sr |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:600:22` | 6 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[9,1963],  indices: i32[9] closest seen |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:881:18` | 6 | never seen function: broadcast_in_dim id=139663683745760 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:890:10` | 6 | never seen function: broadcast_in_dim id=139663683229856 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/common/isdf_fitting.py:1638:42` | 6 | never seen function: broadcast_in_dim id=139986004219584 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/file_io/wfn_loader.py:604:18` | 6 | never seen function: _phdf5_unfold_kernel.<locals>._per_rank id=140014794038848 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorra |

## Persistent cache misses

_None._
