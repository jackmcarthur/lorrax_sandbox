# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_rebased_D_usage_profile_2026-05-13/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 1350 | 1.624 | 0.101 |
| jaxpr→MLIR | 453 | 1.898 | 0.125 |
| XLA compile | 478 | 15.484 | 1.324 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 14 | 5.399 | 1.324 |
| `_per_rank` | 47 | 1.631 | 0.104 |
| `broadcast_in_dim` | 45 | 0.835 | 0.179 |
| `sigma_sx` | 7 | 0.805 | 0.263 |
| `_take` | 24 | 0.739 | 0.083 |
| `convert_element_type` | 42 | 0.717 | 0.304 |
| `dynamic_slice` | 20 | 0.368 | 0.056 |
| `eigh` | 3 | 0.335 | 0.166 |
| `true_divide` | 15 | 0.322 | 0.098 |
| `multiply` | 17 | 0.300 | 0.069 |
| `add` | 9 | 0.292 | 0.063 |
| `transpose` | 14 | 0.270 | 0.228 |
| `squeeze` | 21 | 0.252 | 0.031 |
| `_psum` | 43 | 0.247 | 0.041 |
| `cumsum` | 3 | 0.211 | 0.080 |
| `swapaxes` | 4 | 0.208 | 0.104 |
| `concatenate` | 8 | 0.182 | 0.057 |
| `_multi_slice` | 5 | 0.178 | 0.064 |
| `_local_fft` | 1 | 0.170 | 0.170 |
| `conjugate` | 6 | 0.142 | 0.068 |
| `floor_divide` | 2 | 0.141 | 0.071 |
| `_expand_band_diagonal_to_kij_jit` | 4 | 0.136 | 0.068 |
| `gather` | 8 | 0.131 | 0.057 |
| `_einsum` | 5 | 0.124 | 0.110 |
| `_pair_density` | 4 | 0.116 | 0.107 |
| `select_n` | 3 | 0.110 | 0.054 |
| `less` | 3 | 0.109 | 0.054 |
| `_squeeze` | 6 | 0.098 | 0.037 |
| `_potrf` | 1 | 0.078 | 0.078 |
| `_identity_fn` | 6 | 0.072 | 0.031 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 10 | 0.220 | 0.101 |
| `fn` | 4 | 0.145 | 0.042 |
| `_per_rank` | 47 | 0.096 | 0.014 |
| `_take` | 57 | 0.092 | 0.005 |
| `sigma_sx` | 3 | 0.083 | 0.029 |
| `_compute_S_omega_jit` | 1 | 0.080 | 0.080 |
| `multiply` | 176 | 0.063 | 0.001 |
| `_convolve` | 3 | 0.063 | 0.022 |
| `_ifft_contract_fft` | 3 | 0.061 | 0.023 |
| `_psum` | 46 | 0.055 | 0.002 |
| `_reduce_sum` | 62 | 0.039 | 0.001 |
| `_solve_w` | 1 | 0.037 | 0.037 |
| `minimax_tau_integrate_chi` | 1 | 0.035 | 0.035 |
| `_where` | 40 | 0.033 | 0.002 |
| `add` | 95 | 0.031 | 0.001 |
| `_einsum` | 44 | 0.028 | 0.002 |
| `get_sqrt_v_and_phase` | 1 | 0.027 | 0.027 |
| `_ifft_conj` | 3 | 0.024 | 0.008 |
| `_right_ifft_contract_fft` | 1 | 0.020 | 0.020 |
| `_local_fftn` | 6 | 0.020 | 0.004 |
| `inv` | 1 | 0.019 | 0.019 |
| `_moveaxis` | 79 | 0.019 | 0.002 |
| `floor_divide` | 5 | 0.019 | 0.004 |
| `solve` | 1 | 0.018 | 0.018 |
| `true_divide` | 42 | 0.016 | 0.001 |
| `less` | 40 | 0.016 | 0.001 |
| `broadcast_in_dim` | 48 | 0.012 | 0.000 |
| `_accum` | 6 | 0.011 | 0.002 |
| `_left_ifft_conj` | 2 | 0.011 | 0.005 |
| `convert_element_type` | 41 | 0.010 | 0.001 |

## Tracing cache misses

Total: **525** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/u2/j/jackm/software/lorrax_A/src/common/fft_helpers.py:325:15` | 21 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/_slab_io_ffi.py:706:17` | 20 | never seen function: _get_read_sm.<locals>._per_rank id=140591060133632 defined at /global/u2/j/jackm/software/lorrax_A/src/file_io/_slab_io |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/_slab_io_ffi.py:621:18` | 17 | never seen function: _get_write_sm.<locals>._per_rank id=140593206927712 defined at /global/u2/j/jackm/software/lorrax_A/src/file_io/_slab_i |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:122:10` | 17 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[4],  x: i32[4],  y: i32[4] closes |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:46:11` | 15 | never seen function: cumsum id=140114611789696 defined at /opt/jax/jax/_src/numpy/reductions.py:2030 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/fft_helpers.py:343:15` | 15 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[328,328,3,3,1] closest seen input type signature |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:99:28` | 14 | for dynamic_slice defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[4,4],  args[1]: i64[],  args[2] |
| `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:108:15` | 12 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[16,4,17676],  indices: i32[9,24,24,80] |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:967:18` | 11 | for _where defined at /opt/jax/jax/_src/numpy/ut2026-05-12 20:24:49,058 jax._src.compiler WARNING: Not writing persistent cache entry since  |
| `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:750:9` | 11 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[5760],  x: i32[5760],  y: i32[576 |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:599:18` | 10 | never seen function: _where id=140599803279008 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/_slab_io_ffi.py:132:11` | 10 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: i64[4,3] closest seen input type si |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:99:11` | 9 | never seen function: dynamic_slice id=140592674876096 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:1502:15` | 8 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:601:22` | 7 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[9,1963],  indices: i32[9] closest seen |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:47:17` | 7 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[4],  args[1]: i64[] closest seen input typ |
| `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:1181:16` | 7 | never seen function: compute_V_q_tile.<locals>._init_V id=140591064898816 defined at /global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:603:24` | 6 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: bool[9],  indices: i32[9] closest seen inpu |
| `/global/u2/j/jackm/software/lorrax_A/src/common/load_wfns.py:784:54` | 6 | never seen function: convert_element_type id=140593210646880 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:751:9` | 6 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:305:23` | 6 | never seen function: make_sharded_fftn_3d.<locals>._local_fftn id=140591064461728 defined at /global/u2/j/jackm/software/lorrax_A/src/common |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:573:22` | 5 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: i64[4,5,4] closest seen input type  |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:588:21` | 5 | never seen function: _read_kchunk_union_sharded_cached.<locals>._per_rank id=140593206920832 defined at /global/u2/j/jackm/software/lorrax_A |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:968:18` | 5 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[1,1,9,1],  x: c128[4,2,9,1963],   |
| `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:148:8` | 5 | never seen function: accum_pair_density.<locals>._accum id=140593207230848 defined at /global/u2/j/jackm/software/lorrax_A/src/common/isdf_f |
| `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_bispinor.py:111:20` | 5 | never seen function: add id=140593206677952 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/tagged_arrays.py:294:29` | 5 | never seen function: convert_element_type id=140591057129600 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/runtime/__init__.py:194:12` | 4 | never seen function: _psum id=140620923529440 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/fft_helpers.py:222:22` | 4 | never seen function: fft id=140114614799392 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/fft_helpers.py:83:19` | 4 | never seen function: _make_jittable_local_fft.<locals>._local_fft id=140108613331648 defined at /global/u2/j/jackm/software/lorrax_A/src/com |

## Persistent cache misses

_None._
