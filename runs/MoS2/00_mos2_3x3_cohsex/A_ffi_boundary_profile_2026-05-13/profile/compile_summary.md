# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_profile_2026-05-13/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 1526 | 2.164 | 0.124 |
| jaxpr→MLIR | 549 | 2.025 | 0.124 |
| XLA compile | 582 | 21.509 | 1.111 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 10 | 5.946 | 1.111 |
| `_per_rank` | 50 | 1.956 | 0.131 |
| `broadcast_in_dim` | 59 | 1.264 | 0.174 |
| `sigma_sx` | 8 | 1.095 | 0.275 |
| `convert_element_type` | 54 | 0.973 | 0.298 |
| `_take` | 26 | 0.894 | 0.103 |
| `multiply` | 25 | 0.714 | 0.071 |
| `true_divide` | 18 | 0.670 | 0.099 |
| `add` | 16 | 0.547 | 0.063 |
| `concatenate` | 17 | 0.457 | 0.074 |
| `swapaxes` | 5 | 0.400 | 0.184 |
| `dynamic_slice` | 28 | 0.396 | 0.059 |
| `eigh` | 3 | 0.342 | 0.170 |
| `cumsum` | 4 | 0.306 | 0.097 |
| `_einsum` | 8 | 0.287 | 0.151 |
| `get_sqrt_v_and_phase` | 1 | 0.281 | 0.281 |
| `select_n` | 5 | 0.278 | 0.063 |
| `hartree` | 2 | 0.248 | 0.243 |
| `_ifft_contract_fft` | 3 | 0.247 | 0.234 |
| `_psum` | 40 | 0.241 | 0.045 |
| `squeeze` | 28 | 0.239 | 0.029 |
| `sigma_coh` | 2 | 0.234 | 0.226 |
| `_ifft_conj` | 4 | 0.232 | 0.116 |
| `conjugate` | 6 | 0.212 | 0.131 |
| `gather` | 8 | 0.200 | 0.068 |
| `transpose` | 13 | 0.198 | 0.087 |
| `_multi_slice` | 5 | 0.185 | 0.062 |
| `less` | 3 | 0.165 | 0.057 |
| `_local_fft` | 1 | 0.158 | 0.158 |
| `floor_divide` | 2 | 0.144 | 0.073 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 8 | 0.338 | 0.124 |
| `_per_rank` | 51 | 0.184 | 0.049 |
| `fn` | 5 | 0.178 | 0.042 |
| `_take` | 61 | 0.115 | 0.004 |
| `sigma_sx` | 4 | 0.111 | 0.029 |
| `_solve_all_at_once` | 2 | 0.092 | 0.047 |
| `_convolve` | 4 | 0.084 | 0.022 |
| `multiply` | 179 | 0.069 | 0.002 |
| `solve` | 3 | 0.058 | 0.021 |
| `_right_ifft_contract_fft` | 3 | 0.058 | 0.020 |
| `_ifft_contract_fft` | 3 | 0.056 | 0.021 |
| `_psum` | 42 | 0.047 | 0.002 |
| `_where` | 52 | 0.044 | 0.002 |
| `convert_element_type` | 58 | 0.041 | 0.029 |
| `_reduce_sum` | 66 | 0.039 | 0.001 |
| `add` | 100 | 0.038 | 0.001 |
| `_solve_w` | 1 | 0.037 | 0.037 |
| `minimax_tau_integrate_chi` | 1 | 0.036 | 0.036 |
| `_einsum` | 51 | 0.031 | 0.001 |
| `_compute_S_omega_jit` | 1 | 0.027 | 0.027 |
| `_ifft_conj` | 3 | 0.025 | 0.009 |
| `get_sqrt_v_and_phase` | 1 | 0.025 | 0.025 |
| `_lu_solve` | 8 | 0.024 | 0.004 |
| `_moveaxis` | 91 | 0.022 | 0.001 |
| `less` | 48 | 0.020 | 0.001 |
| `_local_fftn` | 6 | 0.019 | 0.005 |
| `_left_ifft_conj` | 3 | 0.019 | 0.007 |
| `inv` | 1 | 0.018 | 0.018 |
| `true_divide` | 44 | 0.017 | 0.001 |
| `trace` | 4 | 0.017 | 0.004 |

## Tracing cache misses

Total: **629** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/_slab_io_ffi.py:697:17` | 20 | never seen function: _FfiBackend.read_slab.<locals>._per_rank id=139995399834816 defined at /global/u2/j/jackm/software/lorrax_A/src/file_io |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:956:18` | 19 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 tracing context doesn't match, e.g. due to config or context manager closest seen  |
| `/global/u2/j/jackm/software/lorrax_A/src/common/fft_helpers.py:343:15` | 19 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[320,320,3,3,1] closest seen input type signature |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:122:10` | 19 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[4],  x: i32[4],  y: i32[4] closes |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:99:28` | 18 | for dynamic_slice defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[4,4],  args[1]: i64[],  args[2] |
| `/global/u2/j/jackm/software/lorrax_A/src/common/fft_helpers.py:325:15` | 17 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 tracing context doesn't match, e.g. due to config or context manager closest seen context |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/_slab_io_ffi.py:585:18` | 17 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=139996413066624 defined at /global/u2/j/jackm/software/lorrax_A/src/file_i |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:883:9` | 15 | never seen function: dynamic_slice id=140054593360800 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:99:11` | 15 | never seen function: dynamic_slice id=139996410415840 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:46:11` | 13 | never seen function: cumsum id=140060605374176 defined at /opt/jax/jax/_src/numpy/reductions.py:2030 |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:598:18` | 10 | never seen function: _where id=140003385472512 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:108:15` | 10 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[80,4,17676],  indices: i32[9,24,24,80] |
| `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:1408:15` | 10 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/_slab_io_ffi.py:131:11` | 10 | for _psum defined at /opt/jax/jax/experimental/multihost_utils.py:42 never seen input type signature: x: i64[4,3] closest seen input type si |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:957:18` | 8 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[1,1,9,1],  x: c128[20,2,9,1963],  |
| `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:634:9` | 8 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[5760],  x: i32[5760],  y: i32[576 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/wfn_transforms.py:635:9` | 8 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:602:24` | 7 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: bool[9],  indices: i32[9] closest seen inpu |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:770:12` | 7 | never seen function: convert_element_type id=140054590782208 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:881:18` | 7 | never seen function: broadcast_in_dim id=140054593584992 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:891:9` | 7 | never seen function: broadcast_in_dim id=140054593365120 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/gamma_matrices.py:47:17` | 7 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[4],  args[1]: i64[] closest seen input typ |
| `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:1583:40` | 7 | never seen function: broadcast_in_dim id=140054461386400 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:1584:42` | 7 | never seen function: broadcast_in_dim id=140054461390720 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py:1181:16` | 7 | never seen function: compute_V_q_tile.<locals>._init_V id=139995400748896 defined at /global/u2/j/jackm/software/lorrax_A/src/gw/v_q_tile.py |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:587:21` | 6 | never seen function: read_kchunk_union_sharded.<locals>._per_rank id=139996413065984 defined at /global/u2/j/jackm/software/lorrax_A/src/ffi |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:600:22` | 6 | for _take defined at /opt/jax/jax/_src/numpy/indexing.py:136 never seen input type signature: a: c128[9,1963],  indices: i32[9] closest seen |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:890:10` | 6 | never seen function: broadcast_in_dim id=140054593371520 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/file_io/wfn_loader.py:878:16` | 6 | never seen function: convert_element_type id=139996411238720 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:917:35` | 6 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[656,656],  x: c128[656,656],  y:  |

## Persistent cache misses

_None._
