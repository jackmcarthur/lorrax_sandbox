# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/A_low_mem_test/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 70 | 0.054 | 0.019 |
| jaxpr→MLIR | 30 | 0.219 | 0.132 |
| XLA compile | 30 | 0.948 | 0.205 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_fft_gather_reshard` | 1 | 0.205 | 0.205 |
| `broadcast_in_dim` | 7 | 0.149 | 0.062 |
| `convert_element_type` | 1 | 0.144 | 0.144 |
| `true_divide` | 5 | 0.138 | 0.036 |
| `_multi_slice` | 2 | 0.055 | 0.031 |
| `iota` | 2 | 0.044 | 0.022 |
| `_squeeze` | 2 | 0.031 | 0.017 |
| `_identity_fn` | 2 | 0.028 | 0.015 |
| `scatter` | 2 | 0.027 | 0.014 |
| `dynamic_slice` | 1 | 0.026 | 0.026 |
| `maximum` | 1 | 0.026 | 0.026 |
| `add` | 1 | 0.023 | 0.023 |
| `multiply` | 1 | 0.021 | 0.021 |
| `_compute_P_traced` | 1 | 0.020 | 0.020 |
| `squeeze` | 1 | 0.012 | 0.012 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_fft_gather_reshard` | 1 | 0.019 | 0.019 |
| `fft_impl` | 3 | 0.008 | 0.004 |
| `_take` | 1 | 0.004 | 0.004 |
| `true_divide` | 7 | 0.003 | 0.001 |
| `_moveaxis` | 10 | 0.003 | 0.000 |
| `multiply` | 7 | 0.002 | 0.000 |
| `broadcast_in_dim` | 7 | 0.002 | 0.000 |
| `iota` | 2 | 0.002 | 0.002 |
| `_compute_P_traced` | 1 | 0.002 | 0.002 |
| `conjugate` | 3 | 0.001 | 0.001 |
| `add` | 4 | 0.001 | 0.000 |
| `fft` | 4 | 0.001 | 0.000 |
| `negative` | 2 | 0.001 | 0.000 |
| `_einsum` | 1 | 0.001 | 0.001 |
| `_where` | 1 | 0.001 | 0.001 |
| `_multi_slice` | 2 | 0.001 | 0.000 |
| `scatter` | 2 | 0.001 | 0.000 |
| `_identity_fn` | 2 | 0.000 | 0.000 |
| `_squeeze` | 2 | 0.000 | 0.000 |
| `dynamic_slice` | 1 | 0.000 | 0.000 |
| `maximum` | 1 | 0.000 | 0.000 |
| `sqrt` | 1 | 0.000 | 0.000 |
| `less` | 1 | 0.000 | 0.000 |
| `convert_element_type` | 1 | 0.000 | 0.000 |
| `exp` | 1 | 0.000 | 0.000 |
| `squeeze` | 1 | 0.000 | 0.000 |
| `_broadcast_arrays` | 1 | 0.000 | 0.000 |

## Tracing cache misses

Total: **29** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_A/src/common/fft_helpers.py:114:31` | 4 | never seen function: fft id=140431012856992 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:733:32` | 4 | never seen function: get_sharded_wfns_centroids.<locals>._fft_gather_reshard id=140421537199936 defined at /global/homes/j/jackm/software/lo |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:842:18` | 2 | never seen function: convert_element_type id=140431002901760 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:844:18` | 2 | never seen function: _identity_fn id=140431013987968 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:845:19` | 2 | for _identity_fn defined at /opt/jax/jax/_src/pjit.py:2575 never seen input type signature: x: c128[9,640,80,2] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:682:20` | 2 | never seen function: iota id=140427719989568 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:684:20` | 2 | never seen function: iota id=140421537188096 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:690:24` | 2 | never seen function: dynamic_slice id=140421537192896 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:717:22` | 2 | never seen function: _where id=140431011262240 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:843:19` | 1 | never seen function: broadcast_in_dim id=140422486970016 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:683:20` | 1 | never seen function: broadcast_in_dim id=140422486982176 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:876:26` | 1 | never seen function: scatter id=140421537381280 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/load_wfns.py:877:27` | 1 | for scatter defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[9,640,80,2],  args[1]: i32[0],  args[ |
| `/global/homes/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:1076:44` | 1 | never seen function: broadcast_in_dim id=140421536942336 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:1077:46` | 1 | never seen function: broadcast_in_dim id=140422486982336 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_A/src/common/isdf_fitting.py:171:8` | 1 | never seen function: compute_pair_density_spin_traced.<locals>._compute_P_traced id=140421536947296 defined at /global/homes/j/jackm/softwar |

## Persistent cache misses

_None._
