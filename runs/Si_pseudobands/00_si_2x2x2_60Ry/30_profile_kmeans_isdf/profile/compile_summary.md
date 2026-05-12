# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/30_profile_kmeans_isdf/profile/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 94 | 0.064 | 0.017 |
| jaxpr→MLIR | 63 | 0.300 | 0.141 |
| XLA compile | 63 | 3.291 | 0.458 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_unstack` | 1 | 0.458 | 0.458 |
| `broadcast_in_dim` | 11 | 0.441 | 0.048 |
| `convert_element_type` | 9 | 0.278 | 0.047 |
| `kmeans_update_step` | 1 | 0.259 | 0.259 |
| `dynamic_slice` | 5 | 0.238 | 0.048 |
| `scatter` | 3 | 0.209 | 0.081 |
| `multiply` | 3 | 0.146 | 0.052 |
| `concatenate` | 2 | 0.102 | 0.052 |
| `squeeze` | 4 | 0.096 | 0.024 |
| `add` | 2 | 0.090 | 0.045 |
| `_argmin` | 1 | 0.080 | 0.080 |
| `pbc_distance_sq_batch` | 1 | 0.078 | 0.078 |
| `pbc_distance_sq_single` | 1 | 0.072 | 0.072 |
| `_squeeze` | 3 | 0.070 | 0.025 |
| `_reduce_sum` | 1 | 0.066 | 0.066 |
| `true_divide` | 1 | 0.053 | 0.053 |
| `_reduce_max` | 1 | 0.053 | 0.053 |
| `_linspace` | 1 | 0.051 | 0.051 |
| `conjugate` | 1 | 0.049 | 0.049 |
| `reshape` | 2 | 0.049 | 0.024 |
| `transpose` | 1 | 0.048 | 0.048 |
| `less` | 1 | 0.048 | 0.048 |
| `select_n` | 1 | 0.047 | 0.047 |
| `matmul` | 1 | 0.047 | 0.047 |
| `minimum` | 1 | 0.046 | 0.046 |
| `real` | 1 | 0.045 | 0.045 |
| `_broadcast_arrays` | 1 | 0.024 | 0.024 |
| `fft` | 1 | 0.024 | 0.024 |
| `copy` | 1 | 0.023 | 0.023 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `kmeans_update_step` | 1 | 0.017 | 0.017 |
| `pbc_distance_sq_batch` | 1 | 0.005 | 0.005 |
| `pbc_distance_sq_single` | 1 | 0.005 | 0.005 |
| `_linspace` | 1 | 0.004 | 0.004 |
| `_einsum` | 3 | 0.004 | 0.002 |
| `broadcast_in_dim` | 11 | 0.003 | 0.000 |
| `_reduce_sum` | 3 | 0.002 | 0.002 |
| `convert_element_type` | 9 | 0.002 | 0.000 |
| `subtract` | 7 | 0.002 | 0.000 |
| `multiply` | 6 | 0.002 | 0.000 |
| `dynamic_slice` | 5 | 0.001 | 0.000 |
| `matmul` | 1 | 0.001 | 0.001 |
| `true_divide` | 4 | 0.001 | 0.001 |
| `reshape` | 2 | 0.001 | 0.001 |
| `add` | 4 | 0.001 | 0.000 |
| `_where` | 1 | 0.001 | 0.001 |
| `scatter` | 3 | 0.001 | 0.000 |
| `_unstack` | 1 | 0.001 | 0.001 |
| `_one_hot` | 1 | 0.001 | 0.001 |
| `squeeze` | 4 | 0.001 | 0.000 |
| `remainder` | 1 | 0.001 | 0.001 |
| `less` | 1 | 0.001 | 0.001 |
| `round` | 3 | 0.001 | 0.000 |
| `_broadcast_arrays` | 2 | 0.001 | 0.000 |
| `_reduce_max` | 1 | 0.000 | 0.000 |
| `concatenate` | 2 | 0.000 | 0.000 |
| `_squeeze` | 3 | 0.000 | 0.000 |
| `minimum` | 1 | 0.000 | 0.000 |
| `conjugate` | 1 | 0.000 | 0.000 |
| `maximum` | 1 | 0.000 | 0.000 |

## Tracing cache misses

Total: **52** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/get_charge_density.py:37:14` | 7 | never seen function: add id=140076471015104 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:417:14` | 3 | never seen function: broadcast_in_dim id=140076471224896 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:423:16` | 3 | never seen function: broadcast_in_dim id=140076471224736 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:444:16` | 3 | never seen function: convert_element_type id=140077138570080 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/get_charge_density.py:56:21` | 2 | never seen function: convert_element_type id=140081662402112 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/get_charge_density.py:82:48` | 2 | never seen function: dynamic_slice id=140077675007648 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/get_charge_density.py:28:14` | 2 | never seen function: convert_element_type id=140076470810592 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/get_charge_density.py:32:9` | 2 | never seen function: dynamic_slice id=140076470811392 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:439:16` | 2 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f32[] closest seen input type |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:444:36` | 2 | never seen function: dynamic_slice id=140077138566720 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:447:18` | 2 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[] closest seen input type |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:485:42` | 2 | never seen function: dynamic_slice id=140077137711872 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/get_charge_density.py:54:26` | 1 | never seen function: _unstack id=140081663506720 defined at /opt/jax/jax/_src/numpy/array_methods.py:636 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/get_charge_density.py:40:17` | 1 | never seen function: fft id=140081672373728 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:701:14` | 1 | never seen function: convert_element_type id=140076471099904 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:702:15` | 1 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f32[3,3] closest seen input t |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:412:30` | 1 | never seen function: transpose id=140076471102944 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:412:20` | 1 | never seen function: copy id=140076471105824 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:418:8` | 1 | never seen function: _linspace id=140081668480096 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:6468 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:424:15` | 1 | never seen function: reshape id=140076471238496 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:457:23` | 1 | for dynamic_slice defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f32[400,3],  args[1]: i64[],  args[2 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:158:30` | 1 | never seen function: round id=140081668370208 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3451 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:456:25` | 1 | never seen function: pbc_distance_sq_single id=140077674952992 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/km |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:475:14` | 1 | never seen function: broadcast_in_dim id=140077138369632 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:127:30` | 1 | for round defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3451 never seen input type signature: a: f32[46656,400,3] closest seen input type  |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:199:19` | 1 | never seen function: pbc_distance_sq_batch id=140077674948992 defined at /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kme |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:207:11` | 1 | never seen function: _one_hot id=140081663058912 defined at /opt/jax/jax/_src/nn/functions.py:652 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:223:16` | 1 | never seen function: _where id=140081670795360 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:230:25` | 1 | never seen function: remainder id=140081670063584 defined at /opt/jax/jax/_src/numpy/ufuncs.py:3019 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax/src/centroid/kmeans_isdf.py:234:36` | 1 | for round defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3451 never seen input type signature: a: f32[400,3] closest seen input type signat |

## Persistent cache misses

_None._
