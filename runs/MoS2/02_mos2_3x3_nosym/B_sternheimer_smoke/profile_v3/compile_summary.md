# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/profile_v3/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 650 | 1.436 | 0.559 |
| jaxpr→MLIR | 413 | 1.363 | 0.119 |
| XLA compile | 413 | 22.818 | 0.725 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `multiply` | 65 | 3.622 | 0.077 |
| `broadcast_in_dim` | 60 | 2.477 | 0.389 |
| `add` | 26 | 1.388 | 0.063 |
| `concatenate` | 15 | 1.120 | 0.159 |
| `true_divide` | 13 | 0.781 | 0.068 |
| `_reduce_sum` | 12 | 0.763 | 0.077 |
| `compute_V_H_and_V_xc` | 1 | 0.725 | 0.725 |
| `convert_element_type` | 23 | 0.724 | 0.063 |
| `exp` | 7 | 0.664 | 0.120 |
| `gather` | 11 | 0.632 | 0.064 |
| `sqrt` | 10 | 0.595 | 0.125 |
| `integer_pow` | 12 | 0.570 | 0.054 |
| `transpose` | 7 | 0.567 | 0.099 |
| `dynamic_slice` | 11 | 0.566 | 0.057 |
| `_einsum` | 7 | 0.556 | 0.188 |
| `_table_interp` | 5 | 0.547 | 0.123 |
| `scatter` | 7 | 0.518 | 0.084 |
| `apply_H_k` | 1 | 0.515 | 0.515 |
| `subtract` | 9 | 0.483 | 0.059 |
| `matmul` | 13 | 0.424 | 0.071 |
| `_power` | 3 | 0.376 | 0.127 |
| `_pad` | 6 | 0.327 | 0.056 |
| `_squeeze` | 10 | 0.309 | 0.056 |
| `conjugate` | 5 | 0.285 | 0.062 |
| `squeeze` | 11 | 0.274 | 0.027 |
| `less` | 5 | 0.266 | 0.055 |
| `negative` | 5 | 0.262 | 0.059 |
| `scatter-add` | 1 | 0.225 | 0.225 |
| `mul` | 4 | 0.221 | 0.058 |
| `_broadcast_arrays` | 8 | 0.210 | 0.032 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_value_and_jacfwd` | 1 | 0.559 | 0.559 |
| `_chi_sum_over_k` | 1 | 0.216 | 0.216 |
| `compute_V_H_and_V_xc` | 1 | 0.178 | 0.178 |
| `_sternheimer_core` | 3 | 0.110 | 0.046 |
| `_table_interp` | 4 | 0.043 | 0.012 |
| `apply_H_k` | 3 | 0.041 | 0.019 |
| `multiply` | 105 | 0.036 | 0.002 |
| `accumulate_species_on_G` | 1 | 0.022 | 0.022 |
| `interp_uniform_jax` | 1 | 0.020 | 0.020 |
| `broadcast_in_dim` | 60 | 0.017 | 0.002 |
| `add` | 46 | 0.014 | 0.001 |
| `clip` | 10 | 0.013 | 0.002 |
| `species_structure_factors` | 1 | 0.012 | 0.012 |
| `_einsum` | 23 | 0.012 | 0.002 |
| `remainder` | 3 | 0.009 | 0.004 |
| `_where` | 9 | 0.008 | 0.002 |
| `true_divide` | 23 | 0.008 | 0.001 |
| `_reduce_sum` | 14 | 0.007 | 0.001 |
| `subtract` | 22 | 0.007 | 0.001 |
| `convert_element_type` | 23 | 0.006 | 0.001 |
| `matmul` | 10 | 0.006 | 0.001 |
| `_broadcast_arrays` | 18 | 0.004 | 0.001 |
| `concatenate` | 15 | 0.004 | 0.000 |
| `negative` | 21 | 0.004 | 0.000 |
| `maximum` | 11 | 0.004 | 0.001 |
| `_tpa` | 1 | 0.004 | 0.004 |
| `scatter` | 7 | 0.004 | 0.002 |
| `_take` | 1 | 0.004 | 0.004 |
| `_pad` | 6 | 0.004 | 0.001 |
| `gather` | 11 | 0.004 | 0.001 |

## Tracing cache misses

Total: **244** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:371:10` | 13 | never seen function: add id=139972784736672 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/run_full.py:225:11` | 13 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature closest seen input type signature has 9 mismatch |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:72:8` | 10 | never seen function: dynamic_slice id=139973283821216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:356:25` | 9 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen input type si |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:93:43` | 9 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:372:10` | 6 | never seen function: concatenate id=139972984881376 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:364:12` | 6 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 tracing context doesn't match, e.g. due to config or context manager closest s |
| `/global/homes/j/jackm/software/lorrax_B/src/solvers/sternheimer_solve.py:144:14` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[9,1963],  args[1]: i32[] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:225:21` | 5 | never seen function: convert_element_type id=139972986177792 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:373:14` | 5 | for transpose defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[1947,102] closest seen input type s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:763:14` | 5 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:86:12` | 5 | never seen function: broadcast_in_dim id=139972117866560 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 4 | never seen function: fft id=139976072790400 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:8` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:553:11` | 4 | never seen function: convert_element_type id=139972984540352 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:823:11` | 4 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1963],  args[1]: i32[] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:343:11` | 4 | never seen function: convert_element_type id=139972116514112 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:264:9` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:344:13` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:260:10` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: i32[1947],  min: i64[],  max: i64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:262:8` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[1947],  min: f64[],  max: f64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:281:11` | 3 | never seen function: _table_interp id=139973333565664 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:248 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:79:8` | 3 | never seen function: broadcast_in_dim id=139972783743488 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:23` | 3 | never seen function: broadcast_in_dim id=139972783746048 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:114:13` | 3 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:196:10` | 3 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:354:14` | 3 | never seen function: reshape id=139972114068352 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:356:21` | 3 | never seen function: reshape id=139972113124384 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=139973285831488 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:155:11` | 2 | never seen function: _where id=139976071277568 defined at /opt/jax/jax/_src/numpy/util.py:287 |

## Persistent cache misses

_None._
