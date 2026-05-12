# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/profile_v4/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 584 | 1.799 | 0.547 |
| jaxpr→MLIR | 249 | 1.232 | 0.239 |
| XLA compile | 249 | 16.501 | 3.187 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_build_stk_at_q` | 1 | 3.187 | 3.187 |
| `multiply` | 35 | 1.832 | 0.057 |
| `broadcast_in_dim` | 35 | 1.561 | 0.385 |
| `concatenate` | 14 | 1.006 | 0.161 |
| `add` | 16 | 0.856 | 0.060 |
| `compute_V_H_and_V_xc` | 1 | 0.719 | 0.719 |
| `_reduce_sum` | 10 | 0.643 | 0.079 |
| `dynamic_slice` | 9 | 0.458 | 0.056 |
| `convert_element_type` | 16 | 0.457 | 0.050 |
| `gather` | 7 | 0.421 | 0.083 |
| `integer_pow` | 8 | 0.407 | 0.054 |
| `true_divide` | 6 | 0.358 | 0.067 |
| `sqrt` | 6 | 0.354 | 0.127 |
| `exp` | 4 | 0.340 | 0.095 |
| `transpose` | 4 | 0.316 | 0.090 |
| `_pad` | 6 | 0.315 | 0.055 |
| `_table_interp` | 3 | 0.299 | 0.101 |
| `scatter` | 4 | 0.273 | 0.072 |
| `matmul` | 9 | 0.238 | 0.032 |
| `squeeze` | 9 | 0.224 | 0.026 |
| `subtract` | 4 | 0.199 | 0.050 |
| `conjugate` | 3 | 0.162 | 0.059 |
| `real` | 4 | 0.156 | 0.052 |
| `negative` | 3 | 0.154 | 0.057 |
| `accumulate_species_on_G` | 1 | 0.135 | 0.135 |
| `_power` | 1 | 0.124 | 0.124 |
| `_where` | 2 | 0.105 | 0.053 |
| `less` | 2 | 0.105 | 0.054 |
| `select_n` | 2 | 0.105 | 0.052 |
| `compute_per_band_kinetic` | 1 | 0.103 | 0.103 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_value_and_jacfwd` | 1 | 0.547 | 0.547 |
| `_per_k_full_S` | 1 | 0.386 | 0.386 |
| `_chi_sum_over_k` | 1 | 0.206 | 0.206 |
| `compute_V_H_and_V_xc` | 1 | 0.165 | 0.165 |
| `_sternheimer_core` | 3 | 0.110 | 0.045 |
| `_build_stk_at_q` | 1 | 0.055 | 0.055 |
| `_table_interp` | 4 | 0.043 | 0.012 |
| `multiply` | 103 | 0.035 | 0.001 |
| `apply_H_k` | 3 | 0.034 | 0.012 |
| `accumulate_species_on_G` | 1 | 0.021 | 0.021 |
| `interp_uniform_jax` | 1 | 0.019 | 0.019 |
| `add` | 45 | 0.013 | 0.001 |
| `clip` | 10 | 0.013 | 0.002 |
| `species_structure_factors` | 1 | 0.011 | 0.011 |
| `_einsum` | 23 | 0.010 | 0.001 |
| `broadcast_in_dim` | 35 | 0.009 | 0.000 |
| `_reduce_sum` | 14 | 0.008 | 0.001 |
| `remainder` | 3 | 0.007 | 0.003 |
| `subtract` | 23 | 0.007 | 0.001 |
| `_where` | 9 | 0.007 | 0.001 |
| `true_divide` | 23 | 0.007 | 0.001 |
| `matmul` | 10 | 0.006 | 0.001 |
| `concatenate` | 14 | 0.004 | 0.000 |
| `convert_element_type` | 16 | 0.004 | 0.000 |
| `fft` | 17 | 0.004 | 0.001 |
| `maximum` | 11 | 0.004 | 0.001 |
| `less` | 11 | 0.004 | 0.001 |
| `negative` | 20 | 0.004 | 0.000 |
| `dynamic_slice` | 9 | 0.004 | 0.001 |
| `_pad` | 6 | 0.003 | 0.001 |

## Tracing cache misses

Total: **178** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:371:10` | 9 | never seen function: add id=139834541476096 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:72:8` | 6 | never seen function: dynamic_slice id=139833935803904 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:225:21` | 5 | never seen function: convert_element_type id=139833942853728 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:763:14` | 5 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 4 | never seen function: fft id=139837329735904 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:8` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:372:10` | 4 | never seen function: concatenate id=139833938248064 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:553:11` | 4 | never seen function: convert_element_type id=139833937906720 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:823:11` | 4 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1963],  args[1]: i32[] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:264:9` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:344:13` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:356:25` | 3 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen input type si |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:260:10` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: i32[1947],  min: i64[],  max: i64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:262:8` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[1947],  min: f64[],  max: f64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:281:11` | 3 | never seen function: _table_interp id=139834590527328 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:248 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:79:8` | 3 | never seen function: broadcast_in_dim id=139833938747744 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:23` | 3 | never seen function: broadcast_in_dim id=139833938750304 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:93:43` | 3 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:373:14` | 3 | for transpose defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[1947,102] closest seen input type s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:114:13` | 3 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:196:10` | 3 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:364:12` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 tracing context doesn't match, e.g. due to config or context manager closest s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=139834542580384 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:155:11` | 2 | never seen function: _where id=139837328239456 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/run_full.py:137:19` | 2 | never seen function: broadcast_in_dim id=139834070495328 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:448:15` | 2 | never seen function: dynamic_slice id=139833938260384 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:558:17` | 2 | never seen function: _pad id=139837325775936 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:560:13` | 2 | for _pad defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 never seen input type signature: array: i32[1947],  constant_values: i64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:563:16` | 2 | for _pad defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 never seen input type signature: array: c128[102,1947],  constant_values: f64[ |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:565:38` | 2 | never seen function: broadcast_in_dim id=139833937913120 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
