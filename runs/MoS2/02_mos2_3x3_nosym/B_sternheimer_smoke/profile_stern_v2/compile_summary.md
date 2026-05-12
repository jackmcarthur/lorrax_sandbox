# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/profile_stern_v2/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 606 | 0.608 | 0.212 |
| jaxpr→MLIR | 553 | 1.683 | 0.119 |
| XLA compile | 553 | 31.267 | 0.728 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `multiply` | 87 | 4.995 | 0.079 |
| `broadcast_in_dim` | 83 | 3.503 | 0.389 |
| `add` | 30 | 1.668 | 0.077 |
| `concatenate` | 17 | 1.281 | 0.161 |
| `_reduce_sum` | 17 | 1.125 | 0.107 |
| `apply_H_k` | 2 | 1.084 | 0.607 |
| `true_divide` | 15 | 0.924 | 0.070 |
| `subtract` | 16 | 0.917 | 0.078 |
| `scatter` | 10 | 0.900 | 0.162 |
| `sqrt` | 13 | 0.883 | 0.123 |
| `gather` | 14 | 0.877 | 0.109 |
| `_einsum` | 10 | 0.852 | 0.133 |
| `dynamic_slice` | 16 | 0.802 | 0.070 |
| `convert_element_type` | 26 | 0.798 | 0.057 |
| `exp` | 8 | 0.782 | 0.128 |
| `scatter-add` | 4 | 0.758 | 0.239 |
| `compute_V_H_and_V_xc` | 1 | 0.728 | 0.728 |
| `integer_pow` | 12 | 0.562 | 0.053 |
| `matmul` | 15 | 0.558 | 0.070 |
| `transpose` | 7 | 0.545 | 0.098 |
| `_table_interp` | 5 | 0.532 | 0.119 |
| `_squeeze` | 15 | 0.448 | 0.058 |
| `less` | 8 | 0.420 | 0.055 |
| `squeeze` | 16 | 0.406 | 0.028 |
| `conjugate` | 7 | 0.404 | 0.064 |
| `select_n` | 7 | 0.391 | 0.061 |
| `_power` | 3 | 0.372 | 0.125 |
| `mul` | 6 | 0.334 | 0.060 |
| `remainder` | 5 | 0.328 | 0.073 |
| `negative` | 6 | 0.320 | 0.063 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `compute_V_H_and_V_xc` | 1 | 0.212 | 0.212 |
| `_sternheimer_core` | 2 | 0.064 | 0.047 |
| `_table_interp` | 3 | 0.034 | 0.012 |
| `multiply` | 86 | 0.032 | 0.002 |
| `broadcast_in_dim` | 83 | 0.024 | 0.001 |
| `accumulate_species_on_G` | 1 | 0.022 | 0.022 |
| `interp_uniform_jax` | 1 | 0.020 | 0.020 |
| `apply_H_k` | 1 | 0.019 | 0.019 |
| `add` | 43 | 0.014 | 0.001 |
| `species_structure_factors` | 1 | 0.013 | 0.013 |
| `clip` | 8 | 0.011 | 0.002 |
| `convert_element_type` | 26 | 0.008 | 0.001 |
| `_reduce_sum` | 14 | 0.008 | 0.001 |
| `true_divide` | 19 | 0.006 | 0.001 |
| `subtract` | 18 | 0.006 | 0.001 |
| `matmul` | 10 | 0.006 | 0.001 |
| `_einsum` | 11 | 0.006 | 0.001 |
| `remainder` | 3 | 0.006 | 0.003 |
| `_where` | 7 | 0.006 | 0.001 |
| `gather` | 14 | 0.005 | 0.001 |
| `scatter` | 10 | 0.005 | 0.002 |
| `dynamic_slice` | 16 | 0.005 | 0.001 |
| `concatenate` | 17 | 0.005 | 0.000 |
| `squeeze` | 16 | 0.004 | 0.001 |
| `_tpa` | 1 | 0.004 | 0.004 |
| `maximum` | 9 | 0.004 | 0.001 |
| `_take` | 1 | 0.003 | 0.003 |
| `less` | 9 | 0.003 | 0.001 |
| `negative` | 16 | 0.003 | 0.001 |
| `_pad` | 6 | 0.003 | 0.001 |

## Tracing cache misses

Total: **283** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:181:13` | 27 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[9,1963],  args[1]: i32[] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:371:10` | 13 | never seen function: add id=140642651392576 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/run_full.py:210:11` | 13 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature closest seen input type signature has 9 mismatch |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:356:25` | 10 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen input type si |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:72:8` | 10 | never seen function: dynamic_slice id=140642200041280 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:93:43` | 10 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:98:11` | 8 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1],  args[1]: i32[] closest seen input typ |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:86:12` | 7 | never seen function: broadcast_in_dim id=140641060792544 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/run_full.py:348:11` | 7 | never seen function: convert_element_type id=140641057728416 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:747:14` | 7 | never seen function: broadcast_in_dim id=140641055965728 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:372:10` | 6 | never seen function: concatenate id=140642063991168 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:225:21` | 5 | never seen function: convert_element_type id=140642065533344 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:373:14` | 5 | for transpose defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[1947,102] closest seen input type s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 4 | never seen function: fft id=140645438538272 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:8` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:553:11` | 4 | never seen function: convert_element_type id=140642063650144 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:807:11` | 4 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1963],  args[1]: i32[] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:197:11` | 4 | never seen function: dynamic_slice id=140641057379072 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:264:9` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:344:13` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:260:10` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: i32[1947],  min: i64[],  max: i64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:262:8` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[1947],  min: f64[],  max: f64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:281:11` | 3 | never seen function: _table_interp id=140642699460992 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:248 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:79:8` | 3 | never seen function: broadcast_in_dim id=140642064507552 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:23` | 3 | never seen function: broadcast_in_dim id=140642064510112 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:114:13` | 3 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:196:10` | 3 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:364:12` | 3 | never seen function: convert_element_type id=140641056776544 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:335:13` | 3 | never seen function: broadcast_in_dim id=140640186004352 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=140642649859040 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
