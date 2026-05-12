# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/profile_runstern_v3/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 360 | 0.753 | 0.190 |
| jaxpr→MLIR | 92 | 0.772 | 0.310 |
| XLA compile | 92 | 3.961 | 3.354 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_at_q_jit` | 1 | 3.354 | 3.354 |
| `broadcast_in_dim` | 20 | 0.101 | 0.010 |
| `convert_element_type` | 10 | 0.093 | 0.037 |
| `concatenate` | 11 | 0.055 | 0.007 |
| `multiply` | 5 | 0.051 | 0.029 |
| `dynamic_slice` | 7 | 0.037 | 0.008 |
| `_ionic_gspace_jit` | 1 | 0.037 | 0.037 |
| `compute_V_H_and_V_xc` | 1 | 0.036 | 0.036 |
| `_where` | 2 | 0.019 | 0.014 |
| `_pad` | 4 | 0.018 | 0.006 |
| `add` | 4 | 0.017 | 0.005 |
| `squeeze` | 4 | 0.017 | 0.005 |
| `_assemble_Z_jit` | 1 | 0.017 | 0.017 |
| `_reduce_sum` | 3 | 0.014 | 0.005 |
| `sqrt` | 1 | 0.013 | 0.013 |
| `gather` | 2 | 0.011 | 0.007 |
| `_reduce_prod` | 1 | 0.010 | 0.010 |
| `_moveaxis` | 2 | 0.009 | 0.005 |
| `conjugate` | 1 | 0.005 | 0.005 |
| `fft` | 1 | 0.005 | 0.005 |
| `remainder` | 1 | 0.005 | 0.005 |
| `_broadcast_arrays` | 1 | 0.005 | 0.005 |
| `compute_per_band_kinetic` | 1 | 0.005 | 0.005 |
| `integer_pow` | 1 | 0.004 | 0.004 |
| `_mean` | 1 | 0.004 | 0.004 |
| `abs` | 1 | 0.004 | 0.004 |
| `select_n` | 1 | 0.004 | 0.004 |
| `matmul` | 1 | 0.004 | 0.004 |
| `real` | 1 | 0.004 | 0.004 |
| `less` | 1 | 0.004 | 0.004 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `compute_V_H_and_V_xc` | 1 | 0.190 | 0.190 |
| `_chi_at_q_jit` | 1 | 0.181 | 0.181 |
| `_build_stk_at_q` | 1 | 0.063 | 0.063 |
| `_ionic_gspace_jit` | 1 | 0.048 | 0.048 |
| `_assemble_Z_jit` | 1 | 0.042 | 0.042 |
| `_sternheimer_core` | 1 | 0.038 | 0.038 |
| `accumulate_species_on_G` | 1 | 0.019 | 0.019 |
| `multiply` | 66 | 0.019 | 0.001 |
| `_table_interp` | 1 | 0.016 | 0.016 |
| `interp_uniform_jax` | 1 | 0.016 | 0.016 |
| `species_structure_factors` | 1 | 0.011 | 0.011 |
| `apply_H_k_from_G` | 1 | 0.010 | 0.010 |
| `_where` | 11 | 0.008 | 0.001 |
| `add` | 26 | 0.008 | 0.001 |
| `true_divide` | 20 | 0.006 | 0.001 |
| `clip` | 4 | 0.006 | 0.002 |
| `_einsum` | 12 | 0.005 | 0.001 |
| `_tpa` | 1 | 0.004 | 0.004 |
| `broadcast_in_dim` | 20 | 0.004 | 0.000 |
| `subtract` | 12 | 0.004 | 0.001 |
| `remainder` | 2 | 0.004 | 0.003 |
| `_reduce_sum` | 9 | 0.003 | 0.000 |
| `less` | 9 | 0.003 | 0.001 |
| `compute_per_band_kinetic` | 1 | 0.003 | 0.003 |
| `_broadcast_arrays` | 17 | 0.003 | 0.000 |
| `_take` | 1 | 0.003 | 0.003 |
| `concatenate` | 11 | 0.003 | 0.000 |
| `dynamic_slice` | 7 | 0.003 | 0.001 |
| `matmul` | 5 | 0.002 | 0.001 |
| `convert_element_type` | 10 | 0.002 | 0.001 |

## Tracing cache misses

Total: **101** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:824:11` | 6 | never seen function: add id=140443754354848 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 3 | never seen function: fft id=140444948710400 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:264:9` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:566:32` | 3 | never seen function: convert_element_type id=140443491125024 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=140443758195904 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:155:11` | 2 | never seen function: _where id=140444947115648 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:806:19` | 2 | never seen function: broadcast_in_dim id=140443492359744 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:562:17` | 2 | never seen function: _pad id=140444944734048 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:563:13` | 2 | for _pad defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 never seen input type signature: array: i32[1947],  constant_values: i64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:567:32` | 2 | never seen function: broadcast_in_dim id=140443491129504 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:566:15` | 2 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:448:15` | 2 | never seen function: dynamic_slice id=140443491301504 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1147:17` | 2 | never seen function: broadcast_in_dim id=140443492770304 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1205:20` | 2 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1963] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1207:20` | 2 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1963] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1210:20` | 2 | never seen function: broadcast_in_dim id=140443492367584 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1220:29` | 2 | never seen function: dynamic_slice id=140443754702752 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:821:17` | 2 | never seen function: dynamic_slice id=140443754708512 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:821:9` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1221:8` | 2 | never seen function: broadcast_in_dim id=140443754363808 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1219:19` | 2 | never seen function: broadcast_in_dim id=140443754071520 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1224:20` | 2 | never seen function: broadcast_in_dim id=140443754076480 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1236:16` | 2 | never seen function: dynamic_slice id=140443754079840 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1234:25` | 2 | never seen function: broadcast_in_dim id=140443489813984 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:115:13` | 2 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:197:10` | 2 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:154:32` | 1 | never seen function: integer_pow id=140443492552352 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:233:16` | 1 | never seen function: species_structure_factors id=140444291803040 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:152:10` | 1 | never seen function: clip id=140444944689056 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:153:8` | 1 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[46080],  min: f64[],  max: f64[] clo |

## Persistent cache misses

_None._
