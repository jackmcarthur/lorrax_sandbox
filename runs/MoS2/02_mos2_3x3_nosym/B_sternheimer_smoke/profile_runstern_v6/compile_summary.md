# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/profile_runstern_v6/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 404 | 0.868 | 0.175 |
| jaxpr→MLIR | 80 | 0.741 | 0.302 |
| XLA compile | 80 | 11.055 | 10.368 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_at_q_jit` | 1 | 10.368 | 10.368 |
| `spherical_hankel_table_batch_jax` | 10 | 0.143 | 0.058 |
| `convert_element_type` | 9 | 0.086 | 0.040 |
| `broadcast_in_dim` | 14 | 0.082 | 0.022 |
| `abs` | 1 | 0.063 | 0.063 |
| `concatenate` | 8 | 0.041 | 0.007 |
| `compute_V_H_and_V_xc` | 1 | 0.033 | 0.033 |
| `dynamic_slice` | 5 | 0.026 | 0.006 |
| `multiply` | 3 | 0.026 | 0.013 |
| `_ionic_gspace_jit` | 1 | 0.024 | 0.024 |
| `_pad` | 4 | 0.018 | 0.005 |
| `_assemble_Z_jit` | 1 | 0.017 | 0.017 |
| `_per_k_psi_to_masked_G` | 2 | 0.015 | 0.008 |
| `add` | 3 | 0.014 | 0.005 |
| `_reduce_sum` | 3 | 0.014 | 0.005 |
| `sqrt` | 1 | 0.013 | 0.013 |
| `_reduce_prod` | 1 | 0.011 | 0.011 |
| `_per_k_kinetic` | 1 | 0.010 | 0.010 |
| `_moveaxis` | 2 | 0.009 | 0.005 |
| `_where` | 2 | 0.009 | 0.004 |
| `_mean` | 1 | 0.006 | 0.006 |
| `conjugate` | 1 | 0.006 | 0.006 |
| `integer_pow` | 1 | 0.005 | 0.005 |
| `matmul` | 1 | 0.005 | 0.005 |
| `squeeze` | 1 | 0.004 | 0.004 |
| `real` | 1 | 0.004 | 0.004 |
| `fft` | 1 | 0.004 | 0.004 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_at_q_jit` | 1 | 0.175 | 0.175 |
| `compute_V_H_and_V_xc` | 1 | 0.161 | 0.161 |
| `spherical_hankel_table_batch_jax` | 10 | 0.104 | 0.047 |
| `_build_stk_at_q` | 1 | 0.060 | 0.060 |
| `subtract` | 17 | 0.047 | 0.043 |
| `_ionic_gspace_jit` | 1 | 0.040 | 0.040 |
| `_assemble_Z_jit` | 1 | 0.038 | 0.038 |
| `_sternheimer_core` | 1 | 0.037 | 0.037 |
| `multiply` | 75 | 0.021 | 0.001 |
| `accumulate_species_on_G` | 1 | 0.017 | 0.017 |
| `interp_uniform_jax` | 1 | 0.015 | 0.015 |
| `_table_interp` | 1 | 0.014 | 0.014 |
| `_where` | 17 | 0.013 | 0.002 |
| `_per_k_psi_to_masked_G` | 2 | 0.011 | 0.008 |
| `apply_H_k_from_G` | 1 | 0.010 | 0.010 |
| `_einsum` | 16 | 0.009 | 0.001 |
| `true_divide` | 26 | 0.008 | 0.001 |
| `add` | 28 | 0.008 | 0.001 |
| `species_structure_factors` | 1 | 0.007 | 0.007 |
| `clip` | 4 | 0.005 | 0.002 |
| `_broadcast_arrays` | 23 | 0.005 | 0.001 |
| `remainder` | 2 | 0.004 | 0.003 |
| `_tpa` | 1 | 0.004 | 0.004 |
| `_per_k_kinetic` | 1 | 0.004 | 0.004 |
| `less` | 11 | 0.004 | 0.000 |
| `compute_per_band_kinetic` | 1 | 0.003 | 0.003 |
| `_reduce_sum` | 9 | 0.003 | 0.001 |
| `broadcast_in_dim` | 14 | 0.003 | 0.000 |
| `abs` | 5 | 0.003 | 0.001 |
| `_take` | 1 | 0.003 | 0.003 |

## Tracing cache misses

Total: **101** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial_tables.py:205:22` | 4 | for spherical_hankel_table_batch_jax defined at /global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:246 never seen input ty |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial_tables.py:214:22` | 4 | for spherical_hankel_table_batch_jax defined at /global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:246 explanation unavail |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 3 | never seen function: fft id=139828090405888 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:264:9` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:566:32` | 3 | never seen function: convert_element_type id=139825888462784 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=139826488751296 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:171:9` | 2 | never seen function: _where id=139828088811136 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:180:15` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[4000,1640],  x: f64[4000,1640],   |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial_tables.py:182:19` | 2 | never seen function: spherical_hankel_table_batch_jax id=139826493730752 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/radial/r |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:200:20` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: f64[4000,1640],  y: f64[40 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:155:11` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[46080],  x: i32[46080],  y: i32[4 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:806:19` | 2 | never seen function: broadcast_in_dim id=139826487522176 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:562:17` | 2 | never seen function: _pad id=139828086429536 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:563:13` | 2 | for _pad defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 never seen input type signature: array: i32[1947],  constant_values: i64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:567:32` | 2 | never seen function: broadcast_in_dim id=139825888468544 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:566:15` | 2 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:448:15` | 2 | never seen function: dynamic_slice id=139825888296160 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1147:17` | 2 | never seen function: broadcast_in_dim id=139825887977184 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1205:20` | 2 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1963] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1207:20` | 2 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1963] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1210:20` | 2 | never seen function: broadcast_in_dim id=139825888462144 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:821:9` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:115:13` | 2 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:197:10` | 2 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:154:32` | 1 | never seen function: integer_pow id=139825887465120 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:233:16` | 1 | never seen function: species_structure_factors id=139826493859744 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:152:10` | 1 | never seen function: clip id=139828086384544 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:153:8` | 1 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[46080],  min: f64[],  max: f64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:159:11` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[46080],  x: f64[46080],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:101:14` | 1 | never seen function: interp_uniform_jax id=139826493727552 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:1 |

## Persistent cache misses

_None._
