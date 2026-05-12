# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/profile_runstern_v4/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 344 | 0.714 | 0.176 |
| jaxpr→MLIR | 70 | 0.697 | 0.297 |
| XLA compile | 70 | 3.875 | 3.375 |

## Top 12 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_at_q_jit` | 1 | 3.375 | 3.375 |
| `broadcast_in_dim` | 14 | 0.103 | 0.025 |
| `convert_element_type` | 9 | 0.093 | 0.035 |
| `concatenate` | 8 | 0.040 | 0.009 |
| `compute_V_H_and_V_xc` | 1 | 0.034 | 0.034 |
| `dynamic_slice` | 5 | 0.027 | 0.007 |
| `_ionic_gspace_jit` | 1 | 0.027 | 0.027 |
| `multiply` | 3 | 0.023 | 0.008 |
| `_pad` | 4 | 0.017 | 0.004 |
| `_assemble_Z_jit` | 1 | 0.016 | 0.016 |
| `_per_k_psi_to_masked_G` | 2 | 0.014 | 0.007 |
| `_reduce_sum` | 3 | 0.012 | 0.004 |

## Top 12 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_at_q_jit` | 1 | 0.176 | 0.176 |
| `compute_V_H_and_V_xc` | 1 | 0.168 | 0.168 |
| `_build_stk_at_q` | 1 | 0.061 | 0.061 |
| `_ionic_gspace_jit` | 1 | 0.041 | 0.041 |
| `_assemble_Z_jit` | 1 | 0.041 | 0.041 |
| `_sternheimer_core` | 1 | 0.037 | 0.037 |
| `multiply` | 66 | 0.018 | 0.001 |
| `accumulate_species_on_G` | 1 | 0.017 | 0.017 |
| `_table_interp` | 1 | 0.016 | 0.016 |
| `interp_uniform_jax` | 1 | 0.014 | 0.014 |
| `_per_k_psi_to_masked_G` | 2 | 0.011 | 0.007 |
| `apply_H_k_from_G` | 1 | 0.010 | 0.010 |

## Tracing cache misses

Total: **85** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 3 | never seen function: fft id=140421721441280 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:264:9` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:566:32` | 3 | never seen function: convert_element_type id=140420271408928 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=140420538479808 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:155:11` | 2 | never seen function: _where id=140421719846528 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:806:19` | 2 | never seen function: broadcast_in_dim id=140420272315968 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:562:17` | 2 | never seen function: _pad id=140421717464928 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:563:13` | 2 | for _pad defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 never seen input type signature: array: i32[1947],  constant_values: i64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:567:32` | 2 | never seen function: broadcast_in_dim id=140420271413408 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:566:15` | 2 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:448:15` | 2 | never seen function: dynamic_slice id=140420271585408 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1147:17` | 2 | never seen function: broadcast_in_dim id=140420273021440 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
