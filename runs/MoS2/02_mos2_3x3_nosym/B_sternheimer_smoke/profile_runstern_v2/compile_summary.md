# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/profile_runstern_v2/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 360 | 0.726 | 0.177 |
| jaxpr→MLIR | 92 | 0.781 | 0.305 |
| XLA compile | 92 | 10.919 | 10.346 |

## Top 25 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_at_q_jit` | 1 | 10.346 | 10.346 |
| `convert_element_type` | 10 | 0.113 | 0.054 |
| `broadcast_in_dim` | 20 | 0.099 | 0.011 |
| `concatenate` | 11 | 0.054 | 0.006 |
| `dynamic_slice` | 7 | 0.036 | 0.008 |
| `compute_V_H_and_V_xc` | 1 | 0.035 | 0.035 |
| `multiply` | 5 | 0.025 | 0.006 |
| `_ionic_gspace_jit` | 1 | 0.023 | 0.023 |
| `_pad` | 4 | 0.020 | 0.007 |
| `add` | 4 | 0.019 | 0.006 |
| `squeeze` | 4 | 0.018 | 0.005 |
| `_assemble_Z_jit` | 1 | 0.016 | 0.016 |
| `_reduce_sum` | 3 | 0.013 | 0.005 |
| `_reduce_prod` | 1 | 0.011 | 0.011 |
| `_where` | 2 | 0.009 | 0.004 |
| `_moveaxis` | 2 | 0.008 | 0.005 |
| `sqrt` | 1 | 0.008 | 0.008 |
| `gather` | 2 | 0.008 | 0.004 |
| `_mean` | 1 | 0.007 | 0.007 |
| `conjugate` | 1 | 0.006 | 0.006 |
| `abs` | 1 | 0.005 | 0.005 |
| `_broadcast_arrays` | 1 | 0.005 | 0.005 |
| `compute_per_band_kinetic` | 1 | 0.005 | 0.005 |
| `remainder` | 1 | 0.004 | 0.004 |
| `fft` | 1 | 0.004 | 0.004 |

## Top 25 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_at_q_jit` | 1 | 0.177 | 0.177 |
| `compute_V_H_and_V_xc` | 1 | 0.163 | 0.163 |
| `_build_stk_at_q` | 1 | 0.062 | 0.062 |
| `_ionic_gspace_jit` | 1 | 0.050 | 0.050 |
| `_assemble_Z_jit` | 1 | 0.043 | 0.043 |
| `_sternheimer_core` | 1 | 0.038 | 0.038 |
| `accumulate_species_on_G` | 1 | 0.019 | 0.019 |
| `multiply` | 66 | 0.019 | 0.001 |
| `interp_uniform_jax` | 1 | 0.016 | 0.016 |
| `_table_interp` | 1 | 0.016 | 0.016 |
| `species_structure_factors` | 1 | 0.012 | 0.012 |
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
| `_take` | 1 | 0.003 | 0.003 |
| `compute_per_band_kinetic` | 1 | 0.003 | 0.003 |

## Tracing cache misses

Total: **101** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:824:11` | 6 | never seen function: add id=140026874058912 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 3 | never seen function: fft id=140028068971520 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:264:9` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:566:32` | 3 | never seen function: convert_element_type id=140026610894624 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=140026877965504 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:155:11` | 2 | never seen function: _where id=140028067409536 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:806:19` | 2 | never seen function: broadcast_in_dim id=140026875698752 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:562:17` | 2 | never seen function: _pad id=140028065060704 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:563:13` | 2 | for _pad defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 never seen input type signature: array: i32[1947],  constant_values: i64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:567:32` | 2 | never seen function: broadcast_in_dim id=140026610899104 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:566:15` | 2 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:448:15` | 2 | never seen function: dynamic_slice id=140026611071104 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1147:17` | 2 | never seen function: broadcast_in_dim id=140026875470336 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1205:20` | 2 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1963] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1207:20` | 2 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1963] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1210:20` | 2 | never seen function: broadcast_in_dim id=140026875706592 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1220:29` | 2 | never seen function: dynamic_slice id=140026874406816 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:821:17` | 2 | never seen function: dynamic_slice id=140026874412576 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:821:9` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1221:8` | 2 | never seen function: broadcast_in_dim id=140026874067872 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1219:19` | 2 | never seen function: broadcast_in_dim id=140026873791968 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1224:20` | 2 | never seen function: broadcast_in_dim id=140026873796928 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1236:16` | 2 | never seen function: dynamic_slice id=140026873800288 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1234:25` | 2 | never seen function: broadcast_in_dim id=140026609567200 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:115:13` | 2 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |

## Persistent cache misses

_None._
