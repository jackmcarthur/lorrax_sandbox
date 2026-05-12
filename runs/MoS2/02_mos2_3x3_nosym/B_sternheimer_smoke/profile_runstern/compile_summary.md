# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/profile_runstern/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 481 | 0.823 | 0.268 |
| jaxpr→MLIR | 209 | 0.991 | 0.298 |
| XLA compile | 209 | 4.726 | 3.388 |

## Top 60 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_at_q_jit` | 1 | 3.388 | 3.388 |
| `broadcast_in_dim` | 32 | 0.186 | 0.011 |
| `multiply` | 27 | 0.155 | 0.011 |
| `convert_element_type` | 13 | 0.113 | 0.043 |
| `add` | 15 | 0.111 | 0.031 |
| `concatenate` | 13 | 0.071 | 0.008 |
| `dynamic_slice` | 10 | 0.060 | 0.010 |
| `_reduce_sum` | 8 | 0.056 | 0.021 |
| `matmul` | 9 | 0.050 | 0.007 |
| `gather` | 8 | 0.047 | 0.015 |
| `squeeze` | 7 | 0.044 | 0.009 |
| `integer_pow` | 7 | 0.038 | 0.009 |
| `sqrt` | 5 | 0.037 | 0.011 |
| `compute_V_H_and_V_xc` | 1 | 0.036 | 0.036 |
| `_pad` | 6 | 0.032 | 0.007 |
| `_ionic_gspace_jit` | 1 | 0.026 | 0.026 |
| `true_divide` | 4 | 0.024 | 0.008 |
| `_table_interp` | 3 | 0.023 | 0.008 |
| `scatter` | 3 | 0.022 | 0.010 |
| `transpose` | 4 | 0.018 | 0.005 |
| `_reduce_prod` | 1 | 0.018 | 0.018 |
| `exp` | 3 | 0.018 | 0.008 |
| `conjugate` | 3 | 0.016 | 0.007 |
| `_broadcast_arrays` | 3 | 0.016 | 0.007 |
| `_squeeze` | 3 | 0.016 | 0.006 |
| `subtract` | 3 | 0.015 | 0.005 |
| `select_n` | 2 | 0.014 | 0.010 |
| `real` | 3 | 0.014 | 0.005 |
| `less` | 2 | 0.011 | 0.006 |
| `_moveaxis` | 2 | 0.010 | 0.006 |
| `fft` | 1 | 0.009 | 0.009 |
| `negative` | 1 | 0.006 | 0.006 |
| `_power` | 1 | 0.006 | 0.006 |
| `abs` | 1 | 0.005 | 0.005 |
| `compute_per_band_kinetic` | 1 | 0.005 | 0.005 |
| `remainder` | 1 | 0.005 | 0.005 |
| `_mean` | 1 | 0.005 | 0.005 |

## Top 60 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_chi_at_q_jit` | 1 | 0.268 | 0.268 |
| `compute_V_H_and_V_xc` | 1 | 0.176 | 0.176 |
| `_build_stk_at_q` | 1 | 0.062 | 0.062 |
| `_ionic_gspace_jit` | 1 | 0.041 | 0.041 |
| `_sternheimer_core` | 1 | 0.038 | 0.038 |
| `_table_interp` | 3 | 0.037 | 0.013 |
| `multiply` | 84 | 0.023 | 0.001 |
| `accumulate_species_on_G` | 1 | 0.017 | 0.017 |
| `interp_uniform_jax` | 1 | 0.014 | 0.014 |
| `clip` | 8 | 0.013 | 0.002 |
| `add` | 39 | 0.011 | 0.001 |
| `apply_H_k_from_G` | 1 | 0.010 | 0.010 |
| `broadcast_in_dim` | 32 | 0.008 | 0.001 |
| `species_structure_factors` | 1 | 0.007 | 0.007 |
| `_where` | 9 | 0.007 | 0.001 |
| `true_divide` | 21 | 0.006 | 0.000 |
| `subtract` | 19 | 0.005 | 0.000 |
| `_einsum` | 12 | 0.005 | 0.001 |
| `_reduce_sum` | 13 | 0.005 | 0.001 |
| `matmul` | 11 | 0.004 | 0.001 |
| `_tpa` | 1 | 0.004 | 0.004 |
| `maximum` | 9 | 0.004 | 0.001 |
| `convert_element_type` | 13 | 0.004 | 0.001 |
| `remainder` | 2 | 0.004 | 0.003 |
| `minimum` | 8 | 0.004 | 0.001 |
| `less` | 11 | 0.003 | 0.001 |
| `compute_per_band_kinetic` | 1 | 0.003 | 0.003 |
| `concatenate` | 13 | 0.003 | 0.000 |
| `_broadcast_arrays` | 17 | 0.003 | 0.000 |
| `_pad` | 6 | 0.003 | 0.001 |
| `_take` | 1 | 0.002 | 0.002 |
| `dynamic_slice` | 10 | 0.002 | 0.000 |
| `negative` | 14 | 0.002 | 0.000 |
| `fft` | 8 | 0.002 | 0.001 |
| `gather` | 8 | 0.002 | 0.000 |
| `sqrt` | 8 | 0.002 | 0.000 |
| `greater` | 6 | 0.002 | 0.000 |
| `abs` | 3 | 0.001 | 0.001 |
| `exp` | 7 | 0.001 | 0.000 |
| `equal` | 5 | 0.001 | 0.000 |
| `conjugate` | 6 | 0.001 | 0.000 |
| `squeeze` | 7 | 0.001 | 0.000 |
| `integer_pow` | 7 | 0.001 | 0.000 |
| `_mean` | 1 | 0.001 | 0.001 |
| `real` | 6 | 0.001 | 0.000 |
| `bitwise_and` | 5 | 0.001 | 0.000 |
| `transpose` | 4 | 0.001 | 0.000 |
| `floor` | 4 | 0.001 | 0.000 |
| `scatter` | 3 | 0.001 | 0.000 |
| `_power` | 2 | 0.001 | 0.000 |
| `_squeeze` | 7 | 0.001 | 0.000 |
| `absolute` | 3 | 0.001 | 0.000 |
| `_reduce_prod` | 1 | 0.000 | 0.000 |
| `_moveaxis` | 4 | 0.000 | 0.000 |
| `cos` | 2 | 0.000 | 0.000 |
| `_reduce_any` | 1 | 0.000 | 0.000 |
| `select_n` | 2 | 0.000 | 0.000 |
| `logical_and` | 1 | 0.000 | 0.000 |
| `log` | 1 | 0.000 | 0.000 |
| `less_equal` | 1 | 0.000 | 0.000 |

## Tracing cache misses

Total: **150** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:371:10` | 9 | never seen function: add id=140534755387456 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:72:8` | 6 | never seen function: dynamic_slice id=140534753923040 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:8` | 6 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:824:11` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1963],  args[1]: i32[] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:372:10` | 4 | never seen function: concatenate id=140534754452928 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:553:11` | 4 | never seen function: convert_element_type id=140534756553120 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 3 | never seen function: fft id=140536344691712 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:264:9` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:344:13` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:356:25` | 3 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen input type si |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:260:10` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: i32[1947],  min: i64[],  max: i64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:262:8` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[1947],  min: f64[],  max: f64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:281:11` | 3 | never seen function: _table_interp id=140535157739968 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:248 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:79:8` | 3 | never seen function: broadcast_in_dim id=140534753934880 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:23` | 3 | never seen function: broadcast_in_dim id=140534753933920 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:93:43` | 3 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:373:14` | 3 | for transpose defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[1947,102] closest seen input type s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=140534759894816 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:155:11` | 2 | never seen function: _where id=140536343096960 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:806:19` | 2 | never seen function: broadcast_in_dim id=140534754826400 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:448:15` | 2 | never seen function: dynamic_slice id=140534756546400 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:558:17` | 2 | never seen function: _pad id=140536340715360 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:560:13` | 2 | for _pad defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 never seen input type signature: array: i32[1947],  constant_values: i64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:563:16` | 2 | for _pad defined at /opt/jax/jax/_src/numpy/lax_numpy.py:4207 never seen input type signature: array: c128[102,1947],  constant_values: f64[ |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:565:38` | 2 | never seen function: broadcast_in_dim id=140534756084448 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:565:15` | 2 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: bool[1947],  args[1]: bool[16] closest |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1147:17` | 2 | never seen function: broadcast_in_dim id=140534756085888 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1205:20` | 2 | never seen function: broadcast_in_dim id=140534222107040 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1206:20` | 2 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: bool[1963] closest seen input typ |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1207:20` | 2 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1963] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1210:20` | 2 | never seen function: broadcast_in_dim id=140534222108640 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1220:29` | 2 | never seen function: dynamic_slice id=140534222689184 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:821:17` | 2 | for dynamic_slice defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1963,3],  args[1]: i64[],  args[ |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:821:9` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1221:8` | 2 | never seen function: broadcast_in_dim id=140534220966624 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1219:19` | 2 | never seen function: broadcast_in_dim id=140534220969184 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1224:20` | 2 | never seen function: broadcast_in_dim id=140534220974144 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1236:16` | 2 | never seen function: dynamic_slice id=140534220977504 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:1234:25` | 2 | never seen function: broadcast_in_dim id=140534221296864 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:220:11` | 2 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[] closest seen input typ |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:115:13` | 2 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:197:10` | 2 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:154:32` | 1 | never seen function: integer_pow id=140534755379456 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:233:16` | 1 | never seen function: species_structure_factors id=140535157598112 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:152:10` | 1 | never seen function: clip id=140536340670368 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:153:8` | 1 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[46080],  min: f64[],  max: f64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:159:11` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[46080],  x: f64[46080],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:101:14` | 1 | never seen function: interp_uniform_jax id=140535159563072 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:1 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:237:22` | 1 | never seen function: accumulate_species_on_G id=140535157599552 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:7 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:240:26` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[24,24,80] closest seen input type signature has  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:249:14` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[46080],  x: f64[],  y: f64[46080] |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:193:45` | 1 | never seen function: _ionic_gspace_jit id=140535157601312 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:221 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:297:18` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: f64[24,24,80] closest seen input type signature has 1 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:118:14` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[24,24,80],  x: f64[],  y: f64[24, |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/xc.py:137:28` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[24,24,80],  x: f64[24,24,80],  y: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:87:16` | 1 | never seen function: compute_V_H_and_V_xc id=140536319012960 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:277 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:214:21` | 1 | never seen function: convert_element_type id=140534754827200 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:344:53` | 1 | never seen function: broadcast_in_dim id=140534754836000 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:373:47` | 1 | never seen function: transpose id=140534754458368 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:376:8` | 1 | for broadcast_in_dim defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[102] closest seen input type |

## Persistent cache misses

_None._
