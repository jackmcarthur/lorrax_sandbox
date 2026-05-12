# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/probe_dft/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 154 | 0.283 | 0.179 |
| jaxpr→MLIR | 51 | 0.350 | 0.133 |
| XLA compile | 51 | 4.116 | 0.736 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `compute_V_H_and_V_xc` | 1 | 0.736 | 0.736 |
| `multiply` | 11 | 0.649 | 0.072 |
| `convert_element_type` | 7 | 0.582 | 0.346 |
| `broadcast_in_dim` | 3 | 0.249 | 0.188 |
| `_reduce_sum` | 3 | 0.216 | 0.083 |
| `sqrt` | 2 | 0.194 | 0.132 |
| `species_structure_factors` | 1 | 0.177 | 0.177 |
| `accumulate_species_on_G` | 1 | 0.137 | 0.137 |
| `add` | 2 | 0.117 | 0.061 |
| `integer_pow` | 2 | 0.114 | 0.057 |
| `negative` | 2 | 0.111 | 0.056 |
| `abs` | 1 | 0.085 | 0.085 |
| `cos` | 1 | 0.080 | 0.080 |
| `true_divide` | 1 | 0.072 | 0.072 |
| `subtract` | 1 | 0.069 | 0.069 |
| `scatter` | 1 | 0.069 | 0.069 |
| `_reduce_prod` | 1 | 0.066 | 0.066 |
| `fft` | 2 | 0.063 | 0.032 |
| `exp` | 1 | 0.059 | 0.059 |
| `real` | 1 | 0.056 | 0.056 |
| `_where` | 1 | 0.056 | 0.056 |
| `equal` | 1 | 0.055 | 0.055 |
| `_moveaxis` | 2 | 0.037 | 0.033 |
| `_squeeze` | 1 | 0.034 | 0.034 |
| `reshape` | 1 | 0.033 | 0.033 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `compute_V_H_and_V_xc` | 1 | 0.179 | 0.179 |
| `accumulate_species_on_G` | 1 | 0.021 | 0.021 |
| `interp_uniform_jax` | 1 | 0.019 | 0.019 |
| `multiply` | 40 | 0.012 | 0.001 |
| `species_structure_factors` | 1 | 0.010 | 0.010 |
| `_where` | 5 | 0.004 | 0.001 |
| `add` | 15 | 0.004 | 0.001 |
| `clip` | 2 | 0.004 | 0.002 |
| `matmul` | 1 | 0.004 | 0.004 |
| `_take` | 1 | 0.003 | 0.003 |
| `true_divide` | 10 | 0.003 | 0.000 |
| `subtract` | 6 | 0.002 | 0.001 |
| `_reduce_sum` | 3 | 0.002 | 0.001 |
| `convert_element_type` | 7 | 0.002 | 0.000 |
| `less` | 4 | 0.001 | 0.000 |
| `_broadcast_arrays` | 5 | 0.001 | 0.000 |
| `equal` | 4 | 0.001 | 0.000 |
| `bitwise_and` | 4 | 0.001 | 0.000 |
| `maximum` | 3 | 0.001 | 0.000 |
| `broadcast_in_dim` | 3 | 0.001 | 0.000 |
| `_reduce_prod` | 1 | 0.001 | 0.001 |
| `sqrt` | 3 | 0.001 | 0.000 |
| `fft` | 3 | 0.001 | 0.000 |
| `negative` | 4 | 0.001 | 0.000 |
| `minimum` | 2 | 0.001 | 0.000 |
| `abs` | 1 | 0.001 | 0.001 |
| `greater_equal` | 1 | 0.001 | 0.001 |
| `exp` | 3 | 0.001 | 0.000 |
| `less_equal` | 1 | 0.001 | 0.001 |
| `cos` | 2 | 0.000 | 0.000 |

## Tracing cache misses

Total: **29** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:225:21` | 5 | never seen function: convert_element_type id=139934399138688 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=139934403654848 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 2 | never seen function: fft id=139936001475104 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:155:11` | 2 | never seen function: _where id=139935999929504 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:154:32` | 1 | never seen function: integer_pow id=139934398942880 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:182:16` | 1 | never seen function: species_structure_factors id=139935976283552 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:152:10` | 1 | never seen function: clip id=139935997437376 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:153:8` | 1 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[46080],  min: f64[],  max: f64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:159:11` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[46080],  x: f64[46080],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:101:14` | 1 | never seen function: interp_uniform_jax id=139934404621024 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:1 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:190:22` | 1 | never seen function: accumulate_species_on_G id=139934404713568 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:7 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:192:17` | 1 | never seen function: reshape id=139934398951200 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:193:26` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[24,24,80] closest seen input type signature has  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:209:14` | 1 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[46080] closest seen input type sig |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:210:14` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[46080],  x: f64[],  y: f64[46080] |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:211:13` | 1 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[2] closest seen input typ |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:212:25` | 1 | never seen function: broadcast_in_dim id=139933865425984 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:297:18` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: f64[24,24,80] closest seen input type signature has 1 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:118:14` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[24,24,80],  x: f64[],  y: f64[24, |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/xc.py:137:28` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[24,24,80],  x: f64[24,24,80],  y: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:87:16` | 1 | never seen function: compute_V_H_and_V_xc id=139934404718848 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:277 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:214:21` | 1 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[102] closest seen input t |

## Persistent cache misses

_None._
