# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/probe_dft2/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 148 | 0.338 | 0.184 |
| jaxpr→MLIR | 18 | 0.236 | 0.113 |
| XLA compile | 18 | 0.917 | 0.723 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_ionic_gspace_jit` | 1 | 0.723 | 0.723 |
| `convert_element_type` | 3 | 0.063 | 0.038 |
| `compute_V_H_and_V_xc` | 1 | 0.048 | 0.048 |
| `multiply` | 2 | 0.012 | 0.007 |
| `_moveaxis` | 2 | 0.011 | 0.006 |
| `broadcast_in_dim` | 1 | 0.011 | 0.011 |
| `_reduce_sum` | 2 | 0.010 | 0.006 |
| `_reduce_prod` | 1 | 0.010 | 0.010 |
| `sqrt` | 1 | 0.009 | 0.009 |
| `fft` | 1 | 0.008 | 0.008 |
| `add` | 1 | 0.004 | 0.004 |
| `abs` | 1 | 0.004 | 0.004 |
| `integer_pow` | 1 | 0.004 | 0.004 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `compute_V_H_and_V_xc` | 1 | 0.184 | 0.184 |
| `_ionic_gspace_jit` | 1 | 0.044 | 0.044 |
| `true_divide` | 11 | 0.026 | 0.023 |
| `accumulate_species_on_G` | 1 | 0.018 | 0.018 |
| `interp_uniform_jax` | 1 | 0.016 | 0.016 |
| `species_structure_factors` | 1 | 0.010 | 0.010 |
| `multiply` | 40 | 0.010 | 0.000 |
| `add` | 15 | 0.003 | 0.000 |
| `_where` | 5 | 0.003 | 0.001 |
| `_take` | 1 | 0.002 | 0.002 |
| `less` | 4 | 0.002 | 0.001 |
| `clip` | 2 | 0.002 | 0.001 |
| `matmul` | 2 | 0.002 | 0.001 |
| `subtract` | 6 | 0.001 | 0.000 |
| `_squeeze` | 3 | 0.001 | 0.001 |
| `_reduce_sum` | 3 | 0.001 | 0.000 |
| `convert_element_type` | 3 | 0.001 | 0.000 |
| `equal` | 4 | 0.001 | 0.000 |
| `_broadcast_arrays` | 5 | 0.001 | 0.000 |
| `bitwise_and` | 4 | 0.001 | 0.000 |
| `maximum` | 3 | 0.001 | 0.000 |
| `sqrt` | 3 | 0.001 | 0.000 |
| `fft` | 3 | 0.001 | 0.000 |
| `negative` | 4 | 0.001 | 0.000 |
| `abs` | 1 | 0.001 | 0.001 |
| `real` | 2 | 0.001 | 0.000 |
| `exp` | 3 | 0.001 | 0.000 |
| `minimum` | 2 | 0.000 | 0.000 |
| `_reduce_prod` | 1 | 0.000 | 0.000 |
| `_power` | 1 | 0.000 | 0.000 |

## Tracing cache misses

Total: **21** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:147:12` | 2 | never seen function: convert_element_type id=140349466786880 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:153:16` | 2 | never seen function: fft id=140350805213728 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:155:11` | 2 | never seen function: _where id=140350803684512 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:154:32` | 1 | never seen function: integer_pow id=140349201978912 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:233:16` | 1 | never seen function: species_structure_factors id=140350779956800 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:152:10` | 1 | never seen function: clip id=140350801225152 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:153:8` | 1 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[46080],  min: f64[],  max: f64[] clo |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:159:11` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[46080],  x: f64[46080],  y: f64[] |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:101:14` | 1 | never seen function: interp_uniform_jax id=140350204392640 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/radial/radial_jax.py:1 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:237:22` | 1 | never seen function: accumulate_species_on_G id=140350204485184 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:7 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:240:26` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[24,24,80] closest seen input type signature has  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:249:14` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[46080],  x: f64[],  y: f64[46080] |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:193:45` | 1 | never seen function: _ionic_gspace_jit id=140350204486944 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:221 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:297:18` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: f64[24,24,80] closest seen input type signature has 1 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:118:14` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[24,24,80],  x: f64[],  y: f64[24, |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/xc.py:137:28` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[24,24,80],  x: f64[24,24,80],  y: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/scf_potential.py:87:16` | 1 | never seen function: compute_V_H_and_V_xc id=140350204491904 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:277 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:214:21` | 1 | never seen function: convert_element_type id=140350792577120 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
