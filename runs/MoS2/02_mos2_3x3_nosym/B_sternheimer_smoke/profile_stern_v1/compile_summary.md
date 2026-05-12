# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/profile_stern_v1/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 672 | 0.804 | 0.179 |
| jaxpr→MLIR | 563 | 1.765 | 0.116 |
| XLA compile | 563 | 45.681 | 2.007 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `multiply` | 87 | 5.452 | 0.684 |
| `_sternheimer_core` | 6 | 5.318 | 0.917 |
| `broadcast_in_dim` | 76 | 3.336 | 0.393 |
| `sqrt` | 9 | 2.188 | 1.642 |
| `add` | 39 | 2.136 | 0.061 |
| `equal` | 1 | 2.007 | 2.007 |
| `convert_element_type` | 35 | 1.849 | 0.847 |
| `cos` | 1 | 1.606 | 1.606 |
| `_einsum` | 27 | 1.524 | 0.128 |
| `apply_H_k` | 6 | 1.478 | 0.263 |
| `real` | 4 | 1.267 | 1.111 |
| `species_structure_factors` | 1 | 1.265 | 1.265 |
| `scatter` | 11 | 1.142 | 0.142 |
| `subtract` | 19 | 1.031 | 0.065 |
| `_reduce_sum` | 14 | 0.912 | 0.079 |
| `dynamic_slice` | 15 | 0.775 | 0.063 |
| `gather` | 13 | 0.750 | 0.067 |
| `exp` | 8 | 0.728 | 0.098 |
| `compute_V_H_and_V_xc` | 1 | 0.723 | 0.723 |
| `concatenate` | 10 | 0.671 | 0.168 |
| `true_divide` | 11 | 0.665 | 0.071 |
| `_where` | 2 | 0.649 | 0.596 |
| `mul` | 12 | 0.641 | 0.059 |
| `matmul` | 15 | 0.633 | 0.069 |
| `conjugate` | 11 | 0.611 | 0.062 |
| `scatter-add` | 4 | 0.608 | 0.178 |
| `integer_pow` | 14 | 0.569 | 0.055 |
| `select_n` | 9 | 0.495 | 0.061 |
| `remainder` | 7 | 0.439 | 0.067 |
| `squeeze` | 15 | 0.373 | 0.026 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_sternheimer_core` | 6 | 0.194 | 0.080 |
| `compute_V_H_and_V_xc` | 1 | 0.179 | 0.179 |
| `apply_H_k` | 3 | 0.083 | 0.062 |
| `multiply` | 104 | 0.038 | 0.002 |
| `_table_interp` | 3 | 0.034 | 0.012 |
| `accumulate_species_on_G` | 1 | 0.023 | 0.023 |
| `interp_uniform_jax` | 1 | 0.020 | 0.020 |
| `broadcast_in_dim` | 76 | 0.020 | 0.001 |
| `add` | 52 | 0.016 | 0.001 |
| `_einsum` | 31 | 0.016 | 0.001 |
| `_tpa` | 3 | 0.012 | 0.004 |
| `clip` | 8 | 0.011 | 0.002 |
| `remainder` | 7 | 0.011 | 0.003 |
| `species_structure_factors` | 1 | 0.010 | 0.010 |
| `convert_element_type` | 35 | 0.008 | 0.000 |
| `_apply_P_U` | 3 | 0.008 | 0.003 |
| `compute_per_band_kinetic` | 3 | 0.008 | 0.003 |
| `_reduce_sum` | 14 | 0.008 | 0.001 |
| `true_divide` | 23 | 0.008 | 0.001 |
| `_where` | 7 | 0.007 | 0.002 |
| `subtract` | 19 | 0.007 | 0.001 |
| `matmul` | 10 | 0.006 | 0.001 |
| `gather` | 13 | 0.005 | 0.001 |
| `dynamic_slice` | 15 | 0.005 | 0.001 |
| `scatter` | 11 | 0.004 | 0.001 |
| `maximum` | 9 | 0.004 | 0.001 |
| `_take` | 1 | 0.003 | 0.003 |
| `squeeze` | 15 | 0.003 | 0.000 |
| `less` | 9 | 0.003 | 0.001 |
| `integer_pow` | 14 | 0.003 | 0.000 |

## Tracing cache misses

Total: **292** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:181:13` | 39 | never seen function: scatter id=140429595137088 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:747:14` | 13 | never seen function: broadcast_in_dim id=140429332609472 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:356:25` | 12 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen input type si |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:93:43` | 12 | for integer_pow defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947] closest seen input type sign |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:807:11` | 12 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947],  args[1]: i32[] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:371:10` | 9 | never seen function: add id=140430269990848 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:86:12` | 9 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:72:8` | 6 | never seen function: dynamic_slice id=140430466430784 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/run_full.py:140:17` | 6 | never seen function: broadcast_in_dim id=140430130994592 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:804:17` | 6 | for dynamic_slice defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947,3],  args[1]: i64[],  args[ |
| `/global/homes/j/jackm/software/lorrax_B/src/solvers/sternheimer_precond.py:95:11` | 6 | never seen function: _tpa id=140430789745920 defined at /global/homes/j/jackm/software/lorrax_B/src/solvers/sternheimer_precond.py:48 |
| `/global/homes/j/jackm/software/lorrax_B/src/solvers/sternheimer_solve.py:309:11` | 6 | never seen function: _sternheimer_core id=140430789750720 defined at /global/homes/j/jackm/software/lorrax_B/src/solvers/sternheimer_solve.p |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:98:11` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1],  args[1]: i32[] closest seen input typ |
| `/global/homes/j/jackm/software/lorrax_B/src/solvers/sternheimer_solve.py:142:13` | 6 | never seen function: broadcast_in_dim id=140429331035168 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:348:11` | 6 | never seen function: broadcast_in_dim id=140429594134944 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:552:30` | 6 | never seen function: dynamic_slice id=140428929087296 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/ionic_gspace.py:225:21` | 5 | never seen function: convert_element_type id=140430740523424 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:8` | 4 | for concatenate defined at /opt/jax/jax/_src/dispatch.py:96 never seen passing 2 positional args and 0 keyword args with keys: |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:372:10` | 4 | never seen function: concatenate id=140430131347840 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:553:11` | 4 | never seen function: convert_element_type id=140430130990432 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/run_sternheimer.py:804:9` | 4 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: i32[],  y: i32[] closest s |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/dft_operators.py:264:9` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[1947] closest seen input  |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:344:13` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: f64[1947,3] closest seen inpu |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:260:10` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: i32[1947],  min: i64[],  max: i64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:262:8` | 3 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen input type signature: arr: f64[1947],  min: f64[],  max: f64[] clos |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:281:11` | 3 | never seen function: _table_interp id=140430789542272 defined at /global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:248 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:79:8` | 3 | never seen function: broadcast_in_dim id=140429601202848 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/radial/solid_harmonics.py:82:23` | 3 | never seen function: broadcast_in_dim id=140429601205408 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/psp/vnl_ops.py:373:14` | 3 | for transpose defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[1947,102] closest seen input type s |
| `/global/homes/j/jackm/software/lorrax_B/src/solvers/projectors.py:107:19` | 3 | never seen function: _apply_P_U id=140430789551552 defined at /global/homes/j/jackm/software/lorrax_B/src/solvers/projectors.py:37 |

## Persistent cache misses

_None._
