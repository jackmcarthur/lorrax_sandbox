# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_baseline/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 176 | 0.127 | 0.024 |
| jaxpr→MLIR | 75 | 0.441 | 0.138 |
| XLA compile | 76 | 5.562 | 0.666 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_normal` | 2 | 0.744 | 0.383 |
| `scan` | 1 | 0.666 | 0.666 |
| `conjugate` | 1 | 0.448 | 0.448 |
| `multiply` | 6 | 0.319 | 0.056 |
| `broadcast_in_dim` | 10 | 0.303 | 0.054 |
| `add` | 5 | 0.259 | 0.054 |
| `convert_element_type` | 7 | 0.224 | 0.051 |
| `gather` | 4 | 0.223 | 0.063 |
| `dynamic_slice` | 5 | 0.216 | 0.062 |
| `true_divide` | 3 | 0.211 | 0.073 |
| `norm` | 2 | 0.184 | 0.093 |
| `argsort` | 1 | 0.178 | 0.178 |
| `_diag` | 3 | 0.175 | 0.059 |
| `_threefry_split` | 1 | 0.153 | 0.153 |
| `transpose` | 2 | 0.135 | 0.073 |
| `eigh` | 1 | 0.133 | 0.133 |
| `scatter-add` | 1 | 0.106 | 0.106 |
| `maximum` | 2 | 0.104 | 0.054 |
| `less` | 2 | 0.102 | 0.054 |
| `select_n` | 2 | 0.101 | 0.054 |
| `squeeze` | 3 | 0.082 | 0.027 |
| `scatter` | 1 | 0.076 | 0.076 |
| `_einsum` | 1 | 0.063 | 0.063 |
| `_unstack` | 1 | 0.055 | 0.055 |
| `_threefry_seed` | 1 | 0.055 | 0.055 |
| `reshape` | 2 | 0.050 | 0.026 |
| `_squeeze` | 2 | 0.050 | 0.025 |
| `matmul` | 1 | 0.049 | 0.049 |
| `_broadcast_arrays` | 2 | 0.049 | 0.026 |
| `concatenate` | 1 | 0.049 | 0.049 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_matvec_impl` | 1 | 0.024 | 0.024 |
| `_normal` | 2 | 0.011 | 0.006 |
| `_normal_real` | 2 | 0.011 | 0.005 |
| `_uniform` | 2 | 0.009 | 0.004 |
| `_diag` | 3 | 0.008 | 0.005 |
| `norm` | 2 | 0.006 | 0.003 |
| `_einsum` | 9 | 0.006 | 0.001 |
| `add` | 18 | 0.006 | 0.001 |
| `multiply` | 16 | 0.006 | 0.001 |
| `true_divide` | 7 | 0.003 | 0.000 |
| `broadcast_in_dim` | 10 | 0.003 | 0.000 |
| `_where` | 3 | 0.002 | 0.001 |
| `subtract` | 8 | 0.002 | 0.000 |
| `eigh` | 1 | 0.002 | 0.002 |
| `convert_element_type` | 7 | 0.002 | 0.000 |
| `dynamic_slice` | 5 | 0.002 | 0.001 |
| `_threefry_seed` | 1 | 0.002 | 0.002 |
| `conjugate` | 6 | 0.001 | 0.000 |
| `_threefry_split` | 1 | 0.001 | 0.001 |
| `bitwise_or` | 5 | 0.001 | 0.001 |
| `less` | 4 | 0.001 | 0.000 |
| `_reduce_sum` | 2 | 0.001 | 0.001 |
| `bitwise_xor` | 4 | 0.001 | 0.000 |
| `gather` | 4 | 0.001 | 0.000 |
| `vdot` | 1 | 0.001 | 0.001 |
| `_threefry_split_foldlike` | 1 | 0.001 | 0.001 |
| `maximum` | 3 | 0.001 | 0.000 |
| `_broadcast_arrays` | 5 | 0.001 | 0.000 |
| `negative` | 5 | 0.001 | 0.000 |
| `squeeze` | 3 | 0.001 | 0.000 |

## Tracing cache misses

Total: **63** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:650:12` | 6 | never seen function: add id=140477914450656 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:294:18` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[20],  args[1]: i64[] closest seen input ty |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:592:18` | 5 | never seen function: convert_element_type id=139854880405792 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:664:10` | 3 | never seen function: dynamic_slice id=140477915027936 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:668:8` | 3 | never seen function: _uniform id=140480594664864 defined at /opt/jax/jax/_src/random.py:407 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:247:9` | 3 | for _uniform defined at /opt/jax/jax/_src/random.py:407 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:596:54` | 2 | never seen function: dynamic_slice id=140480529349216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:652:12` | 2 | never seen function: gather id=140477914462176 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:661:11` | 2 | never seen function: dynamic_slice id=140477915023776 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:253:12` | 2 | never seen function: convert_element_type id=140477380255936 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:288:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[200,200],  x: f64[200],  y: f64[2 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:293:10` | 2 | never seen function: argsort id=139855054502976 defined at /opt/jax/jax/_src/numpy/sorting.py:92 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:296:24` | 2 | never seen function: gather id=139851525426848 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:83:23` | 2 | never seen function: reshape id=139852190249696 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:590:19` | 1 | never seen function: convert_element_type id=139854880406432 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:647:18` | 1 | never seen function: convert_element_type id=140477914676928 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:667:10` | 1 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[] closest seen input type |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:245:13` | 1 | never seen function: _threefry_split id=140480594635936 defined at /opt/jax/jax/_src/prng.py:1111 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:245:4` | 1 | never seen function: _unstack id=140480599439168 defined at /opt/jax/jax/_src/numpy/array_methods.py:636 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:249:14` | 1 | never seen function: norm id=140480599015104 defined at /opt/jax/jax/_src/numpy/linalg.py:1076 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:251:8` | 1 | never seen function: broadcast_in_dim id=140477380248896 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:252:8` | 1 | never seen function: scatter id=140477380253696 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_serial.py:72:10` | 1 | never seen function: fft id=140480602555392 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_serial.py:73:10` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[480,480,4,4,4] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_serial.py:75:10` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:258:12` | 1 | never seen function: solve_bse.<locals>._matvec_impl id=140477379841376 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/bse_lancz |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:265:20` | 1 | never seen function: _where id=140480600977024 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:271:19` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: c128[],  y: c128[] closest |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:286:24` | 1 | never seen function: scan id=140477380572832 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:289:15` | 1 | never seen function: dynamic_slice id=139852189957984 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
