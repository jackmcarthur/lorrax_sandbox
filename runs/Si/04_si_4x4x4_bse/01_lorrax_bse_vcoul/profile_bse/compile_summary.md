# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/profile_bse/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 162 | 0.133 | 0.024 |
| jaxpr→MLIR | 62 | 0.397 | 0.134 |
| XLA compile | 62 | 4.934 | 0.644 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_normal` | 2 | 0.764 | 0.399 |
| `scan` | 1 | 0.644 | 0.644 |
| `convert_element_type` | 7 | 0.536 | 0.342 |
| `broadcast_in_dim` | 9 | 0.269 | 0.053 |
| `add` | 5 | 0.252 | 0.053 |
| `less` | 2 | 0.238 | 0.189 |
| `gather` | 4 | 0.218 | 0.061 |
| `_diag` | 3 | 0.203 | 0.085 |
| `dynamic_slice` | 4 | 0.190 | 0.060 |
| `argsort` | 1 | 0.180 | 0.180 |
| `norm` | 2 | 0.177 | 0.093 |
| `multiply` | 3 | 0.160 | 0.055 |
| `_threefry_split` | 1 | 0.151 | 0.151 |
| `transpose` | 2 | 0.146 | 0.078 |
| `true_divide` | 2 | 0.138 | 0.070 |
| `eigh` | 1 | 0.131 | 0.131 |
| `select_n` | 2 | 0.104 | 0.052 |
| `scatter` | 1 | 0.070 | 0.070 |
| `matmul` | 1 | 0.058 | 0.058 |
| `_threefry_seed` | 1 | 0.053 | 0.053 |
| `_unstack` | 1 | 0.053 | 0.053 |
| `squeeze` | 2 | 0.052 | 0.026 |
| `maximum` | 1 | 0.052 | 0.052 |
| `_broadcast_arrays` | 2 | 0.049 | 0.025 |
| `reshape` | 1 | 0.024 | 0.024 |
| `_squeeze` | 1 | 0.023 | 0.023 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_matvec_impl` | 1 | 0.024 | 0.024 |
| `_normal` | 2 | 0.014 | 0.008 |
| `_normal_real` | 2 | 0.013 | 0.008 |
| `_uniform` | 2 | 0.009 | 0.005 |
| `_diag` | 3 | 0.009 | 0.006 |
| `norm` | 2 | 0.008 | 0.005 |
| `_einsum` | 8 | 0.006 | 0.001 |
| `add` | 18 | 0.006 | 0.001 |
| `multiply` | 13 | 0.004 | 0.001 |
| `_reduce_sum` | 2 | 0.003 | 0.002 |
| `broadcast_in_dim` | 9 | 0.003 | 0.001 |
| `true_divide` | 6 | 0.002 | 0.000 |
| `subtract` | 8 | 0.002 | 0.000 |
| `eigh` | 1 | 0.002 | 0.002 |
| `_where` | 3 | 0.002 | 0.001 |
| `_threefry_seed` | 1 | 0.002 | 0.002 |
| `convert_element_type` | 7 | 0.002 | 0.001 |
| `matmul` | 1 | 0.002 | 0.002 |
| `conjugate` | 5 | 0.002 | 0.001 |
| `_threefry_split` | 1 | 0.001 | 0.001 |
| `less` | 4 | 0.001 | 0.001 |
| `bitwise_or` | 5 | 0.001 | 0.000 |
| `gather` | 4 | 0.001 | 0.000 |
| `dynamic_slice` | 4 | 0.001 | 0.000 |
| `vdot` | 1 | 0.001 | 0.001 |
| `bitwise_xor` | 4 | 0.001 | 0.000 |
| `select_n` | 2 | 0.001 | 0.001 |
| `_threefry_split_foldlike` | 1 | 0.001 | 0.001 |
| `real` | 4 | 0.001 | 0.000 |
| `maximum` | 3 | 0.001 | 0.000 |

## Tracing cache misses

Total: **57** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_io.py:578:12` | 6 | never seen function: add id=140547305108352 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:294:18` | 6 | for add defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[20],  args[1]: i64[] closest seen input ty |
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_io.py:592:10` | 3 | never seen function: dynamic_slice id=140546838661216 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_io.py:596:8` | 3 | never seen function: _uniform id=140548932902144 defined at /opt/jax/jax/_src/random.py:407 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:247:9` | 3 | for _uniform defined at /opt/jax/jax/_src/random.py:407 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:252:8` | 3 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i32[] closest seen input type |
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_io.py:580:12` | 2 | never seen function: gather id=140547304531232 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_io.py:589:11` | 2 | never seen function: dynamic_slice id=140547305108192 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:251:8` | 2 | never seen function: convert_element_type id=140546838936480 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:253:12` | 2 | never seen function: convert_element_type id=140546838619424 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:288:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[200,200],  x: f64[200],  y: f64[2 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:293:10` | 2 | never seen function: argsort id=140548938660768 defined at /opt/jax/jax/_src/numpy/sorting.py:92 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:296:24` | 2 | never seen function: gather id=140547302298688 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_io.py:575:18` | 1 | never seen function: convert_element_type id=140548932852672 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_io.py:595:10` | 1 | for convert_element_type defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: i64[] closest seen input type |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:245:13` | 1 | never seen function: _threefry_split id=140548932856832 defined at /opt/jax/jax/_src/prng.py:1111 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:245:4` | 1 | never seen function: _unstack id=140548933990048 defined at /opt/jax/jax/_src/numpy/array_methods.py:636 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:249:14` | 1 | never seen function: norm id=140548939382304 defined at /opt/jax/jax/_src/numpy/linalg.py:1076 |
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_serial.py:72:10` | 1 | never seen function: fft id=140548942840672 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_serial.py:73:10` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[480,480,4,4,4] closest seen input type signature |
| `/global/u2/j/jackm/software/lorrax_A/src/bse/bse_serial.py:75:10` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:258:12` | 1 | never seen function: solve_bse.<locals>._matvec_impl id=140546838670976 defined at /global/u2/j/jackm/software/lorrax_A/src/bse/bse_lanczos. |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:265:20` | 1 | never seen function: _where id=140548941295072 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:271:19` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: c128[],  y: c128[] closest |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:286:24` | 1 | never seen function: scan id=140547304209472 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:289:15` | 1 | never seen function: dynamic_slice id=140547303723616 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:290:12` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 never seen input type signature: v: f64[199] closest seen input type signatur |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:290:36` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 explanation unavailable! please open an issue at https://github.com/jax-ml/ja |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:292:22` | 1 | never seen function: eigh id=140548939376064 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/u2/j/jackm/software/lorrax_A/src/solvers/lanczos.py:296:19` | 1 | never seen function: transpose id=140547302305728 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
