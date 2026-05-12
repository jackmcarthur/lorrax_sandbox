# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_lanczos/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 143 | 0.200 | 0.092 |
| jaxpr→MLIR | 21 | 0.293 | 0.125 |
| XLA compile | 22 | 3.012 | 1.882 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `_full_run` | 1 | 1.882 | 1.882 |
| `multiply` | 5 | 0.295 | 0.064 |
| `_multi_slice` | 1 | 0.175 | 0.175 |
| `sqrt` | 2 | 0.115 | 0.059 |
| `scatter-add` | 1 | 0.104 | 0.104 |
| `broadcast_in_dim` | 2 | 0.061 | 0.032 |
| `_squeeze` | 2 | 0.061 | 0.031 |
| `add` | 1 | 0.058 | 0.058 |
| `conjugate` | 1 | 0.057 | 0.057 |
| `convert_element_type` | 2 | 0.055 | 0.029 |
| `concatenate` | 1 | 0.054 | 0.054 |
| `reshape` | 1 | 0.033 | 0.033 |
| `dynamic_slice` | 1 | 0.031 | 0.031 |
| `squeeze` | 1 | 0.030 | 0.030 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_full_run` | 1 | 0.092 | 0.092 |
| `_matvec` | 1 | 0.031 | 0.031 |
| `norm` | 3 | 0.008 | 0.003 |
| `multiply` | 19 | 0.007 | 0.001 |
| `_diag` | 3 | 0.006 | 0.004 |
| `_normal` | 1 | 0.005 | 0.005 |
| `_normal_real` | 1 | 0.004 | 0.004 |
| `_einsum` | 8 | 0.004 | 0.001 |
| `_where` | 3 | 0.003 | 0.001 |
| `add` | 9 | 0.003 | 0.001 |
| `_uniform` | 1 | 0.003 | 0.003 |
| `broadcast_in_dim` | 3 | 0.002 | 0.002 |
| `true_divide` | 7 | 0.002 | 0.000 |
| `vdot` | 1 | 0.002 | 0.002 |
| `_reduce_sum` | 4 | 0.002 | 0.001 |
| `conjugate` | 7 | 0.002 | 0.000 |
| `subtract` | 7 | 0.002 | 0.000 |
| `_threefry_seed` | 1 | 0.001 | 0.001 |
| `eigh` | 1 | 0.001 | 0.001 |
| `_reduce_prod` | 2 | 0.001 | 0.001 |
| `_threefry_split` | 1 | 0.001 | 0.001 |
| `_psum` | 1 | 0.001 | 0.001 |
| `negative` | 6 | 0.001 | 0.000 |
| `_broadcast_arrays` | 4 | 0.001 | 0.000 |
| `sqrt` | 4 | 0.001 | 0.000 |
| `_threefry_split_foldlike` | 1 | 0.001 | 0.001 |
| `convert_element_type` | 4 | 0.001 | 0.000 |
| `less` | 3 | 0.001 | 0.000 |
| `scatter-add` | 2 | 0.001 | 0.000 |
| `real` | 4 | 0.001 | 0.000 |

## Tracing cache misses

Total: **35** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:659:14` | 3 | never seen function: convert_element_type id=140123980919360 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:325:15` | 3 | never seen function: fft id=140126831592448 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:247:9` | 3 | never seen function: _uniform id=139993608986016 defined at /opt/jax/jax/_src/random.py:407 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:420:19` | 2 | never seen function: _psum id=140606896414112 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:656:54` | 2 | never seen function: dynamic_slice id=140123980913920 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_simple.py:69:23` | 2 | never seen function: convert_element_type id=140123980466912 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:288:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,60],  x: f64[60],  y: f64[60]  |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_io.py:403:18` | 1 | never seen function: _identity_fn id=139993617974752 defined at /opt/jax/jax/_src/pjit.py:2575 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:648:11` | 1 | never seen function: broadcast_in_dim id=140123980806912 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:648:37` | 1 | never seen function: broadcast_in_dim id=140123980807872 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/gw/head_correction.py:651:19` | 1 | never seen function: convert_element_type id=140123980908640 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:249:14` | 1 | never seen function: norm id=139993609649856 defined at /opt/jax/jax/_src/numpy/linalg.py:1076 |
| `/global/homes/j/jackm/software/lorrax_C/src/common/fft_helpers.py:343:15` | 1 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 explanation unavailable! please open an issue at https://github.com/jax-ml/jax |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:287:21` | 1 | never seen function: build_bse_simple_matvec.<locals>._matvec id=139990239224320 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/ |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:265:20` | 1 | never seen function: _where id=139993615265408 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:271:19` | 1 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[],  x: c128[],  y: c128[] closest |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:277:17` | 1 | for norm defined at /opt/jax/jax/_src/numpy/linalg.py:1076 never seen input type signature: x: c128[4096] closest seen input type signature  |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:290:12` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 never seen input type signature: v: f64[59] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:290:36` | 1 | for _diag defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7967 explanation unavailable! please open an issue at https://github.com/jax-ml/ja |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:292:22` | 1 | never seen function: eigh id=139993609643616 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:293:10` | 1 | never seen function: argsort id=139993614744640 defined at /opt/jax/jax/_src/numpy/sorting.py:92 |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:297:12` | 1 | for norm defined at /opt/jax/jax/_src/numpy/linalg.py:1076 never seen input type signature: x: c128[20,4096] closest seen input type signatu |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:337:45` | 1 | never seen function: solve_bse_sharded.<locals>._full_run id=139990239227520 defined at /global/homes/j/jackm/software/lorrax_C/src/bse/bse_ |
| `/global/homes/j/jackm/software/lorrax_C/src/solvers/lanczos.py:245:13` | 1 | never seen function: _threefry_split id=139993608957088 defined at /opt/jax/jax/_src/prng.py:1111 |
| `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:343:19` | 1 | never seen function: reshape id=140123980478112 defined at /opt/jax/jax/_src/dispatch.py:96 |

## Persistent cache misses

_None._
