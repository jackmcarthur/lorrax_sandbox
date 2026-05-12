# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/C_mos2_4x4_200b_htransform/profile_4gpu/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 305 | 0.142 | 0.007 |
| jaxpr→MLIR | 228 | 0.800 | 0.123 |
| XLA compile | 228 | 14.492 | 0.453 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `multiply` | 24 | 1.421 | 0.068 |
| `transpose` | 11 | 1.168 | 0.453 |
| `add` | 15 | 0.922 | 0.083 |
| `subtract` | 11 | 0.634 | 0.064 |
| `conjugate` | 10 | 0.626 | 0.077 |
| `_reduce_max` | 6 | 0.588 | 0.141 |
| `broadcast_in_dim` | 16 | 0.547 | 0.056 |
| `sort` | 4 | 0.535 | 0.222 |
| `_solve_triangular` | 4 | 0.468 | 0.120 |
| `gather` | 8 | 0.449 | 0.063 |
| `exp` | 5 | 0.418 | 0.100 |
| `matmul` | 8 | 0.365 | 0.135 |
| `norm` | 2 | 0.356 | 0.295 |
| `_where` | 6 | 0.342 | 0.059 |
| `reshape` | 12 | 0.342 | 0.033 |
| `iota` | 6 | 0.319 | 0.058 |
| `eigvalsh` | 3 | 0.309 | 0.105 |
| `swapaxes` | 4 | 0.298 | 0.077 |
| `dynamic_slice` | 5 | 0.279 | 0.058 |
| `abs` | 4 | 0.272 | 0.083 |
| `_reduce_min` | 3 | 0.271 | 0.140 |
| `scan` | 2 | 0.260 | 0.134 |
| `cholesky` | 2 | 0.253 | 0.129 |
| `convert_element_type` | 6 | 0.223 | 0.060 |
| `greater` | 3 | 0.192 | 0.067 |
| `_identity_fn` | 4 | 0.184 | 0.067 |
| `negative` | 3 | 0.178 | 0.065 |
| `greater_equal` | 3 | 0.169 | 0.059 |
| `_argmin` | 2 | 0.156 | 0.078 |
| `real` | 3 | 0.145 | 0.063 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `multiply` | 29 | 0.012 | 0.001 |
| `_where` | 9 | 0.007 | 0.001 |
| `_diag` | 1 | 0.007 | 0.007 |
| `add` | 19 | 0.007 | 0.001 |
| `diagonal` | 1 | 0.007 | 0.007 |
| `norm` | 2 | 0.005 | 0.003 |
| `subtract` | 14 | 0.005 | 0.001 |
| `cholesky` | 2 | 0.005 | 0.003 |
| `matmul` | 9 | 0.005 | 0.001 |
| `negative` | 15 | 0.005 | 0.001 |
| `_accum_G` | 1 | 0.005 | 0.005 |
| `clip` | 3 | 0.004 | 0.002 |
| `broadcast_in_dim` | 16 | 0.004 | 0.000 |
| `transpose` | 11 | 0.004 | 0.001 |
| `reshape` | 12 | 0.004 | 0.000 |
| `abs` | 5 | 0.003 | 0.001 |
| `greater` | 5 | 0.003 | 0.002 |
| `_reduce_max` | 6 | 0.003 | 0.001 |
| `conjugate` | 11 | 0.003 | 0.000 |
| `_broadcast_arrays` | 10 | 0.002 | 0.000 |
| `true_divide` | 7 | 0.002 | 0.000 |
| `gather` | 8 | 0.002 | 0.000 |
| `_reduce_sum` | 4 | 0.002 | 0.001 |
| `exp` | 6 | 0.002 | 0.000 |
| `greater_equal` | 4 | 0.002 | 0.000 |
| `eigvalsh` | 1 | 0.002 | 0.002 |
| `convert_element_type` | 6 | 0.002 | 0.000 |
| `less` | 5 | 0.002 | 0.000 |
| `_reduce_min` | 3 | 0.002 | 0.001 |
| `maximum` | 3 | 0.001 | 0.001 |

## Tracing cache misses

Total: **109** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:648:37` | 6 | for sort defined at /opt/jax/jax/_src/numpy/sorting.py:31 never seen input type signature: a: f64[181,960] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:206:8` | 4 | never seen function: convert_element_type id=139735356990432 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:164:47` | 3 | never seen function: convert_element_type id=139772202441984 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:173:11` | 3 | for transpose defined at /opt/jax/jax/_src/dispatch.py:96 never seen input type signature: args[0]: c128[960,960] closest seen input type si |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:212:44` | 3 | never seen function: iota id=139735890130272 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:610:23` | 3 | never seen function: broadcast_in_dim id=139769048613248 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:610:13` | 3 | never seen function: broadcast_in_dim id=139771558030656 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:614:20` | 3 | never seen function: reshape id=139769046582912 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:80:12` | 2 | never seen function: fft id=139738505790464 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:82:33` | 2 | never seen function: reshape id=139738439411776 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:211:10` | 2 | never seen function: gather id=139735890120992 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:277:14` | 2 | never seen function: dynamic_slice id=139772597932960 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:290:15` | 2 | never seen function: convert_element_type id=139769846655520 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:299:8` | 2 | never seen function: _where id=139772654890624 defined at /opt/jax/jax/_src/numpy/util.py:287 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:300:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[60,16],  y: f64[6 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:301:8` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[60,16],  x: f64[],  y: f64[60,16] |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:585:45` | 2 | never seen function: gather id=139771558306944 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:585:25` | 2 | never seen function: eigh id=139772649268832 defined at /opt/jax/jax/_src/numpy/linalg.py:809 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:586:28` | 2 | never seen function: dynamic_slice id=140097137201920 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:587:39` | 2 | never seen function: gather id=139735358695008 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:590:41` | 2 | never seen function: gather id=139735358375232 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:595:39` | 2 | never seen function: diagonal id=139772648953216 defined at /opt/jax/jax/_src/numpy/lax_numpy.py:7842 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:603:14` | 2 | never seen function: reshape id=139769047234048 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:607:11` | 2 | never seen function: transpose id=139769046765856 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:408:10` | 2 | never seen function: iota id=139769046772896 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:409:11` | 2 | for _where defined at /opt/jax/jax/_src/numpy/util.py:287 never seen input type signature: condition: bool[4],  x: f64[4],  y: f64[4] closes |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:619:9` | 2 | never seen function: convert_element_type id=139769048627648 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:641:22` | 2 | never seen function: dynamic_slice id=139772598159456 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:601:40` | 2 | never seen function: transpose id=139769048267520 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_C/src/bandstructure/htransform.py:339:13` | 2 | for clip defined at /opt/jax/jax/_src/numpy/lax_numpy.py:3379 never seen passing 3 positional args and 0 keyword args with keys: |

## Persistent cache misses

_None._
