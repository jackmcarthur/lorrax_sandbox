# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 6 | 0.006 | 0.003 |
| jaxpr→MLIR | 3 | 0.121 | 0.113 |
| XLA compile | 3 | 0.523 | 0.264 |

## Top 30 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `convert_element_type` | 1 | 0.264 | 0.264 |
| `broadcast_in_dim` | 1 | 0.156 | 0.156 |
| `kernel` | 1 | 0.104 | 0.104 |

## Top 30 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `kernel` | 2 | 0.004 | 0.003 |
| `matmul` | 1 | 0.001 | 0.001 |
| `_reduce_sum` | 1 | 0.001 | 0.001 |
| `broadcast_in_dim` | 1 | 0.000 | 0.000 |
| `convert_element_type` | 1 | 0.000 | 0.000 |

## Tracing cache misses

Total: **4** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke_driver.py:12:4` | 2 | never seen function: convert_element_type id=140282327061888 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke_driver.py:18:12` | 1 | never seen function: kernel id=140285395003616 defined at /pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke_driver.py:8 |
| `/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/pf.py:282:47` | 1 | never seen function: kernel id=140282326759952 defined at /pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke_driver.py:8 |

## Persistent cache misses

_None._
