# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile_v4/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 666 | 0.625 | 0.035 |
| jaxpr→MLIR | 231 | 0.929 | 0.151 |
| XLA compile | 239 | 8.955 | 0.692 |

## Top 5 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `sigma_sx` | 3 | 1.031 | 0.361 |
| `sigma_coh` | 2 | 0.787 | 0.424 |
| `get_sqrt_v_and_phase` | 2 | 0.712 | 0.641 |
| `_kernel` | 1 | 0.692 | 0.692 |
| `convert_element_type` | 26 | 0.580 | 0.286 |

## Top 5 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_fft_and_rslice` | 3 | 0.060 | 0.020 |
| `_solve_w` | 1 | 0.035 | 0.035 |
| `get_sqrt_v_and_phase` | 1 | 0.031 | 0.031 |
| `_psum` | 15 | 0.030 | 0.014 |
| `_reduce_sum` | 30 | 0.029 | 0.014 |

## Tracing cache misses

Total: **225** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:214:31` | 9 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[4,4,480,480,4] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:222:22` | 6 | never seen function: fft id=140069458052096 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:385:18` | 5 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=140701588796608 defined at /global/homes/j/jackm/software/lorrax_B/src/fil |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=140709128341120 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:195:12` | 4 | never seen function: _psum id=140415712354016 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |

## Persistent cache misses

_None._
