# Compilation log summary

**Log:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_compile_floor_check/gwjax_profile_v4b/compile.log`

## Wall-clock totals across the run

| Stage | Count | Total seconds | Max single |
|---|---:|---:|---:|
| trace+transform | 661 | 0.651 | 0.054 |
| jaxpr→MLIR | 219 | 0.786 | 0.149 |
| XLA compile | 242 | 8.169 | 0.698 |

## Top 5 XLA compilations by total time

| jit() name | Count | Total s | Max s |
|---|---:|---:|---:|
| `sigma_sx` | 3 | 1.028 | 0.368 |
| `sigma_coh` | 2 | 0.724 | 0.364 |
| `_kernel` | 1 | 0.698 | 0.698 |
| `convert_element_type` | 26 | 0.441 | 0.286 |
| `_compute_CCT_LR` | 2 | 0.425 | 0.215 |

## Top 5 pjit trace+transform by total time

| function | Count | Total s | Max s |
|---|---:|---:|---:|
| `_kernel` | 2 | 0.068 | 0.054 |
| `_fft_and_rslice` | 2 | 0.040 | 0.020 |
| `_solve_w` | 1 | 0.034 | 0.034 |
| `_compute_S_omega_jit` | 1 | 0.026 | 0.026 |
| `get_sqrt_v_and_phase` | 1 | 0.025 | 0.025 |

## Tracing cache misses

Total: **231** cache misses. Each one is a retrace event — look at the **because** line to find the root cause (new shape, new static arg, new jaxpr, etc.).

| Location | Misses | Sample reason |
|---|---:|---|
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:214:31` | 8 | for fft defined at /opt/jax/jax/_src/lax/fft.py:68 never seen input type signature: x: c128[4,4,480,480,4] closest seen input type signature |
| `/global/homes/j/jackm/software/lorrax_B/src/common/fft_helpers.py:222:22` | 6 | never seen function: fft id=140201959605248 defined at /opt/jax/jax/_src/lax/fft.py:68 |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/_slab_io_ffi.py:385:18` | 5 | never seen function: _FfiBackend.write_slab.<locals>._per_rank id=140480462140608 defined at /global/homes/j/jackm/software/lorrax_B/src/fil |
| `/global/homes/j/jackm/software/lorrax_B/src/file_io/tagged_arrays.py:224:29` | 5 | never seen function: convert_element_type id=140487245563520 defined at /opt/jax/jax/_src/dispatch.py:96 |
| `/global/homes/j/jackm/software/lorrax_B/src/runtime/__init__.py:195:12` | 4 | never seen function: _psum id=140503895921408 defined at /opt/jax/jax/experimental/multihost_utils.py:42 |

## Persistent cache misses

_None._
