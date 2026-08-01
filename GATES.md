# GATES.md — gated paths: defaults and certified settings

Every major runtime gate, its default, the setting certified by
measurement, and whether a default flip is pending. Gates announce
themselves at resolution; an explicit request that cannot be honored
refuses rather than silently downgrading (`src/ffi/gate.py`). Verify a
default against the repo before relying on it; update this table when a
gate is added or flipped. Enumerate candidates with
`grep -rhoE "LORRAX_[A-Z0-9_]+" src/ | sort -u` in the repo.

| Gate | Default | Certified setting | Flip pending? |
|---|---|---|---|
| `LORRAX_FFT_FFI` (`src/ffi/fft.py`) | `off` (modes off/on, no auto) | `on` — CPU: MKL FFT (DFTI API) strided flat-k; CUDA: cuFFT `cufftPlanMany64` mirror. sigma.exec 3.78x (CLAIMS row 6) | Yes: becomes the permanent backend of `make_sharded_ifftn_3d` after the FFTW-many + dlsym rework (repo `docs/architecture/ffi_layout.md` §7), measurement-gated. An `auto` mode is an owner call. |
| `LORRAX_FFT_FFI_FUSED` (`src/ffi/fft.py`) | `off` | `on` together with `LORRAX_FFT_FFI` (fused gw_conv target) | Rides the same §7 plan. |
| `LORRAX_FFT_FFI_LOG` / `_THREADS` / `_CHUNK` | unset | tuning knobs; no certified values | No. |
| `LORRAX_BANDS_GEMM_FFI` (`src/ffi/gemm.py`) | `auto` (owner order) | `auto` — vendor GEMM (MKL host / cuBLAS CUDA) when probe passes | No. |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | jax default (`gloo`) | `mpi` — set by `config/frontera/mpi_transport_env.sh`; gloo banned at distributed tiers (CLAIMS rows 3-4) | Owner-gated docs rescoping gloo to P<=16; production already `mpi`. |
| `FI_PROVIDER` | libfabric autodetect | `mlx` in-container; `tcp` only as rtx escape hatch (CLAIMS row 13) | No. |
| `slab_io` (input file key, `src/file_io/`) | `auto` | `auto` -> tier 1 parallel-HDF5 FFI writer when the probe passes; falls back to the allgather tier | No. |
| `LORRAX_CHECK_REPLICA` | `0` | debug-only; keep off in production | No. |
| `LORRAX_MPI_FORCE_THREAD_MAIN` and transport glue | per `mpi_transport_env.sh` | exactly as the certified template sets them — do not hand-tune | No. |

Not gates: `LORRAX_IMPI_ROOT`, `LORRAX_MKL_ROOT`, `LORRAX_SIF`,
`LORRAX_BUNDLE`, `LORRAX_FFI_*_DIR` and similar are build/launch
environment plumbing, owned by `config/frontera/build_cpu_runtime_bundle.sh`
and the sbatch template.
