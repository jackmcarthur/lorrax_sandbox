# Portability — where the answers live

This page is a router, not a spec. The repo (`/work2/08271/jackmc/frontera/lorrax`)
owns the details; do not duplicate them here.

## Machines

| Topic | Authority |
|---|---|
| Frontera (current target: CLX CPU, 2x28 cores, 192 GB, IMPI+MKL, apptainer) | repo `docs/environment/machines/frontera.md`; 9-layer stack in `docs/environment/overview.md`; transports in `docs/environment/transports.md` |
| Perlmutter (dormant GPU leg; Cray SS11, Shifter era) | repo `docs/environment/machines/perlmutter.md`. The old sandbox recipes for it are in `_archive/` and are not runnable. |
| Launch on Frontera | `config/frontera/templates/gw_dev.sbatch` + `config/frontera/mpi_transport_env.sh`; bundles via `config/frontera/build_cpu_runtime_bundle.sh` |

## Vendor libraries

| Topic | State | Authority |
|---|---|---|
| FFT engine (MKL DFTI today; FFTW-many + dlsym as the portable spelling — MKL exports FFTW3 symbols natively, `cray-fftw` on Cray legs, cuFFT `cufftPlanMany64` mirror on GPU) | plan owner-directed 2026-07-31 | repo `docs/architecture/ffi_layout.md` §7 |
| GEMM (vendor BLAS via runtime dlsym per symbol with announced refusal; fallback to XLA path) | certified, `auto` | repo `src/ffi/gemm.py`, `gemm_batch_ffi.cc`; pattern description in `ffi_layout.md` §7 |
| Dense-solver legs: SLATE and ScaLAPACK FFI (`LORRAX_SLATE_*`, `LORRAX_SCALAPACK_*`) | built on Frontera; `test_slate_cholesky_trsm_cpu` hang is OPEN (CLAIMS row 15) | repo `src/ffi/slate/`, `src/ffi/scalapack/`, `docs/architecture/ffi_layout.md` |
| Parallel HDF5 writer (`slab_io=auto` tier 1) | certified | repo `src/ffi/phdf5/`, `src/file_io/` |
| CUDA wheel pin | `pyproject.toml` is authoritative for all Python-side pins (jax 0.9 line; `jax[cuda12]` for uv installs (switched from cuda13: Frontera rtx driver 535 supports CUDA <= 12; flip back to `jax[cuda13]` only on a newer-driver machine, per the comment in pyproject.toml)). The Frontera container venv is CPU; bundles strip GPU plugin files. | repo `docs/installation/index.md`, `pyproject.toml` |

Rule of thumb: every vendor dependency enters through an FFI facade with
runtime resolution and announced refusal — absent engine means slower and
loud, never broken. New legs copy that pattern (`src/ffi/TEMPLATE.md`).
