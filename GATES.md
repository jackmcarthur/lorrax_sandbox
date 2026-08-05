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
| `LORRAX_FFT_FFI` (`src/ffi/fft.py`) | `on` — REQUIRED (repo decisions.md 2026-08-01); missing library = startup refusal naming the .so; `=0` REFUSES (the XLA flat-k twin is deleted) | `on` — CPU: MKL FFT (DFTI API) strided flat-k; CUDA: cuFFT `cufftPlanMany64` mirror. sigma.exec 3.78x (CLAIMS row 6) | Flip LANDED 2026-08-01 for the flat-k layer. The `make_sharded_ifftn_3d` shard_map-interior layer still has NO FFI route (repo `docs/architecture/ffi_layout.md` §7 rework still open; that layer is KEPT XLA by the ruling until a route exists). |
| `LORRAX_FFT_FFI_FUSED` (`src/ffi/fft.py`) | `on` (2026-08-01); `=0` = announced opt-out to the decomposed chain (itself FFI-served) | `on` together with `LORRAX_FFT_FFI` (fused gw_conv target) | No — flip landed. |
| `LORRAX_FFT_FFI_LOG` / `_THREADS` / `_CHUNK` | unset | tuning knobs; no certified values | No. |
| `LORRAX_BANDS_GEMM_FFI` (`src/ffi/gemm.py`) | `on` — REQUIRED (2026-08-01); `auto` mode DELETED (stale `=auto` resolves to default, announced); `=0` = announced UNCERTIFIED opt-out onto the XLA einsum arm (retained for the structural `extra='minor'` case) | `on` — vendor GEMM (MKL host; CUDA runs XLA's native cuBLAS dot — the dial does not exist there) | No — flip landed. |
| `JAX_CPU_COLLECTIVES_IMPLEMENTATION` | jax default (`gloo`) | `mpi` — set by `config/frontera/mpi_transport_env.sh`; gloo banned at distributed tiers (CLAIMS rows 3-4) | Owner-gated docs rescoping gloo to P<=16; production already `mpi`. |
| `FI_PROVIDER` | libfabric autodetect | `mlx` in-container; `tcp` only as rtx escape hatch (CLAIMS row 13) | No. |
| `slab_io` (input file key, `src/file_io/`) | `auto` | `auto` -> tier 1 parallel-HDF5 FFI writer when the probe passes; falls back to the allgather tier | No. |
| `LORRAX_CHECK_REPLICA` | `0` | debug-only; keep off in production | No. |
| `LORRAX_MPI_FORCE_THREAD_MAIN` and transport glue | per `mpi_transport_env.sh` | the certified template deliberately leaves `LORRAX_MPI_FORCE_THREAD_MAIN` UNSET — its role is superseded by `warm_mesh_cliques`; other transport glue exactly as the template sets it, do not hand-tune | No. |

Not gates: `LORRAX_IMPI_ROOT`, `LORRAX_MKL_ROOT`, `LORRAX_SIF`,
`LORRAX_BUNDLE`, `LORRAX_FFI_*_DIR` and similar are build/launch
environment plumbing, owned by `config/frontera/build_cpu_runtime_bundle.sh`
and the sbatch template.

Running fully distributed (no N_mu^2 tile on any rank): the per-stage
deck keys (`distributed_zeta_solve` / `distributed_lu` / `w_dyson_solver`
/ `eigh_backend`), their auto-thresholds with calibration, and the
certified example jobs live in repo `docs/dev/large_nmu_operation.md`.

Numerics-affecting opt-in deck keys (NOT env gates): `ppm_probe_chi_reuse`
(off | auto, default off) folds the GN probe χ₀ into the static τ sweep on
augmented nodes — same quadrature-error contract, not bit-identical, and
NO net win at b300 scale (planning cost ≈ node saving; CLAIMS 27, repo
`docs/input_reference.md`). Keep off for pinned-baseline decks.
