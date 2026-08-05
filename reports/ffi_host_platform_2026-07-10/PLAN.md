# FFI host-platform port — SLATE on JAX CPU backend (plan)

_Executes the P2 follow-up spec'd in `reports/slate_linalg_ffi_2026-07-10/report.md`
("FFI CPU story"). Branch: `agent/ffi-host-platform` on lorrax_C, base `adc2197`
(origin/main). Goal: the distributed-linalg FFI works when JAX runs on CPU devices,
with GPU/CPU switching structured like JAX upstream (same target name registered
per platform; platform-specific handler libraries)._

## Model (JAX upstream parallel)

jaxlib registers `lapack_*` kernels under `platform="cpu"` and `cusolver_*` under
`platform="CUDA"`; `jax.ffi.ffi_call` resolves the target by the lowering platform,
so call sites are platform-agnostic. LORRAX mirror:

| piece | CUDA (exists) | Host (new) |
|---|---|---|
| handler lib | `liblorrax_ffi.so` (cuSOLVERMp+cuBLASMp+NCCL+phdf5+slate) | `liblorrax_ffi_host.so` (slate + MPI only, zero CUDA) |
| handler symbols | `SlatePotrfFfi` … | `SlatePotrfHostFfi` … (one TU: `slate/cpp/host_ffi.cc`) |
| staging | `cudaMemcpyAsync` D2D + stream sync | `std::memcpy` |
| matrix wrap | `fromDevices()` (device ptrs) | `fromScaLAPACK()` (host ptrs, same block-cyclic layout + GridOrder::Col) |
| execution | `Target::Devices` | `Target::HostTask` |
| registration | `platform="CUDA"` | `platform="cpu"` |
| build | in-container (`build.sh` via `run_shifter.sh`) | host-side Cray PE (`host/build_host.sh`), SLATE `cpu` install, staged container XLA-FFI headers |

Unchanged: `slate/cpp/context.cc` (pure MPI — compiled into both libs),
`slate/*.py` shard_map/local-transpose plumbing, `SlateCtx` (POD, handles
interchangeable across libs), all layout/mesh validation in `context.py`.

## Verified API facts

- `Matrix::fromScaLAPACK(m, n, A, lld, mb, nb, p, q, comm)` per-dimension-tile
  overload exists (needed for trsm rectangular X); Hermitian/Triangular variants
  take single `nb`. Default GridOrder::Col matches `fromDevices` → the comm
  rank-remap in `context.cc` carries unchanged.
- Host `tileGetForReading(i, j, LayoutConvert)` overload exists (eigh writeback —
  same MOSI lesson as the 2026-07-10 device fix, host-side).
- XLA FFI API: container jax 0.7.2; lorrax_C/.venv jax 0.9.1 ships API 0.3 —
  newer-headers-on-older-runtime is unsafe, so host build stages the CONTAINER's
  `jax.ffi.include_dir()` headers to `$HOME/software/lorrax_xla_ffi_headers/`.

## Work items

1. `src/ffi/slate/cpp/host_ffi.cc` — five host handlers (potrf, trsm, eigh,
   batched_potrf, batched_trsm), `Ffi::Bind()` without `Ctx<PlatformStream>`,
   same attrs/target semantics as device TUs. Device TUs untouched.
2. `src/ffi/common/cpp/host/CMakeLists.txt` + `build_host.sh` — self-contained
   CUDA-free build: CC wrapper, slate cpu install (`LORRAX_SLATE_HOST_INSTALL_DIR`,
   default `$HOME/software/slate_builds/cpu/install`), staged XLA headers, stand-in
   `MPI::MPI_CXX`. Output `host/build/liblorrax_ffi_host.so`. readelf must show no
   CUDA/NCCL NEEDED entries.
3. `ffi_loader.py` — per-platform tables: `{platform: (so name, env override,
   symbol map, argtype decl set)}`; `get_lib(platform=None)` defaulting via
   `jax.default_backend()`; registration per platform on load; slate lifecycle
   helpers route to any lib that exports them. Loud FileNotFoundError with
   build pointers per platform.
4. Config: `gw_config.py` CPU-backend force — `slate` now passes through on CPU;
   `auto`/`cusolvermp` still forced `off` (auto never picks slate; cusolvermp is
   CUDA-only). `isdf/core.py` `_require_slate_ffi` message gains the host build
   pointer; 1×q mesh guard unchanged.
5. Tests: extend `tests/test_ffi_linalg_contract.py` — slate cases parametrized
   over platform (gpu, cpu); cpu cases run on the JAX CPU devices of the same
   process (`jax.default_device` / cpu mesh), skipif-clean when
   `liblorrax_ffi_host.so` absent. CLI mode gains `--platform cpu` for
   multi-rank CPU meshes.
6. Docs: `slate/README.md` CPU-story section rewritten (status: done),
   `ffi/AGENTS.md` layout table, `PORTING.md` note.
7. Validation: (a) contract suite GPU — unchanged green; (b) cpu cases on GPU
   node (JAX_PLATFORMS unset, cpu devices in-process); (c) multi-rank 2×2 CPU
   via CLI on compute node; (d) readelf CUDA-free proof; (e) full pytest suite;
   (f) adversarial multi-agent review of the diff before checkpoint.

## Risks / open questions

- SLATE heev + HostTask via fromScaLAPACK-wrapped buffers: does the back-transform
  land in wrapped user memory or internal tiles? Mitigated by the same explicit
  tileGetForReading + copy-out writeback as the device fix.
- Bit-determinism of HostTask (OpenMP) reruns: contract tests assert it on GPU;
  keep for CPU, relax to documented note if flaky.
- cray-mpich singleton init (1-rank pytest case) inside container on CPU backend —
  expected fine (same path the GPU 1×1 tests use).
- scalapack backend (near future, per user): host lib + loader tables are named
  generically (`liblorrax_ffi_host.so`) so its handlers join the same lib later.
