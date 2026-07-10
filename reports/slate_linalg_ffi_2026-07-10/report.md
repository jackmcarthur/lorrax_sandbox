# SLATE linalg FFI — overnight program report

_Branch `agent/slate-linalg-ffi` on lorrax_D (base `9605a0f`). Plan: `PLAN.md`._
_Sections appended per phase._

---

## P2 — SLATE build hardening: scripted GPU + CPU builds (2026-07-10)

### Deliverables

| Item | Where |
|---|---|
| Build script (both variants) | `src/ffi/slate/scripts/build_perlmutter.sh` |
| GPU build (`gpu_backend=cuda`) | `$HOME/software/slate_builds/gpu/{build,install}` |
| CPU build (`gpu_backend=none`) | `$HOME/software/slate_builds/cpu/{build,install}` |
| Pinned source | `$HOME/software/slate_builds/src/slate` @ `ded15290` (v2025.05.28-1 — same commit as the `$HOME/software/slate` evaluation build) |
| FFI vs new build | `$HOME/software/slate_builds/ffi_build_gpu/liblorrax_ffi.so` (separate build dir via new `LORRAX_FFI_BUILD_DIR` knob in `build.sh`) |
| README | `src/ffi/slate/README.md` — new "Building" section |

The `$HOME/software/slate` evaluation install was **not touched**.

### Module stacks (NERSC-recommended; script loads them explicitly)

Per docs.nersc.gov (CUDA + cray-mpich pages): GPU codes build with
`PrgEnv-gnu cudatoolkit craype-accel-nvidia80`; `craype-accel-nvidia80`
additionally makes the `CC` wrapper link `libmpi_gtl_cuda` (the GPU
Transport Layer that Cray MPICH dlopens when `MPICH_GPU_SUPPORT_ENABLED=1`).

| Variant | Modules | Toolchain seen |
|---|---|---|
| gpu | `PrgEnv-gnu cray-libsci cmake cudatoolkit/12.9 craype-accel-nvidia80` | GNU 14.3.0 + nvcc 12.9, sm_80 |
| cpu | `PrgEnv-gnu cray-libsci cmake` + explicit `module unload cudatoolkit craype-accel-nvidia80` | GNU 14.3.0, zero CUDA in the link |

Link-time proof (readelf NEEDED): gpu `libslate.so` needs `libcudart.so.12,
libcublas*, libcusolver, libcuda.so.1, libmpi_gtl_cuda.so.0`; cpu
`libslate.so` needs none of them — only libsci/mpich/gomp/system libs.

### CMake line (both variants; from the script)

```
cmake -S $SRC -B $BUILD
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DCMAKE_Fortran_COMPILER=ftn
  -DCMAKE_INSTALL_PREFIX=$PREFIX
  -Dblas=libsci
  -Dgpu_backend={cuda|none}
  -DSCALAPACK_LIBRARIES=""          # THE gotcha — demystified below
  -Dbuild_tests=yes
  [-DCMAKE_CUDA_ARCHITECTURES=80]   # gpu variant only
```

**`-DSCALAPACK_LIBRARIES=""` demystified** (was cargo-cult until now):
`test/CMakeLists.txt` defaults it to `"scalapack"` → `-lscalapack`, which
does not exist standalone on Cray — ScaLAPACK lives *inside*
`libsci_gnu_mpi`, which the `CC` wrapper already links implicitly.  Empty
string keeps the tester's ScaLAPACK reference checks compiled in
(`SLATE_HAVE_SCALAPACK`) while adding no `-l` flags, so wrapper-provided
libsci satisfies the `p*` symbols.  (`"none"` would compile the reference
path OUT and `--ref=y` cross-checks would be unavailable.)  Validated:
`--ref=y` runs passed on both node types (below), i.e. the libsci
ScaLAPACK really is being exercised.

`-DSLATE_HAVE_MT_BCAST` deliberately NOT set: ICL INSTALL.md warns the
multi-threaded-bcast path hangs "on certain systems, particularly
Frontier" — and Frontier portability is the point of this exercise.

### CPU-story decision: separate `gpu_backend=none` build — YES

1. **SLATE's execution target is a runtime option** (`Option::Target`:
   `Devices` vs `HostTask`), and blaspp's `get_device_count()` returns 0
   rather than erroring when CUDA reports no devices.  Verified: the cuda
   build ran `--target=t` (HostTask) successfully on the GPU node **and on
   the Milan CPU node** — the latter works only because Perlmutter CPU
   nodes happen to ship `/usr/lib64/libcuda.so.1` (site quirk, NOT
   portable; the cuda build hard-NEEDs `libcuda.so.1` + GTL).
2. **The `none` build is the config that carries** to CPU-only or
   non-NVIDIA machines unchanged, and never drags CUDA/GTL into a CPU-node
   link.  Cost: one more invocation of the same script.

GPU nodes → `gpu/` build (Target::Devices; host-target available for
debugging). CPU nodes → `cpu/` build.

### SLATE tester results (all `pass`, `--check=y`; potrf/trsm also `--ref=y` vs libsci ScaLAPACK)

GPU node (nid001008), gpu build, 4 ranks × 1 A100 (`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`), grid 2×2:

| routine | target | type | n (nb) | error | status |
|---|---|---|---|---|---|
| potrf | devices | d | 512 (128) | 6.12e-19 | pass (+ref) |
| potrf | devices | z | 512 (128) | 6.44e-19 | pass (+ref) |
| trsm  | devices | d | 512 (128) | 9.22e-20 | pass (+ref) |
| trsm  | devices | z | 512 (128) | 1.62e-19 | pass (+ref) |
| heev (vec) | devices | d | 256 (64) | 2.06e-17 / 1.83e-16 | pass |
| heev (vec) | devices | z | 256 (64) | 2.35e-17 / 2.70e-16 | pass |
| potrf | **host-task** | z | 512 (128) | 5.63e-19 | pass (cuda build, host exec) |

CPU node (nid004157, Milan), cpu (`none`) build, 4 ranks × 8 cores,
`MPICH_GPU_SUPPORT_ENABLED=0`, grid 2×2:

| routine | target | type | n (nb) | error | status |
|---|---|---|---|---|---|
| potrf | host-task | d | 512 (128) | 5.02e-19 | pass (+ref) |
| potrf | host-task | z | 512 (128) | 5.70e-19 | pass (+ref) |
| trsm  | host-task | d | 512 (128) | 8.20e-20 | pass (+ref) |
| trsm  | host-task | z | 512 (128) | 1.43e-19 | pass (+ref) |
| heev (vec) | host-task | d | 256 (64) | 1.76e-17 / 1.94e-16 | pass |
| heev (vec) | host-task | z | 256 (64) | 6.42e-17 / 2.31e-16 | pass |

(Testsweeper syntax note for future agents: flags are `--flag=value`;
`--flag value` silently parses the value as a routine name.)

### LORRAX FFI vs new GPU build — PASS

`liblorrax_ffi.so` rebuilt in a **separate** build dir against the new
install (`LORRAX_SLATE_INSTALL_DIR=…/slate_builds/gpu/install`,
`LORRAX_FFI_BUILD_DIR=…/slate_builds/ffi_build_gpu`, then runtime
`LORRAX_FFI_SO=` override) — the in-tree `build/liblorrax_ffi.so` that P1
may be testing against was not modified.  cmake logged
`SLATE: found via …/slate_builds/gpu/install`.

Smoke on GPU node, 4 ranks, 2×2 mesh, c128:

| test | residuals | verdict |
|---|---|---|
| `common.slate_cholesky_trsm_test -n 256` | potrf 3.08e-16 (vs numpy 1.68e-16), trsm N 1.59e-16, trsm C 5.38e-16 | PASS |
| `common.slate_batched_test --nbatch 8 -n 128 --mesh 2x2` | potrf 1.54e-16, trsm N 2.28e-16, trsm C 5.50e-16 | PASS |

### FFI CPU story — verdict: GPU-only today; port is bounded, deferred

Three independent blockers, all verified:

1. All five op handlers (`potrf/trsm/eigh/batched_{potrf,trsm}_ffi.cc`)
   bind `Ctx<PlatformStream<cudaStream_t>>`, stage with `cudaMemcpyAsync`
   (D2D), build matrices via `fromDevices()` (device pointers), and
   hardcode `{Option::Target, Target::Devices}`.  (`context.cc` is pure
   MPI — reusable as-is.)
2. `ffi_loader.py` registers every target with `platform="CUDA"` only.
3. `liblorrax_ffi.so` hard-links the CUDA stack — empirically, on the CPU
   node `ctypes.CDLL` fails at `libnccl.so.2` before any registration
   question arises (tried with `JAX_PLATFORMS=cpu` + sandbox venv python).

What a port needs (est. 1–2 days, honest):
- Host handler variants: no stream ctx, `memcpy`, `fromScaLAPACK()`
  (host pointers, same 2-D block-cyclic layout), `Target::HostTask`;
  registered `platform="Host"`.
- A CUDA-free .so target: current `common/cpp/CMakeLists.txt`
  hard-requires cuSOLVERMp + compiles `.cu` TUs unconditionally, so the
  slate-host handlers need a separable `liblorrax_ffi_host.so` (slate +
  MPICH + phdf5 only), built host-side (no container) against the `cpu`
  install.
- Loader: per-platform symbol table + .so selection by
  `jax.default_backend()`.
- The Python-side shard_map local-transpose plumbing is backend-agnostic
  and should carry unchanged.

### Reproducing / adapting per-machine

```
bash src/ffi/slate/scripts/build_perlmutter.sh gpu   # or cpu; --fresh to wipe
```

Idempotent.  Env overrides: `LORRAX_SLATE_BUILDS_DIR`, `LORRAX_SLATE_REPO`,
`LORRAX_SLATE_COMMIT`, `LORRAX_SLATE_CUDATOOLKIT`, `LORRAX_SLATE_MAKE_J`.
Frontier sketch: swap the CUDA module pair for
`rocm + craype-accel-amd-gfx90a`, `gpu_backend=hip`; the wrapper/libsci/
SCALAPACK-gotcha structure carries over.

### P2 loose ends

- Login-node shifter builds hit a transient bind-mount failure tonight
  (logged in KNOWN_SANDBOX_ERRORS.md); the compute-node path works.
- SLATE tester + FFI smoke both leave the eigh eigvec layout artifact
  question open (SLATE's own heev *passes* — supports the P1/PLAN view
  that the artifact is in our wrapper's layout handling, not SLATE).
