# Overnight: block-cyclic linalg FFI stability + SLATE portability — EXECUTIVE SUMMARY

_Branch `agent/slate-linalg-ffi` (7 commits atop `9605a0f`), **UNPUSHED for morning
review**.  Final suite: **205 passed / 0 failed (4:02, plain 1-GPU invocation)**._

1. **Every block-cyclic FFI swept** (cusolvermp potrf/potrs/getrf/getrs/syevd, cusolvermg,
   cublasmp, slate potrf/trsm/heev/batched) across 1×1/2×2/4×1/1×4 meshes, divisible +
   padded + edge sizes, with residual and bit-determinism checks — permanent contract
   suite: `tests/test_ffi_linalg_contract.py` (skips cleanly without the FFI stack).
2. **Production bug: cuBLASMp had been dead since 2026-05-10** (stage drift → wrong
   CAL-ABI dispatch; `screening_solver=cublasmp_ffi` unusable).  Fixed + reproducible
   restage script.
3. **SLATE eigh eigenvector defect root-caused and FIXED** — stale MOSI device tiles +
   a use-after-free, not layout.  `distributed_eigh` now returns true eigenvectors.
4. **SLATE builds hardened**: scripted, pinned GPU (cuda/sm80) + CPU (gpu_backend=none)
   builds under `$HOME/software/slate_builds/`; SLATE's own testers pass on BOTH GPU and
   CPU nodes (1e-16..1e-19).  The `-DSCALAPACK_LIBRARIES=""` mystery documented.
5. **SLATE selectable from the input file**: portable axes
   `distributed_cholesky = auto|off|cusolvermp|slate`, `distributed_lu = auto|off|cusolvermp`
   (legacy cusolvermp_* keys = deprecated aliases).  Optional-dependency semantics:
   never auto-picked, loud failure with build pointers when absent.  Validated e2e:
   slate ≡ default to 2e-6 (COHSEX) / 4e-6 eV (GN-PPM) — print-precision floor.
6. **Honest scope line**: the SLATE *library* is CPU-ready and validated on CPU nodes;
   the LORRAX *FFI layer* is CUDA-only today (CUDA-typed handlers + NCCL dlopen) — a
   ~1-2 day host-handler port, spec'd in the P2 section.  SLATE getrf (for the LU axis)
   and slate::trsm back-solve wiring are the other follow-ups.

Sections below: P1 sweep matrix + failure catalog · P2 builds · P3 config integration
· P4 e2e validation.

---

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

---

## P1 — block-cyclic linalg FFI stability sweep + contract tests (2026-07-10)

Systematic sweep of every distributed-linalg FFI surface on process
meshes 1×1 / 2×2 / 4×1 / 1×4 (c128 + f64; divisible, non-divisible and
tiny sizes; every cell = residual vs numpy AND bit-exact rerun
determinism).  Sweep driver + raw logs: `p1_sweep/` (`sweep.py`,
`log_*.txt`).  Permanent artifact: `tests/test_ffi_linalg_contract.py`
— 21 pytest cases on a 1×1 mesh (~19 s, skipif-clean without the FFI
stack) + a CLI mode that reruns the same check bodies on multi-rank
meshes (`lxrun python3 -m tests.test_ffi_linalg_contract --mesh 2x2`).

### Final matrix (after fixes; residual = max over cells)

| op | 1×1 | 2×2 | 4×1 | 1×4 |
|---|---|---|---|---|
| cusolvermp potrf+potrs (c128/f64, NRHS<N, n=4..64) | PASS ≤6e-16 | PASS ≤6e-16 | GUARD¹ | GUARD¹ |
| cusolvermp getrf+getrs (herm-indef + general) | PASS ≤9e-14 | PASS ≤9e-14 | PASS ≤9e-14 | PASS ≤9e-14 |
| cusolvermp syevd (eigvals + Q^H contract) | PASS ≤2.5e-14 | PASS ≤2.6e-14 | GUARD² | GUARD² |
| cusolvermg syevd (f64, n=64/100) | PASS ≤3.4e-14 | — | — | — |
| cublasmp gemm (op(A)∈{N,T,C}, op(B)=N) | PASS ≤4e-16 | PASS ≤4e-16 | GUARD³ | GUARD³ |
| cublasmp gemm op(B)∈{T,C} | PASS (1×1 only) | GUARD⁴ | GUARD | GUARD |
| cublasmp fused W-solve | PASS ≤1.3e-14 | PASS ≤1.2e-14 | GUARD³ | GUARD³ |
| slate potrf+trsm (incl. RECT m=32/128≠n=64) | PASS ≤4e-16 | PASS ≤4e-16 | PASS ≤3.5e-16 | GUARD⁵ |
| slate batched potrf+trsm | PASS ≤2.4e-16 | PASS ≤1.9e-16 | PASS ≤2.4e-16 | PASS ≤1.9e-16 |
| slate heev (eigvals + TRUE eigvecs) | PASS ≤3.8e-14 | PASS ≤3.9e-14 | square-only | square-only |
| non-divisible N / NRHS | — | RAISE ✓ | RAISE ✓ | RAISE ✓ |

Every PASS cell is also bit-deterministic across a same-process rerun
(fresh device buffers, donated inputs re-materialized).

GUARD = wrapper now raises a descriptive ValueError instead of the
library failure mode listed below.

### Failure catalog (all root-caused; minimal repros in `p1_sweep/sweep.py` cells)

1. **cusolverMpPotrf requires square ScaLAPACK blocks (mb == nb).**
   One-tile-per-rank layout ⇒ square meshes only; on 4×1/1×4
   `cusolverMpPotrf_bufferSize` fails status=3 at EVERY size.  The isdf
   `auto` resolver already routed 1-D meshes to `sharded_cholesky`; the
   explicit-override path now raises with the same routing hint.
2. **cusolverMpSyevd on a non-square mesh DEADLOCKS** — no error
   status; some ranks reject mb≠nb, the rest park in a collective.
   Wrapper now rejects p≠q up front.  Also: the returned Q buffer is
   the conj-transpose of the eigenvector matrix (docstring previously
   claimed plain eigenvectors); contract pinned in the tests, docstring
   fixed.  (`ffi.slate.distributed_eigh` returns true column eigvecs.)
3. **cuBLASMp was DEAD on every mesh — mixed-generation stage drift**
   (the production `screening_solver = cublasmp_ffi` path).  The 0.7.2
   cuSOLVERMp stage (2026-05-10) contains no libcublasmp, so the loader
   fell back through the .so RUNPATH to the 25.5 HPC-SDK's cuBLASMp
   0.4.0 (CAL comm ABI) while cuSOLVERMp 0.7.2 (NCCL ABI) resolved via
   LD_LIBRARY_PATH.  `ensure_cublasmp` keyed the comm type off the
   cuSOLVERMp version → passed an ncclComm_t to a CAL-ABI grid create →
   status=6 everywhere.  Fixes: staged cuBLASMp 0.5.1 (NCCL ABI) + its
   nvshmem dependency into the 0.7.2 stage
   (`scripts/stage_cublasmp_redist.sh`, reproducible); `ensure_cublasmp`
   now asks the LOADED library (`cublasMpGetVersion`) and raises a
   stack-mismatch error naming the pairing rule; libcal linked
   explicitly in CMake (it used to resolve transitively through
   cuBLASMp 0.4.0 — dlopen broke once 0.5.1 dropped that NEED).
   Post-fix: gemm + fused W-solve at 1e-16..1e-14 on 1×1/2×2.
3a. **cuBLASMp rejects 1-D process grids** (`Matmul_bufferSize`
   status=3 on 4×1/1×4, every size/combo; 1×1 and 2×2 fine) — guarded.
4. **cuBLASMp Matmul with op(B)≠N on a multi-rank grid is
   rank-DIVERGENT**: two ranks return status=6, the others block in the
   collective ⇒ job deadlock (observed 2×2 NC).  Wrapper rejects
   transb≠'N' on multi-rank meshes (pre-transpose B instead).  The
   fused W-solve only uses (C,N)/(N,N) internally — unaffected.
5. **slate trsm with rectangular RHS (m ≠ n) aborted every rank**
   (uncatchable `blas::Error` from a SLATE OpenMP task on 2×2; silent
   mis-assembly risk on 1-D meshes).  Root cause: X was built with
   A's square tile size — tile (i,j) of a 64×32 X with nb=32 on a 2×2
   grid lands on ranks that hold no X data.  FIX: per-dimension X tiles
   in `cpp/{trsm,batched_trsm}_ffi.cc` (solve dim conforms with A's nb;
   free dim = one tile per rank).  Also fixed: side='R' validated the
   wrong axes in trsm.py, and side='R' lld was wrong in the handler.
6. **slate eigh "layout artifact" ROOT-CAUSED and FIXED** — it was
   never a layout transform: the writeback loop read `Z(ti,tj,dev)`
   directly, but heev's back-transformation (redistribute →
   unmtr_hb2st → unmtr_he2hb) leaves the valid copy of Z's tiles on the
   HOST — the device instances still held the *pre-back-transform*
   bytes (why the diag test returned exact I while random-A eigvecs
   failed every transpose/conj hypothesis).  FIX: `tileGetForReading`
   (MOSI-correct) before each tile copy + stream sync before Z's
   destructor frees the tiles (latent use-after-free race) + the
   missing local-transpose pair on A/Q (cholesky convention) + A-shard
   copy into the Q buffer so heev no longer scribbles on an XLA INPUT
   buffer.  Contract upgraded: `distributed_eigh` now returns TRUE
   column eigenvectors, `A@Q == Q@diag(W)` at ≤3.9e-14 on 1×1 AND 2×2,
   orthonormality ≤1e-11, bit-deterministic.
7. **slate single-matrix ops on 1×q meshes SIGABRT** (size-dependent
   `internal_batch.hh:290` `group.ld[m] == Mij.stride()` assert): local
   stride lld=n ≠ tile nb=n/q; p ≥ q meshes have lld == nb and are
   safe.  This is also the root of the README's historical batched-1×4
   assert.  `validate_tile_layout` rejects 1×q for single-matrix ops
   (use q×1); the batched (1,Py) sub-grid keeps an explicit
   `allow_row_grid` opt-out (production-validated on 2×2; 1×4 passed
   at nbatch=4/n=32 in this sweep, the old nbatch=8/n=128 repro remains
   an accepted documented risk).
8. **`block_size` overrides on multi-rank meshes were silently wrong**
   for every slate op (JAX block shards == SLATE block-cyclic tiles
   ONLY at one tile per rank) — `validate_tile_layout` now rejects any
   non-default nb on a multi-rank mesh; 1×1 stays free.
9. **Non-PD input surfaces as opaque status=7, not a clean info** —
   resolved as NOT a tiny-size bug: the one deterministic
   `cusolverMpPotrf (q=1) status=7` cell (2×2, n=4, sweep seed 7)
   turned out to have eigmin = −0.032 for batch element q=1 (the n·I
   shift wasn't enough for that draw) — cuSOLVERMp correctly refuses;
   it just reports CUSOLVER_STATUS_INTERNAL_ERROR instead of the
   info-based "not positive definite at leading minor" path.  Genuinely
   PD tiny sizes (n=4..32, mb=nb down to 2) all pass at ≤5e-16
   (`mp_chol_edge` cells).  Diagnosability note only.

### Not fixed / reported only

- **cusolvermg**: no live consumer (only `common/eigh_benchmark.py` and
  its own smoke test import it) — bench-only target, kept as-is.
- Config-level fallback for a missing/broken cuBLASMp on the GPU
  backend does not exist (`screening_solver = cublasmp_ffi` fails at
  first solve) — acceptable-loud, and P3's slate integration is adding
  the probe-and-fallback pattern on these axes anyway.

### Test-suite state

`tests/test_ffi_linalg_contract.py`: 21 passed in ~18 s on 1 GPU
(cusolvermp potrf/potrs ×3, solve_lu ×3, eigh ×2, cusolvermg ×1,
cublasmp gemm ×2 + wsolve ×1, slate chol+trsm ×4 (incl. rect), batched
×1, eigh ×2, padding-through-FFI ×1, layout-guard logic ×1).  Full
suite: **197 passed / 0 failed in 4:03** (plain `LORRAX_NGPU=1 lxrun
python3 -m pytest -q tests`; 176 pre-existing + 21 new).

CLI matrix (same check bodies, multi-rank; `p1_sweep/cli_*.txt`):
2×2 = 22/22 PASS; 4×1 = lu + all slate ops PASS, square-only SKIP;
1×4 = lu + batched PASS, slate single-matrix GUARD, square-only SKIP —
`done: 0 failures` on all three.

Commits (branch `agent/slate-linalg-ffi`): `4e252f0` slate trsm
rect-tiles + layout validation, `1e5c7de` slate eigh fix, `842767e`
cuBLASMp restore + mesh-limit guards, `fa35636` + `22c6ef2` contract
suite.  `liblorrax_ffi.so` rebuilt (compute node) against the updated
sources; cuBLASMp 0.5.1 staged at
`~/software/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64/`.

## P3 — SLATE as an input-file-selectable optional backend (executed by orchestrator)

Portable distributed-linalg config axes (the values name libraries, not vendor key names):

```
distributed_cholesky = auto | off | cusolvermp | slate    # charge-channel ζ-fit Cholesky
distributed_lu       = auto | off | cusolvermp            # transverse LU (no SLATE getrf yet)
```

- Legacy `cusolvermp_charge` / `cusolvermp_lu` (auto|on|off) are parsed-but-deprecated
  aliases: warned at load, honored only when the portable key is left at `auto`
  (`on` → `cusolvermp`).  Explicit portable key beats the alias.
- **Optional-dependency semantics**: `slate` is never auto-picked; an explicit request
  fails LOUDLY (actionable message pointing at the build scripts) if the FFI/library is
  absent — mirroring the explicit `slab_io` semantics — instead of silently running a
  different backend.  The guarded 1×q mesh geometry (SLATE stride assert, P1) is
  rejected at resolve with a workaround message.
- Wiring: `isdf.core.factor_c_q` gained a `slate_cholesky` branch — one whole-mesh
  block-cyclic `slate::potrf` per q (SLATE's batched API distributes the batch over the
  mesh 'x' axis, which mismatches this call site's replicated-q layout; per-q at nq ≲
  tens is the right shape).  `to_jax_lower()` returns a conventional L, so
  `solve_zeta` consumes it through the SAME triangular-solve branch as
  `sharded_cholesky` — zero solve-side changes.  slate::trsm back-solve wiring = perf
  follow-up.
- CPU JAX backend: both cuSolverMp and the SLATE FFI are CUDA-only today (P2 verdict),
  so the CPU-config path forces `off` with a printed note.
- Tests: +6 config-resolution cases (aliases, precedence, invalid values, slate-for-LU
  rejected) in test_qp_solver_config.py; +2 functional contract tests
  (factor_c_q slate ≡ numpy Cholesky at 1e-12 + bit-determinism; auto never picks
  slate) in test_ffi_linalg_contract.py.
- One orchestrator bug caught by the first e2e attempt: the alias lookup used a
  variable from the wrong parser scope (`section` vs `params`) — restructured to the
  established deprecated-key pattern (warn at parse scope, honor at resolve scope).

## P4 — e2e validation through the input file (GPU, 1×1 mesh)

Four fresh gnppm-fixture runs (`p4_e2e/`): {static COHSEX (cohsex_ibz_test.in),
GN-PPM} × {default, `distributed_cholesky = slate`}.  Log banners confirm the path:
`path=slate_cholesky` vs `path=sharded_cholesky`.

| case | slate vs default max\|ΔΣ\| |
|---|---|
| static COHSEX (sigSX/sigCOH/sigTOT/VH) | **2e-6 eV** |
| GN-PPM (sigX/sigC/sigXC/VH) | **4e-6 eV** |

Both at the sigma_diag print-precision floor (last printed digit = 1e-6 eV) —
backend-equivalent physics; the residual difference is the legitimate last-bit change
from a different (but correct) factorization order, which GN-PPM's known sensitivity
does not amplify on the redesigned fixture.

Absence/misuse behavior covered by construction: config validation rejects invalid
values and slate-for-LU; explicit-slate-without-library raises with build instructions
(exercised implicitly — the resolver probe runs the same loader the contract tests
skip on).
