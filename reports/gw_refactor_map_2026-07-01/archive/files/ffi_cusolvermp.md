# src/ffi/cusolvermp/ — deep-read notes (gw_refactor_map 2026-07-01)

Package: JAX FFI wrappers around NVIDIA cuSOLVERMp distributed dense linear
algebra (potrf/potrs/getrf+getrs/syevd) on a 2-D JAX process mesh ('x','y'),
one JAX process per GPU. C++ handlers live in `src/ffi/cusolvermp/cpp/`
(batched_potrf_ffi.cc, batched_potrs_ffi.cc, batched_solve_lu_ffi.cc,
context.cc, eigh_ffi.cc); FFI target registration + ctypes bindings are in
`src/ffi/common/ffi_loader.py` (targets `lorrax_cusolvermp_eigh`,
`lorrax_cusolvermp_batched_potrf`, `lorrax_cusolvermp_batched_potrs`,
`lorrax_cusolvermp_batched_solve_lu`; C entry points
`lrx_create_cusolvermp_context` / `lrx_destroy_cusolvermp_context`).
`scripts/` holds `stage_nvhpc.sh` / `stage_pypi.sh` (library staging).

Production consumer: the ζ-fit stage of the ISDF pipeline
(`src/common/isdf_fitting.py`). Solver selection is driven by cohsex.in keys
`cusolvermp_charge` / `cusolvermp_lu` (defaults "auto" in
`src/gw/gw_config.py:247-248`, threaded through `gw_init.py:721/843` →
`isdf_fitting._resolve_solver_kind_{charge,transverse}`; "auto" → cuSOLVERMp
iff mesh is truly 2-D, i.e. Px≥2 and Py≥2). Note `src/ffi/cublasmp/batched.py`
REUSES this package's context (`from ..cusolvermp.context import
get_or_init_context`), so context.py is shared infrastructure for two FFI
backends.

---

## batched.py (398 loc)

**Purpose.** Per-q batched distributed Cholesky (potrf), triangular solve
(potrs), and general LU solve (getrf+getrs) for stacks of Nq matrices each
2-D-sharded `P(None,'x','y')` across the full mesh. The FFI handler loops
over q, one cuSOLVERMp call per matrix on the world-wide (Px,Py) grid. This
is the linear-solve backend for the ζ-fit: charge channel (Hermitian PD CCT)
uses potrf+potrs; transverse bispinor channels (indefinite CCT^μ) use LU.

**Category.** distributed linalg FFI.

**Module-level machinery.**
- `_JIT_CACHE` (line 67): dict of `jax.jit(shard_map(...))` closures keyed by
  (op, mesh key, dtype, shapes, ctx_handle). Rationale comment cites
  `src/ffi/phdf5/ARCHITECTURE.md §2.4` (eager shard_map re-traces per call).
- `_mesh_key(mesh)` (69-70): (axis names, axis sizes).
- Layout convention: JAX row-major local shard `(N/Px, N/Py)` is handed to
  cuSOLVERMp (col-major ScaLAPACK tiles) after an inner-dim `jnp.transpose`
  per q-slice; the context is created with `col_major=False` so tile (i,j) →
  rank i*Py+j matches JAX's row-major mesh reshape.

### Function table

| function | lines | role |
|---|---|---|
| `_mesh_key` | 69-70 | jit-cache key helper |
| `CusolverMpBatchedLowerL` (dataclass) | 73-97 | opaque handle for the batched Cholesky factor: `raw` (Nq,N,N) sharded `P(None,'y','x')`, bytes = cuSOLVERMp col-major L tiles; plus mesh, n, mb, nb, nbatch |
| `_validate_mesh` | 100-109 | requires axes ('x','y'), Px·Py == jax.process_count() |
| `batched_distributed_cholesky` | 112-167 | A[q] = L[q] L[q]^H via cusolverMpPotrf |
| `batched_distributed_potrs` | 170-255 | solve A[q] X[q] = B[q] from the potrf factor |
| `cholesky_handle_to_natural_L` | 258-307 | materialize handle → conventional row-major lower-tri L, `P(None,'x','y')` |
| `batched_distributed_solve_lu` | 310-398 | solve A[q] X[q] = B[q] via getrf+getrs (general/indefinite A) |

### batched_distributed_cholesky (112-167)
- Math: `A[q] = L[q] L[q]^H` (Hermitian PD, per q).
- In: `A (Nq,N,N)` device, `P(None,'x','y')` (local `(Nq, N/Px, N/Py)`);
  requires N % Px == 0, N % Py == 0; dtype F64/C128.
- Inside shard_map (in_specs `P(None,'x','y')`, out_specs `P(None,'y','x')`,
  check_rep=False): `local_A_T = jnp.transpose(local_A, (0,2,1))` then
  `jax.ffi.ffi_call(_POTRF_TARGET, L_local_T, input_output_aliases={0:0})`
  with attrs `nq, n, mb=N/Px, nb=N/Py, ctx_handle`. Aliasing A→L factors in
  place ("saves ~1 GB transient VRAM per rank at Si scale"); the jit wrapper
  uses `donate_argnums=(0,)`.
- Out: `CusolverMpBatchedLowerL` handle wrapping raw
  `(Nq, N/Py, N/Px)`-local array (global `P(None,'y','x')`).
- Callers (grep over src/, tests/, tools/, scripts/ for
  `batched_distributed_cholesky` excluding ffi/slate's namesake):
  - `src/common/isdf_fitting.py:1104` — charge-channel `factor_c_q` when
    `solver_kind == 'cusolvermp_cholesky'`; note it then returns
    `L_handle.raw` (a bare array) and the downstream solve
    (`isdf_fitting.py:1286-1301`) **re-manufactures** the
    `CusolverMpBatchedLowerL` handle from the raw array + shape metadata.
  - bench/tests: `src/common/chol_natural_test.py`,
    `src/common/potrs_rhs_test.py`, `src/common/cusolvermp_batched_test.py`,
    `src/ffi/cusolvermp/profile_batched.py`.

### batched_distributed_potrs (170-255)
- Math: `X[q] = A[q]^{-1} B[q]` given the potrf factor.
- In: handle L + `B (Nq,N,Mrhs)` `P(None,'x','y')`; Mrhs % Py == 0;
  dtype match enforced. descB blocks: `mb_b = L.mb = N/Px`,
  `nb_b = Mrhs/Py` (one col-tile per rank, contiguous slab layout).
- shard_map in_specs `(P(None,'y','x'), P(None,'x','y'))`, out
  `P(None,'x','y')`; B is inner-transposed, FFI aliases B→X
  (`input_output_aliases={1:0}`, "saves ~1.7 GB transient VRAM per rank at
  Si scale"), result un-transposed inside the shard_map;
  `donate_argnums=(1,)` (B only; L reused across r-chunks).
- **Documented library bug** (docstring, 182-189): cuSOLVERMp **0.6.0**
  returns quietly wrong answers when `NRHS ≤ N` on a 2-D grid (Px>1 AND
  Py>1); `NRHS ≥ N + Py` is correct to machine precision. Callers with
  small natural NRHS must zero-pad columns and slice. Docstring says "See
  ``w_isdf.solve_w_low_mem`` for a concrete workaround" — **stale**, see
  weird-code below. The ζ-fit chunked path escapes the bug because
  NRHS = n_rchunk ≫ N.
- Callers: `src/common/isdf_fitting.py:1301` (`solve_zeta` charge channel;
  pads Z_q columns to a multiple of Py, rebuilds the handle, then reshards
  output `P(None,'x','y')` → `P(None,('x','y'),None)` via
  `_reshard_zeta_mu_X_r_Y_to_mu_XY`); tests `potrs_rhs_test.py`,
  `cusolvermp_batched_test.py`; bench `profile_batched.py`.

### cholesky_handle_to_natural_L (258-307)
- Role: raw handle → conventional jax.Array L: inner transpose
  (`P(None,'y','x')` col-major bytes reinterpreted, i.e. bytes are L^T
  row-major), then per-rank tril mask built from
  `jax.lax.axis_index('x'/'y')`: keep entry iff
  `global_row = x_idx*mb + i >= global_col = y_idx*nb + j` (zeroes the
  garbage upper triangle cuSOLVERMp leaves = whatever was in donated A's
  upper).
- Out: `(Nq,N,N)` `P(None,'x','y')` row-major lower-triangular L.
- NOT jit-cached (plain eager shard_map, unlike the other three — re-traces
  per call; harmless for a test-only routine, inconsistent with the file's
  own _JIT_CACHE discipline).
- Callers: **only** `src/common/chol_natural_test.py:55` (grepped
  `cholesky_handle_to_natural_L` across src/, tests/, tools/, scripts/).
  Docstring claims it is "needed by consumers ... e.g. the symmetric
  W-solve path W = X H⁻¹ X†", but the cuBLASMp W-solve
  (`src/ffi/cublasmp/batched.py`, `w_isdf`) has no reference to it →
  dead-in-production suspect.

### batched_distributed_solve_lu (310-398)
- Math: `X[q] = A[q]^{-1} B[q]` via per-q getrf (LU w/ pivoting) + getrs;
  A general/indefinite Hermitian.
- Validation notes in docstring: correct on cuSOLVERMp **0.7.2** across
  2×2/1×4/4×1 meshes, residuals 1e-13–1e-15 C128 up to N=512; **0.6.0
  returned garbage (info=0 but wrong X) on Px>1 AND Py>1** — 0.7+ required.
- In: A `(Nq,N,N)`, B `(Nq,N,NRHS)`, both `P(None,'x','y')`; N % Px, N % Py,
  NRHS % Py all zero. Both inputs inner-transposed; A donated (getrf
  scribbles LU factors into A's buffer, factors discarded), B→X aliased
  (`input_output_aliases={1:0}`); `donate_argnums=(0,1)`. Pivot vectors
  allocated inside the FFI, never surfaced to Python.
- Out: X, same shape/sharding as B.
- Callers: `src/common/isdf_fitting.py:1341` (transverse-channel ζ solve,
  `solver_kind == 'cusolvermp_lu'`; adds per-q ridge
  `1e-12·|tr(CCT)|/n_rmu · I` before the call, pads NRHS to multiple of Py);
  test `src/common/cusolvermp_solve_lu_test.py:137`.

### Flags consumed
None read directly in this file. Upstream: cohsex.in `cusolvermp_charge`,
`cusolvermp_lu` (gw_config.py:247-248 defaults "auto"; forced "off" when the
FFI .so is unavailable, gw_config.py:1019-1032) select whether
isdf_fitting dispatches here.

### I/O
None. Pure in-memory device arrays across the FFI boundary.

### Suspects
- **dead**: `cholesky_handle_to_natural_L` — sole caller is the standalone
  test `src/common/chol_natural_test.py` (grep evidence above). The
  docstring's claimed production consumer (symmetric W-solve) doesn't call
  it.
- **redundancy**: `src/ffi/slate/batched.py` defines a parallel
  `batched_distributed_cholesky` (SLATE backend, explicitly "shape contract
  mirrors ffi.cusolvermp.batched_* (forthcoming)") — a second backend path
  for the same op; SLATE is evaluation-grade per memory notes. Also the
  handle round-trip in isdf_fitting (return `.raw`, rebuild
  `CusolverMpBatchedLowerL` later from bare metadata) duplicates the
  handle's metadata bookkeeping in two places.
- **weird**:
  - Lines 187, 324: docstrings reference `w_isdf.solve_w_low_mem`; grep for
    `solve_w_low_mem` across src/ hits ONLY these two docstrings. The
    function is gone — `w_isdf.solve_w` (w_isdf.py:354) now dispatches to
    JAX_NATIVE / CUBLASMP_FFI solvers and never calls this package. Stale
    cross-reference; the NRHS≤N-bug workaround knowledge now lives only
    here and in isdf_fitting's padding code.
  - The cuSOLVERMp 0.6.0 silent-wrong-answer bugs (potrs NRHS≤N on 2-D
    grids; solve_lu garbage on 2-D grids) are version-dependent
    correctness landmines encoded only in docstrings — no runtime version
    check anywhere in the package.
  - `_JIT_CACHE` keyed partly by `int(ctx_handle)` — correct but means a
    ctx re-init (new handle int) grows the cache; never evicted.
  - Opaque-bytes contract: `raw` sharded `P(None,'y','x')` holding
    col-major tiles; any consumer touching `.raw` directly (isdf_fitting
    does, to store it) must preserve this invariant with no type-level
    protection.

---

## context.py (130 loc)

**Purpose.** Per-process singleton cache of cuSOLVERMp contexts — a context
bundles (NCCL comm, CUDA stream, cusolverMp handle, process grid,
workspace) — keyed by `MeshKey = (Px, Py, col_major)`. Bootstraps the
dedicated NCCL communicator by broadcasting rank 0's `ncclUniqueId` (raw
bytes) through the JAX distributed KV store. atexit teardown.

**Category.** distributed linalg FFI — resource/context management.

### Function table

| function | lines | role |
|---|---|---|
| `_mesh_key` | 47-54 | Mesh → (Px, Py, col_major); requires axes ('x','y') |
| `_broadcast_unique_id` | 57-66 | rank 0 `ffi_loader.fill_nccl_unique_id` into a uint8 buffer, then `broadcast_bytes(buf, key=...)` (jax multihost KV) |
| `_make_ctx` | 69-94 | validates Px·Py == process_count; unique KV key `lorrax_ffi/cusolvermp/nccl_unique_id/v0/{p}x{q}/{col|row}`; calls `ffi_loader.create_cusolvermp_context(rank, world, uid ptr, p, q, grid_layout_col_major)` → opaque int64 |
| `get_or_init_context` | 97-111 | thread-safe (module `_LOCK`) get-or-create from `_CACHE` |
| `_atexit_teardown` | 117-130 | best-effort `destroy_cusolvermp_context` per cached handle |

### Entry points / callers (grep `get_or_init_context` under src/, tests/, tools/, scripts/, filtering the slate package's own namesake)
- `ffi/cusolvermp/batched.py` ×3 (all `col_major=False`)
- `ffi/cusolvermp/eigh.py:98` (**default `col_major=True`**)
- `src/ffi/cublasmp/batched.py:33,129,251` — cuBLASMp **reuses this
  package's context** (`from ..cusolvermp.context import
  get_or_init_context`, `col_major=False`). Cross-backend coupling: a
  refactor moving/renaming this module breaks cublasmp.
- `MeshKey` is in `__all__` but grep finds no importer outside this file →
  export-only, mild dead suspect.

### Suspects
- **weird**: default `col_major=True` on `get_or_init_context` while every
  batched/cublasmp caller passes `col_major=False`; only eigh.py uses the
  default. So a run using both eigh and batched holds **two** NCCL comms /
  grids for the same mesh (cache keys differ in the layout bit). Legal per
  the design ("multiple contexts can coexist") but an easy trap and doubles
  NCCL resources.
- **weird**: teardown is registered at import time and swallows all
  exceptions; on signal/segfault exits contexts leak (documented as
  acceptable).
- No I/O, no config flags.

---

## eigh.py (124 loc)

**Purpose.** `distributed_eigh(A, mesh, compute_evecs, block_size)` — JAX FFI
wrapper around `cusolverMpSyevd` for a single (n,n) Hermitian/symmetric
matrix sharded `P('x','y')`. Explicitly written flat as "the template to
copy when adding the next distributed routine", with the three
per-routine decisions ((1) FFI target name, (2) output shapes/specs,
(3) attrs) called out in comments.

**Category.** distributed linalg FFI (currently bench/validation-only).

### Function table

| function | lines | role |
|---|---|---|
| `distributed_eigh` | 43-124 | validate (square, mesh axes, Px·Py == process_count, n divisible by both), `get_lib()`, `get_or_init_context(mesh)` (col-major default), build attrs `n, mb, nb, ctx_handle, compute_evecs`, shard_map in `P('x','y')` → out `(P(), P('x','y'))`, single `jax.ffi.ffi_call("lorrax_cusolvermp_eigh", (W_local, Q_local))` |

- Math: `A Q = Q diag(W)`, W ascending float64 replicated, Q same dtype as A
  sharded `P('x','y')`.
- Layout trick (docstring 16-23): cuSOLVERMp sees the row-major shard as
  col-major, i.e. gets A^T; for Hermitian A eigenvalues of A^T equal those
  of A, so W is correct. (The argument covers eigenvalues only; Q's basis
  convention is not argued in-file.)
- `block_size` param: default n/p = one tile per rank (matches JAX block
  sharding, eigenvectors come out in-place); smaller blocks are faster
  (block=256 gives 2.4× at n=16k) **but return eigenvectors in a
  block-cyclic permutation of the input basis** — silent output-layout
  change controlled by a perf knob.
- `compute_evecs=False`: Q is still allocated/returned but contents garbage
  ("should be ignored").
- Not jit-cached (eager shard_map each call, unlike batched.py; fine for
  its one-shot bench usage).

### Callers (grep `distributed_eigh` across src/, tests/, tools/, scripts/, excluding slate/ and cusolvermg/ namesakes)
- `src/common/cusolvermp_eigh_test.py:117,166` (correctness test)
- `src/common/eigh_benchmark.py:125`
- `src/common/eigh_block_sweep.py:48`
- `src/common/slate_vs_cusolvermp_bench.py:114`
- **No caller in src/gw, src/common pipeline code, src/solvers, src/bse** →
  dead-in-production suspect (validated infrastructure awaiting a
  consumer, e.g. a future dense-χ eigendecomposition path).

### Suspects
- **dead (production)**: `distributed_eigh` — only test/bench callers (grep
  evidence above).
- **redundancy**: three parallel `distributed_eigh` implementations —
  `ffi/cusolvermp/eigh.py`, `ffi/slate/eigh.py` ("shape contract mirrors
  ffi.cusolvermp.distributed_eigh"; slate `__init__` notes "eigenvectors
  buggy; W ok"), `ffi/cusolvermg/eigh.py` ("simpler counterpart"). Classic
  parallel-backend cruft per the no-redundancy policy.
- **weird**: uses context `col_major=True` (default) vs batched's `False` —
  see context.py note; also the returned-garbage-Q and block-cyclic-Q
  behaviors above.

---

## __init__.py (25 loc)

**Purpose.** Package re-exports: `CusolverMpBatchedLowerL`,
`batched_distributed_cholesky`, `batched_distributed_potrs`,
`batched_distributed_solve_lu`, `cholesky_handle_to_natural_L`,
`distributed_eigh`.

**Category.** distributed linalg FFI (package surface).

- Docstring only advertises `distributed_eigh` although the batched solvers
  are the actual production API — stale/misleading doc.
- All external imports go through this surface (`from ffi.cusolvermp import
  ...`) except cublasmp, which imports `..cusolvermp.context` directly.

---

## profile_batched.py (336 loc)

**Purpose.** Benchmark/Nsight-profiling driver for the batched potrf/potrs
wrappers at GWJAX-like call shapes (defaults nq=9, n=640, mrhs=640, c128,
2×2 mesh — "MoS2 3×3-ish"). Explicitly "not a correctness unit test";
exists to produce Nsight traces that transfer to the in-situ GWJAX calls.

**Category.** diagnostic/bench script.

### Entry point
`main` via `LORRAX_NGPU=4 lxrun python3 -u -m ffi.cusolvermp.profile_batched
--mode potrf --nq 9 -n 640 --iters 8 --warmup 2` (usage documented only in
its own module docstring; grep for `profile_batched` across src/, tests/,
tools/, scripts/, docs/ finds no other reference — sandbox-level reports
`reports/cusolvermp_ffi_profile*_2026-05-12/` outside this repo are its
historical output).

### Function table

| function | lines | role |
|---|---|---|
| module top | 35-37 | `from runtime import set_default_env; set_default_env()` BEFORE importing jax — env side effect ordering matters |
| `_log` | 61-63 | rank-0 print |
| `_parse_mesh` | 66-72 | "2x2" → (2,2) |
| `_dtype` | 75-76 | "c128"/"f64" |
| `_load_cudart` | 79-81 | ctypes CDLL libcudart |
| `cuda_profiler_api` | 84-105 | ctx mgr: block_until_ready + `multihost_utils.sync_global_devices` fences, then `cudaProfilerStart/Stop` via ctypes for `nsys --capture-range=cudaProfilerApi` |
| `trace_annotation` | 108-116 | jax.profiler.TraceAnnotation if available, else no-op |
| `make_mesh` | 119-126 | devices.reshape(px,py) → Mesh('x','y') |
| `build_inputs` | 129-174 | jit-built random Hermitian A, symmetrized, `A += 2.0*n * I` ("strongly PD, like the CCT charge-channel solve"); B random; both `with_sharding_constraint(P(None,'x','y'))` |
| `fresh_for_donation` | 177-191 | `A + step*1e-12*I`, `B + 0` — forces fresh buffers because the wrappers donate inputs (reusing donated buffers would fail or measure donation side effects) |
| `prebuild_donated_inputs` | 194-217 | materialize per-iteration donated pairs BEFORE the profiler range (keeps JAX alloc/copy noise out of the trace); multihost sync after |
| `timed_call` | 220-229 | mode "potrf" → cholesky only; "potrf_potrs" → cholesky + potrs; block_until_ready |
| `summarize_ms` | 232-241 | mean/median/p90/min/max |
| `main` | 244-332 | argparse (--mode, --mesh, --nq, -n, --mrhs, --dtype, --warmup, --iters, --seed, --no-prebuild-inputs, --sync-each-iter, --cuda-profiler-api, --sync-label); `init_jax_distributed()`, `fallback_to_cpu_if_no_gpu_backend()`, `jax_enable_x64`; warmup then timed loop with perf_counter |

### Flags / config
CLI-only; no cohsex.in / LorraxConfig. Env: `LORRAX_NGPU`, `SLURM_JOBID`
(via lxrun), plus whatever `runtime.set_default_env` sets.

### I/O
stdout timing lines only (rank 0). Nsight output written by nsys itself,
not this script.

### Suspects
- **dead-ish**: no in-repo caller (grep evidence above); manual bench tool
  by design.
- **redundancy**: overlaps a family of standalone drivers in `src/common/`
  — `cusolvermp_batched_test.py` (correctness twin), `potrs_rhs_test.py`
  (NRHS-bug prober), `chol_natural_test.py`, `cusolvermp_solve_lu_test.py`,
  `slate_vs_cusolvermp_bench.py`, `eigh_benchmark.py`, `eigh_block_sweep.py`.
  A refactor could consolidate into one parametrized harness.
- **weird**:
  - `set_default_env()` executed at import, before the jax import block —
    intentional but fragile ordering (an import re-sort breaks it).
  - `--mode` covers only potrf / potrf_potrs; `solve_lu` (the transverse
    production path) is not profileable here.
  - Docstring hardcodes a path to the **lorrax_D** checkout
    (`.../sources/lorrax_D`) — will go stale for other letters.
  - Magic constants: `2.0*n` diagonal shift (PD margin), `1e-12` donation
    eps, `step0=1000` for timed vs warmup input streams.

---

## Cross-file refactor observations

1. **Backend triplication**: distributed eigh exists in cusolvermp, slate,
   and cusolvermg; batched Cholesky in cusolvermp and slate. Per the
   no-redundancy policy these are parallel paths; only cusolvermp's batched
   solvers have production callers (isdf_fitting).
2. **Context sharing**: `cublasmp` depends on `cusolvermp.context` — the
   context module is really `ffi/common`-level infrastructure misfiled
   under cusolvermp.
3. **Layout split-brain**: eigh uses a col-major grid context, batched (and
   cublasmp) a row-major one; two NCCL comms per mesh if both are used.
4. **Version landmines**: 0.6.0 silent-corruption bugs (potrs NRHS≤N,
   solve_lu on 2-D grids) documented in docstrings only; no runtime
   cuSOLVERMp version assertion.
5. **Stale docs**: `w_isdf.solve_w_low_mem` references (batched.py:187,324);
   `__init__` docstring advertising only distributed_eigh; profile_batched
   docstring hardcoding lorrax_D.
