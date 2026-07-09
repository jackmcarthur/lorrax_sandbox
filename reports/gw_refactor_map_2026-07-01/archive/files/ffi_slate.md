# ffi/slate — SLATE distributed dense linalg FFI (deep-read notes)

Group: `src/ffi/slate/{__init__,context,cholesky,trsm,eigh,batched}.py` (848 LOC total).
Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.

## Group-level summary

JAX-FFI wrappers around SLATE (ICL's tile-based MPI+GPU dense linear algebra
library): distributed Cholesky (`slate::potrf`), triangular solve
(`slate::trsm`), Hermitian eigendecomposition (`slate::heev`), plus 3-D batched
potrf/trsm variants for the GWJAX `(Nq, Nmu, Nmu)` workload. One process per
GPU on a 2-D `('x','y')` JAX mesh; C++ handlers live in `src/ffi/slate/cpp/`
(`potrf_ffi.cc`, `trsm_ffi.cc`, `eigh_ffi.cc`, `batched_potrf_ffi.cc`,
`batched_trsm_ffi.cc`, `context.cc`) loaded via `ffi.common.ffi_loader`
(FFI target names: `lorrax_slate_{potrf,trsm,eigh,batched_potrf,batched_trsm}`,
context symbols `lrx_slate_init_mpi`, `create_slate_context`,
`create_slate_subrow_context`, `destroy_slate_context`).

Design tripod (per README):
1. **Local transpose** — JAX row-major vs SLATE col-major; each rank
   `jnp.transpose`s its own shard inside `shard_map` (no inter-rank comm).
2. **MPI rank remap** (cpp/context.cc) — SLATE `fromDevices` hardcodes
   `GridOrder::Col` (tile (i,j) → rank `i + j*p`) while JAX's C-order mesh puts
   shard (mx,my) on rank `mx*q + my`; the comm is rebuilt via `MPI_Comm_split`
   with key `= (jax_rank / q) + (jax_rank % q) * p`.
3. **GPU-aware Cray MPICH** — `MPICH_GPU_SUPPORT_ENABLED=1` +
   `libmpi_gtl_cuda.so` LD_PRELOAD (handled by module/scripts, not Python).

**Key fact for the refactor map: NO production caller exists.** Every consumer
of `ffi.slate` is a test/bench script under `src/common/` (grep of
src/tests/tools/scripts for `ffi.slate|distributed_cholesky|distributed_trsm|
distributed_eigh|batched_distributed|SlateLowerL|SlateBatchedLowerL`):

- `src/common/slate_cholesky_trsm_test.py` — cholesky + trsm (handle path, op N/C) + `to_jax_lower`
- `src/common/slate_trsm_isolated_test.py` — plain-array trsm across side/uplo/op grid
- `src/common/slate_eigh_test.py` — eigh smoke test
- `src/common/slate_batched_test.py` — batched cholesky + batched trsm (handle path)
- `src/common/slate_chol_trsm_bench.py` — bench; also `to_jax_lower`
- `src/common/slate_vs_cusolvermp_bench.py` — eigh backend A/B bench

Production distributed linalg in GW code goes through `ffi.cusolvermp` /
`ffi.cublasmp` (e.g. `src/gw/w_isdf.py:675` primes the cuBLASMp context).
SLATE is the documented evaluation build / future AMD-GPU (HIP/Frontier)
fallback (README "Future work"; user memory `project_slate_install`).

No LorraxConfig flags or cohsex.in keys are consumed anywhere in this package.
No file I/O (only dlopen of `lorrax_ffi` shared lib via ffi_loader).

Known gaps (documented in README/docstrings):
- `heev` eigenvectors wrong by an unpinned layout transform (eigvals correct).
- `p*q == jax.process_count()` required; no sub-mesh/partial-world calls.
- `heev` needs square mesh; potrf/trsm work on any p×q.
- Batched path on a 1×4 mesh trips SLATE internal assert `internal_batch.hh:290`.

---

## src/ffi/slate/__init__.py (39 LOC)

Pure re-export module + package docstring stating the public API and the
"eigenvectors buggy; W ok" caveat.

| Symbol | Re-exported from |
|---|---|
| `SlateBatchedLowerL`, `batched_distributed_cholesky`, `batched_distributed_trsm` | `.batched` |
| `SlateLowerL`, `distributed_cholesky` | `.cholesky` |
| `distributed_eigh` | `.eigh` |
| `distributed_trsm` | `.trsm` |

Callers: the six `src/common/slate_*` test/bench scripts (see group summary).

---

## src/ffi/slate/context.py (145 LOC)

Per-process singleton cache of opaque SLATE context handles (int), one per
mesh shape `(p, q)`. A context = dup'd + rank-remapped MPI_Comm + identity.
MPI bootstrap piggybacks on phdf5's `phdf5_init_mpi()` if present, else
`lrx_slate_init_mpi` (cpp/context.cc).

| Function | Lines | Role |
|---|---|---|
| `validate_mesh(mesh, *, require_square=False) -> (p,q)` | 51–71 | Checks axes named `('x','y')`; `p*q == jax.process_count()`; optional `p==q` (heev). Raises ValueError otherwise. |
| `_make_ctx(mesh) -> int` | 74–80 | `ffi_loader.get_lib()`; `ffi_loader.create_slate_context(rank, world_size, p, q)`. Collective (dups MPI_COMM_WORLD). |
| `get_or_init_context(mesh) -> int` | 83–95 | Thread-safe (`_LOCK`) memo in `_CACHE: Dict[(p,q), int]`. |
| `_make_subrow_ctx(mesh) -> int` | 98–104 | `ffi_loader.create_slate_subrow_context(rank, world, Px, Py)`. |
| `get_or_init_subrow_context(mesh) -> int` | 107–123 | Memo in `_SUBROW_CACHE`; sub-comm = MPI_COMM_WORLD split by x-coordinate (one comm of size Py per X-row) for batched ops. |
| `_atexit_teardown()` | 126–142 | `atexit`-registered; `destroy_slate_context` on all cached handles, swallowing every exception. |

Callers (grep):
- `validate_mesh` ← cholesky.py:84, trsm.py:62, eigh.py:63, batched.py:115/186.
- `get_or_init_context` ← cholesky.py:92, trsm.py:107, eigh.py:70.
- `get_or_init_subrow_context` ← batched.py:128/244.
- Nothing outside the package imports context.py directly.

Weird/notable:
- Cache key is **only `(p, q)`** — two meshes of the same shape but different
  device/axis permutations share one context handle. Safe today because the
  handle only encodes (rank, world, p, q) and rank comes from
  `jax.process_index()`, but a silent trap if sub-mesh support lands.
- `_atexit_teardown` bare `except Exception: pass` (intentional shutdown
  robustness, but hides teardown failures).
- Note the parallel `ffi.cusolvermp.context.get_or_init_context(mesh,
  col_major=...)` — same name, different signature, different package.

---

## src/ffi/slate/cholesky.py (116 LOC)

Single-matrix distributed Cholesky. Math: `A = L L^H` (Hermitian PD),
`slate::potrf`, `Uplo::Lower` only, dtypes F64/C128.

| Function | Lines | Role |
|---|---|---|
| `SlateLowerL` (frozen dataclass) | 40–73 | Opaque handle: `raw` (n,n) sharded `P('y','x')` holding SLATE col-major L tile bytes — reading `raw` directly in JAX gives `L.T` / `L.conj().T`; plus `mesh`, `n`, `nb`. |
| `SlateLowerL.to_jax_lower()` | 62–73 | shard_map local transpose `in_specs=P('y','x') → out_specs=P('x','y')`, then `jnp.tril` to strip the zeroed strict-upper; returns conventional row-major lower L. |
| `distributed_cholesky(A, *, mesh, block_size=None) -> SlateLowerL` | 76–116 | Validates square A, `n % p == 0 and n % q == 0`; default tile `nb = n // max(p, q)`; shard_map `in_specs=P('x','y')`, `out_specs=P('y','x')`, `check_rep=False`: local `jnp.transpose(local_A, (1, 0))` then `jax.ffi.ffi_call("lorrax_slate_potrf", L_local_T)(local_A_T, n=, nb=, ctx_handle=)` with `L_local_T = ShapeDtypeStruct((n//q, n//p), A.dtype)`. |

Boundary arrays: `A` (n,n) device, `P('x','y')`; per-rank FFI input
`(n/q, n/p)` (transposed shard, col-major bytes); handle `raw` (n,n)
`P('y','x')`.

Callers: `distributed_cholesky` ← src/common/slate_cholesky_trsm_test.py:50,
slate_chol_trsm_bench.py:50; `to_jax_lower` ← slate_cholesky_trsm_test.py:131,
slate_chol_trsm_bench.py:177; `SlateLowerL` isinstance-checked in trsm.py:64.

Weird/notable:
- No dtype validation in Python (docstring says F64/C128 only; enforcement
  presumably in C++).
- Unlike `batched.py`, no jit/shard_map trace cache — every call re-traces the
  eager shard_map (batched.py:62 comment says this "re-traces per call";
  acceptable for a factor-once op but inconsistent).

---

## src/ffi/slate/trsm.py (153 LOC)

Single-matrix distributed triangular solve:
`op(A) @ X = alpha * B` (side='L') or `X @ op(A) = alpha * B` (side='R'),
`slate::trsm`. Enum maps `_SIDE={'L':0,'R':1}`, `_UPLO={'L':0,'U':1}`,
`_OP={'N':0,'T':1,'C':2}`, `_DIAG={'N':0,'U':1}`.

| Function | Lines | Role |
|---|---|---|
| `distributed_trsm(A, B, *, mesh, side='L', uplo='L', op='N', diag='N', alpha=1.0, block_size=None) -> jax.Array` | 41–153 | Two A paths: (a) `SlateLowerL` handle — side/uplo pinned to 'L'/'L', op restricted to 'N' (forward `L X = B`) / 'C' (adjoint `L^H X = B`), `A.raw` fed straight in with `in_specs=(P('y','x'), P('x','y'))` (no A transpose — bytes already SLATE col-major); (b) plain (n,n) array sharded `P('x','y')` — full side/uplo/op/diag surface, local-transposes both A and B. Shape checks: side='L' ⇒ `B.shape[0]==n`, m=B.shape[1]; side='R' ⇒ `B.shape[1]==n`, m=B.shape[0]; `n%p==0 and m%q==0`. Attrs to FFI: `n, m, nb, side, uplo, op, diag, alpha_re, alpha_im, ctx_handle`. FFI out `X_local_T = ShapeDtypeStruct((B.shape[1]//q, B.shape[0]//p), B.dtype)` in `P('y','x')`; a second `_untranspose` shard_map (lines 148–153) flips back to user `P('x','y')`. |

Callers: ← src/common/slate_cholesky_trsm_test.py:163/169 (handle, op N/C),
slate_trsm_isolated_test.py:85 (plain array, side/uplo/op grid),
slate_chol_trsm_bench.py:146/163 (handle).

Weird/notable:
- The output untranspose is a **separate** eager shard_map pass (README: "3
  local transposes per cholesky+trsm chain... below ~1k the transposes are a
  visible fraction of total time"), whereas `batched.py` deliberately folds
  the untranspose inside the same shard_map ("skip a second pass",
  batched.py:280–281). Copy-paste divergence between the single and batched
  paths.
- `alpha` is threaded as `alpha_re/alpha_im` even for real dtypes.
- Same missing dtype validation and missing jit cache as cholesky.py.
- Divisibility check `n%p, m%q` is asymmetric versus cholesky's `n%p and n%q`
  — consistent with B's (n,m) sharding but easy to trip when side='R'.

---

## src/ffi/slate/eigh.py (98 LOC)

Distributed Hermitian eigendecomposition via `slate::heev`. Math:
`A = Q diag(W) Q^H`, W ascending. Shape contract deliberately mirrors
`ffi.cusolvermp.distributed_eigh` so call sites can swap backends with a
one-line import change.

| Function | Lines | Role |
|---|---|---|
| `distributed_eigh(A, *, mesh, compute_evecs=True, block_size=None) -> (W, Q)` | 24–98 | `validate_mesh(require_square=True)` (heev rejects rectangular grids); `n % p == 0`; default `nb = n // p` (one tile per rank, "eigenvectors come out in-place"); `w_dtype = float64 if A.dtype in (complex128, float64) else float32`; outputs `W_local = ShapeDtypeStruct((n,), w_dtype)` replicated `P()` and `Q_local = ShapeDtypeStruct((n//p, n//q), A.dtype)` `P('x','y')`; shard_map `in_specs=P('x','y')`, `out_specs=(P(), P('x','y'))`, `check_rep=False`; FFI `lorrax_slate_eigh` with attrs `n, nb, ctx_handle, compute_evecs`. No transpose anywhere (heev input is Hermitian so A^T = conj(A); the eigvec output layout is the unresolved bug). |

Callers: ← src/common/slate_eigh_test.py:53/119,
slate_vs_cusolvermp_bench.py:107.

Weird/notable:
- **Known-broken eigenvectors**: README — "heev eigenvectors are wrong by a
  layout transform we haven't fully pinned down — eigvals are correct."
  __init__.py docstring: "eigenvectors buggy; W ok". Q is returned anyway.
- With `compute_evecs=False`, Q is *still allocated and returned* with
  unspecified contents (documented).
- Unlike cholesky/trsm there is no local transpose on A — plausibly connected
  to the eigenvector layout artifact (Hermitian input hides a transpose;
  the eigvec matrix is not Hermitian so the missing transform bites there).
- `w_dtype` float32 fallback branch is dead in practice (package is F64/C128
  only), but no explicit dtype check rejects F32/C64 input before the FFI.

---

## src/ffi/slate/batched.py (297 LOC)

3-D batched wrappers for the GWJAX `(Nq, Nmu, Nmu)` scenario: `Nbatch`
independent N×N Hermitian PD matrices, each too big for one GPU. Sharding
contract `P('x', None, 'y')` — batch split along 'x', per-matrix distribution
along 'y'; each X-row of the mesh gets its own MPI sub-comm of size Py
(via `get_or_init_subrow_context`), SLATE grid p=1, q=Py per row. SLATE has no
native batched potrf/trsm — the C++ handler is a plain `for` loop over the
per-rank batch; only the sub-comm/device setup is amortised. Requires
`Nbatch % Px == 0`, `N % Py == 0` (and `Nrhs % Py == 0` for trsm); F64/C128.
Module-level `_JIT_CACHE: dict` memoizes `jax.jit(shard_map(...))` per full
signature (mesh key, dtype, shapes, enums, ctx handle) to kill eager shard_map
re-tracing when looping same-shape calls (comment cites
`src/ffi/phdf5/ARCHITECTURE.md §2.4`).

| Function | Lines | Role |
|---|---|---|
| `_mesh_key(mesh)` | 67–68 | `(axis_names, shape)` tuple for the jit cache key. |
| `SlateBatchedLowerL` (frozen dataclass) | 71–91 | Handle: `raw` (Nbatch, N, N) sharded `P('x','y',None)` — inner (N,N) bytes are SLATE col-major L tiles (JAX read = `L.T`/`L.conj().T`); `mesh, n, nb, nbatch`. Note: **no** `to_jax_lower` equivalent, unlike `SlateLowerL`. |
| `batched_distributed_cholesky(A, *, mesh, block_size=None) -> SlateBatchedLowerL` | 94–161 | Per-slice `A[q] = L[q] L[q]^H` via `slate::potrf`. Default `nb = n // Py`. shard_map `in_specs=P('x',None,'y')`, `out_specs=P('x','y',None)`, `check_rep=False`; local `jnp.transpose(local_A, (0, 2, 1))` (per-slice col-major), FFI `lorrax_slate_batched_potrf` with attrs `nbatch_local, n, nb, ctx_handle`; out `ShapeDtypeStruct((nb_batch_local, n//Py, n), A.dtype)`. |
| `batched_distributed_trsm(A, B, *, mesh, side='L', uplo='L', op='N', diag='N', alpha=1.0, block_size=None) -> jax.Array` | 164–297 | One `slate::trsm` per batch slice. Handle path: op ∈ {'N','C'}, side/uplo pinned 'L'/'L', `in_specs=(P('x','y',None), P('x',None,'y'))`; plain-array path: `in_specs=(P('x',None,'y'),)*2`, local-transposes A too. B is (Nbatch, N, Nrhs) `P('x',None,'y')`; output same shape/sharding (untranspose folded inside the shard_map, lines 280–281 / 292). FFI `lorrax_slate_batched_trsm` attrs: `nbatch_local, n, m, nb, side, uplo, op, diag, alpha_re, alpha_im, ctx_handle`; out `ShapeDtypeStruct((nbatch_local, B.shape[2]//Py, B.shape[1]), B.dtype)`. Duplicates trsm.py's `_SIDE/_UPLO/_OP/_DIAG` maps verbatim (lines 56–59). |

Callers: `batched_distributed_cholesky`, `batched_distributed_trsm` ←
src/common/slate_batched_test.py:52–55, 165, 171 (handle path only; the
plain-array A branch of batched trsm has no caller anywhere — grep of
src/tests/tools/scripts). Docstring mentions a "cuSOLVERMp twin ...
(forthcoming)"; `ffi.cusolvermp.batched_*` now exists (`batched.py` there has
`batched_distributed_cholesky` / `batched_distributed_potrs` /
`batched_distributed_solve_lu`), i.e. the twin landed and is what production
uses.

Weird/notable:
- `_JIT_CACHE` is unbounded and keyed on the int `ctx_handle`; harmless in
  practice (handles are cached per mesh shape) but a stale-handle footgun if
  contexts were ever destroyed/recreated mid-process.
- Handle vs plain-array trsm: for the handle path `side`/`uplo` args are
  silently overwritten (`side, uplo = "L", "L"`, line 193) rather than
  rejected if the caller passed conflicting values — same pattern in trsm.py:69.
- Mesh compatibility check for the handle compares only `axis_names`
  (line 198), not shape/devices, unlike the (p,q)-checked context.
- README: correctness verified 2×2 mesh (nbatch=8, n=128, c128, ~2.3e-16);
  **1×4 mesh trips SLATE assert `internal_batch.hh:290`** — documented
  out-of-scope, "cuSOLVERMp is the answer if you need rectangular meshes."

---

## Cross-package redundancy map (for the refactor)

Four parallel distributed-linalg FFI backends exist under `src/ffi/`:

| Op | slate | cusolvermp | cusolvermg | cublasmp |
|---|---|---|---|---|
| eigh | `distributed_eigh` (eigvecs broken) | `distributed_eigh` (production) | `eigh.py` (simpler variant) | — |
| cholesky | `distributed_cholesky`, `batched_distributed_cholesky` | `batched_distributed_cholesky` | — | — |
| solve | `distributed_trsm`, `batched_distributed_trsm` | `batched_distributed_potrs`, `batched_distributed_solve_lu` | — | `batched_fused_w_solve` |
| gemm | — | — | — | `batched_distributed_gemm` |

The SLATE column is entirely test/bench-reachable only. Documented raison
d'être: AMD-GPU (HIP/Frontier) fallback + evaluation build. Given the
"no redundancy / no parallel old-new paths" project rule, the refactor should
make an explicit keep-as-fallback vs delete decision; if kept, the duplicated
`validate_mesh` variants (slate/context.py, cusolvermp/batched.py
`_validate_mesh`, cublasmp/batched.py `_validate_mesh`, phdf5/context.py
`validate_mesh_2d`) and the `_SIDE/_UPLO/_OP/_DIAG` enum maps (trsm.py +
batched.py) are consolidation candidates.
