# FFI group: cublasmp + cusolvermg

Files: `src/ffi/cublasmp/batched.py`, `src/ffi/cublasmp/__init__.py`,
`src/ffi/cusolvermg/eigh.py`, `src/ffi/cusolvermg/__init__.py`.
Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.
Grep scope for all caller claims below: `src/`, `tests/`, `tools/`, `scripts/`
(`grep -rn "batched_distributed_gemm|batched_fused_w_solve|eigh_mg|cublasmp|cusolvermg"`).
tests/, tools/, scripts/ had **zero** hits for any of these symbols; all callers live in `src/`.

---

## src/ffi/cublasmp/batched.py — 290 loc

**Purpose.** Python/JAX side of the cuBLASMp distributed-GEMM and fused
W-Dyson-solve FFI. All matrices flow in their natural `P(None,'x','y')`
2D-sharded layout on the (Px,Py) mesh; the C++ handler runs one
`cublasMpMatmul`/solve per batch element q with a single reusable
descriptor set. Category: **distributed linalg FFI (W/Dyson stage backend)**.

**Layout convention.** JAX row-major vs cuBLASMp col-major is reconciled by a
"pre-transpose trick": inside `shard_map` the inner two dims are transposed
(`jnp.transpose(x, (0, 2, 1))`) so raw local bytes are col-major with
`lld = local_rows`; result is transposed back. `transa/transb` then have
standard BLAS semantics. One-tile-per-rank restriction: every global dim must
divide by Px/Py as appropriate. Dtypes F64/C128.

### Function table

| Function | Lines | Role |
|---|---|---|
| `_mesh_key(mesh)` | 47–48 | Hashable (axis_names, shape) key for `_JIT_CACHE`. |
| `_validate_mesh(mesh)` | 51–59 | Requires axes ('x','y') and Px·Py == `jax.process_count()`. Returns (Px,Py). |
| `batched_distributed_gemm(A,B,C,*,mesh,alpha,beta,transa,transb)` | 62–187 | `D[q] = alpha·op(A[q]) @ op(B[q]) + beta·C[q]` per q via FFI target `lorrax_cublasmp_batched_gemm`. |
| `batched_fused_w_solve(V,chi,pref,*,mesh,stop_after_step=0)` | 190–226 | Thin shape/dtype validator; delegates to `batched_fused_w_solve_jit(...)` then calls it. |
| `batched_fused_w_solve_jit(*,dtype,nq,n,pref,mesh,stop_after_step=0)` | 229–290 | Builds/caches the jitted fused W-solve; exposed separately so callers can AOT `.lower(V,chi).compile()` (precompile path). FFI target `lorrax_cublasmp_batched_w_solve`. |

### batched_distributed_gemm (62–187)

- Validation: 3D inputs `(Nq, rows, cols)`, matching batch dim and dtype,
  `transa/transb ∈ {N,T,C}` (codes `{"N":0,"T":1,"C":2}` at line 41),
  contraction-k agreement, C shape = (m,n), divisibility
  `m%Px == n%Py == k%Py == k%Px == 0` (line 123).
- `get_lib()` loads/registers FFI targets; `get_or_init_context(mesh, col_major=False)`
  (from `ffi.cusolvermp.context`) supplies `ctx_handle` — cuBLASMp shares the
  cuSOLVERMp communicator/grid context.
- Attrs passed to FFI (lines 153–164): `nq,m,n,k, mb_a,nb_a,mb_b,nb_b,mb_c,nb_c,
  lld_a,lld_b,lld_c, transa,transb, alpha_re/im, beta_re/im, ctx_handle`.
  `lld_X = X_rows // Px` (local rows in the col-major view).
- shard_map body `_gemm` (172–182): pre-transposes A,B,C; `jax.ffi.ffi_call`
  with `input_output_aliases={2: 0}` (D written in place of C); output local
  shape `(nq, n//Py, m//Px)` (transposed back on return). Wrapped
  `jax.jit(_gemm, donate_argnums=(2,))` (line 184) and cached in module-level
  `_JIT_CACHE` keyed by ("gemm", mesh_key, dtype, shapes, trans, alpha, beta, ctx_handle).
- **Callers:** `src/common/cublasmp_gemm_test.py:96` ONLY. Not used anywhere in
  the production GW pipeline (w_isdf uses only the fused solve). Test-harness-only.

### batched_fused_w_solve / _jit (190–290)

Physics (docstring + w_isdf `_get_w_solve_fn_low_mem`, w_isdf.py:269–301):
symmetric-Cholesky Dyson solve of `W(q) = v(q) [1 − pref·χ₀(q) v(q)]⁻¹ v(q)`
recast as

```
v = X X†                  (cusolverMp potrf on V)
H = I − X† (pref·χ) X     (2 cublasMp gemms + identity−T CUDA kernel)
L_H = chol(H)             (cusolverMp potrf)
W = X H⁻¹ X†              (2 cublasMp trsms + 1 cublasMp gemm)
```

all inside ONE FFI call (`lorrax_cublasmp_batched_w_solve`,
implemented in `cpp/batched_w_solve_ffi.cc` + `cpp/w_solve_kernels.cu`) so XLA
never sees intermediates → no reshard/remat. `pref` is the Dyson prefactor
`2/(√N_k · n_spin · n_spinor)` computed by `w_isdf._w_solve_pref_scalar`
(w_isdf.py:304–311); it enters as a **compile-time complex attr**
(`pref_re/pref_im`), not a runtime arg — hence the different `.lower(V, chi)`
signature vs the JAX_NATIVE path noted in `precompile_solve_w`.

- `batched_fused_w_solve` (190–226): checks V is (Nq,N,N) square, chi matches
  shape/dtype, then builds/fetches the jit and calls it.
- `batched_fused_w_solve_jit` (229–290): validates `n % Px == n % Py == 0`;
  attrs = `nq, n, pref_re, pref_im, ctx_handle, stop_after_step`; output local
  shape `(nq, n//Py, n//Px)`; shard_map in/out specs all `P(None,'x','y')`,
  `check_rep=False`. **V is deliberately NOT aliased to the output** (comment
  lines 281–284): downstream `sigma_coh` still needs the original v; the FFI
  does an internal `cudaMemcpyAsync V_in → V_work`. Note the top-level
  docstring of `batched_fused_w_solve` says "V ... DONATED" (line 207) which
  contradicts the actual non-aliased implementation — stale docstring.
- `stop_after_step` (0–9): early-exit debug attr; C++ side gates each pipeline
  stage (`batched_w_solve_ffi.cc:302,311,330,349,359,375,387,411,430`). No
  Python caller ever passes a nonzero value (grep over src/tests/tools/scripts:
  only the definition sites hit) — debug scaffolding.

**Callers (grep evidence):**
- `batched_fused_w_solve <- src/gw/w_isdf.py:282,298` (`_get_w_solve_fn_low_mem`,
  the ScreeningSolver.CUBLASMP_FFI branch of the W-solve dispatch),
  `src/common/cublasmp_w_solve_test.py:28,81`.
- `batched_fused_w_solve_jit <- src/gw/w_isdf.py:676–681` (`precompile_solve_w`
  AOT branch), and internally from `batched_fused_w_solve`.
- `batched_distributed_gemm <- src/common/cublasmp_gemm_test.py:36,96` only.

**Flags consumed (indirect, via w_isdf dispatch):** cohsex.in
`isdf_memory_mode` (`auto|high_mem|low_mem`, gw_config.py:262) →
`_LEGACY_ISDF_MEMORY_MODE` (gw_config.py:137–141) →
`config.backend.screening_solver = ScreeningSolver.CUBLASMP_FFI`
(value string `"cublasmp_ffi"`, gw_config.py:132); consumed at
gw_jax.py:262,265. batched.py itself reads no config.

**Arrays crossing the boundary:** A/B/C and V/chi: `(Nq, N, N)` device arrays,
sharded `P(None,'x','y')`, one tile per rank; locally transposed views handed
to FFI as col-major `(nq, cols//Py, rows//Px)` buffers. Output W same layout.
All device-resident; no host staging in this file.

**I/O:** none (no files read/written). FFI shared library loaded via
`ffi.common.ffi_loader.get_lib()`; targets registered in ffi_loader.py:45–46
(`lorrax_cublasmp_batched_gemm → CublasMpBatchedGemmFfi`,
`lorrax_cublasmp_batched_w_solve → CublasMpBatchedWSolveFfi`).

**Cross-module deps:** `ffi.common.ffi_loader.get_lib`,
`ffi.cusolvermp.context.get_or_init_context` (shared multi-process GPU
context), consumed by `gw.w_isdf` (production) and `common.cublasmp_*_test`.

### Suspects

- **dead-ish:** `batched_distributed_gemm` — zero production callers; only
  `src/common/cublasmp_gemm_test.py` (grep over src/tests/tools/scripts).
  Likely a stepping-stone kept as FFI validation harness; the fused solve
  subsumed its production role.
- **dead knob:** `stop_after_step` — plumbed through Python API + attrs but
  never set ≠ 0 by any caller; pure C++-side debug ladder.
- **weird:** module-level unbounded `_JIT_CACHE` keyed partly on
  `int(ctx_handle)` — if the context were ever re-created the stale jit for
  the old handle would linger (handle reuse could alias entries).
- **weird:** docstring says V is "DONATED (XLA reuses its buffer)" while the
  code explicitly avoids `input_output_aliases` for V and comments the
  opposite — stale doc, real behavior is copy-into-workspace.
- **weird:** `check_rep=False` on both shard_maps (required for FFI, but means
  no replication checking on the `None` batch axis).
- **redundancy (intentional dispatch, not cruft):** fused CUBLASMP_FFI path
  vs `JAX_NATIVE` LU path (`w_isdf._get_w_solve_fn` / `lu_factor+lu_solve`)
  are parallel implementations of the same Dyson solve, selected by
  `screening_solver`; both kept on purpose (high-mem default vs low-mem FFI).

---

## src/ffi/cublasmp/__init__.py — 21 loc

Pure re-export shim: `batched_distributed_gemm`, `batched_fused_w_solve`,
`batched_fused_w_solve_jit` from `.batched`, plus `__all__` and a usage
docstring. No logic, no suspects. Category: package init.

---

## src/ffi/cusolvermg/eigh.py — 86 loc

**Purpose.** `eigh_mg` — JAX FFI wrapper around `cusolverMgSyevd`:
single-process multi-GPU real-symmetric F64 eigensolve. The simpler
counterpart to `ffi.cusolvermp.distributed_eigh` (no NCCL/MPI bootstrap;
cuSOLVERMg does the column-tile block-cyclic distribution internally —
input/output live on device 0). Category: **distributed linalg FFI
(bench/experimental eigensolver — not on any GW production path)**.

### Function table

| Function | Lines | Role |
|---|---|---|
| `eigh_mg(A, *, tile_size=32, max_gpus=0, compute_evecs=True)` | 27–86 | Validates square F64 A (complex Hermitian explicitly NOT implemented, TypeError at 64–67); `get_lib()`; builds `_call` closure invoking FFI target `lorrax_cusolvermg_eigh_f64` with attrs `n, tile_size, max_gpus, compute_evecs`; returns `(evals (n,) ascending, Q (n,n))` via `jax.jit(_call)(A)`. |

- **Convention flag:** returned Q are eigenvectors of **A^T** (the col-major
  view of the row-major input) — docstring lines 58–59 argue equality for
  symmetric A by transposition. For real symmetric input this is exact only
  up to the transpose identity; any future complex extension would inherit a
  conjugation trap here.
- `compute_evecs=False`: Q output buffer still allocated/returned but
  "contents should be ignored" (lines 50–52) — wasted (n,n) buffer.
- `jax.jit(_call)` is constructed fresh inside every `eigh_mg` call (line 85);
  relies on JAX's global trace cache rather than an explicit cache
  (contrast `_JIT_CACHE` in cublasmp/batched.py) — minor inconsistency,
  functionally fine.
- `tile_size` default 32 chosen "for small test cases" vs cuSOLVERMg's own
  256 default for large matrices (docstring 44–48) — a bench-oriented magic
  default that would be suboptimal at production n.

**Callers (grep evidence over src/tests/tools/scripts):**
- `eigh_mg <- src/common/cusolvermg_eigh_test.py:37,62,65` (smoke test, run as
  `python3 -u -m common.cusolvermg_eigh_test` per `src/ffi/AGENTS.md:93`;
  validated F64 @ n∈{128, 2048}, err ≲ 2e-11 per AGENTS table),
  `src/common/eigh_benchmark.py:74,95,97,103,105` (bench vs `jnp.linalg.eigh`).
- No caller in `src/gw/`, tests/, tools/, or scripts/ → not on any physics
  pipeline. The whole subpackage is a bench/eval artifact.

**Flags consumed:** none (no LorraxConfig / cohsex.in keys).
**I/O:** none. FFI target registered ffi_loader.py:47
(`lorrax_cusolvermg_eigh_f64 → EighMgF64`), C++ in `cpp/eigh_mg_ffi.cc`.
**Cross-module deps:** `ffi.common.ffi_loader.get_lib` only.

### Suspects

- **dead-ish (production):** entire `eigh_mg` path — grep shows only
  `common.cusolvermg_eigh_test` and `common.eigh_benchmark` callers; the GW
  code never eigendecomposes via this route. Candidate for
  quarantine-as-bench or deletion in the refactor, alongside the docstring's
  own admission that the cusolverMp `distributed_eigh` is the intended
  successor "once that path is unblocked".
- **weird:** eigenvectors-of-A^T convention; F64-only with complex explicitly
  deferred; Q returned-but-garbage when `compute_evecs=False`.

---

## src/ffi/cusolvermg/__init__.py — 11 loc

Pure re-export shim: `eigh_mg` from `.eigh` + `__all__`. No logic, no
suspects. Category: package init.
