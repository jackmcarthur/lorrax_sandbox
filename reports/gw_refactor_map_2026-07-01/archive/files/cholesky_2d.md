# src/common/cholesky_2d.py — deep-read notes (2026-07-01)

LOC: 365. Pure JAX distributed-linalg module; no file I/O, no config parsing.

## Purpose

In-tree 2D block-distributed Cholesky factorization for Hermitian matrices sharded
across a 2D device mesh ('x','y'), designed for the ISDF ζ-fit normal-equation
matrix C_q(μ_X, ν_Y) = (C·Cᵀ). It exists specifically to avoid the "involuntary
full rematerialization" that occurs when resharding P(None,'x','y') →
P(('x','y'), None, None) before a per-q `vmap(jnp.linalg.cholesky)` (module
docstring cites 1.6 GB/device reshard vs 5 MB/device blocked, for n=10k, P=128).
It is the CPU-backend / no-cuSolverMp fallback to the `ffi/cusolvermp`
distributed potrf (see `src/gw/gw_config.py:981` comment: `cusolvermp_charge` /
`cusolvermp_lu` = "off" routes to "in-tree cholesky_2d and per-q jnp.linalg.solve
paths").

Category: **distributed linalg (JAX shard_map kernel), zeta-fit solver support**.

## Entry points (grep over src/, tests/, tools/, scripts/)

Grep commands used:
`grep -rn "cholesky_2d|tiles_to_dense|dense_to_tiles|cholesky_solve_2d|solve_triangular_2d|cholesky_2d_batched|cholesky_2d_single" --include="*.py" src tests tools scripts`

| symbol | callers found |
|---|---|
| `cholesky_2d_batched` | `src/common/isdf_fitting.py:1146` (inside `factor_c_q`, cached in `_chol_2d_cache`); re-exported `src/common/__init__.py:8`; algorithmically mirrored (not imported) by `tests/archive/test_blocked_cholesky.py` / `tests/archive/benchmark_cholesky_sharding.py` |
| `dense_to_tiles` | `src/common/isdf_fitting.py:1150`; `src/common/__init__.py:10` |
| `tiles_to_dense` | `src/common/isdf_fitting.py:1158`; `src/common/__init__.py:11` |
| `cholesky_2d_single` | only via `cholesky_2d_batched` internally; exported in `__init__.py:7`; referenced by name in comments in `tests/test_padding.py:287,351` (documents the 1×1-mesh VMA/scan-carry failure mode) — **no direct external caller** |
| `cholesky_solve_2d` | exported in `__init__.py:9` but **zero call sites** anywhere in src/tests/tools/scripts |
| `solve_triangular_2d` | **zero references anywhere** outside this file; not even exported in `__init__.py` |

Production call path: `gw.gw_init:817 → common.isdf_fitting.factor_c_q → cholesky_2d_batched` (only when `vertex_mu_L == 0`, solver_kind not `cusolvermp_cholesky`, and mesh > 1×1; 1×1 meshes use a dense ridge-regularized `jnp.linalg.cholesky` fallback inside `factor_c_q` itself).

## Function-by-function

### `dense_to_tiles(A, b)` — lines 44–70
Reshape `(..., n, n)` dense → `(..., J, J, b, b)` tiles, J = n//b, then **zeros the
upper-triangular tiles** (`mask = i_idx >= j_idx`). Pure layout transform + mask;
no physics. Note the masking means this is NOT a general dense→tile converter —
it destroys strict-upper tile content (fine for a lower factor / Hermitian input
whose upper half is redundant, but a landmine if reused generically; it *is*
reused generically at the end of `solve_triangular_2d`/`cholesky_solve_2d`, see
weird-code below). Callers: `isdf_fitting.factor_c_q:1150`, internal
(`solve_triangular_2d:317`, `cholesky_solve_2d:364`).

### `tiles_to_dense(tiles, b)` — lines 73–93
Inverse layout transform `(..., J, J, b, b)` → `(..., n, n)` via moveaxis+reshape.
No mask. Callers: `isdf_fitting.factor_c_q:1158`, internal (297–298, 351–352).

### `cholesky_2d_single(mesh, J, b)` — lines 96–228
Factory returning a `shard_map` kernel (`in_specs=out_specs=P('x','y',None,None)`)
that performs right-looking blocked Cholesky A = L·Lᴴ on `(J,J,b,b)` tiles, block
row-index sharded on 'x', block col-index on 'y' (JAX BLOCK distribution:
`my_row_start = px * J_row`). Requires `J % Pr == 0` and `J % Pc == 0` (asserted).
Algorithm per column step k (driven by `lax.scan` over `jnp.arange(J)`):

1. **POTRF** (159–163): diagonal owner (`px == k//J_row and py == k//J_col`)
   computes `Lkk = jnp.linalg.cholesky(A[k%J_row, k%J_col])`; every device runs
   the cholesky but only the owner's write survives the `jnp.where`.
2. **Broadcast Lkk** (166–167): `Lkk = lax.psum(where(is_diag_owner, Lkk_local, 0), ('x','y'))`
   — psum-as-broadcast (owner contributes, everyone else zeros).
3. **TRSM** (169–178): column-k owners solve `L[i,k] @ Lkk^H = A[i,k]` for global
   rows i > k via `right_solve_LH` (141–143): implemented as
   `solve_triangular(L.conj(), block.T, lower=True, trans='N').T`, i.e.
   X Lᴴ = B ⇔ L* Xᵀ = Bᵀ. `lax.fori_loop` over local rows with `jnp.where` gating.
4. **Panel broadcast** (180–198): each device fills its rows of a full
   `(J, b, b)` panel buffer (rows i_glob ≥ k of column k), then
   `panel = lax.psum(panel_local, ('x','y'))`. The panel init is wrapped in
   `lax.pcast(..., ('x','y'), to='varying')` — an explicit VMA (varying-manual-axes)
   annotation required by JAX 0.9+ strict checking on the CPU backend (comment
   at 182–185 says GPU is more forgiving).
5. **SYRK** (200–220): nested `fori_loop` over local (j_loc, i_loc);
   `update = Lik @ Ljk.conj().T`; applied only where
   `i_glob > k and j_glob > k and i_glob >= j_glob` (trailing lower triangle),
   again via compute-everything-then-`jnp.where` masking.

Equation: standard blocked Cholesky recurrence
`L[k,k] = chol(A[k,k] − Σ_{p<k} L[k,p]L[k,p]ᴴ)` realized in right-looking form
(the subtraction happens incrementally in step 5 of earlier iterations).

Communication: 2 psums per column step → O(J log P) rounds as advertised.

### `cholesky_2d_batched(mesh, J, b)` — lines 231–259
Wraps `cholesky_2d_single` in `jax.jit(lax.map(chol_single, all_C_q))` for shape
`(nq, J, J, b, b)` sharded `P(None,'x','y',None,None)` — one XLA dispatch for all
q-points (docstring claims ~18× over a Python loop). **This is the only
production entry point** (via `isdf_fitting.factor_c_q`, which caches the built
fn in `_chol_2d_cache` keyed on `('chol_2d', id(mesh_xy), J, block_size)`).

### `solve_triangular_2d(L_tiles, B_tiles, mesh, lower, trans)` — lines 262–317
"Distributed" triangular solve that in fact **gathers/replicates L to dense and
solves locally** (`tiles_to_dense` → `jax.scipy.linalg.solve_triangular` with
trans='N'/'C'/'T' handled by pre-transposing L) then re-tiles. Explicit
`TODO: Implement distributed triangular solve for very large L` at line 291.
`mesh` arg is **accepted but never used**. Zero callers → dead.

### `cholesky_solve_2d(C_q_tiles, Z_q_tiles, mesh, J, b)` — lines 320–364
Full pipeline `C_q ζ_q = Z_q`: (1) 2D-blocked chol, (2) forward `L Y = Z`,
(3) backward `Lᴴ ζ = Y`; steps 2–3 done densely with two `jax.vmap`'d
`solve_triangular` calls (again "For now, use dense solve"). Builds
`cholesky_2d_batched` fresh on every call (no cache). Zero callers → dead;
the production pipeline instead does chol in `factor_c_q` and the solve in
`isdf_fitting.solve_zeta` (which also handles the SVD/pivoted-LU indefinite-CCT
path for `vertex_mu_L != 0` — Cholesky is charge-channel-only, matching the
bispinor finding that μ_L=i CCT is indefinite).

## Flags consumed

None directly in this file. Routing to/away from it is governed upstream:
`cohsex.in` keys `cusolvermp_charge` / `cusolvermp_lu` (parsed in
`src/gw/gw_config.py`, auto-forced to "off" on CPU backend per comment at
gw_config.py:970–984) and `isdf_fitting._resolve_solver_kind` /
`vertex_mu_L` channel index decide whether `factor_c_q` reaches
`cholesky_2d_batched`.

## I/O

None. Pure device compute.

## Suspects

### Dead
- `solve_triangular_2d` (lines 262–317): grep for the name across src/tests/tools/scripts
  hits only its own definition; not exported in `common/__init__.py`.
- `cholesky_solve_2d` (lines 320–364): exported in `common/__init__.py:9` but grep
  finds zero call sites; production uses `factor_c_q` + `solve_zeta` instead.
- `cholesky_2d_single` as a *public* symbol: exported in `__init__.py:7` yet only
  consumed internally by `cholesky_2d_batched`; external references are test
  docstring comments only (`tests/test_padding.py:287,351`).

### Redundancy
- `cholesky_solve_2d`'s inline dense vmap-solve (354–362) duplicates
  `solve_triangular_2d`'s dense solve (300–314) — two parallel "temporary" dense
  fallbacks for the same missing distributed TRSM, both dead.
- Third parallel Cholesky path overall: (a) this 2D-blocked kernel, (b)
  `factor_c_q`'s dense ridge-regularized `jnp.linalg.cholesky` 1×1-mesh branch,
  (c) `ffi.cusolvermp.batched_distributed_cholesky`. The dispatch lives in
  `isdf_fitting.factor_c_q`, but a refactor should treat these as one solver
  surface.
- `tests/archive/test_blocked_cholesky.py` contains its own
  `dense_to_tiles_lower` (line 57) re-implementing `dense_to_tiles` — archived
  copy-paste, and `tests/archive/benchmark_cholesky_sharding.py` imports from
  that archived test rather than from this module.

### Weird code
- **Masked-compute idiom throughout the kernel** (lines 159–163, 173–176,
  192–195, 208–216): every device computes every POTRF/TRSM/SYRK block and
  discards results via `jnp.where`, including `jnp.linalg.cholesky` on garbage
  (possibly non-PD) trailing blocks on non-owner devices — NaNs are produced
  and masked away. Correct under IEEE-masking but wasteful (full J_row×J_col
  SYRK work at every k instead of trailing-submatrix-only) and NaN-debugging
  hostile (`jax_debug_nans` would trip on the discarded cholesky/solve lanes).
- **Replicated panel buffer** (186–198): `(J, b, b)` = full n×b panel replicated
  on every device per column step via psum. This is the advertised √P bandwidth
  win vs 1D, but it is a per-device replicated intermediate of size n·b —
  relevant to the "zero replicated intermediates" refactor principle.
- **`lax.pcast(..., to='varying')`** (186): JAX-0.9 VMA annotation with a
  version/backend-specific rationale comment (CPU strict, GPU forgiving);
  fragile against JAX upgrades, and the 1×1-mesh scan-carry VMA failure it
  relates to is worked around *outside* this module (in `factor_c_q`, and
  documented in `tests/test_padding.py`), i.e. the kernel silently doesn't
  support 1×1 meshes and relies on callers knowing that.
- **`right_solve_LH`** (141–143): solves X Lᴴ = B as
  `solve_triangular(L.conj(), B.T, lower=True, trans='N').T` — algebraically
  correct (L* Xᵀ = Bᵀ) but an unconventional conj/transpose dance; recorded
  verbatim, not judged.
- **`dense_to_tiles` upper-tile zeroing** (64–68) applied to *solve results* at
  `solve_triangular_2d:317` and `cholesky_solve_2d:364` — the solutions ζ/X are
  generally dense, so re-tiling through a lower-triangular-masking converter
  would silently zero the upper tiles of the answer. Line 316's own comment
  ("though result may not be triangular") flags awareness but the mask is
  applied anyway. Both call sites are dead code, so no live bug, but this is a
  latent correctness trap if resurrected.
- **TODO marker** line 291: distributed triangular solve never implemented.
- Docstring magic numbers (lines 27–29: "1.6 GB/device", "5 MB/device",
  n=10k P=128; line 235: "~18x speedup") — unverified perf folklore baked into
  docs.
