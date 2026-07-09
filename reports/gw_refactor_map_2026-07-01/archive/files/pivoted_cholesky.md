# src/centroid/pivoted_cholesky.py — deep-read notes (993 LOC)

Group: pivoted_cholesky. Repo root: /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D.

## Purpose

Pivoted-Cholesky pruning of over-sampled ISDF candidate points. K-means (with
`--oversample > 1`) produces M > N_mu candidate grid points; this module builds
the q=0 pair-product Gram matrix G_ab over those candidates and runs greedy
pivoted Cholesky to keep the N_mu candidates that best span the
valence-conduction pair-product space. Optionally orbit-aware (one pivot per
symmetry orbit, output orbit-closed) — this is the gate for the IBZ-only
cascade (see memory: project_lorrax_ibz_cascade). Deliberately single-process
in structure but supports multi-device row-sharded Gram + select via shard_map.

**Category:** preprocessing tool — ISDF centroid selection (k-means
post-processing stage), with distributed-linalg kernels inside.

## Entry points (grep over src/, tests/, tools/, scripts/)

| function | callers |
|---|---|
| `prune_candidates_by_pivoted_cholesky` | src/centroid/kmeans_cli.py:345,384 (production, when `--oversample > 1`); tests/test_pivoted_cholesky.py:468-511 (validation-error tests) |
| `pivoted_cholesky_select` | tests/test_pivoted_cholesky.py only (reference impl for sharded version) |
| `build_candidate_gram_q0` | tests/test_pivoted_cholesky.py only |
| `make_sharded_gram_q0` | tests/test_pivoted_cholesky.py:212,240 only |
| `make_sharded_pivoted_cholesky_select` | pivoted_cholesky.py:412 (from `prune_candidates_by_pivoted_cholesky`); tests/test_pivoted_cholesky.py:318,418 |
| `build_gram_q0_via_loadwfns` | pivoted_cholesky.py:386 (from `prune_candidates_by_pivoted_cholesky`) only |
| `gather_wfn_at_candidates` | **zero callers anywhere** (grep `gather_wfn_at_candidates` over the whole checkout hits only the definition) |
| `_fold_spin_into_band` | **zero callers** (not even in this file or tests) |

Production call chain: `python -m centroid.kmeans_cli --oversample 1.5 ...` →
`prune_candidates_by_pivoted_cholesky` → `build_gram_q0_via_loadwfns` (Gram) →
`make_sharded_pivoted_cholesky_select` (select) → keep_idx back to kmeans_cli →
written to `centroids_frac_{N}{suffix}.txt`.

## Function-by-function

### `gather_wfn_at_candidates(wfn, sym, cand_idx, band_start, band_end)` — L60-119
- Role: evaluate psi_{n,k}(r~_a) at M candidate FFT-grid points, IBZ k-points
  only. Raw c(G) → scatter to FFT box → iFFT (`norm='ortho'`) → gather at
  candidates, via `WfnLoader` + `common.wfn_transforms.to_rmu` on a hard-coded
  1×1 single-device mesh (`jax.devices()[:1]`).
- Returns `(nkpts, band_end-band_start, nspinor, M)` complex128.
- `sym` arg is `del`-ed at L97 ("reserved; see docstring").
- Accesses private `wfn._filename` (exists: file_io/wfn_loader.py:133, "legacy
  WFNReader compat").
- **DEAD**: no callers found. Superseded by
  `common.load_wfns.load_centroids_band_chunked` used in
  `build_gram_q0_via_loadwfns`.

### `_fold_spin_into_band(psi)` — L127-130
- `(nk, nb, nspinor, M) → (nk, nb*ns, M)` pure reshape. **DEAD**: zero callers.

### `build_candidate_gram_q0(phi_val_cand, psi_cond_cand, k_weights=None, enforce_hermitian=True)` — L133-184, `@jax.jit(static: enforce_hermitian)`
- Physics: `G_ab = Σ_k w_k · [Σ_v φ_{v,k}(a) φ*_{v,k}(b)] · [Σ_c ψ*_{c,k}(a) ψ_{c,k}(b)]`
  (the q=0 ISDF pair-product Gram). Implemented as per-k matmuls in a
  `lax.fori_loop`:
  - `P_v = phi_k.T @ jnp.conj(phi_k)` (valence projector, (M,M))
  - `P_c_tilde = jnp.conj(psi_k).T @ psi_k` (conduction projector, (M,M))
  - `G += w_k * (P_v * P_c_tilde)` elementwise product.
- Optional `(G + G^H)/2` symmetrization at exit.
- Inputs `(nk, nv_eff, M)` / `(nk, nc_eff, M)` complex, spin folded into band.
- Consumers: tests only. Serves as the single-device reference implementation.
- Minor: accumulator allocated `(M, M2)` at L179 with M from phi, M2 from psi —
  no shape assertion that they match.

### `pivoted_cholesky_select(G, k_keep, orbit_id=None)` — L192-269, `@jax.jit(static: k_keep)`
- Role: single-device greedy pivoted Cholesky on Hermitian PSD G. Always runs
  exactly `k_keep` iterations (no early stop). Per iteration j:
  - pivot p = argmax of active residual diagonal d;
  - `L[:, j] = (G[:, p] − Σ_{i<j} L[:, i]·conj(L[p, i])) / sqrt(d[p])`, with the
    pivot entry force-set to exactly `sqrt(d[p])` ("kills rounding drift");
  - Schur update `d ← max(d − |L[:,j]|², 0)`;
  - deactivate p — or, if `orbit_id` given, the whole orbit
    (`kill_mask = orbit_id == orbit_id[p]`), i.e. one pivot per sym orbit.
- Returns `(piv (k_keep,) int32 −1-padded, L (M,k_keep), rank, d_final (M,),
  d_taken (k_keep,), trR_over_trG (k_keep+1,))`.
- rank = #pivots with `d_taken > sqrt(eps)·max(diag G)`; post-rank `d_taken`
  zeroed.
- `pivot_val = max(masked_d[p], eps)` — eps floor for division safety.
- Consumers: tests only; production path uses the sharded factory below even on
  a 1×1 mesh.

### `prune_candidates_by_pivoted_cholesky(wfn, sym, cand_idx, n_keep, mesh, *, n_val, n_cond, band_range_left, band_range_right, band_norms, k_weights, verbose, bispinor, orbit_id, use_phdf5)` — L279-449
- Role: end-to-end wrapper: validate → build Gram via
  `build_gram_q0_via_loadwfns` → reshard to row-shard
  `P(('x','y'), None)` → sharded select → keep-index unfold.
- Validation:
  - band-window vs `wfn.nbands`;
  - PW-basis sanity: raise if `max_band > 0.5 · ngk_max · nspinor` ("centroid
    pruning ill-posed; prune the real-space grid directly");
  - requires 2-D mesh with both 'x' and 'y' axes (gw_jax convention; 1×1 for
    single device);
  - `M % (mesh.x · mesh.y) == 0` (sharded select splits M evenly; error message
    suggests `--no-shard`).
- Orbit mode: `n_keep` counts ORBITS; after select, `keep_idx` = union of
  orbits of picked pivots (`np.isin(orbit_id, orbit_id[piv])`) — orbit-closed
  by construction.
- `d_final` gathered across processes via
  `multihost_utils.process_allgather(d_final, tiled=True)`.
- Returns `(keep_idx, rank, G, d_final_np, d_taken, trR_over_trG)` — note G
  (device, row-sharded) is returned but kmeans_cli discards it (`keep_idx,
  rank, *_`).
- `k_weights` kwarg is accepted but **never used** in the body (the Gram
  builder hard-codes uniform 1/nk_tot weights).
- Legacy `(n_val, n_cond)` defaults: `n_val = wfn.nelec`,
  `n_cond = min(n_val, nbands − n_val)`.

### `make_sharded_gram_q0(mesh, M, *, enforce_hermitian=True)` — L475-573
- Factory returning jitted closure `(phi, psi, kw) → G` with G row-sharded
  `P('x', None)` on a 1-D mesh. Inputs replicated. Inside `shard_map`
  (`check_rep=False`): each device slices its M_slab columns via
  `lax.dynamic_slice_in_dim(phi_full, my_idx*M_slab, M_slab, axis=2)` and does
  the same per-k projector matmuls as `build_candidate_gram_q0` but
  `(M_slab, M)`:
  - `P_v_slab = phi_k_block.T @ jnp.conj(phi_k)`; `P_c_slab = jnp.conj(psi_k_block).T @ psi_k`.
- `enforce_hermitian=True` gathers full G to every device
  (`with_sharding_constraint` to `P()`), symmetrizes, reshards — an admitted
  all-gather cost (comment L557-562).
- Consumers: tests only. **Not on the production path** —
  `build_gram_q0_via_loadwfns` is used instead.

### `make_sharded_pivoted_cholesky_select(mesh, M, k_keep, *, mesh_axis='x')` — L601-749
- Factory returning jitted `step(G, orbit_id=None)`; G row-sharded on
  `mesh_axis` (str or tuple, e.g. `('x','y')` flattened). Same greedy algorithm
  as `pivoted_cholesky_select`, distributed:
  - per-device argmax → `lax.pmax(local_pv)` for global pivot value;
  - tie-break to lowest global index: `winner_p = where(local_pv >= global_pv,
    local_global_p, int32(2**30))`; `global_p = -pmax(-winner_p)`;
  - column `G[:, global_p]` local (row-sharded — no collective; header comment
    L594-598 explains why row-sharding is chosen);
  - row `L[p, :]` broadcast from owning shard via masked `psum`
    (zeros elsewhere); same idiom broadcasts `orbit_id[p]` in orbit mode;
  - pivot-row entry force-set to `sqrt(d[p])` only on the owner shard.
- Comm per iteration: 2 scalar pmax + one (k_keep,) psum (+1 int psum in orbit
  mode) → O(k_keep²) total comm, per the header comment.
- Output shardings: `(rep, row_shard, rep, row_shard_1d, rep, rep)` for
  `(piv, L, rank, d_final, d_taken, trR_over_trG)`.
- Consumers: `prune_candidates_by_pivoted_cholesky` (production);
  tests/test_pivoted_cholesky.py (equivalence vs single-device).

### `build_gram_q0_via_loadwfns(wfn, sym, cand_idx, n_val, n_cond, mesh_xy, *, bispinor, verbose, band_range_left, band_range_right, band_norms, band_chunk_size=64, use_phdf5=False, memory_per_device_gb=None)` — L785-993
- Role: production Gram build reusing gw_jax's exact data path (full-BZ
  unfolded, 2-D mesh):
  - lazy imports: `common.meta.Meta`, `common.load_wfns.load_centroids_band_chunked`,
    `common.isdf_fitting.{pair_density, gram_q0_from_pair}` ("don't charge the
    single-device prune path for the gw_jax dep chain").
  - `load_centroids_band_chunked(wfn, sym, meta, cand_idx, bispinor, mesh_xy,
    range, band_chunk_size, use_phdf5)` → `(psi_rmu_Y, psi_rmuT_X)`; Y shape
    `(nk, nb, ns, n_rmu)`, X shape `(nk, n_rmu, nb, ns)` (X pre-conjugated),
    once for the "left" window, once for "right".
  - `pair_density(psi_rmuT_X, psi_rmu_Y, mesh_xy)` →
    `P_k[mu_X, nu_Y] = Σ_{n,s} ψ*(μ) ψ(ν)` (gw_jax convention; header comment
    L752-782).
  - `gram_q0_from_pair(P_l_k, P_r_k, kw, mesh_xy)` →
    `G[mu_X, nu_Y] = Σ_k w_k · conj(P_l_k) · P_r_k` — the `conj()` flips
    gw_jax's `Σ_v φ*(μ)φ(ν)` into the valence-projector form `Σ_v φ(a)φ*(b)`.
    Comment L989-990: "γ̃ identity (charge channel) — open-spin Frobenius
    reduction."
  - k-weights hard-coded uniform `1/nk_tot` (full-BZ unfold).
- Window semantics (**important convention change**): legacy `(n_val, n_cond)`
  mode maps to left=(0, n_val), right=(0, n_val+n_cond) — i.e. **v×(v+c)**,
  NOT the literal legacy v×c. Justified in comment L870-880: "On MoS2 4×4 this
  cut V_H |err| at the CBM ~3× vs the legacy v×c window"; callers wanting
  strict v×c must pass explicit band ranges. But the docstring at L806-808
  still calls (n_val, n_cond) "the literal valence × conduction pair-product
  Gram" — self-contradictory.
- Pseudoband `band_norms`: `ψ /= max(norm, 1.0)` on both windows (applied on Y
  axis 1 and X axis 2). Docstring cites "same clamp recipe as
  isdf_fitting.py:838-847" — **stale line reference** (those lines are now the
  pair-pipeline jit cache).
- Memory budget: sets `meta.memory_per_device_gb` from arg or
  `common.gpu_utils.get_device_memory_gb()` (bare `except Exception:` → 0.0,
  "falls back to the 36 GB default" downstream).
- Returns G sharded `P('x','y')`.

## Flags consumed

No LorraxConfig / cohsex.in keys directly. Reached via `centroid.kmeans_cli`
CLI flags: `--oversample` (>1 activates prune), `--prune-n-val`,
`--prune-n-cond`, `--prune-window {v_x_c, v_x_vc, vc_x_vc}`, `--use-phdf5`,
`--no-shard` / `--force-shard` (mesh shape). Device memory auto-detect via
`common.gpu_utils.get_device_memory_gb`.

## Cross-module deps

- `file_io.WfnLoader` (imported as `WFNReader` alias, L51; also re-imported
  inside `gather_wfn_at_candidates`)
- `common.symmetry_maps.SymMaps` (type only; nk_tot used)
- `common.timing` (sections: prune.gram, prune.select, left.load, left.pair,
  right.load, right.pair, q0_sum)
- `common.wfn_transforms.to_rmu` (dead function only)
- `common.meta.Meta.from_system`
- `common.load_wfns.load_centroids_band_chunked`
- `common.isdf_fitting.pair_density`, `common.isdf_fitting.gram_q0_from_pair`
- `common.gpu_utils.get_device_memory_gb`
- `jax.experimental.multihost_utils.process_allgather`

## I/O

- Reads: WFN HDF5 via `WfnLoader` / `load_centroids_band_chunked` (WFNReader or
  phdf5 backend; `use_phdf5` flag). No direct file reads otherwise.
- Writes: nothing directly. The caller (kmeans_cli) writes the pruned
  centroids to `centroids_frac_{N}{suffix}.txt` (np.savetxt, fractional
  coords + provenance header).

## Dead suspects

1. `gather_wfn_at_candidates` (L60-119) — grep for the name across the whole
   checkout (src, tests, tools, scripts) hits only the definition. Superseded
   by the load_wfns path in `build_gram_q0_via_loadwfns`.
2. `_fold_spin_into_band` (L127-130) — zero references anywhere.
3. `k_weights` parameter of `prune_candidates_by_pivoted_cholesky` (L291) —
   accepted, never read in the body (Gram builder hard-codes uniform weights).
4. Test-only-in-practice: `build_candidate_gram_q0`, `pivoted_cholesky_select`
   (single-device), `make_sharded_gram_q0` — no production callers; kept as
   reference implementations for the test suite.

## Redundancy suspects

1. **Three Gram builders**: `build_candidate_gram_q0` (single-device,
   replicated), `make_sharded_gram_q0` (1-D row-sharded, replicated inputs),
   `build_gram_q0_via_loadwfns` (2-D, production). Only the last is on the
   production path; the first two survive only for tests. Classic old/new
   parallel-path cruft.
2. **Two select kernels**: `pivoted_cholesky_select` vs
   `make_sharded_pivoted_cholesky_select`. The sharded one on a 1×1 mesh
   subsumes the single-device one (production already always uses it via
   `prune_candidates_by_pivoted_cholesky`); the single-device version exists
   only as the test oracle. ~80 lines of duplicated algorithm.
3. `gather_wfn_at_candidates` duplicates what
   `load_centroids_band_chunked` + `to_rmu` already do (dead, see above).
4. Caller-side: `kmeans_cli --prune-window v_x_c` ("legacy") passes
   (n_val, n_cond), which `build_gram_q0_via_loadwfns` now silently maps to
   the v×(v+c) window — identical to `--prune-window v_x_vc`. Two flag values
   produce the same Gram while kmeans_cli prints
   "prune window: v×c ... right=({n_val},{max}) [legacy]" (kmeans_cli.py:384
   area), which does NOT match the executed window (0, n_val+n_cond).

## Weird code

1. L51 `from file_io import WfnLoader as WFNReader` — aliases the new loader
   under the legacy name for type hints; dead `gather_wfn_at_candidates` then
   re-imports `WfnLoader` under its real name (L98) and reads private
   `wfn._filename` (L114). Naming shear from the WFNReader→WfnLoader
   migration.
2. L452-472 header comment: "The sharded pivoted-Cholesky SELECT path is not
   in this commit ... Coming in a follow-up" — **stale**; the sharded select
   is implemented ~130 lines below in the same file.
3. Module docstring L3-4 cites "``pivoted_cholesky.md`` (sandbox root)" — no
   such file at /pscratch/sd/j/jackm/lorrax_sandbox/pivoted_cholesky.md.
4. Docstring L816 cites "isdf_fitting.py:838-847" for the norm clamp — stale
   line numbers (that region is now a jit-cache block).
5. L675 tie-break sentinel `jnp.int32(2**30)` magic constant in the sharded
   pivot argmin-index reduction (`-pmax(-winner_p)` idiom).
6. L870-882: silent semantic change of the "legacy (n_val, n_cond)" mode to
   v×(v+c) with an empirical justification comment ("cut V_H |err| at the CBM
   ~3×"); contradicts both the function's own docstring (L806-808 "literal
   valence × conduction") and kmeans_cli's printed window.
7. L920-926: `except Exception: memory_per_device_gb = 0.0` — swallows any
   detection failure, relying on a downstream "36 GB default" that is only
   documented in a comment.
8. L557-566 (`make_sharded_gram_q0`): Hermitian symmetrization deliberately
   all-gathers the full (M,M) G to every device and reshards — flagged in-code
   as acceptable "while G still fits post-gather".
9. L243 / L697-699: pivot entry of L force-set to exactly `sqrt(d[p])`
   ("kills rounding drift") — intentional numerics fix, notable convention.
10. L179: Gram accumulator shaped `(M, M2)` mixing phi's M and psi's M2
    without asserting M == M2.
