# src/centroid/orbit_syms.py — deep-read notes (2026-07-01)

LOC: 504. Pure symmetry machinery: real-space space-group action helpers bridging
`SymMaps` / `WFNReader` (BGW conventions) to (a) orbit-aware k-means centroid
generation and (b) the IBZ→full-BZ unfold permutation tables consumed by V_q and
ζ loaders. No I/O of its own; host-side numpy except two small jitted kernels.

## Convention (module docstring, lines 1-21)

Row-vector fractional position `r`:

- `image_row(r, s) = r @ Rinv[s].T + tau[s]  (mod 1)`
- `fold_back(δ, s) = δ @ R[s].T`
- `R[s] = sym.R_grid[s]` acts on G-vectors (column convention); `Rinv[s] = sym.Rinv_grid[s]` acts on real-space r.
- `tau[s] = wfn.translations[s] / (2π)` — BGW sign already baked in.
- BGW r-action: `r' = mtrx⁻¹ · r + τ`. Verified via `validate_atomic_symmetries` on Si Fd-3m (96/96 pass with this direction, 48/96 with `mtrx·r + τ`). For symmorphic systems (CrI3, MoS2, τ=0) direction is moot.

## Function table

### `build_real_space_syms(wfn, sym, validate=True)` — lines 33-59
- Role: host-side table builder. Optionally runs `sym.validate_atomic_symmetries(wfn)` and raises on failures. Returns `R, Rinv` as `(n_sym,3,3)` int32 **jax** arrays and `tau = wfn.translations[:ntran]/(2π)` as `(n_sym,3)` fp64 jax array.
- Callers (grep across src/tests/tools/scripts): `src/centroid/kmeans_cli.py:269-270` (orbit-aware k-means CLI); referenced by name in `kmeans_isdf.py:816` docstring of `kmeans` (the "pass the table from build_real_space_syms" contract).
- Flags: none directly (CLI decides orbit-awareness).

### `orbit_images(reps, Rinv, tau)` — lines 66-85, `@jax.jit`
- Physics: `images[s, r, :] = (reps @ Rinv[s].T + tau[s]) mod 1` — all sym images of each representative point. Implemented as `jax.vmap(lambda Ri, t: (reps @ Ri.T + t) % 1.0)(Rinv, tau)`.
- Shapes: reps `(n_rep,3)` fp64 device; output `(n_sym, n_rep, 3)` fp64 device.
- Callers: ONLY internal — `canonicalize_orbit` (line 120). Grep for `orbit_images` in src/tests/tools/scripts finds no external caller. Not dead (internal helper) but public-looking with zero external users.

### `_CANON_INV = jnp.int64(10**12)` — line 88
- Magic constant: 12-digit integer canonicalisation key ≈ 1 ppm resolution on fractional coords. Documented rationale: coarser than fp64 noise by ~3 orders, fine enough not to collapse distinct orbit members.

### `_orbit_lex_winner(images)` — lines 95-111
- Role: for each rep, index `s*` of lex-smallest orbit image. Integer lex ordering: `keys = jnp.round(images * _CANON_INV).astype(jnp.int64)`; per-rep `jnp.lexsort((keys[:,i,2], keys[:,i,1], keys[:,i,0]))` (lexsort primary = LAST key → x primary, y secondary, z tertiary), take `order[0]`. vmapped over reps.
- Callers: `canonicalize_orbit` only (internal).

### `canonicalize_orbit(reps, Rinv, tau)` — lines 114-124, `@jax.jit`
- Role: map each rep to lex-smallest member of its orbit; idempotent; static shape `(n_rep, 3)`. `take_along_axis(images, best_s[None,:,None], axis=0)[0]`.
- Callers: `src/centroid/kmeans_isdf.py:720-722` via `_canonicalize_rep` (single-rep wrapper used inside orbit-aware Lloyd loop).

### `unfold_orbit_unique_with_id(reps_np, Rinv, tau, tol=1e-6)` — lines 127-173, host numpy
- Role: unfold k-means reps into all distinct orbit images + per-candidate dense `orbit_id` (same id iff same physical orbit). Key for orbit-aware pivoted Cholesky: reps that drifted into the same orbit during Lloyd get one id.
- Einsum signatures VERBATIM:
  - `np.einsum('ri,sji->srj', reps_np, Rinv)` (+ `tau[:,None,:]`, mod 1) — unfold step.
  - `np.einsum('ci,sji->scj', flat, Rinv)` — canonical-member step per unique candidate.
- Dedup: integer keys `np.round(flat*inv).astype(int64) % inv` with `inv = round(1/tol)`; `np.unique(..., return_index)` keeping first occurrence in original order (`flat[np.sort(first_idx)]`). Canonical member per candidate via per-row `np.lexsort((z,y,x))[0]` Python loop; `orbit_id = np.unique(canonical_keys, axis=0, return_inverse=True)`.
- Comment at line 149: "int! avoid fp64-precision loss at 1e18".
- Callers: `src/centroid/kmeans_cli.py:313-318` only.
- Note: uses tol-based key `1e-6` (inv=10^6) here vs `_CANON_INV = 10^12` in the jax path — two different canonicalisation resolutions for the same concept.

### `compute_centroid_sym_perm(r_mu_fft_idx, sym_matrices, translations, fft_grid, *, validate=True, extend_trs=False)` — lines 180-388, host numpy
- Role: THE central table for the V_q IBZ→full-BZ unfold. For each target centroid μ and sym s, computes source centroid α(μ)=`sym_perm[s,μ]` and integer lattice wrap `L_table[s,μ]` such that `y_μ = mtrx·(x_μ − τ) = x_{α(μ)} + L_μ, L_μ ∈ ℤ³`.
- Physics equation (V_q unfold consumer): `V_full[q1, μ', ν'] = exp(2π i q · (L_{μ'} − L_{ν'})) · V_ibz[parent, α(μ'), α(ν')]`.
- Einsum VERBATIM: `images_raw = np.einsum('sij,srj->sri', S.astype(np.float64), r_shifted)` where `r_shifted = r_frac[None,:,:] - tau_frac[:,None,:]`.
- Algorithm: frac coords from FFT indices → apply `mtrx·(x−τ)` → `images_int = np.rint(images_raw * fft_grid)` (snap BEFORE floor, see weird_code) → `L_wrap = floor_divide(images_int, grid)` int8 → mod back to grid → radix-flatten (`i_x*ny*nz + i_y*nz + i_z`) → lookup via `flat_to_mu` dense table (`-1` = miss).
- Validation (validate=True): raises RuntimeError on orbit-closure failure (image not in centroid table; message tells caller to regenerate centroids orbit-aware) and on non-permutation rows (τ×fft_grid non-integer collision).
- `extend_trs=True`: duplicates rows to `(2·n_sym, ...)` — TRS keeps r fixed, so TRS-augmented ops reuse spatial rows; makes the table index-compatible with `SymMaps.irr_idx_q / sym_idx_q` values in `[0, 2·ntran)`. This is the Phase-1 fix for the TRS-blind sym bug (silent OOB clip under JAX `mode='promise_in_bounds'` → wrong V_q at every TRS-folded q; see `reports/trs_sym_audit_2026-05-14/agent_1_scope_report.md` Site #1). ζ-leg conjugation under TRS (`V_{TRS-q,μ,ν} = conj(V_{q,μ,ν}) = V_{q,ν,μ}`) is applied at the V_q-unfold level (`gw.v_q_tile._unfold_v_q_ibz_to_full`), NOT here.
- Returns: `sym_perm (n_sym|2n_sym, n_rmu) int32`, `L_table (…, n_rmu, 3) int8`.
- Callers (grep evidence):
  - `src/gw/v_q_g_flat.py:176,186` — builds sym_perm/L_table for flat-G V_q unfold.
  - `src/gw/compute_vcoul.py:937,947` — same for vcoul path.
  - `src/common/isdf_fitting.py:2046-2059` — used as an orbit-closure **pre-check** (`_check_perm`) before write_ibz_only ζ output; RuntimeError caught → fallback (charge) or loud-fail (transverse).
  - `src/file_io/zeta_loader.py:422-448` — ZetaLoader q='full_bz' unfold (`mu_perm, _mu_L`).
  - `tests/test_trs_unfold_centroid_perm.py` (lines 39,77,80,101,155,222), `tests/test_q_ibz_and_centroid_perm.py` (lines 15,162,177,201,234).
  - Docstring cross-references (not calls): `src/gw/v_q_tile.py:1447,1587,1612`, `src/common/symmetry_maps.py:175,208,227,263,271,839`, `src/file_io/isdf_header.py:8`.
- Arrays crossing boundary: all host numpy; consumers move sym_perm/L_table to device themselves.

### `compute_rgrid_sym_perm(sym_matrices, translations, fft_grid, *, validate=True)` — lines 395-502, host numpy
- Role: per-sym permutation over the FULL FFT grid. Returns `sym_perm[s, r_new] = r_old` (pull-back gather table) with `r_{r_new} ≡ S_s·r_{r_old} + τ_s`; flat C-order `r_flat = i_x*ny*nz + i_y*nz + i_z`. Used for expanding IBZ-q ζ onto the full BZ: `ζ_full[q, r_new, μ] = ζ_ibz[i(q), sym_perm[s(q), r_new], π_{s(q)}^{-1}(μ)]`.
- Einsum VERBATIM: `images = (np.einsum('rj,sij->sri', r_frac, Rinv.astype(np.float64)) + tau_frac[:,None,:])` with `Rinv = np.rint(np.linalg.inv(S)).astype(np.int64)`.
- NOTE direction difference vs `compute_centroid_sym_perm`: this one computes the FORWARD image `r' = Rinv·r + τ` then INVERTS the permutation (scatter loop `sym_perm[s, img_flat[s]] = base`, lines 484-487); the centroid one computes the SOURCE `mtrx·(x−τ)` directly (no inversion). Both are documented as the same BGW convention seen from opposite ends.
- Also uses plain `np.floor` wrap (`images - np.floor(images)`, line 470) rather than the snap-before-floor integer path used in `compute_centroid_sym_perm` — grid points here are exactly commensurate so the ISDF-noise concern doesn't arise, but it is a second, slightly different implementation of the same wrap.
- Memory: builds `(n_sym, n_rtot, 3)` fp64 `images` — for a 100³ grid × 48 syms that's ~1.1 GB host; fine for typical grids but worth noting for the refactor.
- Validation: per-row permutation check; error message names τ×fft_grid non-integer cause ("Off-grid fractional translations are not yet supported by the ζ full-BZ unfold path").
- Callers: `src/file_io/zeta_loader.py:423,446` (q='full_bz' unfold; zeta_loader.py:48 records it as the prerequisite that had to land). `tests/test_zeta_loader.py:191,213` (expects RuntimeError on singular random sym matrices).

## Cross-module dependencies
- Consumes: `SymMaps` (`sym.R_grid`, `sym.Rinv_grid`, `sym.validate_atomic_symmetries`, `sym.sym_matrices`), `WFNReader` (`wfn.ntran`, `wfn.translations`, `wfn.sym_matrices`).
- Consumed by: `centroid.kmeans_cli`, `centroid.kmeans_isdf`, `gw.v_q_g_flat`, `gw.compute_vcoul`, `common.isdf_fitting`, `file_io.zeta_loader`; contract cross-refs in `gw.v_q_tile`, `common.symmetry_maps`, `file_io.isdf_header`.

## I/O
None. Pure computation; no files read or written.

## Flags / cohsex.in keys
None consumed directly. Orbit-awareness and extend_trs are decided by callers (kmeans_cli CLI flags like `--no-orbit`; V_q unfold paths).

## Suspects

### dead_suspects
- `orbit_images` (line 66): grepped `orbit_images` across src/, tests/, tools/, scripts/ — only caller is `canonicalize_orbit` in the same file. Public jitted API with zero external users; could be folded into `canonicalize_orbit` or de-exported.

### redundancy_suspects
1. Two canonicalisation implementations of the same "lex-min orbit member with integer keys" idea: jax path (`_orbit_lex_winner`/`canonicalize_orbit`, key = 10^12) vs numpy path inside `unfold_orbit_unique_with_id` (key = round(1/tol)=10^6, per-row Python-loop lexsort). Different resolutions, duplicated logic.
2. Two grid-snap/wrap implementations: `compute_centroid_sym_perm` (integer snap-before-floor, lines 322-326, with the documented fp-noise fix) vs `compute_rgrid_sym_perm` (float `images - np.floor(images)` then rint, lines 470-473). If the fp-noise fix ever matters on the full grid, the second path lacks it.
3. Two directions of "the same" table: `compute_centroid_sym_perm` builds a source map α directly; `compute_rgrid_sym_perm` builds forward images then inverts. Consistent with the user's "unified sym action" memory note — these are candidates for routing through one canonical sym-action helper.
4. `compute_centroid_sym_perm` is called at 4 independent sites (v_q_g_flat, compute_vcoul, isdf_fitting pre-check, zeta_loader), each re-deriving `ntran`/translations slicing locally — table construction is recomputed rather than cached on SymMaps.

### weird_code
- Line 88: `_CANON_INV = jnp.int64(10**12)` magic constant (documented; hypothesis: fine, but should share a constant with the 1e-6 tol used in `unfold_orbit_unique_with_id`).
- Line 149: `inv = np.int64(round(1.0 / tol))` with comment "int! avoid fp64-precision loss at 1e18" — historical fp bug scar.
- Lines 312-322: snap-to-int BEFORE floor with long comment documenting the exact off-by-one L-wrap bug (spurious exp(±iπ/2) phase in unfold_v_q, 14/64 q's at rel err ~0.8 on Si Fd-3m 24³). Load-bearing; any refactor must preserve it.
- Lines 386-387: `np.concatenate([sym_perm, sym_perm.copy()])` — `.copy()` on the second operand is redundant (concatenate always copies); harmless.
- `L_table` dtype int8 (line 324): silently assumes |L| ≤ 127; true for any physical τ/centroid in [0,1) (|L| ≤ ~2) but undocumented as a bound.
- `compute_rgrid_sym_perm` materialises `(n_sym, n_rtot, 3)` fp64 intermediates — O(GB) host peak on large FFT grids.
