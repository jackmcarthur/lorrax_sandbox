# src/gw/v_q_g_flat.py — deep-read notes (refactor map, 2026-07-01)

557 LOC. Post-G-flat rewrite of the V_q hot loop; replaces `gw.v_q_tile.compute_V_q_tile`
whenever the on-disk ζ is in WFN.h5-style per-q sphere ("G_flat") layout.

**Physics:**
```
V_q[μ, ν] = Σ_G  conj(ζ̃_{q,μ}(G)) · v(q+G) · ζ̃_{q,ν}(G)
g0_μ(q)   = ζ̃_{q,μ}(G=0)     # = ζ̃[μ, 0] by sphere convention
```
IBZ→full-BZ unfold post-loop via centroid double-permute + umklapp L-phase
`exp(2π i q·(L_μ − L_ν))` + TRS conjugation (V_q bilinear in ζ, no τ-phase).

**Category:** physics: bare-Coulomb V_q stage (post ζ-fit, pre W-solve), with
IBZ/symmetry orchestration.

## Entry points (grep across src/, tests/, tools/, scripts/)

| symbol | callers |
|---|---|
| `compute_all_V_q_g_flat` | `gw/compute_vcoul.py:899-916` (`compute_all_V_q` dispatcher, G-flat branch — itself called from `gw/gw_init.py:1097`); `tests/test_compute_all_V_q_g_flat.py:37,144,185,194,219`; `tests/test_compute_V_q_bispinor_g_flat.py:207,237` |
| `_compute_V_q_g_flat_one_tile` | `gw/v_q_bispinor.py:531,640` (bispinor per-tile loop) |
| `_resolve_ibz_q_list` | `gw/gw_jax.py:213-219` (W IBZ cascade tables); `gw/v_q_bispinor.py:559-564`; referenced in comment `gw/gw_init.py:957` |
| `_pick_g_chunk` | internal only (line 349). External mentions are docstrings/comments only: `gw/compute_vcoul.py:868`, `gw/gw_config.py:241,612` |
| `_make_read_all_ibz` | internal only (lines 390-392); exported in `__all__` but zero external callers |
| `_make_per_q_kernel` | internal only (line 386); not exported |

## Function-by-function

### `_PER_Q_KERNEL_CACHE` (line 51)
Module-level dict caching compiled kernels, keyed
`(id(mesh_xy), n_rmu_L, n_rmu_R, ngkmax, g_chunk, write_g0, same_zeta)`.
Never evicted. `id()` in a cache key is a footgun (address reuse after GC → stale
kernel with a dead mesh), though in practice one mesh lives per run.

### `_make_per_q_kernel(mesh_xy, n_rmu_L, n_rmu_R, ngkmax, g_chunk, *, write_g0, same_zeta)` — lines 54-144
Compile-once jitted per-q contract kernel. Returns
`fn(V_acc, g0_acc, zeta_L_q, zeta_R_q, v_q, q_idx) -> (V_new, g0_new)` with
`donate_argnums=(0,1)` (accumulators updated in place).
- Inputs: `zeta_*_q` shape `(1, n_rmu_*, ngkmax)` at `P(('x','y'), None)` (device),
  resharded to `zeta_L: P('x', None)`, `zeta_R: P('y', None)`; `v_q (ngkmax,)`
  replicated `P(None)`.
- Contraction is a `lax.scan` over `n_chunks = ngkmax // g_chunk` G-chunks
  (replaced a Python-unrolled loop that grew HLO/compile time linearly; CrI3
  6×6 80 Ry has ~14 chunks, MoS2 3×3 has 1). Per-chunk math (verbatim):
  ```python
  L_w = jnp.conj(L_chunk) * v_chunk[None, :]
  return V_carry + L_w @ R_chunk.T, None
  ```
  i.e. `V[μ,ν] += Σ_g conj(ζ_L[μ,g])·v[g]·ζ_R[ν,g]` (matmul, no einsum string).
- `dynamic_update_slice` writes the `(n_rmu_L, n_rmu_R)` block into
  `V_acc (n_q, n_rmu_L, n_rmu_R)` at `P(None,'x','y')`; if `write_g0`,
  `g0_q = zeta_L[:, 0]` written into `g0_acc (n_q, n_rmu_L)` at `P(None,'x')`.
- `same_zeta=True` (charge/diagonal tiles): caller aliases `zeta_R_q is zeta_L_q`;
  kernel reshards one buffer twice. `same_zeta=False`: bispinor off-diagonal
  tiles with potentially different `n_rmu_*`.

### `_resolve_ibz_q_list(*, sym, centroid_indices, kgrid, fft_grid, verbose)` — lines 152-221
Picks IBZ q's via centroid orbit closure; falls back to full BZ. Returns
`(q_irr_kgrid_int, q_irr_frac, q_full_to_irr_idx, q_full_to_irr_sym, sym_perm, L_table, use_ibz)`.
- Calls `centroid.orbit_syms.compute_centroid_sym_perm(..., extend_trs=True)`
  → `sym_perm (2·n_tran, n_rmu)` (second half = TRS-augmented rows, indexed by
  `sym.irr_idx_q/sym_idx_q` values; see reports/trs_sym_audit_2026-05-14) and
  `L_table` per-(sym,μ) integer lattice wraps for the umklapp phase.
- `RuntimeError` from closure check → full-BZ fallback (all `nkx·nky·nkz` q's).
- Env `LORRAX_FORCE_FULL_BZ=1` bypasses IBZ entirely (debug knob, also honored by
  gw_init ζ-writer and gw_jax W cascade so all three gates stay consistent).
- BGW wrap: `q > kg/2 → q − kg` (strict >), matching the writer's per-q
  gvec_components convention; `q_irr_frac = wrapped / kgrid`.

### `_pick_g_chunk(ngkmax, target=4096)` — lines 224-229
Largest divisor of ngkmax ≤ target (magic default 4096). Linear downward scan.

### `_make_read_all_ibz(zeta_loader, n_rmu_padded, mesh_xy)` — lines 232-264
Returns `read_all_ibz(n_q_ibz) -> (n_q_ibz, n_rmu_padded, ngkmax)` device array at
`P(None, ('x','y'), None)`. Duck-type dispatch:
- `ZetaLoader.load(q=list(...), layout='G_flat', sharding=spec)` (test bench), or
- `ZetaReader.read_zeta_G_slab(q_offset=0, q_count, mu_offset=0, mu_count=n_rmu_padded, qvec_batch_frac=zeros, sphere_idx=None, mesh, valid_mu=n_rmu_logical)` (production PHDF5/FFI).
Raises TypeError if neither method exists. Batched single call avoids n_q_ibz
distinct `_FfiBackend.read_slab._per_rank` closures → JAX trace-cache misses.

### `_compute_V_q_g_flat_one_tile(zeta_L_loader, zeta_R_loader, *, v_per_G_builder, kgrid, fft_grid, bvec, mesh_xy, g_chunk, sym, centroid_indices, write_g0, timing_label, verbose)` — lines 271-465
Per-tile core (one (μ_L, ν_L) tile end-to-end). Shared by the charge wrapper and
`gw.v_q_bispinor`'s 7-tile loop. Flow:
1. `same_zeta = zeta_R_loader is None or is zeta_L_loader`; validates both loaders'
   `zeta_layout == 'G_flat'` (ValueError otherwise).
2. `_resolve_ibz_q_list` → n_q_ibz; validates on-disk q-count
   (`zeta_L_loader.gvec_components.shape[0] == n_q_ibz`, error message points at
   `write_ibz_only` mismatch) and, when two loaders, matching gvec shapes
   (must share `zeta_cutoff_ry` + q-layout).
3. `v_q_table = v_per_G_builder(q_irr_frac, gvec_components)` — `(n_q_ibz, ngkmax)`
   complex128 host, then `device_put` replicated `P(None, None)`.
4. `g_chunk` = arg or `_pick_g_chunk(ngkmax)`; must divide ngkmax.
5. μ-padding: each side padded to multiple of `p_x·p_y` (mesh device product).
6. Accumulators: `V_acc (n_q_ibz, n_rmu_L_pad, n_rmu_R_pad) c128 @ P(None,'x','y')`,
   `g0_acc (n_q_ibz, n_rmu_L_pad) c128 @ P(None,'x')` — g0_acc allocated even when
   `write_g0=False` (donate target only, contents unread).
7. Pre-read ALL IBZ ζ̃ slabs in ONE batched call (lines 394-420; ~50 MB/rank MoS2
   3×3, ~0.8 GB/rank CrI3 6×6 80 Ry). 2026-05-12 change note: 63 → 7 read_slab
   calls on MoS2 3×3 bispinor. The historical per-q PHDF5 read inside the loop
   deadlocked against NCCL collectives (root cause of async-prefetch removal).
8. Sync per-q loop: `dynamic_slice_in_dim` a `(1, n_rmu, ngkmax)` q-slab, call
   kernel, `jax.block_until_ready(V_acc)` each q (timing print per q).
9. If `use_ibz`: `common.symmetry_maps.unfold_v_q(V_acc, irr_idx, sym_idx,
   sym_perm, L_table, q_irr_frac, mesh_xy, n_sym_spatial=sym_perm.shape[0]//2)`;
   if `write_g0` also `gw.v_q_tile._unfold_g0_ibz_to_full(...)`.
Returns `(V_qmunu @ P(None,'x','y'), g0 @ P(None,'x') or None)`.

NOTE: parameter `bvec` (line 276) is never referenced in the body — dead
parameter (both callers pass it; only the charge wrapper's closure needs bvec,
and that closes over its own copy).

### `compute_all_V_q_g_flat(zeta_loader, *, kgrid, fft_grid, bvec, cell_volume, mesh_xy, sys_dim, bdot=None, bare_coulomb_cutoff_ry=None, bgw_v_grid_fn=None, mc_average_vcoul_body=True, g_chunk=None, verbose=True, sym=None, centroid_indices=None, async_prefetch=False)` — lines 472-553
Public charge entry point (CC tile, `same_zeta`, `write_g0=True`,
`timing_label='CC'`). `sys_dim` must be 2 or 3 (0-D box not wired →
NotImplementedError). Builds `_bare_v_per_G` closure:
- `gw.compute_vcoul.compute_v_q_per_G(q_irr_frac, gvec_components, bvec,
  cell_volume, sys_dim, vcoul_cutoff_ry=bare_coulomb_cutoff_ry, bdot,
  v_head_miniBZ=table)` — bare Coulomb on the per-q sphere.
- 3D + `mc_average_vcoul_body`: `build_v_head_miniBZ_avg_3d(kgrid, bvec,
  cell_volume)` precomputes mini-BZ-averaged v(q, G=0); injected at IBZ q's only
  (unfold is bilinear and inherits the head). 2D: `f2d → 0` already regularizes
  G=0; MC flag silently no-op'd.
- Optional `bgw_v_grid_fn` overlay (lines 530-540): host-side scatter of BGW's
  full-FFT-grid v into the sphere: `idx = (m0%nx)*ny*nz + (m1%ny)*nz + (m2%nz)`;
  `v[qi] = np.where(v_at_sphere != 0.0, v_at_sphere, v[qi])` — 0.0 is a sentinel
  for "BGW has no value here" (falls back to LORRAX v).
- `async_prefetch` accepted for back-compat with the legacy dispatcher's env knob
  (`LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH`, read in compute_vcoul.py:900-901) but has
  NO effect — sync loop already ~6× faster than legacy μ×ν tile driver (MoS2 3×3).

## Flags / config consumed
- Via dispatcher `gw/compute_vcoul.compute_all_V_q` ← gw_init: cohsex.in keys
  `vq_g_chunk_size` (0 = auto `_pick_g_chunk`; gw_config.py:243,612,969),
  `mc_average_vcoul_body` (gw_config.py:269,538,922), `bare_coulomb_cutoff`,
  `sys_dim` / cell geometry.
- Env vars: `LORRAX_FORCE_FULL_BZ` (read directly, line 174),
  `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH` (read by dispatcher, currently a no-op here).

## I/O
- **Reads:** ζ̃ G-flat slabs from the zeta_q HDF5 via `ZetaLoader.load` /
  `ZetaReader.read_zeta_G_slab` (per-q WFN.h5-style sphere layout,
  `(n_q, n_rmu, ngkmax)` complex; loader attrs used: `n_rmu`, `zeta_layout`,
  `gvec_components (n_q, 3, ngkmax) int32`). One batched read per tile side.
- **Writes:** nothing. Output arrays returned in device memory
  (`V_qmunu @ P(None,'x','y')`, `g0 @ P(None,'x')`).

## Dead suspects
- `async_prefetch` kwarg (line 489): explicitly documented "currently has no
  effect"; the env knob `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH` plumbs into it and is
  therefore dead. `tests/test_compute_all_V_q_g_flat.py:168`
  (`test_..._async_matches_sync`) trivially passes since both paths are identical.
- `bvec` parameter of `_compute_V_q_g_flat_one_tile` (line 276): grep shows no
  use in the function body; callers (charge wrapper line 546, v_q_bispinor)
  still pass it.
- `_make_read_all_ibz` and `_pick_g_chunk` in `__all__` (line 556-557): grep of
  src/tests/tools/scripts finds no external code callers (only comment/docstring
  mentions in compute_vcoul.py:868 and gw_config.py:241,612) — exported surface
  larger than actual use.

## Redundancy suspects
- Parallel old/new V_q paths: this module supersedes `gw/v_q_tile.compute_V_q_tile`,
  but the legacy r-space path remains live in the `compute_all_V_q` dispatcher
  (compute_vcoul.py:876-886, "Reachable only for legacy r-space ζ files; not
  exercised by the current writer"). Classic old-next-to-new; also drags legacy
  kwargs `mu_chunk_size` / `q_batch_size` (ignored) through the dispatcher API.
- Unfold helpers split across modules: `unfold_v_q` from `common.symmetry_maps`
  but `_unfold_g0_ibz_to_full` still imported from `gw.v_q_tile` (line 310) —
  half-migrated symmetry machinery.
- `_resolve_ibz_q_list` is called independently by three sites (this file,
  v_q_bispinor.py:564, gw_jax.py:214) recomputing the same orbit-closure tables
  per call; gw_init.py:957 comment notes it runs "per tile".

## Weird code
- **Stale module docstring** (lines 16-20): claims "Async I/O — kept from the
  legacy driver ... a worker thread reads ζ̃_{q+1} while the compute thread
  contracts ζ̃_q". The implementation is the opposite: one batched pre-read +
  synchronous per-q loop; comment at 394-406 says the prefetcher was the root
  cause of a deadlock and was removed. Docstring contradicts code.
- `gflat_memory_model.py:4` cites `gw.v_q_g_flat.compute_V_q` — a function name
  that does not exist in this module (stale reference to a renamed entry point).
- `_PER_Q_KERNEL_CACHE` keyed on `id(mesh_xy)` (line 71), unbounded, never
  evicted; `id()` reuse after mesh GC could return a kernel bound to a dead mesh.
- `jax.block_until_ready(V_acc)` inside the per-q loop (line 432): serializes
  dispatch every q for the timing print — timing instrumentation left in the hot
  path even when `verbose=False`.
- Magic constant `target=4096` in `_pick_g_chunk` (line 224); no derivation.
- BGW v overlay sentinel: `v_at_sphere != 0.0` (line 540) treats exact-0.0 as
  "missing from BGW grid" — a legitimately zero v (e.g. truncated Coulomb) would
  silently fall back to the LORRAX value.
- BGW wrap uses strict `q_irr_kgrid_int > kg/2` (line 216) — even-grid boundary
  point kg/2 stays positive; matches writer convention per comment, but is the
  kind of half-grid convention worth pinning in the unified sym layer.
- `g0_acc` always allocated (n_q × n_rmu_L_pad c128) even when `write_g0=False`
  (lines 379-382) purely as a donate target.
- `n_chunks` recomputed at line 354 solely for the verbose print (kernel derives
  its own copy at line 86).
