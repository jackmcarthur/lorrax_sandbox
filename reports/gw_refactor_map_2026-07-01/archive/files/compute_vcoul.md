# src/gw/compute_vcoul.py (1119 LOC)

Deep-read notes for the GW refactor map. Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.

## Purpose

Bare-Coulomb factory + V_q dispatcher for the ISDF pipeline. Builds `v(q+G)` /
`sqrt(v(q+G))` for sys_dim = 0 (WS-box), 2 (slab), 3 (bulk, with mini-BZ MC head
average), plus the phase `exp(-2pi i q.r)` on the FFT box; and provides
`compute_all_V_q`, the top-level dispatcher that routes to the G-flat driver
(`gw.v_q_g_flat`) or the legacy r-space tile driver (`gw.v_q_tile`) and applies
the IBZ->full-BZ unfold of V_q via the centroid double-permute.

Category: **physics: bare Coulomb v(q+G) construction + V_q stage dispatcher**.

The module's own header says the actual tile logic moved to `gw.v_q_tile`; this
file retains the Coulomb factory, the dispatcher, and (apparently orphaned)
per-q G-chunked kernels.

## Entry points (grep across src/, tests/, tools/, scripts/)

| Function | Callers (grep evidence) |
|---|---|
| `compute_all_V_q` | `src/gw/gw_init.py:884,1097` (scalar V_q branch of GW driver). Docstring x-ref in `src/gw/w_isdf.py:429`. |
| `make_v_munu_chunked_kernel` | `src/gw/gw_init.py:949,1037` (bispinor legacy r-space branch — only `sphere_idx`, `get_sqrt_v_and_phase` consumed downstream via `src/gw/v_q_bispinor.py:331,336`); internally by `compute_all_V_q`. Mimicked (copy) by `src/gw/aot_memory_model/kernels/vq_mu_chunk.py`. |
| `compute_v_q_per_G` | `src/gw/v_q_g_flat.py:507,521`; `src/gw/v_q_bispinor.py:206,218`; `tests/test_compute_all_V_q_g_flat.py:102,106`. |
| `build_v_head_miniBZ_avg_3d` | `src/gw/v_q_g_flat.py:507,517`; internally by `make_v_munu_chunked_kernel` (sys_dim=3). |
| `fft_integer_axes` | `src/gw/v_q_bispinor.py:104,107`; internal. |
| `exp_ikr_fftbox` | internal only (`make_v_munu_chunked_kernel`). |
| `compute_sqrt_vcoul_0d` | internal only (`make_v_munu_chunked_kernel` sys_dim=0 path). |
| `compute_v_q_per_q_g_chunked` / `_v_q_per_q_g_chunked_jit` | **ZERO callers** (grep `compute_v_q_per_q_g_chunked|_v_q_per_q_g_chunked_jit` over src/tests/tools/scripts: no hits outside this file). |
| re-exported `_choose_v_q_chunks`, `_unfold_g0_ibz_to_full`, `_compute_V_q_tile` (from `gw.v_q_tile`), `unfold_v_q` (from `common.symmetry_maps`) | `_unfold_g0_ibz_to_full` + `unfold_v_q` used by `compute_all_V_q` IBZ branch; `_choose_v_q_chunks` re-export has no importer anywhere (grep `_choose_v_q_chunks` -> only `v_q_tile.py` defines/uses it). |

## Function-by-function

### `exp_ikr_fftbox(fft_nx, fft_ny, fft_nz)` — L39-52
Host-side numpy builders of fractional coordinate grids fx,fy,fz (shapes
(1,nx,1,1) etc., /N) for `exp(ik.r)` on the FFT box; returned as jax arrays.
Carries a "NOTE TO FUTURE EDITORS — numpy USAGE IS INTENTIONAL" block (commit
bbff26f 2026-04-18: jnp fired 3 standalone pjit compiles). Internal consumer:
phase factor inside `get_sqrt_v_and_phase`.

### `fft_integer_axes(fft_nx, fft_ny, fft_nz)` — L55-67
Integer FFT frequency grids in `np.fft.fftfreq` order, reshaped (nx,1,1)/(1,ny,1)/(1,1,nz).
Same "intentional numpy" note. Consumers: `make_v_munu_chunked_kernel` (G_cart
construction) and `src/gw/v_q_bispinor.py:104-107`.

### `compute_sqrt_vcoul_0d(fft_nx, fft_ny, fft_nz, bdot, cell_volume)` — L74-181
0-D (molecule/box) truncated Coulomb: builds `V_trunc(r) = 2 sqrt(det adot)/r`
with Wigner-Seitz minimum-image truncation on a dense grid (dNfft = fft_grid *
N_IN_BOX, rounded up by `_round_up_fft_size`), FFTs to G-space (unnormalized,
BGW convention), then samples onto the WFN FFT grid with a phase correction
`exp(-2pi i (j1*TRUNC_SHIFT/dN1 + ...))` undoing the half-cell trunc shift.
Returns `sqrt(v(G)/cell_volume)` as complex128 on the FFT grid. Only q=0 valid.
Constants `N_IN_BOX, NCELL, TRUNC_SHIFT` imported from `gw.compute_vcoul_0d`.
Metric: `adot = inv(bdot) * 4 pi^2`, per-element divided by dNfft_i*dNfft_j.
Physics: v0D(G) = FT[ theta_WS(r) * 2/r ] (Rydberg). Note the G-extraction is a
**triple pure-Python loop over the full WFN FFT grid** (O(n_G) python
iterations) — slow but one-shot.

### `make_v_munu_chunked_kernel(...)` — L191-487
Factory (module-level cache `_v_munu_kernel_cache`, key = fft grid + kgrid +
bvec + cell_volume + sys_dim; NOTE: `vcoul_cutoff_ry` and
`mc_average_vcoul_body` are NOT in the cache key — see weird_code). Builds:

- Sphere gather: `common.coulomb_sphere.compute_bare_coulomb_sphere_idx`
  (single source of truth shared with the G-flat zeta writer in
  `common.isdf_fitting`) -> `sphere_idx (n_sph,) int32` or None.
- sys_dim=0: static `sqrt_v_0d_flat` from `compute_sqrt_vcoul_0d`;
  `get_sqrt_v_and_phase(q)` returns it plus trivial phase (q must be 0).
- sys_dim=3 `get_sqrt_v_and_phase(qvec_wrapped)` (jit, L312-358):
  phase = `exp(-2j pi (q1/nkx * fx + q2/nky * fy + q3/nkz * fz))`;
  `q_cart = einsum('a,ab->b', q_frac, bvec_j)` (VERBATIM);
  `v = 8 pi / |q+G|^2 * (1/cell_volume)` (Rydberg), G=0 slot replaced by
  mini-BZ MC average `_v_head_avg_j[qx,qy,qz]` when `mc_average_vcoul_body`
  (q=0 head stays 0 — injected later as rank-1 in Sigma_X); optional
  `vcoul_cutoff_ry` zeroing for `|q+G|^2 > cutoff`; returns
  `sqrt_v = sqrt(v)` (complex128), gathered to sphere if active.
- sys_dim=2 (L360-396): same, times slab truncation factor
  `f2d = 1 - exp(-zc*kxy) * cos(kz*zc)` with `zc = pi / bvec[2,2]`
  (slab must be z-oriented).
- Contraction kernels (L402-466): `fft_and_weight_inner` (FFT zeta_r box via
  `make_sharded_fftn_3d` with spec `P(('x','y'),None,None,None)`, multiply
  phase, weight by sqrt_v, gather sphere; also returns `g0_chunk =
  zeta_G_flat[:,0]`), `contract_block_inner` with einsum
  `'mG,nG->mn'` on `(conj(zeta_mu), zeta_nu)` (VERBATIM), plus five JIT'd
  wrappers: `fft_and_weight`, `contract_block`, `fft_weight_contract`,
  `fft_weight_contract_diag`, `fft_weight_contract_offdiag`,
  `fft_and_weight_keep`.
- Returns a `SimpleNamespace` bundle with all kernels + `n_G`, `n_sph`,
  `sphere_idx`, `fft_shape`.

Physics: V_q[mu,nu] = Sigma_G conj(zeta~_mu(G)) v(q+G) zeta~_nu(G);
v3D = 8pi/|q+G|^2, v2D = 8pi/|q+G|^2 * (1 - e^{-zc k_xy} cos(k_z zc)),
head(q!=0, G=0) = <8pi/|q+dq|^2>_miniBZ.

External consumption of the bundle: **only** `sphere_idx` and
`get_sqrt_v_and_phase` (`v_q_bispinor.py:331,336`) and, inside this file,
`sphere_idx`/`n_sph`/`get_sqrt_v_and_phase` in `compute_all_V_q`. The six
FFT/contract kernels have zero external callers (grep
`\.(fft_and_weight|contract_block|fft_weight_contract|fft_and_weight_keep|fft_and_weight_inner|contract_block_inner)\b`
over src/tests/tools/scripts: 0 hits). `aot_memory_model/kernels/vq_mu_chunk.py`
contains a hand-copied `_fft_weight_contract_diag` replica for AOT memory
modeling, not a call.

### `_v_q_per_q_g_chunked_jit(V_acc, zeta_q_L, zeta_q_R, v_q, g_chunk)` — L562-598
jit (donate V_acc, static g_chunk). Python-unrolled loop of
`lax.dynamic_slice_in_dim` G-chunks; per chunk:
`L_weighted = conj(L_chunk) * v_chunk[None,:]; V += L_weighted @ R_chunk.T`.
Physics: V += Sigma_G conj(zeta_L) v zeta_R^T. Requires ngkmax % g_chunk == 0.

### `compute_v_q_per_q_g_chunked(zeta_q_L, zeta_q_R, v_q, *, g_chunk=4096, V_acc=None)` — L601-676
Public wrapper: shape validation, dtype cast of v_q, zero-init accumulator.
Docstring covers G-flat per-q sphere convention (pad slots zeta=0), bispinor
signed/complex v, sharding inheritance, `TODO(q-batch)` marker.
**ZERO callers anywhere** — the same algorithm is independently implemented as
a `lax.scan` inside `v_q_g_flat.py` (`_g_chunk_body`, L112-123 there). Dead +
redundant.

### `build_v_head_miniBZ_avg_3d(kgrid, bvec, cell_volume, *, nmc=2**18, seed=42)` — L679-725
Host-side MC table `<v(q+dq, G=0)>_miniBZ` for 3D bulk: draws nmc uniform
points, wraps to the Voronoi mini-BZ cell via `gw.vcoul.wrap_points_to_voronoi`
(nmax=1), maps by `randlims = bvec.T @ diag(1/kgrid) @ inv(bvec.T)`, then for
each q on the k-grid (BGW wrap: q > kgrid/2 -> q - kgrid) averages
`8 pi / |q_cart + dq_cart|^2`; q=0 slot = 0. Returns (nkx,nky,nkz)*1/V_cell.
Consumed by the sys_dim=3 kernel above and by `v_q_g_flat.py:517` — docstring
warns "keep them in lock-step".

### `compute_v_q_per_G(q_irr_frac, gvec_components, *, bvec, cell_volume, sys_dim, vcoul_cutoff_ry=None, bdot=None, v_head_miniBZ=None)` — L728-843
Host-side (pure numpy, unjitted) v(q+G) on the writer's per-q WFN.h5-style
G-list `isdf_header/gvec_components` (n_q, 3, ngkmax). Per q:
`qG_cart = bvec.T @ (q_frac[:,None] + gvec[qi])`; sys_dim=3: `8 pi/|qG|^2 /V`,
G=0 Miller slot replaced by `v_head_miniBZ[round(qf*kgrid) % kgrid]`;
sys_dim=2: times f2d as above; sys_dim=0: `NotImplementedError` ("Plumb when
needed"). Optional cutoff zeroing. Explicitly documented as a mirror of
`get_sqrt_v_and_phase` (second source of truth). Consumers: `v_q_g_flat.py:521`,
`v_q_bispinor.py:218`, `tests/test_compute_all_V_q_g_flat.py:106`.

### `compute_all_V_q(zeta_io, *, kgrid, fft_grid, bvec, cell_volume, mesh_xy, n_rmu, n_rtot, sys_dim=2, bdot=None, mc_average_vcoul_body=True, bare_coulomb_cutoff=None, bgw_v_grid_fn=None, mu_chunk_size=None, q_batch_size=None, budget_bytes=None, verbose=True, sym=None, centroid_indices=None, use_g_flat_zeta=False, g_chunk_size=0)` — L846-1119
Top-level dispatcher, called from `gw_init.py:1097`.

1. **G-flat dispatch** (L898-915): if `zeta_io.zeta_layout == 'G_flat'` ->
   `gw.v_q_g_flat.compute_all_V_q_g_flat` (per-q G-chunked contract on writer
   sphere; async prefetch gated by env `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH`,
   default 0).
2. **IBZ orchestration** (L927-972): if `sym` + `centroid_indices` given, build
   `sym_perm (2*ntran, n_rmu)` + umklapp `L_table` via
   `centroid.orbit_syms.compute_centroid_sym_perm(extend_trs=True)` (TRS-aware
   — see reports/trs_sym_audit_2026-05-14); on `RuntimeError` (orbit closure
   failure) fall back to full BZ. If closed, iterate only
   `sym.q_irr_kgrid_int`.
3. Build kernels via `make_v_munu_chunked_kernel`; `v_per_G_fn` =
   vmap(get_sqrt_v_and_phase) then `sqrt_v_batch * sqrt_v_batch` (square back
   to v; unified kernel applies v once on L side — mathematically identical to
   symmetric sqrt-v form for real v >= 0). `phase_fn` is gone (phase now
   applied in reader via `common.wfn_transforms.apply_bloch_phase`).
4. Optional `bgw_overlay_fn` (L1019-1034): host-side per-q replacement of
   native v(q+G) with BGW's MC-averaged grid (`bgw_v_grid_fn`), where
   BGW value != 0, for byte-reproducible comparison.
5. Env `LORRAX_V_Q_MU_CHUNK` forces the tile driver's Case B mu_chunk (debug
   knob). `budget_bytes` default 24.0e9.
6. Dispatch to `gw.v_q_tile.compute_V_q_tile` (same_zeta=True, wants g0).
7. IBZ post-loop unfold (L1078-1109): `unfold_v_q` (common.symmetry_maps) with
   per-centroid umklapp phase from L_table, q wrapped `(q > kg/2 -> q - kg)/kg`;
   `_unfold_g0_ibz_to_full` for g0 ("only Gamma slot consumed downstream").
8. Output sharding constraints: `V_qmunu (nq, n_rmu, n_rmu)` ->
   `P(None,'x','y')`; `g0_mu_all (nq, n_rmu)` -> `P(None,'x')`. Flat-q
   convention (q axis 1-D throughout).

`mu_chunk_size` / `q_batch_size` args are accepted and **ignored** (legacy
back-compat, marked in signature comments).

## Flags / config consumed

- cohsex.in / LorraxConfig (via gw_init call site): `cfg.head.bare_coulomb_cutoff`
  (-> `bare_coulomb_cutoff`; default `wfn.ecutwfc` — the known BGW-mismatch
  default), `cfg.head.mc_average_vcoul_body`, `cfg.memory.per_device_gb` /
  `mem_est['available_vcoul_gb']` (-> `budget_bytes`), `cfg.bispinor`
  (selects the branch that only uses this module's factory),
  `cfg.backend.slab_io` (reader backend, upstream of this module).
- Env vars read directly here: `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH` (default '0'),
  `LORRAX_V_Q_MU_CHUNK` (default '0' = off).

## I/O

No direct file I/O. Consumes zeta through injected loader handles
(`file_io.zeta_reader.ZetaReader` / `SlabIO` over `zeta_q.h5`; G-flat layout
uses `isdf_header/gvec_components`, per-q `ngk`, sentinel-padded zeta=0 slots).
Output V_qmunu/g0 returned in-memory (device, sharded); bispinor tiles written
by `v_q_bispinor`, not here.

## Cross-module deps

`common.timing`, `common.fft_helpers.make_sharded_fftn_3d`,
`common.coulomb_sphere.compute_bare_coulomb_sphere_idx`,
`common.symmetry_maps.unfold_v_q`, `gw.v_q_tile` (compute_V_q_tile,
`_choose_v_q_chunks`, `_unfold_g0_ibz_to_full`),
`gw.v_q_g_flat.compute_all_V_q_g_flat`, `gw.compute_vcoul_0d`
(compute_vcoul_box constants), `gw.vcoul.wrap_points_to_voronoi`,
`centroid.orbit_syms.compute_centroid_sym_perm`.

## Dead suspects

1. `compute_v_q_per_q_g_chunked` + `_v_q_per_q_g_chunked_jit` (L562-676):
   grep `compute_v_q_per_q_g_chunked|_v_q_per_q_g_chunked_jit` over
   src/tests/tools/scripts -> zero hits outside this file. The live
   implementation of the same contract is the `lax.scan` in
   `v_q_g_flat.py:106-123`.
2. Kernel-bundle members `fft_and_weight`, `contract_block`,
   `fft_weight_contract`, `fft_weight_contract_diag`,
   `fft_weight_contract_offdiag`, `fft_and_weight_keep`,
   `fft_and_weight_inner`, `contract_block_inner`: grep for attribute access
   `\.(fft_and_weight|contract_block|fft_weight_contract|...)` -> 0 hits;
   only `sphere_idx` / `n_sph` / `get_sqrt_v_and_phase` of the bundle are
   consumed (v_q_bispinor.py:331,336 + this file). v_q_tile does its own FFT.
3. `_choose_v_q_chunks` re-export (L516-520): no external importer (grep
   `_choose_v_q_chunks` -> only v_q_tile.py defines/calls it, plus this
   re-export).
4. `compute_sqrt_vcoul_0d` / sys_dim=0 path: only reachable via
   `make_v_munu_chunked_kernel(sys_dim=0)`; the per-q G-flat path raises
   `NotImplementedError` for sys_dim=0 (L831-839), and `compute_all_V_q_g_flat`
   rejects sys_dim not in (2,3) (v_q_g_flat.py:505). So 0-D works only on the
   legacy r-space route. Semi-live (used by scripts/checks + coulomb/box_0d via
   compute_vcoul_0d, but the copy here has no external caller).
5. `tests/archive/test_chunked_wfn_loading.py:52-54` imports
   `compute_all_V_q_from_zeta_h5` from `.compute_vcoul` — that symbol no
   longer exists in this file (archived test is import-broken).

## Redundancy suspects

1. Three parallel implementations of "v(q+G) formula": (a) jitted
   `get_sqrt_v_and_phase` (full FFT grid, per q), (b) host-side
   `compute_v_q_per_G` (per-q G-list) — docstring itself says "Mirrors the
   formula inside make_v_munu_chunked_kernel", (c) `v_q_bispinor` re-derives
   variants on top of both. Head table shared but "keep them in lock-step"
   comment (L697) admits manual synchronization.
2. `compute_v_q_per_q_g_chunked` duplicates `v_q_g_flat._g_chunk_body` scan
   kernel (python-unrolled vs lax.scan variants of the same contraction).
3. `fft_and_weight` vs `fft_and_weight_keep` (L429-432 vs L463-466): byte-identical
   bodies (both call `fft_and_weight_inner` with unused static `B_mu`), classic
   "fetch_X next to fetch_X_dyn" cruft.
4. Fused-kernel family `fft_weight_contract` / `_diag` / `_offdiag` (L439-461):
   three parallel variants, none externally consumed.
5. `aot_memory_model/kernels/vq_mu_chunk.py:77-87` hand-copies
   `fft_weight_contract_diag` (intentional for AOT modeling, but a copy).
6. Legacy r-space branch of `compute_all_V_q` (v_q_tile route): module comments
   say "Reachable only for legacy r-space zeta files; not exercised by the
   current writer" — an entire parallel old path kept alive.
7. `mu_chunk_size` / `q_batch_size` accepted-and-ignored legacy args.

## Weird code

- L231: `cache_key` for `_v_munu_kernel_cache` omits `vcoul_cutoff_ry`,
  `mc_average_vcoul_body`, and `mesh_xy` — two calls with different cutoff or
  MC setting but same grids would return the FIRST cached kernel bundle
  (stale closure over cutoff). Hypothesis: latent bug masked because each run
  uses one cutoff; dangerous for tests that sweep cutoffs in-process.
- L207-227: `make_v_munu_chunked_kernel` docstring has the `vcoul_cutoff_ry`
  parameter description spliced mid-sentence between "This creates two
  kernels:" and the enumerated list — copy-paste docstring damage.
- L42-47, L58-62: defensive "NOTE TO FUTURE EDITORS — numpy IS INTENTIONAL /
  DO NOT fix back to jnp" blocks (pjit-compile-storm history, commit bbff26f).
- L158-174: pure-Python triple loop over the whole WFN FFT grid in
  `compute_sqrt_vcoul_0d` (fine for one-shot 0-D, but O(n_G) interpreter
  iterations).
- Magic constants: `budget_bytes = 24.0e9` default (L1052); `g_chunk = 4096`
  (L606); `nmc = 2**18`, `seed = 42` (L684-685); `denom < 1e-12` G=0 threshold
  (L337, L381, L719, L809); sphere-radius enlargement by |q_max| commented at
  L250-257.
- L298/L790: `zc = pi / bvec[2,2]` — 2D slab truncation hard-assumes the
  out-of-plane axis is z and bvec is diagonal in z.
- L1003-1007: `v_per_G_fn` computes sqrt(v) then squares it back
  (`sqrt_v_batch * sqrt_v_batch`) — a documented artifact of reusing the
  sqrt-v kernel after the unified driver switched to one-sided v weighting.
- L349 vs L824: 3D head replacement uses `.at[0,0,0].set` on the FFT box in
  the jit path but a `g0_mask = all(gvec==0)` Miller-slot mask in the host
  path — two index conventions for the same physics.
- L720/L819: q=0, G=0 head deliberately left at 0 — the true head is injected
  via a separate rank-1 correction in Sigma_X (cross-module invariant,
  see head_correction.py:735 and bse_io.py:803 which both re-state it).
- L632: `TODO(q-batch)` marker in the dead `compute_v_q_per_q_g_chunked`.
- L537: module comment self-references "compute_vcoul.py:419-454" — stale line
  numbers after edits.
- Sign/convention notes: Rydberg `8 pi/|q+G|^2` (not 4 pi — Ry units);
  phase `exp(-2j pi q.r)` negative sign; BGW wrap `q > kgrid/2 -> q - kgrid`.
