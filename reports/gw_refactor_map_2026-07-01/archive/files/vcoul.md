# src/gw/vcoul.py — deep-read notes (2026-07-01)

LOC: 234 (of which ~113 are a fully commented-out legacy function).
Module docstring: "Coulomb utilities: Voronoi-cell sampling and per-q V(q,G). This module now
supports Sobol QMC sampling for the q=0 averages by default and keeps the V(q,G) head zero so
head averages are injected explicitly later."

## Purpose

Small utility module for the bare-Coulomb / q=0 head-average side of the GW pipeline. Live
content is: (1) a jitted Voronoi-cell point-wrapping helper used by mini-BZ Monte-Carlo/QMC
averages, (2) a thin compatibility wrapper `compute_q0_averages` that forwards to the
dimension-aware `gw.coulomb.get_kernel(sys_dim).q0_average(...)`, and (3) two apparently dead
functions (`compute_vcoul_comps_for_q`, `compute_wcoul0_with_S`). The heavy V(q,G) machinery
that used to live here (`get_V_qG`) is commented out; its successors live in
`src/gw/compute_vcoul.py` and `src/gw/coulomb/*.py`.

Category guess: **physics: bare Coulomb / q=0 head averages (mostly superseded shim layer)**.

## Imports / cross-module deps

- `common.Meta` (only for type hints / `meta.nkx, nky, nkz`, `meta.sys_dim`).
- `gw.coulomb.get_kernel` (lazy import inside `compute_q0_averages`, line 190).
- Duck-typed `wfn` (uses `wfn.kpoints`, `wfn.get_gvec_nk`) and `sym` = `common.symmetry_maps`
  SymMaps object (`find_qpoint_index` at symmetry_maps.py:1408, `irr_idx_k`, `sym_mats_k`,
  `sym_idx_k`) — only inside dead-suspect `compute_vcoul_comps_for_q`.

## Entry points and callers (grep over src/, tests/, tools/, scripts/)

| Function | Callers found |
|---|---|
| `wrap_points_to_voronoi` | `src/gw/compute_vcoul.py:699,705` (inside `build_v_head_miniBZ_avg_3d`); `src/gw/coulomb/base.py:124,144,158` (q0 sampling in `CoulombKernel` base); internal use in `compute_wcoul0_with_S` (vcoul.py:220); mentioned in `src/gw/PERFORMANCE.md:109` |
| `compute_q0_averages` | `src/gw/head_correction.py:131-134` and `:156-177` (two call sites); `scripts/checks/sigma_direct_check.py:70,172,201`; `scripts/checks/w_from_eps0_0d_check.py:59,124` (docstring at :32 documents its 0D contract vc0==wc0) |
| `compute_vcoul_comps_for_q` | **none** (grep `compute_vcoul_comps_for_q` across src/tests/tools/scripts: only the def at vcoul.py:160) |
| `compute_wcoul0_with_S` | **none** (grep `compute_wcoul0_with_S` and `compute_wcoul0` across src/tests/tools/scripts: only the def at vcoul.py:198) |
| `get_V_qG` | commented out entirely (lines 43-155); grep `get_V_qG` finds only vcoul.py:43 |

No `python -m` usage of this module found.

## Function-by-function

### `wrap_points_to_voronoi(randcart, bvec, nmax=1)` — lines 17-40
- Decorator: `@functools.partial(jax.jit, static_argnames=('nmax',))`. Docstring explains the
  jit exists to collapse ~10 eager-pjit cache misses per call into one cached XLA module
  (cross-ref PERFORMANCE.md:109: "~32 misses → 1").
- Role: wrap Cartesian sample points into the Voronoi cell of the reciprocal lattice: builds
  candidate shifts `n·B` for n ∈ {-nmax..nmax}³ (`shifts @ bvec_j`, shape (M,3)), picks the
  nearest shift per point (argmin over `||r - shift||`), returns `r - shift_best`.
- Physics: minimum-image / first-BZ wrap: q_wrapped = q − argmin_{G∈lattice} |q − G|.
- Arrays: randcart (N,3) float64 device; bvec (3,3); returns (N,3) float64 device. No sharding.
- Flags: none.

### commented-out `get_V_qG(wfn, sym, q0, epshead, sys_dim, meta, do_Dmunu=False, mesh_xy=None)` — lines 43-155 (ALL COMMENTED)
Legacy all-q bare-Coulomb builder, kept as a comment block. Notable contents (for archaeology):
- V(q,G,G') = 4π/|q+G|² δ_GG' with 2D slab truncation; 3D raised NotImplementedError (line 56).
- Rydberg convention `v = 8.0*jnp.pi/denom` (line 97) times 2D truncation factor
  `(1 - exp(-zc·k_xy)·cos(k_z·zc))` with `zc = π/bvec[2,2]` (slab half-height along z).
- Optional Breit block `npol=4`: transverse projector `proj = I − q̂q̂ᵀ` stored in
  `Vq[1:4,1:4,:]` (lines 105-113) via `jax.lax.cond(do_Dmunu, ...)`.
- mini-BZ Monte Carlo with hardcoded 2_500_000 uniform samples (line 125) for
  V(q=0,G=0) mean and Ismail-Beigi wcoul0:
  `gamma = (1/eps_head − 1)/(q0² vc(q→0))`, `wq = vc_q/(1 + vc_q·k_xy²·gamma)`,
  `wcoul0 = 8π·mean(wq)` (lines 136-140).
- einsum (commented, line 131): `jnp.einsum("ij,ij->i", randqcart, randqcart)`.
- G-axis Y-sharding via `NamedSharding(mesh_xy, P(None,None,None,'y'))` (line 153) —
  references `NamedSharding`/`P`/`partial` that are NOT imported in the live module (proof it
  cannot be un-commented as-is).
- Superseded by: `compute_vcoul.py:compute_v_q_per_G / compute_all_V_q /
  build_v_head_miniBZ_avg_3d` and `gw/coulomb/{base,slab_2d,bulk_3d,box_0d}.py`.

### `compute_vcoul_comps_for_q(wfn, sym, meta, qvec_nonneg)` — lines 160-172
- Role: for a single q given in non-negative grid coordinates, fold into the symmetric BZ
  (`qvec_wrapped = where(q > kgrid//2, q−kgrid, q)/kgrid`, line 164 — note `kgrid` is float so
  `//` is float floor-div), find its full-BZ index `iq = sym.find_qpoint_index(qvec_wrapped)`,
  its IBZ rep `iqbar = sym.irr_idx_k[iq]`, rotation `Sq = sym.sym_mats_k[sym.sym_idx_k[iq]]`,
  umklapp `G_Sq = round(q_wrapped − Sq @ k_iqbar)`, and rotate the IBZ G-list:
  einsum VERBATIM (line 171):
  `np.einsum("ij,kj->ki", Sq.astype(jnp.int32), wfn.get_gvec_nk(iqbar)) - G_Sq[np.newaxis, :]`
  i.e. G'_k,i = Σ_j Sq_ij G_k,j − G_Sq (rotate-then-shift of integer G components).
- Returns `(iq_cpu, iqbar, qvec_wrapped, vcoul_comps.astype(jnp.int32))`; vcoul_comps shape
  (nG,3) int32, host numpy.
- Oddity: `.astype(jnp.int32)` / `jnp.float64` applied to numpy arrays (works, but mixes
  jnp dtypes into np.einsum); `int(iq) if not hasattr(iq, "get") else int(iq.get())`
  (line 167) is a cupy-style `.get()` guard that is dead weight under JAX.
- Callers: **none found** → dead suspect. Note memory `feedback_unified_sym_action.md`: this is
  exactly a parallel "rotate X at q" helper of the kind slated for consolidation.

### `compute_q0_averages(wfn, epshead, meta, S_cart=None, nsamples=2**18, method="sobol", qmc_reps=10)` — lines 175-195
- Role: compute (vc0_mean, wcoul0) — the mini-BZ average of the bare Coulomb v(q) over the
  q=0 Voronoi cell and the screened head W(q=0) average. Now a **thin compatibility wrapper**:
  `return get_kernel(getattr(meta,'sys_dim',None)).q0_average(wfn, meta, S_cart=..., epshead=...,
  nsamples=..., method=..., qmc_reps=...)`. Docstring: "The branching logic that used to live
  here is now distributed across the per-dim kernel modules."
- Flags/keys: `meta.sys_dim` (via getattr with None default), `meta.nkx/nky/nkz` consumed
  downstream by the kernels. Defaults nsamples=2**18, method="sobol", qmc_reps=10 (these map to
  the QMC settings; the real flag surface is in gw/coulomb/*).
- Callers: gw/head_correction.py (twice), scripts/checks/sigma_direct_check.py,
  scripts/checks/w_from_eps0_0d_check.py.
- Refactor note: pure forwarding shim; callers could import `gw.coulomb.get_kernel` directly.

### `compute_wcoul0_with_S(bvec, nkx, nky, nkz, S_cart, nsamples=2_500_000)` — lines 198-234
- Role/physics: Ismail-Beigi-style generalized head average with a small-q tensor S(ω):
  wcoul0 = ⟨ v(q) / (1 − v(q)·qᵀ S q) ⟩ over the Γ mini-BZ Voronoi cell, with 2D slab
  truncation `v(q) = (4π/|q|²)·2·(1 − exp(−zc·k_xy)·cos(k_z·zc))`, `zc = π/bvec[2,2]`
  (4π·2 = 8π Rydberg; hard-wired 2D — no sys_dim branch).
- Sampling: plain uniform MC, `jax.random.PRNGKey(0)` fixed seed, default 2_500_000 samples;
  scale wrapped points into the Γ-tile via `(bvec.T @ (diag(1/kgrid) @ inv(bvec.T)) @ wrapped.T).T`
  (line 223).
- einsums VERBATIM: line 227 `jnp.einsum('ij,ij->i', randqcart, randqcart)`;
  line 230 `jnp.einsum('qi,ij,qj->q', randqcart, S, randqcart)`.
- Returns scalar complex128 (device→host on cast).
- Callers: **none found** → dead suspect. The S-tensor head average now lives behind
  `compute_q0_averages(S_cart=...)` → `gw.coulomb.*.q0_average`, making this a leftover
  parallel path (module docstring itself says Sobol QMC is the default; this is uniform MC).

## I/O

None. Pure in-memory compute; no files read or written, no HDF5.

## Flags consumed

- `meta.sys_dim` (getattr, default None) — dispatch to gw.coulomb kernel.
- `meta.nkx, meta.nky, meta.nkz` — mini-BZ tile scaling (also in dead functions).
- No LorraxConfig / cohsex.in keys touched directly in this file.

## Suspects

### Dead
1. `compute_vcoul_comps_for_q` (lines 160-172): grep for the name across
   src/, tests/, tools/, scripts/ → only its definition.
2. `compute_wcoul0_with_S` (lines 198-234): grep for `compute_wcoul0` across the same trees →
   only its definition.
3. `get_V_qG` (lines 43-155): entire function is commented out; grep for `get_V_qG` → only the
   commented def line. References `partial`, `NamedSharding`, `P` which are not imported.

### Redundancy
- `compute_wcoul0_with_S` duplicates the S-tensor wcoul0 averaging path that
  `compute_q0_averages(S_cart=...)` → `gw/coulomb/*.q0_average` now owns (uniform-MC old path
  next to Sobol-QMC new path — the classic parallel-path pattern).
- `compute_q0_averages` itself is declared a "thin compatibility wrapper"; once the 4 call
  sites (head_correction.py + 2 scripts) import `gw.coulomb.get_kernel` directly it can go.
- Commented `get_V_qG` overlaps `compute_vcoul.py` (`compute_v_q_per_G`, `compute_all_V_q`,
  `build_v_head_miniBZ_avg_3d`) and `gw/coulomb/` kernels, including a second copy of the
  Voronoi-MC head average.
- `compute_vcoul_comps_for_q` is a private "rotate G-list at q" helper parallel to SymMaps
  canonical sym-action machinery (see unified-sym-action memory).

### Weird
- vcoul.py:125 & :204 — magic constant 2_500_000 MC samples (commented block and live default).
- vcoul.py:234 & :142(commented) — fixed RNG seed `jax.random.PRNGKey(0)`; every call gives
  identical "random" average (deterministic, but no rep-averaging like qmc_reps elsewhere).
- vcoul.py:97 vs :131/:227 — 8π vs 4π·2 forms of the Rydberg Coulomb kernel side by side.
- vcoul.py:164 — `kgrid // 2` floor-division on a float64 array for the BZ fold; correct but
  subtle for odd grids.
- vcoul.py:167 — `hasattr(iq, "get")` cupy-era guard under JAX.
- vcoul.py:171 — jnp dtypes fed into np.einsum / np.round (`Sq.astype(jnp.int32)`,
  `.astype(jnp.float64)`) — works because jnp dtypes alias numpy dtypes, but inconsistent.
- vcoul.py:213/:228 — `zc = π/bvec[2,2]` assumes c-axis ⟂ slab and z-aligned; 2D truncation
  hard-coded in `compute_wcoul0_with_S` with no sys_dim dispatch.
- File uses tab indentation (most of the codebase uses spaces).
