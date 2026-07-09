# src/gw/qsgw_utils.py — deep-read notes (2026-07-01)

353 LOC. Module docstring (lines 1–18) declares the architectural seam: all post-self-energy
plumbing in `gw_jax` operates on **replicated** `(nk, nb, nb)` arrays; the only sharded object
is the dynamic correlation `Σ_c(ω, k, m_X, n_Y)` from `ppm_sigma` (ω fan-out too large to
replicate). Everything here is structured around that seam.

## Purpose

Post-Σ utilities for the dynamic (PPM/QSGW) path: (1) host-NumPy vectorised ω-axis linear
interpolation, (2) diagonal Σ(E) quasiparticle fixed point E = h0 + Re Σ(E), (3) diagonal
extraction from the sharded Σ_c(ω) tensor, (4) the QSGW static Hermitian Σ_xc build, and
(5) one (apparently dead) QP-energy comparison plot.

Category: **physics: post-Σ QP solve / QSGW Hermitisation stage** (host+device seam utilities).

## Function table

### `interp_along_omega(values_w_kn, omega_grid, eval_kn) -> (nk, nb)` — lines 33–71
- Role: vectorised linear interp of `values_w_kn[ω, k, n]` along ω at per-(k,n) points
  `eval_kn[k, n]`, edge-clamped. One `np.searchsorted` + two fancy-index gathers; no Python loops.
- Math: `out[k,n] = w_lo·V[i_lo(k,n),k,n] + w_hi·V[i_hi(k,n),k,n]`, with
  `i_hi = clip(searchsorted(ω, clip(E, ω0, ωN)), 1, nω−1)`, `w_hi = (E−ω_lo)/(ω_hi−ω_lo)`
  (`denom` guarded with `where(ω_hi>ω_lo, ·, 1.0)` against duplicate grid points).
- Arrays: `values_w_kn (nω, nk, nb)` real or complex, host NumPy; `omega_grid (nω,)`;
  `eval_kn (nk, nb)`. All host-side, replicated.
- Callers (grep across src/, tests/, tools/, scripts/):
  - `src/gw/gw_jax.py:859–866` — freq_debug column `sig_c_head(Edft)` (PPM analytic head
    interpolated at E_DFT − E_F, in Ry then ×RYD_TO_EV).
  - `src/gw/ppm_pipeline.py:222–223` — Σ_c(E_DFT) eqp.dat extractor (`sigma_c_at_dft_ev`).
  - `src/gw/eqp_bgw.py:162–167` — Σ_c(E_DFT) plus ±dE central difference →
    `dReΣ_c/dω` → Z-factor `Z = 1/(1 − dΣ/dE)`.
  - Internal: `solve_diagonal_sigma_fixed_point` (line 129).
- Flags: none.

### `solve_diagonal_sigma_fixed_point(h0_diag_ev, sigma_omega_diag_ev, omega_ev, *, max_iter=80, tol_ev=1e-6, mixing=0.6) -> (E, converged, n_iter)` — lines 78–136
- Role: solve `E_kn = h0_kn + Re Σ_xc(E_kn)` per (k, n) by linear mixing
  `E ← (1−mix)·E + mix·(h0 + Re Σ(E))`. Vectorised over (k, n) via `interp_along_omega`.
  Caller must pre-add static Σ_x diagonal to dynamic Σ_c diagonal (docstring line 97–98).
- Init: `E0 = h0 + Re Σ(ω closest to 0)` (line 124–125: `i0 = argmin|ω|`).
- Out-of-grid bands: Σ evaluated with ω clamped at grid edge; docstring defers patching to
  `gw.scissor` (which gw_jax indeed applies afterward).
- Callers: `src/gw/gw_jax.py:649` — called in Ry despite `_ev` parameter names
  (`h0_diag_ry − efermi_ry`, `sigma_xc_diag_w_kn_ry`, `omega_grid_ry`,
  `tol_ev=1.0e-7/RYD_TO_EV`, `max_iter=120`, `mixing=0.6`). Unit-agnostic math, so correct,
  but the `_ev` suffixes lie at that call site. Also referenced in `src/gw/scissor.py:211`
  docstring (not a call).
- Flags: none directly (gw_jax gates the block on `mode.is_dynamic`).

### `extract_sigma_diag_replicated(sigma_w_kij, mesh_xy) -> jax.Array (nω, nk, nb)` — lines 143–171
- Role: pull `Σ[..., n, n]` from the sharded `(nω, nk, nb, nb)` ω-tensor with sharding
  `P(None, None, 'x', 'y')`. Because m and n live on **different** mesh axes, a naive per-shard
  `einsum('...ii->...i')` is only correct on the `ix == iy` diagonal shards; the function forces
  an allgather to fully-replicated (`with_sharding_constraint` to `P(None,None,None,None)`)
  before the diagonal einsum, then constrains the result replicated 3-D.
- Einsum (verbatim): `jnp.einsum("...ii->...i", M_full)`.
- Memory note in docstring: materialised tensor is `nω·nk·nb²·16 B` per device (~270 MB for
  MoS2 4×4×1, 80 bands, 41 ω); "a shard_map specialisation for the very-large-system case is
  left for follow-up" (line 160) — explicit deferred-work marker.
- Callers: `src/gw/gw_jax.py:639` (diag Σ_c for the fixed point, Ry),
  `src/gw/gw_jax.py:781` (eV diag for eqp/freq-debug output),
  `src/gw/ppm_pipeline.py:203–205` (eqp extractor, non-streamed branch).
- Flags: none.

### `build_qsgw_sigma_xc(sigma_c_omega_ry, sigma_x_kij_ry, omega_ev, e_qp_kn_ev, mesh_xy) -> (sigma_xc_qsgw_kij_ry, diagnostics)` — lines 178–299
- Role: static Hermitian QSGW self-energy ansatz
  `Σ_xc^QSGW_ij(k) = ½[ Σ_xc_ij(k, E_i(k)) + Σ_xc_ij(k, E_j(k)) ]ʰ`, `[·]ʰ` = Hermitian part.
  Σ_xc = Σ_c(ω interp) + Σ_x (ω-independent, added once).
- Implementation: host-side searchsorted/clip/weight prep (lines 240–251, duplicating the
  `interp_along_omega` weight recipe), `device_put` of idx/weight `(nk, nb)` arrays replicated
  (`P(None,None)`; comment lines 255–256: `jnp.asarray` wrap "would force a single-device
  staging that turns device_put into an all-reduce"). JIT `_kernel` (lines 262–285):
  - `A[k,m,n] = Σ_c[idx[k,m], k, m, n]` via `jnp.take_along_axis(sig_w, broadcast ilo[None,:,:,None], axis=0)[0]`
    with weights `wlo[:, :, None]` (interp at E_m(k));
  - `B[k,m,n] = Σ_c[idx[k,n], k, m, n]` via `ilo[None,:,None,:]`, weights `wlo[:, None, :]`
    (interp at E_n(k));
  - `M = 0.5·(A+B) + sig_x`; force replicated `P(None,None,None)`; return
    `0.5·(M + conj(swapaxes(M, −1, −2)))`.
  - Per-shard gathers are local: ω-axis is replicated, idx/weights broadcast to all shards.
    A/B inherit `P(None,'x','y')` before the constraint; replication happens before
    Hermitisation to avoid a sharded transpose comm.
- Energy domain: `omega_ev` and `e_qp_kn_ev` must share a common reference (Fermi-relative;
  ω-grid centered on E_F by `ppm_pipeline`).
- Values in Ry, interp abscissae in eV — mixed on purpose (weights are unit-agnostic as long
  as grid and eval points share units).
- Diagnostics: `n_clipped`, `frac_clipped` (E outside [ω_min, ω_max] clamped), `omega_min/max_ev`.
- Weights cast to complex128 (lines 259–260) to avoid dtype promotion in the weighted sum.
- `sigma_xc_qsgw.block_until_ready()` at line 291 (sync point).
- Callers:
  - `src/gw/gw_jax.py:703` (one-shot G0W0/QSGW step-0 block; passes
    `omega_grid_ry * RYD_TO_EV`, `E_sc_rel_ev`; result Ry → `sigma_total = Σ_xc^QSGW + sig_h`).
  - `src/gw/sigma_dispatch.py:156, 232` (SC/QSGW iteration path via `sc_iteration.py`,
    `screening.py`; requires `e_qp_ev`, passes `ppm_outputs.ppm_options.omega_grid_ev`,
    `e_qp_rel_ev = e_qp_ev − efermi·RYD_TO_EV`).
- Flags: none directly.

### `plot_qp_energy_comparison(output_png, e_ref_kn_ev, e_static_kn_ev, e_dyn0_kn_ev, e_diag_sc_kn_ev) -> str` — lines 306–344
- Role: matplotlib scatter (all (k, n) vs reference, y=x line) + k=0 band trend for three
  approximations: "Static COHSEX", "Bare X + Σ_c(0)", "Diagonal SC Σ(E)". Writes `output_png`
  at dpi=160.
- Callers: **none.** Grepped `plot_qp_energy_comparison` across the entire lorrax_D tree
  (`--include=*.py`): only the definition and `__all__` entry in qsgw_utils.py. Dead suspect.
- I/O: writes a PNG (only file I/O in the module).

## Cross-module dependencies
- Imported by: `src/gw/gw_jax.py` (top-level import of build/extract/solve, lines 48–51;
  lazy `interp_along_omega` at 859), `src/gw/ppm_pipeline.py` (lazy, lines 203, 222),
  `src/gw/eqp_bgw.py` (lazy, line 162), `src/gw/sigma_dispatch.py` (lazy, line 156).
- Imports: numpy, jax, jax.numpy, jax.sharding (Mesh, NamedSharding, PartitionSpec). No
  intra-package imports — leaf module.
- Not referenced by tests/, tools/, or scripts/ directly (grep `qsgw_utils` in tests → none).

## I/O
- No HDF5/dat reads or writes. Only `plot_qp_energy_comparison` writes a PNG. Streamed-mode
  h5 reading seen in caller `ppm_pipeline.py` (sigma_kij_h5) is outside this module.

## Flags consumed
- None. LorraxConfig / cohsex.in are not touched here; callers gate use via `mode.is_dynamic`
  (gw_jax) and `config.ppm.sigma_at_dft_extrapolate` (gw_jax scissor block, adjacent not inside).

## Dead suspects
- `plot_qp_energy_comparison` (lines 306–344): zero callers. Evidence: `grep -rn
  plot_qp_energy_comparison` over the whole lorrax_D repo (`*.py`) returns only
  qsgw_utils.py definition + `__all__`. Likely leftover from an early COHSEX-vs-dynamic
  diagnostic session.

## Redundancy suspects
- **Interp recipe duplicated**: `build_qsgw_sigma_xc` lines 240–251 re-implement the exact
  searchsorted/clip/w_lo/w_hi logic of `interp_along_omega` lines 56–67 (needed host-side to
  produce device idx/weight arrays, but the weight-prep could be factored into one shared
  helper returning (idx_lo, idx_hi, w_lo, w_hi)).
- **Two orchestration sites for the same QSGW build**: `gw_jax.py:698–707` and
  `sigma_dispatch.py:225–236` each do the identical
  `sig_x_rep = jax.device_put(jnp.asarray(sig_x), NamedSharding(mesh_xy, P(None,None,None)))`
  + `build_qsgw_sigma_xc(...)` + `print("QSGW: N clipped (x%)")` block — classic parallel
  old(inline)/new(dispatch) path; sigma_dispatch is used by `sc_iteration.py`/`screening.py`
  while gw_jax keeps its own inline copy for the one-shot path.

## Weird code
- `extract_sigma_diag_replicated` (lines 147–171): deliberately allgathers the full
  `(nω, nk, nb, nb)` tensor to every device just to take a diagonal, because the (m, n) axes
  sit on different mesh axes and a naive per-shard `'...ii->...i'` is silently wrong off the
  mesh diagonal. Comment defers a shard_map specialisation — memory blows up for very large
  systems (nω·nk·nb²·16 B per device).
- `solve_diagonal_sigma_fixed_point` non-converged return (line 136): `np.abs(E_new - E) <
  tol_ev` uses the **unmixed** update E_new vs the mixed E — a different convergence measure
  than the in-loop `|E_next − E|` check (line 132–134). Also `E_new` is referenced after the
  loop, which would `NameError` if `max_iter=0` (harmless with defaults).
- Unit naming: parameters suffixed `_ev` are actually unit-agnostic; `gw_jax.py:649` calls the
  fixed point entirely in Ry with `tol_ev=1.0e-7/RYD_TO_EV`; `build_qsgw_sigma_xc` mixes Ry
  values with eV abscissae by design. The gw_jax comment "kernel internals convert; result is
  Ry" (gw_jax.py:~702) is misleading — the kernel does no unit conversion.
- Interp weights `w_lo/w_hi` cast to `complex128` (lines 259–260) though mathematically real —
  dtype-promotion convenience for the complex Σ tensor.
- `denom = np.where(ω_hi > ω_lo, ω_hi − ω_lo, 1.0)` guard appears in both interp copies
  (lines 65, 249) — protects against duplicate ω grid points; benign but repeated.
- Fixed-point init at `argmin|ω|` (line 124) assumes the ω-grid brackets 0 (Fermi-centered);
  no assertion that it does.
