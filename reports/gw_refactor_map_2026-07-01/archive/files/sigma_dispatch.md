# src/gw/sigma_dispatch.py (253 LOC)

Deep-read notes for the GW refactor map, 2026-07-01. Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.

## Purpose

Mode-orthogonal Σ_xc dispatcher. Single entry point `compute_sigma_xc(mode, ...)` that the
QSGW self-consistency iteration map calls each step regardless of compute mode
(`X_ONLY`, `COHSEX`, `GN_PPM`, `HL_PPM`). Owns **no compute of its own** (per its own
docstring): every kernel lives in `cohsex_sigma` (static channels), `ppm_pipeline`
(dynamic Σ_c(ω)) or `qsgw_utils` (QSGW Hermitisation). It selects kernels, wires the
`W_by_role` screened-Coulomb dict into them, and packs everything into a frozen
`SigmaResult` dataclass with a uniform shape contract.

**Category:** physics orchestration — Σ stage dispatch / result container (glue between
screening stage and Σ kernels inside the SC loop).

## Entry points and callers (grep evidence)

Grepped `sigma_dispatch|compute_sigma_xc|SigmaResult` across `src/`, `tests/`, `tools/`,
`scripts/`:

- `compute_sigma_xc <- src/gw/sc_iteration.py:294` (inside `gw_iteration_map`, the per-QSGW-iteration step; only real call site). Chain: `gw_jax.main → sc_iteration.run_self_consistency → gw_iteration_map → compute_sigma_xc`.
- `SigmaResult <- src/gw/sc_iteration.py:73,138` (import + `SCState.last_sigma_result: SigmaResult | None` field; final result consumed by `gw_jax.main`, `dump_sigma_omega_h5_final`).
- Docstring/comment references only (no imports): `src/gw/screening.py:11,13,81` (documents the "add a mode = add a screening request + a dispatch case" contract), `src/gw/gw_jax.py:409,441,478,586,591,595`, `src/gw/ppm_pipeline.py:385`.
- **No** imports found under `tests/`, `tools/`, `scripts/`.
- The **one-shot** (non-SC) path in `gw_jax.py:358` does **not** go through this dispatcher — it calls `cohsex_sigma.compute_cohsex_sigma` directly with extra kwargs (`Gij`, `do_screened=config.do_screened`, `wfns_transverse`, `bispinor_v_q_path`, `backend`). See redundancy suspects.

## Function/dataclass table

### `SigmaResult` (dataclass, frozen) — lines 39–81

Result container for one full Σ pipeline call. Fields (all Ry unless suffixed `_ev`):

| field | shape / type | populated by | residency/sharding (as documented) |
|---|---|---|---|
| `v_h_kij_ry` | (nk, nb, nb) jax.Array | all modes | replicated |
| `sigma_x_kij_ry` | (nk, nb, nb) jax.Array | all modes | replicated |
| `sigma_xc_kij_ry` | (nk, nb, nb) jax.Array | all modes; static: Σ_SX+Σ_COH (with head); PPM: Σ_x + Σ_c^QSGW | replicated |
| `sigma_sx_kij_ry` | (nk, nb, nb) or None | COHSEX only | — |
| `sigma_coh_kij_ry` | (nk, nb, nb) or None | COHSEX only | — |
| `sigma_c_omega_kij_ry` | (nω, nk, nb, nb) or None | PPM only | device, sharded `P(None,None,'x','y')` per docstring |
| `sigma_c_at_dft_diag_ev` | (nk, nb) np.ndarray | PPM only | host |
| `omega_dft_rel_ev` | (nk, nb) np.ndarray | PPM only | host; E_DFT − E_F (eV) |
| `omega_grid_ev` / `omega_grid_ry` | (nω,) np.ndarray | PPM only | host |
| `head_sigma_diag_w_kn_ry` | (nω, nk, nb) np.ndarray | PPM only | host; PPM analytic head diagonal |
| `sigma_omega_h5_path` | str or None | PPM only | on-disk Σ_c(ω) HDF5 path |

Docstring convention: `H_QP = kin_ion + V_H + Σ_xc`; `sigma_c_omega_kij_ry` drives the
eqp1 Z-factor central difference downstream.

### `compute_sigma_xc(...)` — lines 88–250

Signature (line 88): `compute_sigma_xc(mode: ComputeMode, *, wfns, V_q, W_by_role: dict, e_qp_ev, static_head_terms, head_resolver, quad, e_ref, config, meta, mesh_xy, sym, wfn, band_slices, input_dir, write_sigma_omega_h5=True, print_fn=print) -> SigmaResult`.

Lazy imports at 154–156: `cohsex_sigma.{compute_cohsex_sigma, compute_v_h_sigma_x}`,
`ppm_pipeline.compute_ppm_sigma_pipeline`, `qsgw_utils.build_qsgw_sigma_xc`.

Control flow:

1. **L163** `W_static = W_by_role.get("static", V_q)` — silent fallback to bare Coulomb (see weird_code).
2. **L164–177 static channels.** If `mode is ComputeMode.COHSEX`: `compute_cohsex_sigma(wfns, V_q, W_static, meta, mesh_xy, Gij=None, do_screened=True, static_head_terms=..., compute_bare_x=True)`. Else (V-only path, per comment giving each path its own jit-cached graph): `compute_v_h_sigma_x(wfns, V_q, meta, mesh_xy, Gij=None, static_head_terms=...)`. Both return a dict with `sig_h` (V_H), `sig_x` (bare exchange Σ_X = −G·V), `sig_sx`, `sig_coh` (zero placeholders on the V-only path, per comment at L180).
3. **L183–189 X_ONLY:** `sigma_xc = sig_x`; return SigmaResult with only the three always-populated fields.
4. **L190–198 COHSEX:** `sigma_xc = sig_sx + sig_coh` (Σ_SX and Σ_COH each already include their q→0 head per docstring); return with sx/coh diagnostics.
5. **L200–208 PPM guards:** raises `ValueError` if `e_qp_ev is None` (QSGW Σ_c(E_m,E_n) evaluation needs QP energies); raises `KeyError` if `"probe" not in W_by_role`. NOTE: no analogous guard for `"static"` (fallback at L163 instead).
6. **L210–222 dynamic Σ_c:** `compute_ppm_sigma_pipeline(wfns=..., V_q=..., W_static_q=W_static, W_probe_q=W_by_role["probe"], sig_x=..., sig_h=..., quad=..., e_ref=..., config=..., meta=..., mesh_xy=..., head_resolver=..., band_slices=..., wfn=..., sym=..., input_dir=..., write_sigma_omega_h5=..., print_fn=...)`. Two-point plasmon-pole fit anchored at W(ω=0) ("static" role) and W at the GN (iω_p) / HL (real Ω) probe frequency ("probe" role).
7. **L226–235 QSGW build:** `omega_grid_ev` from `ppm_outputs.ppm_options`; `e_qp_rel_ev = e_qp_ev − wfn.efermi·RYD_TO_EV` (L229; `wfn.efermi` is in Ry); `sig_x_rep = jax.device_put(jnp.asarray(sig_x), NamedSharding(mesh_xy, P(None, None, None)))` (explicit full replication of the (nk,nb,nb) Σ_X, L230–231); `sigma_xc_qsgw, qsgw_diag = build_qsgw_sigma_xc(ppm_outputs.sigma_c_omega, sig_x_rep, omega_grid_ev, e_qp_rel_ev, mesh_xy)`. Comment at L224–225: Σ_x is added *inside* the kernel, so `sigma_xc_qsgw` already includes Σ_x. The QSGW Hermitisation this implements is the mode-B-style Σ_xc^QSGW ≈ ½{Σ_c(E_m) + Σ_c(E_n)}_mn + Σ_x (kernel lives in `qsgw_utils`, not here).
8. **L236–237:** prints QSGW clip diagnostic: `n_clipped` / `frac_clipped` from `qsgw_diag` (energies clipped to the ω-grid range in the interpolation).
9. **L239–250:** pack PPM `SigmaResult`.

**Physics equations touched (all delegated):** Σ_X = −G·V (X_ONLY); Σ_xc^COHSEX = Σ_SX + Σ_COH at W(ω=0); PPM: Σ_c(ω) from two-point pole fit {W(0), W(ω_probe)}; QSGW: Σ_xc = Σ_x + symmetrised Σ_c(E_QP) with out-of-grid clipping.

**Einsums:** none in this file (pure orchestration).

**Flags / config consumed directly:** `mode` (= `config.compute_mode`, passed explicitly by caller), `wfn.efermi`, `write_sigma_omega_h5` kwarg. `config` is passed through opaquely to `compute_ppm_sigma_pipeline`. Notably `config.do_screened`, `Gij`, and all bispinor flags are **not** consulted (see weird_code / redundancy). Roles read from `W_by_role`: `"static"`, `"probe"` (contract defined jointly with `gw.screening.screening_requests_for`).

**Key arrays crossing the boundary:** `V_q` (bare Coulomb, flat-q ISDF basis, device); `W_by_role[role]` (screened W in same basis, device); `e_qp_ev` (nk,nb host np); `sig_h/sig_x/sig_sx/sig_coh` (nk,nb,nb device, replicated); `ppm_outputs.sigma_c_omega` (nω,nk,nb,nb device, `P(None,None,'x','y')`).

## I/O

None performed directly by this module. Indirect: `write_sigma_omega_h5` (default True; sc_iteration passes False during intermediate iterations to avoid thrashing) toggles the Σ_c(ω) HDF5 dump inside `ppm_pipeline`; the resulting path is surfaced as `SigmaResult.sigma_omega_h5_path` (converged tensor written once post-SC via `sc_iteration.dump_sigma_omega_h5_final`).

## Dead suspects

None. Both public symbols (`compute_sigma_xc`, `SigmaResult`, the whole `__all__`) are imported and used by `src/gw/sc_iteration.py:73` (grep of `src`, `tests`, `tools`, `scripts` as above).

## Redundancy suspects

1. **Two parallel COHSEX call sites with divergent capabilities.** `gw_jax.py:358` (one-shot path) calls `compute_cohsex_sigma` directly with `Gij=Gij, do_screened=config.do_screened, wfns_transverse=..., bispinor_v_q_path=..., backend=config.backend.slab_io`; this dispatcher (SC path, L165–171) calls it with `Gij=None, do_screened=True` hardcoded and **no** bispinor/backend kwargs. Consequences: (a) the SC path silently drops the bispinor Σ^B channel (matches the known "SC path silently drops Σ^B" issue); (b) `do_screened=False` and custom occupation projectors `Gij` are unreachable via SC. A refactor should make gw_jax's one-shot call go through this dispatcher (or vice versa) — single source of truth.
2. **`compute_cohsex_sigma` vs `compute_v_h_sigma_x` branch pair** (L164–177): deliberate per the comment (V-only path never touches W kernels, separate jit-cached graphs), but it is exactly the "fetch_X next to fetch_X_dyn" shape — worth confirming during refactor that `compute_v_h_sigma_x` is a true subset entry point inside `cohsex_sigma` and not a copy.

## Weird code

1. **L163 `W_static = W_by_role.get("static", V_q)`** — silent fallback to *bare* Coulomb if the screening stage produced no `"static"` role. Asymmetric with L205–208 where a missing `"probe"` raises `KeyError`. For X_ONLY the fallback value is unused; for COHSEX/PPM a missing "static" would silently substitute unscreened V into Σ_SX/Σ_COH or the PPM ω=0 anchor, producing wrong-but-plausible numbers instead of an error. Hypothesis: convenience default so X_ONLY needn't populate the dict; should be tightened to per-mode required-role validation.
2. **L180 comment "zero placeholders for V-only path"** — `compute_v_h_sigma_x` returns zero `sig_sx`/`sig_coh` arrays that are then never used on the X_ONLY/PPM branches (dead allocations of shape (nk,nb,nb)); packing zeros only to unpack and discard them is dict-shape-compat cruft.
3. **L166 / L174 `Gij=None`, L169 `do_screened=True` hardcoded** — the SC dispatch pins choices that are config-driven on the one-shot path (`config.do_screened` at gw_jax.py:360). Hypothesis: dispatch was written for the SC loop only and never absorbed the one-shot knobs.
4. **L230–231 `sig_x_rep = jax.device_put(..., NamedSharding(mesh_xy, P(None, None, None)))`** — explicit re-replication of Σ_X across the whole mesh before `build_qsgw_sigma_xc`, while `sigma_c_omega` stays sharded `P(None,None,'x','y')`. Deliberate (kernel presumably needs replicated Σ_x against sharded Σ_c), but it is an (nk,nb,nb) replicated buffer per device — flagged given the zero-replicated-intermediates principle.
5. **L229 unit mix** — `e_qp_ev` (eV) minus `float(wfn.efermi)` (Ry) × `RYD_TO_EV`: correct but the eV/Ry conversion happens inline at the call boundary rather than in the kernel; a known class of sign/unit hazard in this codebase.
6. No TODO/FIXME/ponytail markers, no commented-out code blocks in the file.
