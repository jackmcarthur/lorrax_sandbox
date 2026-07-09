# src/gw/minimax_screening.py (935 LOC)

Deep-read notes for the GW refactor map, 2026-07-01. All line numbers refer to
`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/minimax_screening.py`.

## Purpose

Minimax-quadrature factory for the static/imag-freq/real-freq chi0 pipelines and
for sigma's crossing/non-crossing tau windows, plus Godby-Needs PPM parameter
extraction from precomputed chi or W^c matrices. It wraps the exact solvers in
`common.minimax` with two acceleration layers: shipped precomputed tables
(`common/minimax_assets/catalog.json` + `.npz`) and a persistent per-user disk
cache (`~/.cache/lorrax/minimax_quadratures`). Contains **no** FFT/einsum kernels —
by design it only produces `(tau, alpha)` node sets consumed by `w_isdf.compute_chi0`
and `ppm_sigma`.

Category: **physics support: quadrature/PPM parameter factory for chi0/W and sigma stages**
(numerics + small I/O cache layer; no device-resident heavy arrays except the GN-PPM
elementwise fit).

## Module header

- Module docstring (lines 1–7): explicitly scoped to the static path; "Reuse existing
  sharded kernels (no duplicate FFT kernels here)".
- `_TINY = 1.0e-12` (line 29): floor for `x_min` (see weird code).
- Imports: `common.minimax` (exact solvers), `gw.minimax_config.MinimaxConfig`,
  `jax.experimental.multihost_utils`.

## Function-by-function table

### Host/gather helpers

| Function | Lines | Role |
|---|---|---|
| `_to_host_np(a, dtype=complex128)` | 32–37 | Gather possibly-sharded JAX array to host numpy via `multihost_utils.process_allgather(tiled=True)`; **bare `except Exception`** falls back to `jax.device_get`. Used only by `extract_gn_ppm_parameters` (the legacy chi-based path). |
| `_scalar_to_host_float(a)` | 40–46 | Multihost-safe scalar fetch: allgather when `process_count()>1`, else `device_get`. Used by `fit_gn_ppm_from_wc_pair` for `unfulfilled_fraction`. |

### Shipped-table layer (package data)

| Function | Lines | Role |
|---|---|---|
| `_minimax_disk_cache_dir()` | 49–59 | Resolves persistent cache dir. Env vars: `LORRAX_DISABLE_MINIMAX_DISK_CACHE` (1/true/yes disables), `LORRAX_MINIMAX_CACHE_DIR` (override); default `~/.cache/lorrax/minimax_quadratures`. Creates it. |
| `_load_shipped_minimax_catalog()` | 62–76 | `@lru_cache(1)`. Loads `common/minimax_assets/catalog.json` via `importlib.resources`. Returns None on any error (three separate silent except paths). |
| `_load_shipped_minimax_table(entry)` | 79–94 | Loads one `.npz` referenced by `entry["file"]` (relative to `common/minimax_assets/`). Datasets: `tau` (f64), `alpha` (f64), `max_error` (scalar). Silent None on any exception. |
| `_find_shipped_table_entry(family, *, range_value, target_error, max_nodes, target_kind=None, eps_q=None)` | 97–156 | Selects best catalog entry: `range_max >= range_value` (tol 1e-12), `error_bound <= target_error` (tol 1e-18), `node_count <= max_nodes`, exact `target_kind` string match, `eps_q` match within 1e-12. Sort key `(entry_range, -entry_err, node_count)`: smallest sufficient range, then **loosest** acceptable error, then fewest nodes. Called by `_pick_shipped_table` and directly by `tests/test_minimax_assets.py` (lines 36, 90). |
| `_pick_shipped_table(...)` | 159–189 | Thin compose of the two above. Docstring restates the conservative selection rule (subset-interval safety argument). |

### Disk-cache layer

| Function | Lines | Role |
|---|---|---|
| `_minimax_disk_cache_path(namespace, payload)` | 192–198 | Cache filename = `{namespace}_{sha256(json(payload,sorted))}.npz`. |
| `_load_minimax_disk_cache(...)` | 201–212 | Loads `tau`, `w`, `err` from the npz; silent None on error. |
| `_store_minimax_disk_cache(...)` | 215–240 | Atomic write: `path.tmp.{pid}` then `os.replace`; nested silent `except Exception: pass` cleanup. |

Note asymmetry vs shipped tables: disk-cache npz keys are `tau/w/err` while shipped
npz keys are `tau/alpha/max_error`.

### Dataclasses

| Class | Lines | Role |
|---|---|---|
| `EnergyWindow` | 243–254 | frozen dataclass `(start_energy, end_energy, index=0, count=1)`; property `upper_inclusive` (index >= count-1). Declared "compatible with `w_isdf.compute_chi0`" but no external consumer reads it (see dead suspects). |
| `MinimaxWindowPair` | 257–287 | `(val_window, cond_window, epsq, tau_i, w_i, z_lm, alpha_i)`. Method `with_imag_freq_modulation(omega_imag)` (269–287): returns copy with `w_i = alpha_i * exp(-tau_i) * cos(omega_imag * tau_i)`, implementing the Laplace identity `Delta/(Delta^2 + w^2) = int exp(-Delta t) cos(w t) dt`; combined resonant+antiresonant factor `-2*Delta/(Delta^2+omega^2)`. **No callers anywhere** — and `build_imag_freq_minimax_window_pair`'s docstring (787–788) says the fresh-node approach "replaces the incorrect cos-reweighting of static nodes", i.e. this method is the deprecated, known-incorrect path left in place. |
| `MinimaxNodes` | 290–305 | frozen dataclass `(t: complex128 (n,), alpha: complex128 (n,))`, registered as a JAX pytree via `jax.tree_util.register_dataclass` (data fields t, alpha). The one boundary-crossing type: closed over / passed into jits in `w_isdf` (lines 127, 602, 641) and `ppm_sigma` (lines 156, 786, 861–863, 877, 1259). Replicated across devices (per w_isdf.py:126 comment). |
| `LaplaceMinimaxQuadrature` | 352–374 | frozen: `(x_min, x_max, tau, alpha, max_error)` for `1/x` on `[x_min,x_max]`; `node_count` property; `to_minimax_nodes(time_axis=...)` → `_laplace_to_minimax_nodes`. Also **constructed externally**: `w_isdf.py:549` builds a fused mixed-sign-tau quadrature by hand for the HL-PPM real-axis path. |
| `CrossingMinimaxQuadrature` | 377–394 | frozen: `(A_dim, tau, alpha, max_error, target_kind)`; `to_minimax_nodes(time_axis='crossing_hgl')`. |
| `GodbyNeedsPPM` | 397–405 | frozen: `(omega_p, omega_qmunu, b_qmunu, valid_qmunu, unfulfilled_fraction)`; jnp arrays. Returned by both extract paths. |

### Node-conversion helpers

| Function | Lines | Role |
|---|---|---|
| `_laplace_to_minimax_nodes(tau, alpha, *, time_axis)` | 308–328 | `'real'` → `t = tau + 0j` (chi0 Laplace, exp(-t*dE) real); `'imag'` → `t = -1j*tau` (sigma Laplace windows single/a_stripe/b_slab); else ValueError. alpha cast to complex128. |
| `_crossing_to_minimax_nodes(tau, alpha, *, time_axis)` | 331–349 | Only accepts `'crossing_hgl'`; keeps tau real (cast complex). Caller-side rescale by 1/xi is applied externally (ppm_sigma.py:863 `MinimaxNodes(t=raw.t / xi, alpha=raw.alpha / xi)`). Essentially a validation wrapper around the same two casts as the 'real' branch above (mild redundancy). |

### GN-PPM fits

| Function | Lines | Role / physics |
|---|---|---|
| `fit_gn_ppm_from_wc_pair(Wc0_qmunu, Wc_probe_qmunu, probe_omega, *, fallback_omega)` | 408–457 | Elementwise GN-PPM fit on already-sharded `(nkx,nky,nkz,n_rmu,n_rmu)` complex128 tensors; pure local algebra, no host gathers except the final scalar mean. Physics: `ratio = W(z)/(W(0)-W(z))`; `Omega^2 = -z_probe^2 * ratio`; valid iff `|W(0)-W(z)|>1e-14`, `Re(Omega^2)` finite and `>0`; `Omega = sqrt(Re(Omega^2))` else `fallback_omega`; residue `B = -0.5 * W(0) * Omega`. Returns `(omega_vals, B_vals, good_mask, unfulfilled_fraction)`. Callers: `ppm_sigma.py:690`; internal `extract_gn_ppm_parameters_from_Wc`. |
| `extract_gn_ppm_parameters(V_qmunu, chi0_q, chi_iwp_q, *, omega_p, fallback_omega=2.0)` | 838–896 | **Legacy chi-based path.** Gathers V, chi0, chi(i*omega_p) to host (`_to_host_np`), squeezes 7D `[:, :, :, 0, :, 0, :]` → `(n_q, n_rmu, n_rmu)`, then per-q host loop: `Pi = chi (I - V chi)^{-1}` via `np.linalg.solve(A.T, chi.T).T` ("right-side solve via transpose for stability"); `ratio = Re[Pi_i/(Pi_0 - Pi_i)]`, `Omega = omega_p * sqrt(ratio)` where valid else fallback; `B = -0.5 * Pi_0 * Omega`. NB: B is built from the **polarizability** Pi_0 here, from **W^c(0)** in the Wc path — different objects, different consumers by construction. **Zero external callers** (see dead suspects). |
| `extract_gn_ppm_parameters_from_Wc(Wc0_q, Wc_iwp_q, *, omega_p, fallback_omega=2.0)` | 899–935 | Current path. Accepts flat-q `(nq, mu, mu)` (ndim==3) or 7D `(nkx,nky,nkz,1,mu,1,mu)` (squeezed via `[:, :, :, 0, :, 0, :]`); delegates to `fit_gn_ppm_from_wc_pair` with `probe_omega = 1j*omega_p`; wraps result in `GodbyNeedsPPM`. Caller: `scripts/checks/sigma_direct_check.py:504` (only). |

### Cached exact-solver wrappers (host-side, expensive)

| Function | Lines | Role |
|---|---|---|
| `_solve_noncrossing_scaled_cached(logR_key, target_key, max_nodes)` | 460–482 | `@lru_cache(64)` + disk cache namespace `"noncrossing"`. Calls `common.minimax.noncrossing_grids(R=exp(logR_key), target, N_start=2, N_max=max_nodes)` for `1/y` on `[1, R]`. Also called directly by `tools/generate_minimax_assets.py:350` (private-API use by the asset generator). |
| `_solve_noncrossing_imag_scaled_cached(logR_key, omega_hat_key, target_key, max_nodes)` | 485–512 | `@lru_cache(64)` + disk namespace `"noncrossing_imag"`. Calls `common.minimax.noncrossing_imag_grids(R, omega_hat, target, ...)` for `y/(y^2+omega_hat^2)` on `[1,R]`. **No shipped-table family exists for this** (imag path never consults the catalog). |
| `_solve_crossing_scaled_cached(A_key, target_key, max_nodes, eps_q_key, target_kind)` | 515–557 | `@lru_cache(128)` + disk namespace `"crossing"`. Dispatches `target_kind` to `common.minimax.{G_hgl, tau_max_hgl}` or `{G_fermi, tau_max_fermi}`; calls `crossing_grids(A_dim, target, G_func, tau_max_func, eps_q=eps_q, N_max=max_nodes)`. Also called directly by `tools/generate_minimax_assets.py:301`. |

### Public solver entry points

| Function | Lines | Role / physics / callers |
|---|---|---|
| `solve_laplace_minimax_interval(x_min, x_max, *, target_error=1e-6, max_nodes=64, use_shipped_tables=True)` | 560–615 | Fits `1/x ≈ sum_l alpha_l exp(-tau_l x)` on `[x_min, x_max]`. Error convention (documented 570–577): solver works on scaled `[1, R]`, `R = x_max/x_min`; `target_error` is L-inf **absolute** error on the scaled problem; physical absolute error after rescale = `target_error / x_min` ("not a relative-at-endpoint tolerance"). Clamps `x_min >= _TINY`, `x_max >= x_min*(1+1e-9)`, `target_error >= 1e-14`, `max_nodes >= 4`. Tries shipped table (family `"noncrossing"`), else cached exact solver with rounded lru keys `round(logR, 12)`, `round(target, 14)`. Rescale: `tau = tau_hat/x_min`, `alpha = w_hat/x_min`, `err = err_hat/x_min`. Callers: `ppm_sigma.py:776` (sigma noncrossing single-window), `ppm_sigma.py:871` (a_stripe/b_slab windows), `w_isdf.py:508+521+532` (`build_real_quadrature` — HL-PPM real-axis via two shifted `1/y` fits with mixed-sign tau), `tests/test_real_axis_quadrature.py:50`, `tests/test_minimax_assets.py:119,173`. |
| `solve_laplace_minimax_imag_interval(x_min, x_max, omega_p, *, target_error=1e-6, max_nodes=64)` | 618–659 | Fits `x/(x^2+omega_p^2) ≈ sum alpha_l exp(-tau_l x)` for `chi0(i*omega_p)` (resonant+antiresonant sum gives `2x/(x^2+omega_p^2)`, `x = E_c - E_v`). Same clamps/rescale as static; scales `omega_hat = omega_p/x_min` too. **No `use_shipped_tables` parameter** — always disk-cache/exact-solver (asymmetric with the other two public solvers). Caller: `w_isdf.py:467-468` (`build_imag_quadrature`, deferred import). Also imported at `ppm_sigma.py:74` but **never used there** (unused import). |
| `solve_phase_minimax_bandwidth(A_dim, *, target_error=1e-6, max_nodes=500, eps_q=1e-3, target_kind="hgl", use_shipped_tables=True)` | 662–712 | Fits crossing regularization target `G(u) ≈ sum alpha_l sin(tau_l u)` on `[0, A_dim]` (HGL or Fermi kind). `target_error` = L-inf absolute error on G itself. Tries shipped family `"crossing"` (matching target_kind, eps_q), else cached exact solver. No rescale (A_dim passed through). Callers: `ppm_sigma.py:853` (crossing window; caller then rescales nodes by 1/xi at 863), `tests/test_minimax_assets.py:144`. |

### Window-pair builders

| Function | Lines | Role / callers |
|---|---|---|
| `build_static_minimax_window_pair(enk_v, enk_c, *, minimax_config=None, target_error=1e-6, max_nodes=64, use_shipped_tables=True, print_fn=None)` | 715–772 | Single non-crossing window spanning all v/c states: `x_min = max(cmin - vmax, _TINY)`, `x_max = max(cmax - vmin, x_min*(1+1e-9))`; calls `solve_laplace_minimax_interval`. If `minimax_config` given, it **overrides** the three keyword args (`target_error`, `max_nodes`, `use_shipped_tables` — the MinimaxConfig fields consumed here). Builds kernel weights `w_i = alpha_i * exp(-tau_i)` with comment (752): "so the chi kernel weight is `-2 * w_i * exp(-tau_i * dE)`" (the -2 lives in the consumer, not here). Returns `([MinimaxWindowPair], LaplaceMinimaxQuadrature)` — a single-element list, vestige of a multi-window design. Live caller: `w_isdf.py:457` as `_, quad = build_static_minimax_window_pair(...)` — **the pair list (and hence MinimaxWindowPair/EnergyWindow content) is discarded**; only `quad` is used. Also imported at `ppm_sigma.py:71` but never called there (unused import). |
| `build_imag_freq_minimax_window_pair(enk_v, enk_c, omega_p, *, target_error=1e-6, max_nodes=64, print_fn=None)` | 775–835 | Copy-paste sibling of the static builder using `solve_laplace_minimax_imag_interval`; docstring: "replaces the incorrect cos-reweighting of static nodes". **Zero callers** — `w_isdf.build_imag_quadrature` (w_isdf.py:462–479) calls `solve_laplace_minimax_imag_interval` directly on the static quad's interval instead. |

## Einsum signatures

None. This file contains no `einsum` calls (verified by read; the only tensor ops are
elementwise `jnp.where/sqrt` in the GN fit and per-q `np.linalg.solve` in the legacy path).

## Boundary-crossing arrays

- `MinimaxNodes.t`, `.alpha`: complex128 `(n_nodes,)`, device, replicated (w_isdf.py:126); jit-safe pytree.
- `fit_gn_ppm_from_wc_pair` inputs/outputs: `(nkx,nky,nkz,n_rmu,n_rmu)` complex128, device,
  "same sharding as attached to inputs", no communication except the scalar `mean(good)` allgather.
- `extract_gn_ppm_parameters` (legacy): full host gather of V/chi0/chi_i (7D `(nkx,nky,nkz,1,n_rmu,1,n_rmu)`), host loop, re-uploads `GodbyNeedsPPM` fields as jnp arrays.
- `build_*_window_pair`: `jax.device_get(enk_v/enk_c)` host pulls of energy arrays (any shape; only min/max used).

## Config flags / env vars consumed

- `MinimaxConfig.target_error` (default 1e-6), `.max_nodes` (64), `.use_shipped_tables` (property) — via `build_static_minimax_window_pair(minimax_config=...)`. (`MinimaxConfig.energy_reference` is consumed by w_isdf's `resolve_minimax_energy_reference`, not here.)
- Env: `LORRAX_DISABLE_MINIMAX_DISK_CACHE` (1/true/yes), `LORRAX_MINIMAX_CACHE_DIR`.
- No direct cohsex.in parsing in this file (flags arrive pre-parsed through MinimaxConfig/SigmaQuadratureConfig in callers).

## I/O

Read:
- `common/minimax_assets/catalog.json` (package data, JSON): `{"tables": [{family, range_max, error_bound, node_count, target_kind, eps_q, file}, ...]}`.
- `common/minimax_assets/{noncrossing,crossing}/*.npz` (per catalog `file`): datasets `tau`, `alpha`, `max_error`. Generated by `tools/generate_minimax_assets.py` (which calls this module's private `_solve_*_cached` / `_find_shipped_table_entry`).
- `$LORRAX_MINIMAX_CACHE_DIR or ~/.cache/lorrax/minimax_quadratures/{noncrossing|noncrossing_imag|crossing}_{sha256}.npz`: datasets `tau`, `w`, `err`.

Write:
- Same disk-cache npz files (atomic `.tmp.{pid}` + `os.replace`).

## Dead suspects (grep evidence)

Grepped across `src/`, `tests/`, `tools/`, `scripts/` (`--include=*.py`, excluding this file) for each public name:

1. `extract_gn_ppm_parameters` (838–896, legacy chi-based GN fit): pattern
   `extract_gn_ppm_parameters[^_]` returns **zero hits** outside the module. Only the
   `_from_Wc` variant is called (`scripts/checks/sigma_direct_check.py:504`). Superseded
   parallel path — prime deletion candidate under the no-redundancy rule.
2. `build_imag_freq_minimax_window_pair` (775–835): **zero hits**. `w_isdf.build_imag_quadrature`
   calls `solve_laplace_minimax_imag_interval` directly instead.
3. `MinimaxWindowPair.with_imag_freq_modulation` (269–287): **zero hits** for
   `with_imag_freq_modulation`; additionally documented-as-incorrect by the replacement's
   docstring (789).
4. `MinimaxWindowPair` / `EnergyWindow` / `EnergyWindow.upper_inclusive` (243–287): zero
   external constructors or attribute reads (`MinimaxWindowPair|EnergyWindow|upper_inclusive|\btau_i\b|\bw_i\b|\.z_lm|\.epsq` — only hits are unrelated `window_pairs` in `solvers/pseudobands_v2.py` and `win.z_lm` on a *different* legacy object in `tests/archive/test_hgl_quadrature.py` / `tests/archive/test_frequency_integration_toy.py` from the old `gw_isdf.w_isdf_dynamic` module). Both live callers of `build_static_minimax_window_pair` discard the pair list (`_, quad = ...` at w_isdf.py:457). The entire window-pair dataclass layer is effectively dead payload; the function could return just the quadrature.
5. Unused imports in consumers (cross-file cruft found while tracing): `ppm_sigma.py:71`
   `build_static_minimax_window_pair` and `ppm_sigma.py:74` `solve_laplace_minimax_imag_interval`
   are imported but each name appears exactly once in that file (the import line).

## Redundancy suspects

1. `extract_gn_ppm_parameters` vs `extract_gn_ppm_parameters_from_Wc` + `fit_gn_ppm_from_wc_pair`:
   classic old/new parallel GN-PPM paths — host-numpy per-q loop vs sharded JAX elementwise.
   The ratio/validity/fallback logic is duplicated nearly verbatim (446–457 vs 876–885).
2. `build_static_minimax_window_pair` (715–772) vs `build_imag_freq_minimax_window_pair`
   (775–835): copy-paste siblings — identical energy min/max extraction, identical
   `w_kernel = alpha*exp(-tau)` transform, identical pair construction; only the solver
   call and print string differ. The imag one is also dead (above).
3. `_laplace_to_minimax_nodes` 'real' branch and `_crossing_to_minimax_nodes` perform
   the identical cast (`tau→complex, alpha→complex`); the crossing function exists only
   to validate a different `time_axis` string.
4. Return convention `([pair], quad)` — single-element list preserved for a multi-window
   API that no longer exists (module docstring: "a single non-crossing minimax window pair").
5. Disk-cache npz field names (`tau/w/err`) vs shipped-asset field names (`tau/alpha/max_error`)
   for the same conceptual payload — two ad-hoc schemas.

## Weird code

1. Line 29 + 579/632/680/742/801: `_TINY = 1.0e-12` floor on `x_min` / `A_dim`. A metallic or
   zero-gap system silently produces `x_min = 1e-12` → `R ≈ 1e16/x_max`-scale intervals and
   an absolute error blow-up of `target_error/x_min = 1e6·target_error` (per the module's own
   error convention at 573–576) with no warning.
2. Lines 599–602, 642–646, 699–704: floats used as `lru_cache` keys with ad-hoc rounding
   (`round(logR, 12)`, `round(target, 14)`, `round(omega_hat, 12)`, `round(A_dim, 12)`,
   `round(eps_q, 12)`); the disk-cache key is sha256 of the JSON of these rounded floats.
   Near-identical intervals that straddle a rounding boundary re-solve and re-store.
3. Lines 66–76, 85–94, 128–133, 143–147, 205–212, 235–240: pervasive bare
   `except Exception: return None / pass` — a corrupted catalog, npz, or cache file silently
   degrades to the expensive exact solver (or silently skips storing) with no log line.
4. Line 137: `if entry_err - 1.0e-18 > target_error: continue` — 1e-18 slop on an error bound
   comparison; combined with sort key `(-entry_err)` at 150, selection deliberately prefers
   the **loosest** acceptable shipped error to minimize nodes (documented, but easy to misread).
5. Line 144: shipped crossing-table `eps_q` must match request within 1e-12 — exact-match
   float gate; any nonstandard `eps_q` silently bypasses all shipped tables.
6. Magic constants: `fallback_omega=2.0` Ry default GN pole (844, 903); probe `2j` Ry cited
   as "standard" (427); validity thresholds `1e-14` on `|denom|` (443, 876); `x_max >= x_min*(1+1e-9)` pads (580, 633, 743, 802).
7. Sign-convention split: the `-2` resonant+antiresonant prefactor of the chi kernel lives in
   the *consumer* while this module bakes `exp(-tau)` into `w_i` (comment 752: "w_i = alpha_i *
   exp(-tau_i) so the chi kernel weight is -2 * w_i * exp(-tau_i * dE)"). Meanwhile
   `MinimaxNodes` consumers use raw `alpha` (no `exp(-tau)` prefold) — two weight conventions
   (`w_i` vs `alpha_i`) coexist inside `MinimaxWindowPair`, which both carries them redundantly
   and is itself dead.
8. `time_axis` string enum ('real' / 'imag' / 'crossing_hgl') selecting sign flips
   (`t = -1j*tau` for sigma windows, 324) — convention knob passed as free-form string;
   the 1/xi crossing rescale is applied by the caller (ppm_sigma.py:863), not here (documented
   at 338–339 but split responsibility).
9. Lines 862–863, 920–921: hard-coded 7D squeeze `[:, :, :, 0, :, 0, :]` assumes the
   `(nkx,nky,nkz,1,mu,1,mu)` spin/aux-singleton layout; a bispinor (ns=2) tensor would be
   silently truncated to one spin block rather than erroring.
10. GN residue built from different objects in the two paths: `B = -0.5 * W^c(0) * Omega`
    (455) vs `B = -0.5 * Pi_0 * Omega` (884). Consistent with each path's consumer but a trap
    if anyone "unifies" them naively.
11. `solve_laplace_minimax_imag_interval` lacks the `use_shipped_tables` knob and any shipped
    family — the imag-freq path always pays solver cost on first use per (R, omega_hat, target)
    tuple (asymmetric with static/crossing).
12. `fit_gn_ppm_from_wc_pair` computes `unfulfilled_fraction` via `_scalar_to_host_float(jnp.mean(...))`
    — a forced host sync + allgather embedded in an otherwise "pure local algebra, no host
    gathers" function (contradicts its own docstring at 434–435, though only by one scalar).

## Cross-module dependency list

- `common.minimax` (`noncrossing_grids`, `noncrossing_imag_grids`, `crossing_grids`, `G_hgl`,
  `G_fermi`, `tau_max_hgl`, `tau_max_fermi`) — exact solvers.
- `gw.minimax_config.MinimaxConfig` — flag bundle.
- `common/minimax_assets` package data (catalog + npz tables); regenerated by
  `tools/generate_minimax_assets.py`, which reaches into this module's private
  `_solve_noncrossing_scaled_cached` / `_solve_crossing_scaled_cached` / `_find_shipped_table_entry`.
- Consumers: `gw.w_isdf` (static/imag/real chi0 quadratures, MinimaxNodes), `gw.ppm_sigma`
  (sigma tau windows: noncrossing/imag/crossing; GN fit from Wc pair),
  `scripts/checks/sigma_direct_check.py` (GN extraction), `tests/test_minimax_assets.py`,
  `tests/test_real_axis_quadrature.py`.
