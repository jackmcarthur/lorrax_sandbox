# gw_small group — deep-read notes (2026-07-01)

Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`. All paths relative.
Files: `src/gw/screening.py` (176 LOC), `src/gw/band_partition.py` (164), `src/gw/degen_average.py` (97), `src/gw/greens_function_kernel.py` (58), `src/gw/minimax_config.py` (36), `src/gw/__init__.py` (11).

Grep scope for all caller claims: `src/`, `tests/`, `tools/`, `scripts/` of lorrax_D (`grep -rn --include='*.py'`).

---

## src/gw/screening.py (176 LOC)

**Purpose.** Screening-frequency planner + executor: each Σ scheme declares which W(ω) evaluations it needs as `(omega_ry, role)` pairs; `compute_screening` produces a `{role: W_q}` dict the Σ builder consumes. Decouples "which W's" from "how Σ is built" so a new self-energy scheme = extend `screening_requests_for` + one dispatch case in `sigma_dispatch.compute_sigma_xc`.

**Category.** physics: chi0/W stage (thin orchestration layer over `w_isdf`).

### Functions

| name | lines | role |
|---|---|---|
| `ScreeningRequest` (frozen dataclass) | 45-63 | One W evaluation: `omega_ry: complex` (pure real → real axis, pure imag → Wick-rotated axis), `role: str` symbolic label ("static", "probe"). |
| `screening_requests_for(mode, config)` | 71-96 | Single source of truth per ComputeMode: `X_ONLY → []`; `COHSEX → [static(ω=0)]`; `GN_PPM → [static, probe(ω = i·config.ppm.omega_p)]`; `HL_PPM → [static, probe(ω = config.ppm.omega_p real)]`. Raises ValueError on unknown mode. |
| `compute_screening(wfns, V_q, requests, *, quad, e_ref, config, meta, mesh_xy, print_fn)` | 103-169 | Loop over requests. "static" role reuses prebuilt minimax `quad`; other roles build a single-frequency quadrature on the fly via `w_isdf.build_imag_quadrature(quad, |Im ω|, config.minimax_config)` or `build_real_quadrature(quad, |Re ω|, ...)` — axis chosen by which component of ω is nonzero; mixed complex ω raises ValueError (lines 144-149). Then per request: `chi0 = compute_chi0(wfns, quad_used, meta, mesh_xy, energy_reference=e_ref)`; `chi0.block_until_ready()`; `W = solve_w(V_q, chi0, meta, mesh_xy, solver=config.backend.screening_solver)`; `del chi0`; `W.block_until_ready()`. Physics: W = (1 − v·χ₀)⁻¹ v evaluated inside `solve_w` (w_isdf.py:354). |

### Entry points / callers
- `screening_requests_for` ← `src/gw/sc_iteration.py:281` (SC iteration map), `src/gw/gw_jax.py:417` (one-shot dynamic path, filters out `role != "static"` since static W already solved).
- `compute_screening` ← `src/gw/sc_iteration.py:283`, `src/gw/gw_jax.py:419`.
- `W_by_role["probe"]` consumed in `src/gw/sigma_dispatch.py:208` (KeyError message references `screening_requests_for`); `gw_jax.py:430` passes `_W_extra["probe"]` as `W_probe_q` to `compute_ppm_sigma_pipeline`.
- No test file imports screening.py directly (grepped `screening_requests` in tests/ — no hits).

### Flags consumed
- `config.ppm.omega_p` ← cohsex.in key `ppm_omega_p` (default 2.0 Ry, gw_config.py:303).
- `config.minimax_config` (property, gw_config.py:765) ← keys `minimax_target_error`, `minimax_max_nodes`, `regenerate_minimax_tables`, `minimax_energy_reference`.
- `config.backend.screening_solver` ← legacy key `isdf_memory_mode` (gw_config.py:830-831, `_LEGACY_ISDF_MEMORY_MODE` map at :1036).

### Arrays crossing the boundary
- In: `V_q` (jax.Array, bare/cut Coulomb in ISDF μ basis), `quad` (static minimax quadrature pytree from `build_static_quadrature`), `wfns` bundle.
- Out: `{role: W_q}` dict of device arrays; χ₀ freed between requests (`del chi0` + block_until_ready serialization = deliberate peak-memory control).

### I/O
None (in-memory only; quadrature table I/O lives in minimax_screening).

### Suspects
- dead: none — all three exports have callers.
- redundancy: none within file; note gw_jax.py:417 re-filters requests (`if r.role != "static"`) because it solved static W earlier — mild duplication of screening knowledge at the call site vs sc_iteration.py which uses the full list.
- weird: `block_until_ready()` on both chi0 and W inside the loop (lines 161, 166) — host-side serialization of the request loop; hypothesis: intentional to keep only one χ₀ live at a time (memory), but it also disables any overlap. Local import of w_isdf inside function body (line 131) — import-cost/circularity dodge, common in this codebase.

---

## src/gw/band_partition.py (164 LOC)

**Purpose.** Three-way band partition for QSGW self-consistency: within the active (sigma) subspace, bands are protected (full off-diagonal Σ), non-protected in ω-range (diagonal Σ at band energy), or out-of-range (scissor diagonal α·E_DFT+β). Provides the static masks and a jit'd per-iteration primitive that masks `H_qp_dft`.

**Category.** physics: QSGW SC-iteration support (Hamiltonian masking primitive).

### Functions

| name | lines | role |
|---|---|---|
| `BandPartition` (frozen dataclass) | 51-101 | `protected_mask: (nb_active,) bool`, `in_range_mask: (nb_active,) bool`; both static across SC iterations. |
| `BandPartition.all_protected(nb_active)` | 72-80 | classmethod; all-ones masks → `apply_band_partition` is identity. Default in `sc_iteration.SCInputs`. |
| `BandPartition.warn_if_protected_outside_grid(*, print_fn)` | 82-101 | All-caps warning if any protected band lies outside the Σ_c(ω) grid (`protected & ~in_range`) — such bands would mix edge-clamped Σ into the eigenproblem. |
| `apply_band_partition(H_full, *, protected_mask, in_range_mask, scissor_E_qp_kn)` | 108-161 | `@jax.jit`. H_full: (nk, nb_active, nb_active) complex. Off-diag keep mask `offdiag_keep = p[:,None]*p[None,:]`; `offdiag_part = H_full * (1-eye) * offdiag_keep[None,:,:]`. Diagonal: `diag_kept = where(in_range_mask, diag(H_full), scissor_E_qp_kn)`. Returns `offdiag_part + diag_kept[:,:,None]*eye[None,:,:]`. Equation: H'_mn = H_mn·P_m·P_n (m≠n); H'_nn = E_scissor(k,n) if ¬in_range else H_nn, with E_scissor = α·E_DFT + β computed by caller. |

### Entry points / callers
- `BandPartition`, `apply_band_partition` ← `src/gw/sc_iteration.py:71` (import), `:114` (SCInputs field `partition: BandPartition`), `:321` (`H_qp_dft_new = apply_band_partition(...)` per SC step); `src/gw/gw_jax.py:485,516` (builds partition from `classify_bands_in_grid` with protected := in-range, default policy); `tests/test_band_partition.py` (identity case, off-diag zeroing, scissor diag, warning text).
- `warn_if_protected_outside_grid` ← `gw_jax.py:518`, `tests/test_band_partition.py:107,119`.

### Flags consumed (via caller gw_jax.py:503-505)
- `config.ppm.omega_min_ev` / `omega_max_ev` ← cohsex.in `sigma_omega_min_ev` / `sigma_omega_max_ev` (defaults −5.0/+5.0 eV relative to E_F, gw_config.py:314-315) define the in-range mask through `scissor.classify_bands_in_grid`.

### Arrays crossing boundary
- `H_full` (nk, nb_active, nb_active) complex, device; masks (nb_active,) bool device; `scissor_E_qp_kn` (nk, nb_active) real device (pass zeros if unused). All small/replicated; no sharding annotations in file.

### I/O
None.

### Suspects
- dead: none.
- redundancy: none.
- weird: `gw_jax.py:511` hardcodes `_protected = _in_range` — the three-way partition currently collapses to two-way in the driver (no user knob for a protected set distinct from in-range); the "non-protected in-range diagonal Σ" middle row of the table is unreachable from gw_jax as written (only via direct SCInputs construction). Masking done with float multiplication (`p.astype(H.dtype)` then products) rather than `jnp.where` — fine numerically, slightly obfuscated.

---

## src/gw/degen_average.py (97 LOC)

**Purpose.** Degenerate-subspace averaging of diagonal Σ matrix elements, mirroring BerkeleyGW `Sigma/shiftenergy.f90` lines 86-122: within each contiguous degenerate group of the DFT spectrum, replace each diagonal Σ value with the group mean (trace/multiplicity is basis-invariant; Schur's lemma argument in docstring). Off-diagonals preserved (BGW convention).

**Category.** physics: Σ post-processing / BGW parity shim (host-side numpy).

### Functions

| name | lines | role |
|---|---|---|
| `TOL_DEGENERACY_RY = 1.0e-6` | 28 | Matches BGW `Common/nrtype.f90 :: TOL_Degeneracy` (1e-6 Ry ≈ 14 µeV). |
| `average_within_degenerate_sets(values_kn, energies_kn_ry, tol_ry)` | 31-71 | Host loop over k; walks bands, group boundary where `|e[k,i]−e[k,i−1]| ≥ tol_ry`; within each group `out[k,i0:i] = mean`. Requires energies sorted ascending per k (adjacent-difference grouping) and in Rydberg. Shapes (nk, nb), real or complex; returns copy. |
| `apply_to_matrix_diagonals(matrix_knn, energies_kn_ry, tol_ry)` | 74-97 | Extract diag of (nk, nb, nb), average via the above, write back; off-diag untouched. Mirrors shiftenergy.f90 averaging of `ax`, `asx`, `ach`. |

### Entry points / callers
- `average_within_degenerate_sets` ← `src/gw/gw_jax.py:378-388` (bare Σ_X diagonal print), `:740` (`sigma_c_at_dft_ev` 1-D for eqp.dat sigC).
- `apply_to_matrix_diagonals` ← `src/gw/gw_jax.py:735` (`_dav` closure applied to `sigma_total`, `sig_sx`, `sig_coh`, `sig_h`, `sig_x` before the single H-build + eigh).
- No dedicated test (grepped `degen_average|average_within_degenerate_sets|apply_to_matrix_diagonals|TOL_DEGENERACY` in tests/ — zero hits). Functions are alive via gw_jax but untested in isolation.

### Flags consumed (at call sites)
- `no_degen_averaging` (cohsex.in, default False, gw_config.py:199) gates both call sites.
- `degen_avg_tol_ry` (cohsex.in, default 1.0e-6, gw_config.py:200) → `tol_ry`.

### Arrays crossing boundary
- All host numpy (`np.asarray(M)` at call site, then `jax.device_put(..., NamedSharding(mesh_xy, P(None,None,None)))` back to replicated device arrays — gw_jax.py:734-736).

### I/O
None.

### Suspects
- dead: none in-file (module-level constant `TOL_DEGENERACY_RY` referenced only as default; fine).
- redundancy: gw_jax applies the averaging twice with identical `_enk_sigma_ry` recomputation (`get_enk_bandrange` called at :381 and :728) — caller-side duplication, acknowledged in gw_jax comment ("the redundancy across components is not a perf concern").
- weird: contiguous-adjacent-difference grouping silently assumes per-k ascending eigenvalues; a non-monotonic energies array would fragment groups without error. Magic constant 1e-6 Ry deliberately mirrors BGW (documented).

---

## src/gw/greens_function_kernel.py (58 LOC)

**Purpose.** Unified ISDF-basis Green's function builder: G_μν(k) = Σ_ij ψ_i(μ) W_ij ψ*_j(ν), plus a time-evolution wrapper G(t) = Σ_n ψ_n(μ) e^{−t(e_n−e_ref)} ψ_n*(ν) shared by χ₀ (real t = imaginary-time) and Σ_c (pure-imaginary t = real-time). Convention: psi_xn direct (μ side), psi_yr conjugated internally (ν side) — matches tested COHSEX convention in gw_jax.

**Category.** physics: shared G kernel (χ₀/Σ common primitive).

### Functions

| name | lines | role |
|---|---|---|
| `build_G(psi_xn, psi_yr, *, Gij=None, phases=None)` | 11-31 | Returns (nk, s, μ_X, s, μ_Y) flat-k. Four branches, einsums VERBATIM: (a) Gij+phases: `'ksxi,kij,kjty->ksxty'` on `psi_xn, p[:, :, None] * Gij * p[:, None, :], jnp.conj(psi_yr)` with `p = phases.astype(jnp.complex128)`; (b) Gij only: `'ksxi,kij,kjty->ksxty'` on `psi_xn, Gij, jnp.conj(psi_yr)`; (c) phases only: `'ksxn,kn,knty->ksxty'` on `psi_xn, phases.astype(jnp.complex128), jnp.conj(psi_yr)`; (d) neither: `'ksxn,knty->ksxty'` on `psi_xn, jnp.conj(psi_yr)`. All `optimize=True`. Inputs: psi_xn (nk, s, μ_X, nb); psi_yr (nk, nb, s, μ_Y); Gij (nk, nb, nb); phases (nk, nb) complex. |
| `build_G_tau(psi_xn, psi_yr, enk, t, *, e_ref=0.0, mask=None)` | 34-58 | `phases = jnp.exp(-t * (enk - e_ref))`; if mask (nk,nb) bool: `phases = where(mask, phases, 0+0j c128)`; delegates to `build_G(psi_xn, psi_yr, phases=phases)`. Equation: G(t)_k(μ,ν) = Σ_n ψ_n(μ)·exp(−t·(e_n−e_ref))·ψ_n*(ν); real t → imaginary-time (χ₀ minimax), pure-imag t → real-time (Σ_c). mask = sigma's mask_A window gate. |

### Entry points / callers
- `build_G` ← `src/gw/cohsex_sigma.py:25,96,104` (`G_occ = build_G(wfns.xn(s.sigma), wfns.yr(s.sigma), Gij=Gij)`; `G_ri = build_G(wfns.xn(s.full), wfns.yr(s.full))`); `src/gw/aot_memory_model/kernels/chi0_tau_step.py:137,161-164` (memory-model mirror, phases-only branch); `src/gw/aot_memory_model/kernels/sigma_kij.py:114,132` (Gij branch). Referenced in `sigma_x_bispinor.py` docstrings only (:28,:75,:77 — the bispinor path reaches build_G via `cohsex_sigma._make_cohsex_kernels`).
- `build_G_tau` ← `src/gw/w_isdf.py:93,117,120` (χ₀: `Gv = build_G_tau(psi_v_xn, psi_v_yr, enk_v, −τ, e_ref=vmax)`, `Gc = build_G_tau(psi_c_xn, psi_c_yr, enk_c, +τ, e_ref=cmin)`, both then `jnp.conj` at the call site — Hermitian-swap conj deliberately NOT inside build_G_tau, per w_isdf.py:122-123 comment); `src/gw/ppm_sigma.py:526,548` (Σ_c: `G(t) = build_G_tau(psi, E_A, 1j·t_node, e_ref=E_ref_A, mask=mask_A)`).
- No dedicated test file (grepped `build_G_tau|greens_function` in tests/ — zero hits); exercised indirectly through χ₀/Σ tests.

### Flags consumed
None directly (t, e_ref, mask supplied by callers from quadrature/window logic).

### Arrays crossing boundary
- psi_xn (nk, s, μ_X, nb) c128, psi_yr (nk, nb, s, μ_Y) c128 — device, sharding imposed by callers (w_isdf wraps in jit with `PSI_XN_SPEC`/`PSI_YR_SPEC` NamedShardings and constrains output to `_G_out_flatk`). Output (nk, s, μ_X, s, μ_Y) flat-k.

### I/O
None.

### Suspects
- dead: the combined `Gij is not None and phases is not None` branch (lines 19-23) has ZERO callers — grepped `phases=` across src/tests/tools/scripts: only chi0_tau_step.py (phases only) and internal build_G_tau (phases only); all Gij callers (cohsex_sigma, sigma_kij) pass no phases. Dead branch, plausible YAGNI leftover from a planned "dressed G with band phases" path.
- redundancy: `aot_memory_model/kernels/{chi0_tau_step,sigma_kij}.py` import the real build_G to mirror kernels for memory modeling — intentional single-source (imports, not copies), so NOT redundancy; noted for the map.
- weird: sign/convention hotspots to preserve in refactor: (1) psi_yr is conjugated INSIDE build_G (docstring line 5, sigma_x_bispinor.py:77 warns about it); (2) the Hermitian-swap `jnp.conj(G)` for χ₀ lives at the w_isdf call site "NOT inside build_G_tau" (w_isdf.py:122-123) — a refactor that moves it breaks the FFT convention; (3) build_G_tau's imag/real-time split is purely the caller's choice of t (real vs 1j·t), no runtime check.

---

## src/gw/minimax_config.py (36 LOC)

**Purpose.** Two frozen dataclasses of math-internal minimax-quadrature settings shared by screening (χ₀) and Σ_c window construction. Pure configuration; no logic beyond a `use_shipped_tables` property (= not regenerate_tables).

**Category.** configuration: quadrature settings container.

### Functions

| name | lines | role |
|---|---|---|
| `MinimaxConfig` | 8-19 | `target_error=1e-6`, `max_nodes=64`, `regenerate_tables=False`, `energy_reference="midgap"` (str|float|int|None). Property `use_shipped_tables`. |
| `SigmaQuadratureConfig` | 22-34 | `target_error=1e-6`, `max_nodes=64`, `crossing_max_nodes=500`, `crossing_eps_q=1e-3`, `regenerate_tables=False`. Property `use_shipped_tables`. |

### Entry points / callers
- `MinimaxConfig` ← `src/gw/gw_config.py:765-772` (built from cohsex.in keys `minimax_target_error`, `minimax_max_nodes`, `regenerate_minimax_tables`, `minimax_energy_reference` — defaults at gw_config.py:292-295); consumed by `src/gw/minimax_screening.py:26,719-730` (grids builder reads target_error/max_nodes/use_shipped_tables), `src/gw/w_isdf.py:20,446-518` (`build_static_quadrature`, `build_imag_quadrature`, `build_real_quadrature`), `src/gw/ppm_sigma.py:68`, `src/gw/screening.py:153,157`, `tests/test_real_axis_quadrature.py:38,70,90,124,142`.
- `SigmaQuadratureConfig` ← `src/gw/gw_config.py:777-785` (from `ppm_sigma_target_error`, `ppm_sigma_max_nodes` — defaults gw_config.py:311-312 — with `crossing_max_nodes=max(500, sigma_max_nodes)` and `crossing_eps_q=1.0e-3` HARDCODED at gw_config.py:782-783, overriding the dataclass defaults with the same values); consumed by `src/gw/ppm_sigma.py:68,1495` (`sigma_window_quad` kwarg; crossing_* fields threaded to window/crossing quadrature builders at ppm_sigma.py:815-816,856-857,1186-1187,1395-1396).

### Flags consumed
Indirect: cohsex.in `minimax_target_error`, `minimax_max_nodes`, `regenerate_minimax_tables`, `minimax_energy_reference`, `ppm_sigma_target_error`, `ppm_sigma_max_nodes`.

### I/O
None (the tables these settings govern are read/written by minimax_screening + tools/generate_minimax_assets.py).

### Suspects
- dead: none — both classes and both properties used.
- redundancy: `crossing_eps_q=1e-3` exists twice (dataclass default line 29 AND hardcoded in gw_config.py:783) — the dataclass default is never allowed to differ; similarly `crossing_max_nodes` default 500 duplicated by `max(500, ...)` at gw_config.py:782. Two identical `use_shipped_tables` properties (copy-paste between the classes).
- weird: `crossing_eps_q` not user-settable despite being a dataclass field — magic constant 1e-3; hypothesis: pole-crossing quadrature regularization pinned during development, exposure deferred.

---

## src/gw/__init__.py (11 LOC)

**Purpose.** Package init for the GW/COHSEX driver: sets `os.environ.setdefault("JAX_ENABLE_X64", "1")` BEFORE any submodule imports JAX (double precision is load-bearing for GW), then re-exports `get_bandranges` (from gw_init) and `read_lorrax_input`/`read_cohsex_input` (from gw_config).

**Category.** package init / environment setup.

### Contents
- Line 6: `os.environ.setdefault("JAX_ENABLE_X64", "1")` — import-time side effect; must run before jax import anywhere in the package.
- Lines 8-9: re-exports. `read_cohsex_input` is itself an alias: `gw_config.py:496 read_cohsex_input = read_lorrax_input`.

### Entry points / callers
- `from gw import ...` users: `tests/test_minimax_assets.py:5` and `tools/generate_minimax_assets.py:28` (`from gw import minimax_screening as ms` — submodule import, triggers the env side effect), `tests/tools/profile_gw_xprof.py:66` (`from gw import gw_jax`).
- NO caller anywhere imports `get_bandranges`, `read_lorrax_input`, or `read_cohsex_input` from the package top level (grepped `from gw import` across src/tests/tools/scripts — only the three submodule imports above). Callers go straight to submodules: `gw.gw_init.read_cohsex_input` (src/bandstructure/htransform.py:879, scripts/checks/w_from_eps0_0d_check.py:58, scripts/checks/sigma_direct_check.py:68 — note gw_init.py:516 re-re-exports it from gw_config), or the SEPARATE `psp.get_DFT_mtxels.read_cohsex_input` implementation.

### Suspects
- dead: the `__all__` re-exports (`get_bandranges`, `read_lorrax_input`, `read_cohsex_input`) have zero top-level importers (evidence: grep `from gw import` hits only minimax_screening/gw_jax submodule imports). The env-var line is the only functional payload.
- redundancy (cross-module, high value for the refactor map):
  1. `get_bandranges` exists TWICE, byte-similar: `src/gw/gw_init.py:519-530` (docstring: "Legacy helper used by psp/get_DFT_mtxels.py. GW code uses BandSlices instead.") and `src/psp/get_DFT_mtxels.py:170-178` — identical body; psp uses its OWN copy (get_DFT_mtxels.py:836), so the gw_init copy's stated raison d'être is false and it appears fully dead.
  2. `read_cohsex_input` has TWO independent implementations: alias of `gw_config.read_lorrax_input` (gw_config.py:496) vs a standalone parser `psp/get_DFT_mtxels.py:93` whose docstring says it mirrors `gw.gw_init.read_cohsex_input`. Classic parallel old/new path.
  3. `JAX_ENABLE_X64` setdefault duplicated in ≥8 modules (gw/__init__, gw/kin_ion_io, runtime/__init__:56, bandstructure/htransform, bse/bse_jax, centroid/kmeans_cli, common/*_test) — a refactor should centralize in runtime/.
- weird: import-time env mutation in `__init__` means merely importing any `gw.*` submodule flips global JAX precision — correct but a known foot-gun for embedding; ordering-sensitive (must precede first jax import process-wide).

---

## Cross-file summary for the refactor map

- **Call graph (this group):** gw_jax / sc_iteration → screening.{screening_requests_for, compute_screening} → w_isdf.{build_*_quadrature, compute_chi0, solve_w}. gw_jax / sc_iteration → band_partition.apply_band_partition (QSGW mask). gw_jax → degen_average (BGW-parity diagonal averaging, host numpy). cohsex_sigma / w_isdf / ppm_sigma / aot_memory_model kernels → greens_function_kernel.{build_G, build_G_tau}. gw_config → minimax_config dataclasses → minimax_screening / w_isdf / ppm_sigma.
- **Top dead/cruft candidates:** build_G's Gij+phases combined branch; gw/__init__ re-exports; gw_init.get_bandranges (duplicate of psp copy that psp doesn't use).
- **Convention landmines:** conj(psi_yr) inside build_G; Hermitian-swap conj at w_isdf call site by design; imag-vs-real axis selection by which component of ω is nonzero (screening.py:144-149); degeneracy grouping assumes sorted-ascending Ry energies.
