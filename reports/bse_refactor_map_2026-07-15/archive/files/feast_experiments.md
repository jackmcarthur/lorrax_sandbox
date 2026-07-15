# src/bse/feast_* experiment drivers — deep-read notes (1465 LOC over 4 files)

Audit date: 2026-07-15, lorrax_D checkout. Task pinned agent/slate-linalg-ffi
(e18d0e5); working checkout was agent/ppm-fit-conditioning (adc2197) — verified
`git diff e18d0e5 HEAD -- <all four files>` is **empty**, so these notes hold at
e18d0e5 verbatim.

Files:

| file | LOC | one-liner |
|---|---|---|
| `bse_feast_dense_debug.py` | 178 | numpy-only FEAST sanity check on a synthetic dense symmetric matrix |
| `feast_sweep.py` | 633 | FEAST parameter sweep (n_quad × γ × n_ritz × window × feast_iter) vs exact diag |
| `feast_zolo_sweep.py` | 333 | Zolotarev-quadrature sweep (n_quad × rho_scale × feast_iter × gmres_tol) |
| `feast_ellipse_mixed_sweep.py` | 321 | ellipse sweep with mixed quadrature (n_quad1 → n_quad2 across 2 FEAST iterations) |

## Purpose

Standalone `python -m bse.<name>` experiment drivers used during the FEAST
BSE-eigensolver development phase to pick contour-quadrature parameters.
They wrap the production machinery in `bse_feast.py` (contour filter + GMRES +
Rayleigh-Ritz) and compare Ritz values against exact eigenvalues of the dense
TDA-BSE Hamiltonian. All physics is the BSE-TDA eigenproblem in the ISDF
representation

```
H_TDA X = E X,   H = D + V − W (Ry),
[D X]_{b,c,v,k}  = (eps_c[k,c] − eps_v[k,v]) · X[b,c,v,k]
M[k,c,v,μ]       = Σ_s conj(psi_c[k,c,s,μ]) · psi_v[k,v,s,μ]        (bse_serial.py:27-29)
[V X]_{b,c,v,k}  = Σ_{μν c'v'k'} conj(M[k,c,v,μ]) V_q0[μ,ν] M[k',c',v',ν] X[b,c',v',k'] / nk
[W X]            = ISDF W_q convolution over the k-difference (FFT ring path)
```

evaluated either through `bse_serial.apply_bse_hamiltonian_single_device`
(dense reference, feast_sweep only) or `bse_ring_comm.build_bse_ring_matvec`
(sharded, zolo/ellipse sweeps). The FEAST filter approximates the spectral
projector onto the eV window [a,b]:

```
P ≈ Σ_j 2·Re[ w_j · (z_j·I − H)^{-1} X ]        (dense_debug:107-110; bse_feast.py:250-252)
```

Contour quadratures as written in code:

Ellipse (dense_debug:26-35 — an independent local copy; the sharded sweeps use
`bse_feast.feast_ellipse_quadrature` → `solvers.quadrature`):

```
c = (a+b)/2 ; r_x = (b−a)/2 ; r_y = γ·r_x
θ_j = π(2j−1)/(2·n_quad),  j = 1..n_quad          # upper half-ellipse only
z_j = c + r_x·cosθ_j + i·r_y·sinθ_j
w_j = (1/(2·n_quad)) · (r_y·cosθ_j + i·r_x·sinθ_j)
```

Index check: (1/2πi)∮dz/(z−λ) with dz/dθ = −r_x sinθ + i·r_y cosθ gives
(1/2πi)·dz = (1/2π)(r_y cosθ + i·r_x sinθ)dθ; midpoint rule with Δθ = π/n over
the upper half plus the 2·Re doubling for the conjugate half reproduces w_j
exactly. Correct standard FEAST ellipse rule.

Zolotarev custom (feast_zolo_sweep.py:63-81):

```
rho     = rho_scale · 0.5·(b−a)
n_left  = n_quad//2 ; n_right = n_quad − n_left
z_L,w_L = _zolotarev_step_poles_weights(a, n_left,  λ_min_eV, λ_max_eV, rho)
z_R,w_R = _zolotarev_step_poles_weights(b, n_right, λ_min_eV, λ_max_eV, rho)
z = [z_L, z_R] ;  w = [w_L, −w_R]                 # h(λ) = step(λ−a) − step(λ−b)
```

This is a **verbatim duplicate** of `bse_feast.feast_zolotarev_quadrature`
(bse_feast.py:916-963) — its only delta, `rho_scale`, has since been upstreamed
into the bse_feast version (bse_feast.py:921, 952).

Units convention (LORRAX-internal, not BGW): windows, spectral bounds, and
quadrature nodes/weights are in **eV**; the matvec operates in **Ry**. The
FEAST runner rescales both node and weight by the same factor,
`z = z_nodes[j]/scale ; w = w_weights[j]/scale` with `scale = ry_to_ev`
(bse_feast.py:237,245-246), which leaves the residues w/(z−λ) invariant:
w_eV/(z_eV−λ_eV) = (w_eV/s)/((z_eV/s)−λ_Ry). Band ordering is LORRAX-internal
(v index ascending from the deepest valence band; **no BGW valence-axis flip**
— per src/bse/STATUS.md:25-34 the flip happens only in
`bse_io.write_eigenvectors_stream`).

Category: **diagnostics — FEAST parameter-exploration drivers, stale/abandoned**.
Three of the four are broken at HEAD (see Suspects); their functionality has
been absorbed into `bse_feast.main()` (mixed n_quad schedule at
bse_feast.py:1272, `--quadrature {zolotarev,ellipse}` at 1124-1126, rho_scale in
`feast_zolotarev_quadrature` at 921).

## Entry points (grep over src/, tests/, tools/, scripts/, docs/ of lorrax_D and sandbox runs/, skills/, scripts/, docs/, templates/)

| symbol | callers (grep evidence) |
|---|---|
| `python -m bse.feast_sweep` / `feast_sweep.main` | only its own docstring (feast_sweep.py:8); `find runs skills scripts … \| xargs grep -l "feast_sweep\|feast_zolo\|feast_ellipse\|dense_debug"` → **no hits** |
| `feast_sweep.EXACT_EIGENVALUES_EV` | `feast_zolo_sweep.py:36`, `feast_ellipse_mixed_sweep.py:31` (only cross-module consumers of anything in these files) |
| `feast_zolo_sweep.main` | **NONE FOUND** (module-level `__main__` only, line 332-333) |
| `feast_ellipse_mixed_sweep.main` | **NONE FOUND** — and unreachable: module fails `py_compile` (IndentationError, line 113) |
| `bse_feast_dense_debug.main` | **NONE FOUND** (module-level `__main__` only, line 177-178) |

`src/bse/__init__.py` is a bare docstring (no re-exports). No pytest file
references any of the four (`grep -rn feast tests/ src/bse/test_bse.py
src/bse/test_davidson_bse.py pyproject.toml` → empty). The only external
mention anywhere is documentation: the GW-map archive note
`reports/gw_refactor_map_2026-07-01/archive/files/solvers_iterative.md:364-366`
lists `feast_ellipse_mixed_sweep.py:24` as an import site of the quadrature.

History: deleted as "dead/abandoned modules" in a0da0a5 (2026-04-29), restored
the same day in fe5e3e8 ("bse: restore feast / pseudopoles / W-exact / test_bse
files") — **already stale at restore time**: `git show
fe5e3e8:src/bse/bse_feast.py` shows `_get_feast_runner` already had the
required `dtype` parameter (line 225) and `quadrature: str = "zolotarev"`
default (line 476) that break these callers (see Suspects).

## Function tables

### bse_feast_dense_debug.py (numpy only, no LORRAX imports — still compiles and runs)

| function | lines | role |
|---|---|---|
| `WindowSpec` | 19-23 | **local 3-field** dataclass (name, a_eV, b_eV) — deliberately not the 4-field `bse_feast.WindowSpec` (name, a_eV, b_eV, note) |
| `feast_ellipse_quadrature(window, n_quad, gamma)` | 26-35 | local copy of the ellipse rule (formula above) |
| `make_test_matrix(n, seed, e_min, e_split, e_max)` | 38-49 | synthetic spectrum: n//5 eigenvalues uniform in [e_min,e_split], rest in [e_split,e_max]; H = Q·diag(λ)·Qᵀ with random orthogonal Q, symmetrized |
| `svd_rayleigh_ritz(H, X)` | 52-78 | S = XᵀX, Hproj = Xᵀ(HX) (real transpose — X is real); keep S-eigenpairs with s > 1e-10·s_max (**truncation**, unlike production `_rayleigh_ritz` which clips/regularizes, bse_feast.py:313-320); A_red = D^{-1/2}UᵀHproj U D^{-1/2}; coeffs = U·D^{-1/2}·evecs |
| `feast_filter_with_iteration(H, window, n_quad, gamma, n_ritz, seed, feast_iter, solve_noise)` | 81-133 | direct `np.linalg.solve(z·I − H, X)` per node; optional per-solve noise `Y += solve_noise·N(0,1)·‖Y‖_col` to model GMRES tolerance; subspace iteration with random padding when n_keep < n_ritz |
| `main` | 136-178 | builds test matrix from CLI, window hardcoded to [0, 2] eV, positional error comparison `|ritz[i] − exact_in_window[i]|` |

### feast_sweep.py

| function | lines | role |
|---|---|---|
| `SweepResult` | 39-58 | per-config record incl. `n_physical`, `n_total`, `s_evals`, errors in meV |
| `build_full_bse_matrix(data, ry_to_ev)` | 61-115 | dense H by unit-vector matvecs (serial TDA path); index math: `c_idx = i // (n_val·nk); v_idx = (i % (n_val·nk)) // nk; k_idx = i % nk` (97-99) — C-order flatten of the (1, n_cond, n_val, nk) layout, matching `.reshape(N)` of the output rows (107). Per element: H[(c′v′k′),(cvk)] = [Ĥ_TDA e_{cvk}]_{c′v′k′}. Symmetrized `0.5·(H + H.T)` (110, real transpose), `np.linalg.eigh`, ×ry_to_ev → eV |
| `match_eigenvalues(ritz_eV, exact_eV, a, b)` | 118-143 | positional match against exact eigenvalues inside window ±5 % margin; returns (max, mean) meV + n_matched |
| `run_sweep(data, mesh_xy, exact_eV, ry_to_ev, configs)` | 146-223 | per-config `run_feast_ritz(...)` wrapped in blanket `except Exception → print FAILED, continue` (187-189); reads `rr_data.ritz_evals/.n_physical/.n_total/.s_evals` (207-210) |
| `generate_configs()` | 226-255 | full grid: 2 windows × n_quad{4,8,16,32} × γ{0.2,0.4,0.8} × n_ritz{4,8,12} × feast_iter{1,2,3} = 216 |
| `generate_focused_configs(exact_eV)` | 258-339 | ~66 configs; auto-picks windows from the exact spectrum |
| `print_report(results, exact_eV)` | 342-486 | 7-section text report + cheapest-config-below-{1,10,50} meV recommendations |
| `generate_minimal_configs()` | 489-546 | default sweep (windows [0,2] and [0,1.91] eV) |
| `EXACT_EIGENVALUES_EV` | 551-554 | **hardcoded** 12 eigenvalues "from full diag with isdf_tensors_600 data, n_val=4, n_cond=4, 3×3×1 k-grid" — the default truth unless `--full-diag` |
| `main(argv)` | 557-629 | hardcodes `_create_mesh_xy(1, 1)` (570) — single-device only, no --px/--py; loads restart, optional full diag, dedups configs, runs sweep, writes JSON |

### feast_zolo_sweep.py

| function | lines | role |
|---|---|---|
| `ZoloSweepConfig` / `ZoloSweepResult` | 41-59 | config = (n_quad, rho_scale, feast_iter, gmres_tol); result adds per-iteration GMRES stats |
| `_zolotarev_quadrature_custom(window, n_quad, λ_min_eV, λ_max_eV, rho_scale)` | 63-81 | duplicate of `bse_feast.feast_zolotarev_quadrature` (see Purpose); reaches into private `_zolotarev_step_poles_weights` |
| `_match_eigenvalues` | 84-95 | no-margin variant of feast_sweep's matcher (duplicated again in ellipse sweep) |
| `_create_mesh_xy(px, py)` | 98-104 | verbatim duplicate of bse_feast.py:686-692 (which feast_sweep imports instead) |
| `run_sweep(...)` | 107-238 | builds ring matvec + `data["W_R"] = ifftn(W_q, axes=(2,3,4), norm="ortho")` (123-124) + diagonal preconditioner; per config: `_get_feast_runner(matvec, data, cfg.n_quad, n_ritz, gmres_max_iter, cfg.gmres_tol, ry_to_ev)` (141-149, **misses required `dtype`**), random complex128 start vectors constrained to `sh.X`, manual FEAST-iteration loop with `_rayleigh_ritz(matvec, filtered, data, s_cutoff=0.01)` (197-199) and `_build_ritz_vectors` re-seeding (204-212); matvec count `iters.sum() + n_ritz·n_quad` (188) |
| `print_summary` | 241-258 | fixed-width table (last-iteration GMRES stats) |
| `main(argv)` | 261-333 | Lanczos spectral bounds via `estimate_spectral_bounds_sharded` → `e_max = e_max_ry_raw·(1+buffer)`; window hardcoded [0, 2] eV; truth = imported `EXACT_EIGENVALUES_EV` — **no --full-diag escape** |

### feast_ellipse_mixed_sweep.py (**does not compile** — see Suspects)

| function | lines | role |
|---|---|---|
| `GAMMA_FIXED = 0.2` | 36 | module constant; duplicates `bse_feast.ELLIPSE_GAMMA_FIXED` (bse_feast.py:30) |
| `EllipseSweepConfig` / `EllipseSweepResult` | 39-57 | config = (n_quad1, n_quad2, gmres_tol) |
| `_match_eigenvalues` | 60-71 | third copy of the matcher |
| `_create_mesh_xy` | 74-80 | second duplicate of bse_feast.py:686-692 |
| `run_sweep(...)` | 83-230 | two hand-unrolled FEAST iterations: runner1(n_quad1) → RR → Ritz restart → runner2(n_quad2) → RR; both `_get_feast_runner` calls (115-123, 124-132) miss `dtype`; **lines 106-113 are the broken indentation block** |
| `print_summary` | 233-250 | nq1→nq2 table with summed matvecs |
| `main(argv)` | 253-321 | same bounds/window/truth pattern as zolo sweep |

## Flags / CLI args consumed

No LorraxConfig keys, no env vars (beyond the docstring-recommended
`JAX_PLATFORMS=cpu`, feast_sweep.py:8). All argparse:

**bse_feast_dense_debug** — `--n` (200), `--seed` (0), `--n-ritz` (4),
`--n-quad` (4), `--gamma` (0.4), `--e-min` (1.0), `--e-split` (2.0),
`--e-max` (80.0), `--feast-iter` (1), `--solve-noise` (0.0).

**feast_sweep** — `-i/--input` (required, COHSEX input file), `--n-val` (4),
`--n-cond` (4), `--full` (216-config grid), `--focused` (~66),
`--output` (feast_sweep_results.json), `--units-ev-per-ry` (13.6056980659),
`--full-diag` (compute truth instead of hardcoded constants).

**feast_zolo_sweep** — `-i/--input`, `--n-val` (4), `--n-cond` (4), `--px` (1),
`--py` (1), `--n-ritz` (8), `--gmres-max-iter` (20), `--gmres-tol` (list,
[1e-2, 1e-3]), `--n-quad` (list, [4,6,8]), `--feast-iter` (list, [1,2]),
`--rho-scale` (list, [0.5,1.0,1.5]), `--buffer` (0.05, E_max headroom),
`--n-lanczos` (10), `--seed` (0), `--units-ev-per-ry`.

**feast_ellipse_mixed_sweep** — `-i/--input`, `--n-val`, `--n-cond`, `--px`,
`--py`, `--n-ritz` (8), `--gmres-max-iter` (20), `--gmres-tol` (list, [1e-2]),
`--n-quad1` (list, [4]), `--n-quad2` (8), `--buffer` (0.05), `--n-lanczos`
(10), `--seed` (0), `--units-ev-per-ry`.

## Sharding / PartitionSpec assumptions

- Mesh is the standard 2-D `("x","y")` mesh (`_create_mesh_xy`, three copies).
  feast_sweep pins it to 1×1 (feast_sweep.py:570) — single device only.
- Excitonic vectors use `make_bse_shardings(mesh_xy).X =
  P(None, "x", "y", None)` over shape `(1, n_cond_pad, n_val_pad, nk)`
  (bse_ring_comm.py:46-49): leading batch axis replicated, **conduction sharded
  on 'x', valence on 'y'**, k replicated. Start vectors are
  `with_sharding_constraint(x, sh.X)` then `.astype(complex128)`
  (zolo:161-169, ellipse:141-149).
- Padded band counts `n_cond_pad`/`n_val_pad` come from the loader
  (multiples of the mesh axes); un-padded `n_cond`/`n_val` are used only in
  feast_sweep's dense reference slicing (feast_sweep.py:76-79).
- Preconditioner diagonal: P("x","y",None) over (nc,nv,nk)
  (bse_feast.py:113-115).
- `data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2,3,4), norm="ortho")`
  (zolo:123-124, ellipse:98-99) inherits W_q's sharding P("x","y",None,None,None)
  — μν ISDF axes sharded, k-difference axes replicated (FFT along replicated
  axes only).

## Host-vs-device residency

- `build_full_bse_matrix`: `jax.device_get`s all six tensors to host, slices
  off band padding, re-uploads unsharded; dense H (N², float64) lives on host
  and is filled by N single-column matvec round-trips (feast_sweep.py:93-107).
  Debug-scale only (N = n_cond·n_val·nk).
- Reduced Rayleigh-Ritz matrices are device_get to host numpy inside
  `_rayleigh_ritz` (bse_feast.py:359-361); the sweeps consume host-side
  `RitzResult` numpy fields.
- GMRES iteration counters device_get'd per config for statistics
  (zolo:183, ellipse:160,189).
- Hardcoded `EXACT_EIGENVALUES_EV` and all report/JSON assembly: host.

## TDA vs full-BSE

All four files are **TDA-only**:
- dense debug: real symmetric matrix, `np.linalg.eigh`.
- feast_sweep dense reference: `apply_bse_hamiltonian_single_device`
  (TDA serial matvec), comment "BSE-TDA is Hermitian" + real symmetrization
  (feast_sweep.py:109-110).
- zolo/ellipse: `build_bse_ring_matvec` (TDA ring path;
  `build_bse_ring_matvec_full` at bse_ring_comm.py:487 is never touched),
  4-D X with `sh.X` (the full-BSE path would use 5-D stacked (X,Y) with
  `sh.X_full`, bse_feast.py:561-567), and `_rayleigh_ritz`/
  `_build_ritz_vectors` left at their `use_tda=True` defaults.
The canonical CLI has moved past this: `bse_feast.main` defaults to full
non-TDA (`--tda` is opt-in, bse_feast.py:1131-1132), so the sweeps exercise
only the legacy default.

## Spin / nspinor

No explicit spin handling in any of the four files. The spinor axis exists
only inside the consumed matvecs — pair amplitude
`M[k,c,v,μ] = Σ_s conj(ψ_c[k,c,s,μ])·ψ_v[k,v,s,μ]` (bse_serial.py:27-29,
same contraction in the preconditioner, bse_feast.py:90-91) — and is summed
out before the excitonic space is formed. The leading size-1 axis of
X `(1, nc, nv, nk)` is a **batch** axis (einsum subscript `b`,
bse_serial.py:62-64), not spin.

## Coupling to gw/ and isdf/

Indirect only, via `bse_io.load_bse_data_from_restart_sharded`
(bse_io.py:358-…): reads the canonical **gw_jax restart bundle**
`tmp/isdf_tensors_*.h5` (`_find_restart_file`, bse_io.py:756-764) — datasets
`V_qmunu`, `W0_qmunu` (used only if attr `W0_ready`, else falls back to V,
bse_io.py:376-379), `psi_full_y`, `enk_full`. I.e. the sweeps consume the GW
stage's ISDF outputs (ψ at ISDF interpolation points, V_q(μν), W₀_q(μν));
they never import `gw.*` or `isdf.*` modules directly. `psi_c_Y` is `psi_c_X`
re-constrained to 'y' sharding — same values (bse_io.py:447-458), which is why
feast_sweep's serial reference using only the X copies is value-consistent.

## Suspects

### Broken (bugs)

1. **`feast_ellipse_mixed_sweep.py` does not compile** — IndentationError at
   line 113. Lines 106-113: the loop opens (`for idx, cfg in
   enumerate(configs, start=1):` line 106, `print` line 107 at loop depth),
   then the multi-line `print(` at 108-112 is dedented to function level and
   line 113 re-indents to loop depth. `python3 -m py_compile
   src/bse/feast_ellipse_mixed_sweep.py` → `IndentationError: unexpected
   indent (line 113)`. Any `python -m bse.feast_ellipse_mixed_sweep` or import
   dies before argparse.

2. **`_get_feast_runner` called without the required `dtype` argument** —
   `feast_zolo_sweep.py:141-149` and `feast_ellipse_mixed_sweep.py:115-123,
   124-132` pass 7 positionals `(matvec, data, n_quad, n_ritz, max_iter, tol,
   ry_to_ev)`; the signature (bse_feast.py:217-227) is `(matvec, data, n_quad,
   n_ritz, max_iter, tol, ry_to_ev, dtype, use_conjugate_symmetry=True)` with
   `dtype` non-defaulted → `TypeError: missing 1 required positional argument:
   'dtype'` on the first config. Present already at the restore commit
   fe5e3e8 (2026-04-29), i.e. the files were restored pre-broken.

3. **`feast_sweep` silently produces zero results** — `run_feast_ritz` is
   called (feast_sweep.py:173-186) without `quadrature=` or
   `lambda_min_eV/lambda_max_eV`; the default is `quadrature="zolotarev"`
   (bse_feast.py:476) which raises `ValueError("Zolotarev quadrature requires
   spectral bounds …")` (bse_feast.py:582-587). The blanket
   `except Exception … continue` (feast_sweep.py:187-189) swallows it for
   every config, so the sweep "completes" printing FAILED per point and an
   empty report ("No config achieved <1 meV accuracy"). The γ values it sweeps
   are only consumed by the ellipse branch (bse_feast.py:592) anyway.
   Ironically `estimate_spectral_bounds_sharded` — the function that would
   supply the missing bounds — is imported at feast_sweep.py:33 and never
   called (sole grep hit in the file).

4. **`SweepResult.n_physical` reads a field `RitzResult` no longer has** —
   feast_sweep.py:208 `n_physical=rr_data.n_physical`; `RitzResult`
   (bse_feast.py:294-304) has fields `ritz_evals, ritz_coeffs, s_evals,
   rayleigh_quotients, rel_residuals, n_total, s_floor` — no `n_physical`
   (the s-cutoff redesign replaced truncation with regularization,
   bse_feast.py:313-320). Latent AttributeError that fires as soon as bug 3
   is fixed.

### Dead

- All four modules have **zero callers at HEAD** by full-mechanism grep
  (direct imports, `__init__` re-exports — none exist —, `python -m`
  invocations across sandbox runs/skills/scripts/docs/templates and lorrax_D
  src/tests/tools/scripts/docs, pyproject, tests): the only inbound edges are
  the two `EXACT_EIGENVALUES_EV` imports **within this same group**. Deleted
  once as "dead/abandoned" (a0da0a5), restored broken (fe5e3e8). Their
  functionality is superseded by `bse_feast.main()`: `--quadrature`
  zolotarev/ellipse choice (bse_feast.py:1124-1126), mixed n_quad schedule
  `[n_quad1] + [n_quad2]*(feast_iter−1)` (bse_feast.py:1272 — exactly what
  feast_ellipse_mixed_sweep prototyped), rho_scale absorbed into
  `feast_zolotarev_quadrature` (bse_feast.py:921). `bse_feast_dense_debug` is
  the only one that still runs, and it needs nothing from LORRAX.

### Redundancy

- `_zolotarev_quadrature_custom` (zolo:63-81) ≡ `feast_zolotarev_quadrature`
  (bse_feast.py:916-963) post-rho_scale-upstreaming.
- `_create_mesh_xy` ×3: bse_feast.py:686-692 (canonical, imported by
  feast_sweep:32), zolo:98-104, ellipse:74-80 (identical copies).
- `_match_eigenvalues` ×3: feast_sweep:118-143 (5 % window margin variant),
  zolo:84-95 and ellipse:60-71 (identical no-margin copies).
- `GAMMA_FIXED = 0.2` (ellipse:36) duplicates `ELLIPSE_GAMMA_FIXED`
  (bse_feast.py:30).
- Two independent `feast_ellipse_quadrature` implementations reachable from
  bse/: `solvers.quadrature` (production, via bse_feast wrapper :965-973) and
  the dense-debug local copy (:26-35). Same math, verified index-by-index.

### Weird / conventions (not bugs)

- **Hardcoded truth**: `EXACT_EIGENVALUES_EV` (feast_sweep.py:551-554) is
  valid only for the isdf_tensors_600 / n_val=4 / n_cond=4 / 3×3×1 dataset;
  feast_sweep can override with `--full-diag`, zolo/ellipse cannot — their
  meV "accuracy" columns are meaningless on any other input.
- **Real-arithmetic TDA convention**: `build_full_bse_matrix` allocates H as
  float64 (feast_sweep.py:93) and assigns complex matvec output into it
  (line 107) — numpy silently discards the imaginary part (ComplexWarning),
  then symmetrizes with `H.T` not `H.conj().T` (line 110). Exact only while
  the TDA Hamiltonian is real in this basis (the codebase's stated
  convention: "BSE-TDA eigenvectors are real", bse_feast.py:453); a genuinely
  complex H (spinor / no-TRS gauge) would yield a wrong reference spectrum
  with no error raised.
- Positional eigenvalue matching (sorted-order pairing, all three matchers)
  misattributes errors when FEAST misses or picks up an extra state — fine
  for a sweep heuristic, not a validation gate.
- `matvec` cost accounting `iters.sum() + n_ritz·n_quad` (zolo:188,
  ellipse:164,193) mirrors bse_feast.py:609 ("+1 initial matvec per solve").
- `import sys` mid-function (feast_sweep.py:163) inside the sweep loop.
