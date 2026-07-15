# src/bse/bse_kpm.py + src/solvers/chebyshev.py + src/solvers/dos.py — deep-read notes (408 + 292 + 359 LOC)

Audit date: 2026-07-15, lorrax_D checkout. **Note on base**: task stated
`agent/slate-linalg-ffi @ e18d0e5`; the working tree is actually at
`agent/ppm-fit-conditioning @ adc2197`. `git diff e18d0e5..HEAD` over all
seven files touched by this audit (the three targets plus `bse_ring_comm.py`,
`bse_feast.py`, `bse_io.py`, `bse_pseudopoles.py`) is **empty** — every
claim below holds identically at both commits.

## Purpose

Stochastic KPM (Kernel Polynomial Method) density-of-states for the BSE
Hamiltonian, used to place DOS-equal-mass spectral windows for the FEAST and
pseudopole solvers, plus the generic Chebyshev/KPM library it delegates to.
Three layers:

- `solvers/chebyshev.py` — physics-free KPM primitives (Jackson damping,
  on-device three-term recurrence, stochastic trace, DOS reconstruction,
  equal-mass window partition).
- `solvers/dos.py` — physics-free end-to-end DOS driver (`compute_dos`) plus
  the pseudobands window planners (`dos_weighted_windows`,
  `geometric_windows`, `compute_window_partition`). Consumed by
  `solvers/pseudobands*.py`, NOT by the BSE path.
- `bse/bse_kpm.py` — BSE-specific glue: rescaled matvec factory, masked
  Rademacher vectors over the padded transition space, spectral-bounds call,
  moments → Jackson → DOS → windows → plot/npz.

Math as written in code:

```
H̃ x            = (H x − c·x) / h                      (bse_kpm.py:57-59)
T_0 x = x;  T_1 x = H̃ x;  T_p x = 2 H̃ T_{p−1} x − T_{p−2} x
                                                      (chebyshev.py:79-92)
μ_p            = (1/(R·D)) Σ_r Re⟨x_r | T_p(H̃) x_r⟩   (chebyshev.py:83-92,171)
σ_p (Jackson)  = [(M−p+1)·cos(πp/(M+1)) + sin(πp/(M+1))/tan(π/(M+1))]/(M+1)
                                                      (chebyshev.py:44-47)
ρ(E)           = [μ_0·T_0(ẽ) + 2 Σ_{p=1}^{M} μ_p·T_p(ẽ)] / (π·h·√(1−ẽ²)),
                 ẽ = clip((E−c)/h, ±(1−1e−10))         (chebyshev.py:205-221)
```

with R = `n_random`, D = `dim` (the argument), M = `n_moments`, c/h =
center/half-width. Because of the D division, **ρ integrates to 1, not to
D** — see "Normalization convention" below.

Random vectors (bse_kpm.py:79-97): Rademacher ±1 per element on the padded
transition space `x[b=1, c∈n_cond_pad, v∈n_val_pad, k∈nk]`, then
`x·mask` where `mask[:, :n_cond, :n_val, :] = 1` else 0. Per-element trace
identity: E[x_i x_j*] = δ_ij·mask_i, so Σ_r ⟨x_r|T_p|x_r⟩/R estimates
tr(P T_p(H̃) P) over the *physical* (unpadded) subspace. Pad rows stay
exactly zero through the recurrence: the D-term is diagonal
(`ΔE(b,c,v,k) = eps_c[k,c] − eps_v[k,v]`, elementwise multiply) and the
V/W terms contract through ψ tensors whose pad columns are zero
(`_pad_axis_to_multiple`, bse_io.py:452-456) — so the (possibly
out-of-[−1,1]) pad diagonal never contaminates the moments. μ_0 =
(#unmasked)/D = 1 exactly since `bse_dim` is computed from the *unpadded*
counts (bse_kpm.py:205-210).

Windows: `partition_windows` (chebyshev.py:229-291) builds a trapezoid CDF
`cdf = concat([0], cumsum(0.5·(ρ_{i+1}+ρ_i)·ΔE_i))` and places edges at
`np.interp(linspace(0, total, n+1), cdf, E)` — equal-mass quantiles,
scale-invariant in ρ (so the unit-normalized ρ is harmless here).

Category: **diagnostics / planning stage** — spectral windowing feeder for
FEAST (`bse_feast --windows-kpm`) and pseudopoles; `solvers/*` doubles as
infra for the pseudobands initiative (`psp`, `solvers/pseudobands*.py`).

## Entry points (grep over src/, tests/, tools/, docs/, sandbox skills/scripts/runs)

| symbol | callers (grep evidence) |
|---|---|
| `bse_kpm.main` | `python -m bse.bse_kpm` (bse_kpm.py:406-407, docstring line 11); `bse_jax.py:513-536` (`--kpm-dos` builds an argv list and calls `bse_kpm.main(kpm_argv)`); sandbox patch copy `scripts/bse_src_patches/bse_jax.py:364-388` |
| `bse_kpm.run_kpm_dos` | `bse_feast.py:1200-1219` (`--windows-kpm`, `emit_outputs=False`); `bse_pseudopoles.py:620-637` (`--windows-kpm`) |
| `bse_kpm.make_bse_h_tilde`, `make_bse_random_vector` | internal to `run_kpm_dos` only (repo-wide grep: no other hits) |
| `chebyshev.jackson_coefficients` | bse_kpm.py:230; dos.py:170; pseudobands.py:82; pseudobands_v2.py:72,284; psp/kpm_dos.py:259; re-export `solvers/__init__.py:5` |
| `chebyshev.chebyshev_moments` | bse_kpm.py:216; dos.py:166; psp/kpm_dos.py:244; re-export `__init__.py:7` |
| `chebyshev.make_chebyshev_recurrence` | internal (chebyshev.py:156); psp/kpm_dos.py:228 (warm-up compile); re-export `__init__.py:6` |
| `chebyshev.reconstruct_dos` | bse_kpm.py:244; dos.py:175; psp/kpm_dos.py:266; re-export `__init__.py:8` |
| `chebyshev.partition_windows` | bse_kpm.py:253 **only** (+ re-export `__init__.py:9`) |
| `dos.compute_dos` | pseudobands.py:308; pseudobands_v2.py:512; re-export `__init__.py:11` |
| `dos.estimate_spectrum` | **no external callers** — only dos.py:147 (inside `compute_dos`) + `__init__.py:11` re-export |
| `dos.dos_weighted_windows` | pseudobands.py:328; docs `solvers/docs/ritz_pseudobands_implementation.md:96` |
| `dos.geometric_windows` | pseudobands.py:332 |
| `dos.compute_window_partition` | pseudobands.py:417 |
| `dos.DOSResult` / `WindowPartition` | pseudobands.py:58-59,264; pseudobands_v2.py:46,474 |

No pytest coverage anywhere: `grep -rn "kpm\|chebyshev\|solvers.dos" tests/`
→ only `test_sternheimer_solvers.py` hits (different `solvers.*` modules).
No sandbox `runs/**/*.sh` or `skills/` references found (bounded search;
`runs/` tree is enormous — first 2000 .sh/.yaml files grepped, zero hits).
`docs/architecture/codebase.md:116` lists `bse_kpm.py`.

## Function tables

### `src/bse/bse_kpm.py`

| function | lines | role |
|---|---|---|
| module init | 13-34 | imports; `jax.config.update("jax_enable_x64", True)` at import (line 32); `RY_TO_EV = 13.6056980659` (line 34) |
| `make_bse_h_tilde(matvec, data, e_center, half_width)` | 37-61 | closes the 8 BSE tensors from `data` over the ring matvec; returns `@jax.jit` `x ↦ (matvec(x, ψ…, ε…, W_R, V_q0) − c·x)/h`. Closure-captured device arrays become jaxpr constants. |
| `make_bse_random_vector(data, use_tda)` | 64-98 | masked Rademacher factory. TDA: shape `(1, n_cond_pad, n_val_pad, nk)`; non-TDA: `jnp.stack([x0, x1], axis=0)` → `(2, 1, ncp, nvp, nk)` with the mask stacked to match. dtype complex64 iff `eps_c` is float32. **No sharding constraint applied** (contrast `bse_feast.py:744-750` which pins `sh.X`); the jit'd matvec's `in_shardings` reshards on first use. |
| `run_kpm_dos(data, mesh_xy, …)` | 101-320 | driver: build ring matvec (122-138, **broken kwarg**, see Suspects), fp32 cast + `W_R = ifftn(W_q, axes=(2,3,4), norm='ortho')` (140-155), Lanczos bounds (159-166), eV overrides (174-179), rescale window (181-199), moments (216-224), Jackson (230-231), reconstruct on eV grid (234-245), `partition_windows` (253-257), plot/npz when `emit_outputs` (259-309), returns dict (311-320). |
| `main(argv)` | 323-403 | argparse (327-357), `timing.reset()`, `_create_mesh_xy(px,py)`, `_find_restart_file(input)`, `load_bse_data_from_restart_sharded(...)` (364-372), prints dims, `run_kpm_dos(...)` under `timing.section("kpm.total")`, timing report. |

Rescale-window math as written (bse_kpm.py:181-199): TDA →
`e_min_buf = max(0, e_min − 0.05·BW)`, `e_max_buf = e_max + 0.05·BW`,
`c = (e_max_buf+e_min_buf)/2`, `h = (e_max_buf−e_min_buf)/2`. Non-TDA →
`h = (1+buffer)·max(|e_min|,|e_max|)`, `c = 0` (symmetric ± spectrum
assumption for `[[A,B],[−B†,−A]]`).

### `src/solvers/chebyshev.py`

| function | lines | role |
|---|---|---|
| `jackson_coefficients(M)` | 31-48 | vectorized Jackson kernel σ_p, p = 0..M (standard form with N = M+1; σ_0 = 1). Host numpy. |
| `make_chebyshev_recurrence(apply_h_tilde, n_moments)` | 55-95 | `@jax.jit` recurrence: μ buffer float64 on device, `lax.fori_loop(2, M+1)` body computes `t_new = 2·apply_h_tilde(t_curr) − t_prev`, `μ_p = Re vdot(x, t_new)`. One device_get per random vector. Cost: M matvecs per vector (does **not** use the μ_{2p} = 2⟨t_p,t_p⟩ − μ_0 doubling trick that would halve matvecs). |
| `chebyshev_moments(apply_h_tilde, dim, n_moments, n_random, *, seed, dtype_real, make_random_vector, verbose)` | 102-172 | Python loop over R vectors, host accumulation, `mu /= n_random * dim` (line 171). Default vector = flat Rademacher of shape `(dim,)` when no factory given (147-151). |
| `reconstruct_dos(mu, E_grid, center, half_width)` | 179-222 | pure-NumPy Chebyshev sum on the grid, formula above; O(M·n_grid). |
| `partition_windows(energy_grid, dos, n_windows, *, energy_min, energy_max)` | 229-291 | mask → sort → clip ρ ≥ 0 → trapezoid CDF → equal-mass edges via `np.interp`; returns `(n_windows, 2)` `[E_lo, E_hi]` pairs in input units. Raises on <2 points or non-positive integral. |

### `src/solvers/dos.py`

| function | lines | role |
|---|---|---|
| `DOSResult` | 41-53 | dataclass: grid, ρ, bounds, c/h, raw+damped moments. |
| `estimate_spectrum(apply_H, dim, n_lanczos=30, pad_fraction=0.02, seed)` | 56-97 | E_min from `simple_lanczos_eig(apply_H, dim, n_eig=1, max_iter=n_lanczos)` (solvers/lanczos.py:138), E_max from the same on `−H` negated; pads both by `pad_fraction·span`. Single-device flat vectors. |
| `compute_dos(apply_H, dim, *, n_moments=2000, n_random=20, n_grid=10000, E_min, E_max, n_lanczos=30, seed, verbose)` | 100-188 | bounds → inline `apply_H_tilde` closure → `chebyshev_moments` → Jackson → `reconstruct_dos` on `linspace(E_min, E_max, n_grid)`. Prints `DOS integral: {trapezoid(rho,E)} states (dim={dim})` (178-180) — prints ≈ 1.0 by construction, see normalization note. |
| `dos_weighted_windows(E_grid, rho, E_cross, E_max, *, galerkin_order=1, tau, n_windows_target)` | 191-295 | Altman-SI-S36-style equal-error window placement: per-trial metric `n_eff · (Δ/√12)^{2k} / max(ε̄,1e−30)^{2k+1} ≥ τ` where `ε̄ = 0.5·(trial+e_lo)` (line 262 — the trailing `− E_cross + E_cross` is a literal no-op) and Δ = trial − e_lo. Despite the comment "Binary search for e_hi" (255) the search is a **linear scan** of 500 evenly spaced trials (256-269). τ-from-target via 50-step geometric bisection (284-295). |
| `WindowPartition` | 298-304 | dataclass: boundaries, n_eff, E_mean, N_S. |
| `compute_window_partition(dos, boundaries)` | 307-341 | per-window `n_eff_j = ∫ρ`, `E_mean_j = ∫Eρ/n_eff` (trapezoid); n_eff is a *fraction* of states — caller rescales (`pseudobands.py:418: n_eff = win_part.n_eff * dim`). |
| `geometric_windows(E_cross, E_max, F=0.10)` | 344-358 | `ε_j = (1+F)^j·ε_cross` boundaries. **Non-terminating if `E_cross ≤ 0`** (while-loop multiplies from 0 or negative); only caller passes `eps_cross = C_m·πB/(M·F) > 0` (pseudobands.py:318-332). |

## Flags / CLI args consumed

`bse_kpm.main` argparse (bse_kpm.py:327-357) — the only config surface; none
of the three files read `LorraxConfig` or env vars:

| flag | meaning | default |
|---|---|---|
| `-i/--input` | COHSEX input file; used to locate `tmp/isdf_tensors_*.h5` restart (`_find_restart_file`, bse_io.py:756) and as `input_file` fallback for kgrid/n_occ | required |
| `--n-val`, `--n-cond` | BSE valence/conduction band counts | 4, 4 |
| `--px`, `--py` | device mesh extents (μ/x, ν/y) | 1, 1 |
| `--n-moments` | Chebyshev order M (cost R·M matvecs) | 200 |
| `--n-random` | stochastic trace vectors R | 4 |
| `--n-lanczos` | min Lanczos steps for bounds | 100 |
| `--buffer` | fractional buffer on bounds | 0.05 |
| `--emin-ev`, `--emax-ev` | manual bound overrides (eV) | None |
| `--seed` | PRNG seed | 0 |
| `--n-energy-pts` | reconstruction grid size | 2000 |
| `--n-windows` | equal-mass window count | 10 |
| `--plot-file` | PNG path; `.npz` sibling derived from it | bse_dos_kpm.png |
| `--ry-to-ev` | Ry→eV conversion factor (yes, a flag) | 13.6056980659 |
| `--rpa` | `include_W=False` — drop the direct W term (D+V only) | off |
| `--tda` | TDA; **default is full non-TDA** (help text, line 354) | off |
| `--nohead` | pass `use_nohead=True` to loader (headless `V_qmunu_nohead`/`W0_qmunu_nohead` datasets) | off |

`bse_jax --kpm-dos` forwards its own `--kpm-n-moments/--kpm-n-random/
--kpm-n-lanczos/--kpm-window-count/--kpm-plot-file/--kpm-emin-ev/
--kpm-emax-ev` plus `--rpa`(-or-not-`--bse`)/`--tda` into this argv
(bse_jax.py:515-535). `bse_feast`/`bse_pseudopoles` call `run_kpm_dos`
directly with their own `--kpm-*`/`--windows-kpm-count` args.

## Sharding / residency

- `bse_kpm` **imports `make_bse_shardings` but never uses it**
  (bse_kpm.py:27; sole occurrence in file). All sharding is inherited:
  - loader output (bse_io.py:358-470): `psi_{c,v}_X` `(nk, nb_pad, ns, μ_pad)`
    sharded `P(None,None,None,"x")`, dual `psi_{c,v}_Y` copies on `"y"`;
    `V_q0` `(μ,ν)` `P("x","y")`; `W_q` `(μ,ν,nkx,nky,nkz)`
    `P("x","y",None,None,None)`; `eps_{c,v}` `(nk, nb_pad)` replicated.
  - the ring matvec's `jax.jit(..., in_shardings=(sh.X, …), out_shardings=sh.X)`
    (bse_ring_comm.py:612-627) pins trial vectors to
    `X = P(None,"x","y",None)` (b, c→x, v→y, k).
- Everything is **device-resident** jnp arrays — no io_callback host caches
  anywhere in this path. During `run_kpm_dos` the fp64 originals AND the
  fp32 copies (`_build_gmres_data_fp32`, bse_feast.py:281-291) are alive
  simultaneously; the bounds call then builds a **second** fp32 copy + W_R
  internally (see Suspects).
- `W_R = jnp.fft.ifftn(W_q, axes=(2,3,4), norm='ortho')` (bse_kpm.py:153)
  uses the plain jnp FFT on a `P("x","y",·,·,·)`-sharded array — the FFT'd
  axes are unsharded, but this is exactly the pattern
  `bse_ring_comm.py:397-405` documents as forcing an all-gather under the
  JAX partitioner bug (it uses `make_sharded_ifftn_3d` for the per-matvec
  FFTs). One-time setup cost here, so tolerable; same pattern in
  `estimate_spectral_bounds_sharded` (bse_feast.py:713-716).
- `solvers/chebyshev.py` is sharding-agnostic (vdot + fori_loop preserve
  whatever sharding `apply_h_tilde` maintains); moments land on host numpy.
  `solvers/dos.py` is single-device flat-vector code (used for the DFT/
  pseudobands Hamiltonian, not the sharded BSE operator).

## TDA vs full BSE

- `use_tda=True` → `build_bse_ring_matvec` (TDA `A` only,
  `Hx = D + V − W` as written at bse_ring_comm.py:583-590); vectors
  `(1, ncp, nvp, nk)`; rescale window clamped at 0 (bse_kpm.py:186).
- `use_tda=False` (the **default** of `bse_kpm.main`) →
  `build_bse_ring_matvec_full` for `S = [[A, B], [−B†, −A]]`
  (docstring bse_ring_comm.py:495; `X_out = A X + B Y`,
  `Y_out = −B X − A Y` under a stated "A and B are Hermitian" reuse,
  bse_ring_comm.py:660-672); vectors `(2, 1, ncp, nvp, nk)`; c = 0,
  symmetric ±h window. KPM on this non-Hermitian-but-real-spectrum operator
  is mathematically fine for the trace (tr T_p(S) = Σ_i T_p(λ_i) for any
  polynomial), though `Re vdot` per-sample variance and non-normal transient
  growth of `T_p(S̃)` are worse than the Hermitian case — undocumented.
- Spectral bounds, however, come from `estimate_spectral_bounds_sharded`,
  which **always Lanczos's the TDA operator** with TDA-shaped vectors
  regardless of `use_tda` (bse_feast.py:734-741 builds
  `build_bse_ring_matvec`, random vector `(1, ncp, nvp, nk)` at 745-750);
  the only non-TDA adjustment is `e_min_ry = −abs(e_max_ry)`
  (bse_feast.py:826-827). Full-BSE |Ω|_max is not bounded by λ_max(A) in
  general — only the 5% buffer absorbs the difference.
- Also mislabeled: `e_min_ry` printed as "E_min (Lanczos)"
  (bse_kpm.py:170) is actually the **diagonal** non-interacting minimum
  `min(eps_c) − max(eps_v)` (bse_feast.py:721-728), not a Lanczos result.
  The true lowest exciton sits *below* it by the binding energy, covered
  only by the buffer — and then excluded again from the window partition by
  `omega_min_eV = max(…, e_min_ry·ry_to_ev)` (bse_kpm.py:248-249).

## Spin / nspinor

No explicit handling in any of the three files. Transition-space vectors
`(b, c, v, k)` carry no spinor axis; `nspinor` lives inside the ψ tensors
`(nk, nb, ns, μ)` and the T-tensor's (s,t) axes inside the ring matvec
(bse_ring_comm.py:415-423). W_R/V_q0 are charge-channel `(μ,ν[,k])` tiles —
consistent with the bispinor roadmap note that BSE/W is charge-only today.
`solvers/*` are dimension-blind by design ("No physics knowledge",
chebyshev.py:10, dos.py:7).

## Units / conventions (LORRAX-internal vs BGW-compat)

- Internal energies in **Ry** (STATUS.md "Index ordering" §2); eV appears
  only at reporting/plotting boundaries via `RY_TO_EV`/`--ry-to-ev`.
  `partition_windows` runs on the eV grid and is divided back to Ry
  (bse_kpm.py:253-257); FEAST/pseudopoles re-multiply to eV for WindowSpecs.
- Valence axis: loader takes `val_indices = arange(n_occ−n_val, n_occ)`
  ascending (bse_io.py:436) — LORRAX-internal ordering; the BGW v-flip
  applies only at eigenvector file write, irrelevant here.
- DOS normalization convention: μ_p are divided by `dim`
  (chebyshev.py:171) so ρ integrates to **1**; downstream re-scales
  (`pseudobands.py:418`). This is a convention, not a bug — but it makes
  the bse_kpm plot label "DOS (states/eV)" (bse_kpm.py:270) and the
  `compute_dos` "DOS integral: … states" print (dos.py:178-180) wrong /
  confusing by a factor of `dim`.

## Cross-module coupling

- `bse/bse_ring_comm.py`: `build_bse_ring_matvec(_full)` (matvec),
  `make_bse_shardings` (imported, unused).
- `bse/bse_feast.py`: `estimate_spectral_bounds_sharded`, `_create_mesh_xy`,
  `_build_gmres_data_fp32` — bse_kpm reaches into two private helpers of a
  *sibling solver* for mesh construction and fp32 casting.
- `bse/bse_io.py`: `_find_restart_file` (private reach-in),
  `load_bse_data_from_restart_sharded`.
- Coupling to `gw/` is **indirect only**: the restart bundle
  `isdf_tensors_*.h5` (`V_qmunu`, `W0_qmunu` gated on `W0_ready` attr,
  `psi_full_y`, `enk_full`, `G0_mu_nu`) is produced by the gw_jax writer
  (bse_io.py:374-386). No `isdf/` imports.
- `solvers/lanczos.py`: `simple_lanczos_eig` (dos.py only).
- `common/timing.py`: sections `kpm.load`, `kpm.moments`, `kpm.total`.
- `psp/kpm_dos.py` is a **deliberate documented mirror** of bse_kpm for the
  DFT Hamiltonian ("The structure mirrors bse_kpm.py as closely as
  possible", psp/kpm_dos.py:10-13) sharing the `solvers.chebyshev` core —
  parallel-but-distinct-physics, not copy-paste cruft.

## Suspects

### BROKEN AT HEAD: phantom `v_couples_k` kwarg → TypeError on every KPM entry
`run_kpm_dos` passes `v_couples_k=bool(not include_W)` to both matvec
builders (bse_kpm.py:120,128,137), but the builders' signatures are
`(mesh_xy, nkx, nky, nkz, timed=False, low_mem=True, include_W=True)` with
no `**kwargs` (bse_ring_comm.py:340-348 and 487-495).
`grep -rn v_couples_k src/` hits only bse_kpm.py:120,128,137 and
bse_pseudopoles.py:231,239,248; `git log --all -S v_couples_k --
src/bse/bse_ring_comm.py` is **empty** — the parameter never existed in
ring_comm in any commit (it arrived with bse_kpm at 345cf0e and survived the
a0da0a5 cleanup / fe5e3e8 restore of pseudopoles). Consequence: *every*
call path — `python -m bse.bse_kpm`, `bse_jax --kpm-dos`,
`bse_feast --windows-kpm`, `bse_pseudopoles` — dies with
`TypeError: build_bse_ring_matvec() got an unexpected keyword argument
'v_couples_k'` before any compute. Fix is either delete the kwarg (RPA then
means "drop W, keep the existing q=0-only V") or implement the intended
k-coupled-V RPA kernel in ring_comm — note the *parsed-but-unread ≠ dead*
pattern: `v_couples_k = not include_W` encodes an intended physics
distinction (RPA exchange coupling k-points) that was never wired.

### Redundancy
- Bounds phase duplicates setup: `run_kpm_dos` builds fp32 data + `W_R`
  (bse_kpm.py:140-155) and its own matvec (122-138); then
  `estimate_spectral_bounds_sharded` internally rebuilds fp32 data + `W_R`
  (bse_feast.py:712-717) *and* a second TDA matvec (734-741). Two full fp32
  copies of ψ/W_q live simultaneously; two W_q IFFTs.
- `n_lanczos_max=max(n_lanczos, 50)` (bse_kpm.py:163) equals `n_lanczos`
  for the default 100 — the adaptive-E_max convergence loop in the bounds
  estimator degenerates to a fixed-iteration run with a single E_max
  evaluation (bse_feast.py:806-820 only checks after `len(alphas) >=
  n_lanczos`). Same pattern hardcoded at bse_pseudopoles.py:608.
- `chebyshev.make_chebyshev_recurrence` spends M matvecs for M+1 moments;
  the standard KPM identities (μ_{2p} = 2⟨t_p,t_p⟩ − μ_0,
  μ_{2p+1} = 2⟨t_{p+1},t_p⟩ − μ_1) would halve the dominant cost.

### Dead / unused
- `make_bse_shardings` import (bse_kpm.py:27) — never referenced in file.
- `dos.estimate_spectrum` — no callers outside `compute_dos` itself despite
  being re-exported and advertised in the module docstring (dos.py:11).

### Weird
- dos.py:262: `eps_bar = 0.5 * (trial + e_lo) - E_cross + E_cross` — literal
  no-op arithmetic left over from an E_F-offset experiment, with three lines
  of contradictory comments (263-265); ε̄ is "distance from Fermi level"
  only under the E_F = 0 proxy.
- dos.py:255 comment "Binary search for e_hi" over a 500-point linear scan.
- `geometric_windows` never terminates if `E_cross ≤ 0` (dos.py:354-357);
  safe with its only current caller.
- Window floor clamps at the *non-interacting* diagonal gap
  (bse_kpm.py:248-249): bound excitons below `min(ε_c)−max(ε_v)` (e.g. a
  strongly-bound LiF-like exciton) fall outside every KPM-weighted window
  handed to FEAST/pseudopoles. Possibly intended (exclude reconstruction
  ringing below the physical edge) but the buffer/DOS grid do extend below —
  the clamp uses the wrong "physical minimum".
- `mu_0 … (should be ~1.0)` print (bse_kpm.py:226): with masked Rademacher
  vectors μ_0 = 1 *exactly*; deviation would indicate a dim-mask mismatch,
  which the message undersells.
- Non-TDA path is the CLI default (`--tda` opts in, bse_kpm.py:353-354)
  while `run_kpm_dos`'s keyword default is `use_tda=True` (bse_kpm.py:117)
  — two defaults for the same knob depending on entry point;
  `bse_feast`/`bse_pseudopoles` pass their own `--tda` through explicitly.
