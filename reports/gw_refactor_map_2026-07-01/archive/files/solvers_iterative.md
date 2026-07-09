# Refactor map — src/solvers/ iterative-solver group

Scope: davidson.py, lanczos.py, chebyshev.py, minres.py, cg_posdef.py, projectors.py,
quadrature.py, dos.py, `__init__.py`. (pseudobands.py, pseudobands_v2.py,
sternheimer_precond.py, sternheimer_solve.py are in this directory but NOT in this
group's assignment; they are referenced below only as callers.)

Global observations for the whole group:

- **No file I/O anywhere in this group.** Every module is pure in-memory numerics
  (jax/numpy). Only side effects are `print()` progress lines.
- **No CLI flags / env vars consumed.** No argparse, no `os.environ` reads.
- **Nothing imports the `solvers` package top-level.** Grep over src/, tests/,
  tools/, scripts/ for `from solvers import` → zero hits; every consumer imports
  submodules directly (`from solvers.davidson import davidson`, etc.). The
  `__init__.py` re-export layer is dead weight and it eagerly imports
  pseudobands/pseudobands_v2 (heavy) on any `import solvers`.
- Physics-free by design: matvec callables come from psp/ (DFT H) or bse/ (BSE ring
  matvec). Category for most of the group is "generic iterative linear-algebra
  kernels (physics-free solver layer)".
- `src/solvers/docs/` holds pseudobands design docs (out of scope for this group).

---

## src/solvers/__init__.py (36 loc)

**Purpose.** Package facade re-exporting davidson, lanczos, chebyshev, dos,
pseudobands(_v2), quadrature symbols under `solvers.*`.

**Entry points / callers.** None — grep shows zero `from solvers import X` or
`import solvers` in src/tests/tools/scripts. All consumers import submodules
directly.

**Functions.** None; just imports + `__all__`.

**Dead suspects.**
- The entire re-export layer: no package-level importers found
  (grepped `from solvers import`, `import solvers$`, `solvers\.` across
  src/, tests/, tools/, scripts/).
- `__all__` is stale relative to lanczos.py: `block_lanczos_eig_jit` and
  `block_lanczos_eig_jit_converged` (the two variants production bse_lanczos.py
  actually uses) are NOT exported; the un-jitted `block_lanczos_eig` is.
- Does NOT export minres, cg_posdef, projectors, sternheimer_* at all —
  half the directory is invisible to the facade.

**Weird.** Eager `from solvers.pseudobands import ...` makes `import solvers`
pull the whole pseudobands machinery; combined with zero users, this facade is a
delete-or-fix candidate in the refactor.

---

## src/solvers/davidson.py (352 loc)

**Purpose.** Shape-agnostic block Davidson eigensolver (after QE cegterg.f90) for
the lowest n_eig eigenvalues of a Hermitian operator given matvec/precond/init
callables. Trailing axes are arbitrary and may be sharded; all contractions use
ellipsis einsum so sharding propagates without collectives. Used both for DFT NSCF
diagonalization (psp) and BSE absorption (bse).

**Category.** generic iterative eigensolver (physics-free solver layer).

**Entry points.**
- `davidson` <- psp/run_nscf.py:229, bse/davidson_absorption.py:177,
  bse/bse_lanczos.py:224 (Davidson branch), bse/test_davidson_bse.py.
- `warmup_davidson_jit` <- psp/run_nscf.py:197, bse/davidson_absorption.py:170,
  bse/bse_lanczos.py:216, bse/test_davidson_bse.py:243.
- `_to_host` (private) <- bse/test_davidson_bse.py:295 (test reaches into private).

**Function table.**
| function | role |
|---|---|
| `_to_host(arr)` | multi-process-safe device→host copy via `process_allgather(tiled=False)` row 0 |
| `_generalized_eigh(A,B)` | Av=λBv via Cholesky reduction, +1e-12·I regularization |
| `_ritz_and_residuals(V,HV,n_eig)` | jitted: Gram project → gen-eigh → Ritz vecs → residuals → per-state norms |
| `_default_precond(R,eig)` | identity precond, normalized residuals |
| `_orthonormalise_batch(P)` | jitted Cholesky self-orthonormalization of batch axis |
| `_ortho_expand(V,P)` | jitted CGS2 against V + Cholesky self-orthonormalization |
| `warmup_davidson_jit(...)` | pre-compile `_ritz_and_residuals` at every subspace size (avoids in-loop recompiles) |
| `davidson(apply_H, ...)` | main driver loop: ritz → precond → converge-check (host) → ortho-expand → restart at m_max |

**I/O.** None. Prints per-iteration progress when verbose.

**Dead suspects.** None — everything reachable.

**Weird code.**
- `_generalized_eigh` line 77 / `_ortho_expand` line 175 / `_orthonormalise_batch`
  line 145: magic `1e-12` Cholesky jitter (three copies of the same constant).
- `davidson` lines 313-318: convergence counts only the *leading contiguous*
  converged block (`for i ...: if conv[i]: n_conv=i+1 else break`) — an interior
  unconverged state blocks credit for all converged states above it. Intentional
  (locked ordering) but undocumented; looks like a bug at first read.
- Line 299 `X = V` "in case the loop never runs" — with `max_iter>=1` the loop
  always runs; defensive dead assignment.
- Restart (line 344) collapses to Ritz vectors *after* concatenating P, so one
  wasted apply_H(P) per restart cycle's final expansion is kept in V but then
  discarded.
- `_orthonormalise_batch` duplicates the second half of `_ortho_expand`
  (copy-paste of the Cholesky block).

**Redundancy suspects.**
- `_orthonormalise_batch` ≡ tail of `_ortho_expand` (same code, could call one
  kernel with V=empty).
- Davidson vs lanczos.py: two full lowest-eigenpair solver families maintained in
  parallel; bse_lanczos.py contains a runtime switch between them.

---

## src/solvers/lanczos.py (582 loc)

**Purpose.** Lanczos eigensolvers for lowest eigenpairs of a Hermitian matvec.
FIVE variants: `simple_lanczos_eig` (Python loop, full reorth),
`lanczos_eig_jit` (fori_loop, partial reorth), `block_lanczos_eig` (Python-loop
block), `block_lanczos_eig_jit` (fori_loop block), and
`block_lanczos_eig_jit_converged` (while_loop with Ritz-stability convergence
checks every `check_every` iters).

**Category.** generic iterative eigensolver (physics-free solver layer).

**Entry points.**
- `simple_lanczos_eig` <- solvers/dos.py:85,90 (spectrum-bound estimation),
  bse/bse_lanczos.py:92, bse/bse_jax.py (re-export).
- `lanczos_eig_jit` <- bse/bse_lanczos.py:87,273, bse/bse_jax.py.
- `block_lanczos_eig` <- bse/bse_lanczos.py:83, bse/bse_jax.py, solvers/__init__.
- `block_lanczos_eig_jit` <- bse/bse_lanczos.py:300.
- `block_lanczos_eig_jit_converged` <- bse/bse_lanczos.py:267,293.
- `_build_block_tridiag` (private) — internal to the two jit block variants.

**Function table.**
| function | role |
|---|---|
| `block_lanczos_eig` | Python-loop block Lanczos, FULL reorth vs all stored blocks, host-side T assembly |
| `simple_lanczos_eig` | single-vector Python-loop Lanczos, full reorth; used for spectrum bounds |
| `lanczos_eig_jit` | fori_loop single-vector, rolling n_reorth-window partial reorth |
| `_build_block_tridiag` | fori_loop assembly of block-tridiagonal T (avoids HLO unroll) |
| `block_lanczos_eig_jit` | fori_loop block Lanczos, fixed max_iter |
| `block_lanczos_eig_jit_converged` | while_loop block Lanczos + periodic Ritz-value convergence check, returns (evals, evecs, n_iter) |

**I/O.** None. `block_lanczos_eig` prints on convergence.

**Dead suspects.** None with zero callers, but `block_lanczos_eig` (non-jit,
line 28) and `simple_lanczos_eig`'s bse usage are legacy paths kept behind
bse_lanczos.py method switches — candidates for pruning to the two jit block
variants + `simple_lanczos_eig` (still needed by dos.py).

**Weird code.**
- `block_lanczos_eig` lines 91-93: full reorthogonalization against *every*
  stored block each iteration (O(j) per iter) right after the standard three-term
  subtraction — makes the three-term recurrence redundant; the jit variants use a
  rolling window instead. Classic old-path/new-path divergence.
- `block_lanczos_eig_jit_converged` lines 541/570: magic `LARGE = 1.0e6` diagonal
  shift to mask inactive T blocks — silently wrong if the physical spectrum
  reaches ~1e6 (fine for Ry/eV scales, but a convention landmine).
- `lanczos_eig_jit` line 281: `Q.at[:, min(j+1, max_iter-1)].set(...)` silently
  overwrites the last column on the final iteration.
- `simple_lanczos_eig` line 190-192: mutates local `max_iter` on breakdown to
  truncate T — subtle host-side control flow.
- Eigenvector re-normalization after Ritz rotation in every variant (norms
  clipped at 1e-15) — hides loss of orthogonality rather than flagging it.

**Redundancy suspects.**
- Five Lanczos variants; `block_lanczos_eig` vs `block_lanczos_eig_jit` are the
  same algorithm in Python-loop vs fori_loop form (explicit "Same algorithm as"
  docstrings). `block_lanczos_eig_jit` vs `..._jit_converged` duplicate the whole
  Lanczos `step` body (lines 392-421 vs 494-517, near-verbatim copy).
- `solvers/__init__.py` exports only the older three, not the two jit block
  variants production code uses.

---

## src/solvers/chebyshev.py (291 loc)

**Purpose.** Kernel Polynomial Method building blocks: Jackson damping
coefficients, jitted Chebyshev three-term recurrence, stochastic-trace moment
estimation, DOS reconstruction, and equal-weight spectral window partitioning.
Operates on a caller-rescaled H̃ with spectrum in [-1,1].

**Category.** generic spectral method / KPM kernel (physics-free solver layer).

**Entry points.**
- `jackson_coefficients` <- solvers/dos.py, solvers/pseudobands.py:82,
  pseudobands_v2.py:72,284, bse/bse_kpm.py:230, psp/kpm_dos.py:259.
- `make_chebyshev_recurrence` <- psp/kpm_dos.py:228 (warmup), and internally by
  `chebyshev_moments`.
- `chebyshev_moments` <- solvers/dos.py:166, bse/bse_kpm.py:216, psp/kpm_dos.py:244.
- `reconstruct_dos` <- solvers/dos.py:175, bse/bse_kpm.py:244, psp/kpm_dos.py:266.
- `partition_windows` <- bse/bse_kpm.py:253.

**Function table.**
| function | role |
|---|---|
| `jackson_coefficients(M)` | (M+1,) Jackson damping σ_p |
| `make_chebyshev_recurrence(matvec, n_moments)` | returns jitted fori_loop recurrence: x_rand → all moments |
| `chebyshev_moments(matvec, dim, M, n_random, ...)` | stochastic-trace loop over Rademacher vectors, host accumulate |
| `reconstruct_dos(mu, E_grid, center, hw)` | NumPy Chebyshev series → ρ(E) with 1/√(1-e²) weight |
| `partition_windows(E, dos, n_windows, ...)` | equal-mass quantile window edges from DOS CDF |

**I/O.** None. Per-vector progress prints.

**Dead suspects.** None.

**Weird code.**
- `reconstruct_dos` line 206: clip `E_tilde` to ±(1−1e-10) — magic epsilon
  guarding the 1/√(1−e²) singularity; DOS values at grid edges are artifacts.
- `chebyshev_moments` normalizes by `n_random * dim` (line 171) — trace
  convention documented only in psp/kpm_dos.py:72's cross-reference.

**Redundancy suspects.**
- `partition_windows` (here) vs `dos_weighted_windows` + `geometric_windows`
  (solvers/dos.py): three window-partitioning routines with different weighting
  heuristics split across two modules.
- Three KPM DOS front-ends downstream (solvers/dos.compute_dos,
  psp/kpm_dos, bse/bse_kpm) all re-implement the bounds→moments→damp→reconstruct
  sequence around these kernels.

---

## src/solvers/minres.py (294 loc)

**Purpose.** Batched (over band axis v) preconditioned + projected Paige–Saunders
MINRES for Hermitian indefinite/semidefinite systems, written for the Sternheimer
operator at ω=0 iterated inside the conduction subspace via a projector Π. Fixed
max_iter fori_loop, no early exit (JIT-shape stability).

**Category.** generic iterative linear solver (Sternheimer support layer).

**Entry points.**
- `minres` <- tests/test_sternheimer_solvers.py:165,187,229 ONLY.
  Production Sternheimer (psp/run_sternheimer.py) uses
  solvers/sternheimer_solve.py, which inlines its own CG core and never imports
  minres. `MinresInfo` likewise test-only.

**Function table.**
| function | role |
|---|---|
| `MinresInfo` | NamedTuple: res_norms, iters_used, converged |
| `_batched_dot` / `_batched_real_norm` | einsum 'vsG,vsG->v' inner product / norm |
| `_identity` | default precond/projector |
| `minres(apply_A, b, ...)` | wrapper: project b, run core, compute final residuals |
| `_minres_core(...)` | jitted Paige–Saunders loop: Lanczos + rolling 2-step Givens, dead-band masking |

**I/O.** None.

**Dead suspects.**
- The whole module is production-dead: grepped `minres` across src/, tests/,
  tools/, scripts/ — only tests/test_sternheimer_solvers.py imports it.
  Kept alive by its test file alone. It was the pre-`cg_posdef`/pre-inline-CG
  Sternheimer solver generation.

**Weird code.**
- Line 201: `_BETA_DEADBAND = 1e-14` magic dead-band; per-batch alive-mask
  machinery (lines 202-209, 282) exists solely to avoid NaN amplification for
  effectively-zero RHS bands — a symptom-level patch documented in comments.
- Line 115-117: `tol` is "informational only (no early exit)" — a tol parameter
  that does not control the iteration is an API trap.
- `iters_used=int(max_iter)` always — never the actual work done.

**Redundancy suspects.**
- `_batched_dot`, `_batched_real_norm`, `_identity` are verbatim-duplicated in
  cg_posdef.py (lines 52-61) and again (dot/norm) in sternheimer_solve.py
  (lines 163-168). Three copies of the same three helpers in one directory.
- minres.py vs cg_posdef.py vs sternheimer_solve.py `_sternheimer_core`: three
  generations of "solve the Sternheimer linear system" (projected MINRES →
  level-shifted CG → inlined pytree-op CG). Only the third is production.

---

## src/solvers/cg_posdef.py (191 loc)

**Purpose.** Batched preconditioned Fletcher–Reeves CG for Hermitian
positive-definite systems, written for the QE-style level-shifted Sternheimer
operator `H − ε + α_pv·P_val` (cgsolve_all.f90 port). Per-band freeze masks in a
fixed-max_iter fori_loop.

**Category.** generic iterative linear solver (Sternheimer support layer) —
currently dead code.

**Entry points.** NONE. Grepped `cg_posdef` across src/, tests/, tools/,
scripts/: the only hit outside the file itself is a *docstring cross-reference*
in solvers/sternheimer_solve.py:9 ("The generic solvers.cg_posdef.cg_posdef
takes apply_A, precond as callables ..."). No import, no call, no test.

**Function table.**
| function | role |
|---|---|
| `CGInfo` | NamedTuple: res_norms, iters_used, converged |
| `_batched_dot` / `_batched_real_norm` / `_identity` | duplicates of minres.py helpers |
| `cg_posdef(apply_A, b, ...)` | wrapper: norms, core call, final residual check |
| `_cg_posdef_core(...)` | jitted PCG fori_loop with per-band alive/freeze masking |

**I/O.** None.

**Dead suspects.**
- Entire file: zero importers, zero tests (evidence above). Its algorithm was
  re-implemented inline as `_sternheimer_core` in sternheimer_solve.py (which
  operates on a `SternheimerOp` pytree instead of callables, to be
  JVP-differentiable). Textbook "parallel old/new path" per the no-redundancy
  house rule; delete or fold into sternheimer_solve.

**Weird code.**
- `iters_used=int(max_iter)` always (same API lie as minres).
- Freeze logic: `alpha`/`beta` zeroed via `alive_mask` but `z_new`/`rho_new`
  still computed for dead bands every iteration (harmless, wasteful).

**Redundancy suspects.**
- Helper triplication with minres.py / sternheimer_solve.py (see above).
- Whole-file duplication of sternheimer_solve.py `_sternheimer_core` algorithm.

---

## src/solvers/projectors.py (116 loc)

**Purpose.** Factories for jitted subspace projectors used by the Sternheimer
solver: P_val = U U† (occupied), P_precond (Schur low-energy block), P_rest =
1−P_val−P_precond, Q_{k−q} = 1−P_val^{k−q}. All shapes (batch, nspinor, nG); one
shared jitted contraction kernel `_apply_P_U`.

**Category.** Sternheimer support layer (subspace projection machinery).

**Entry points.**
- `make_Q_kminq` <- psp/run_sternheimer.py:73, psp/tests/test_sternheimer_jvp.py:61,
  tests/test_sternheimer_solvers.py (multiple).
- `make_P_val` <- tests/test_sternheimer_solvers.py only.
- `make_P_precond` <- tests/test_sternheimer_solvers.py:102 only.
- `make_P_rest` <- tests/test_sternheimer_solvers.py:95,102 only.

**Function table.**
| function | role |
|---|---|
| `_apply_P_U(U, x)` | jitted U·(U†x) einsum pair — the single shared kernel |
| `make_P_val(U_val)` | occupied projector factory |
| `make_P_precond(U_extra)` | Schur T-block projector factory |
| `make_P_rest(U_val, U_extra)` | complement projector; U_extra=None → 1−P_val |
| `make_Q_kminq(U_val_kminq)` | math-notation alias of make_P_rest(U, None) |

**I/O.** None.

**Dead suspects.**
- `make_P_precond`, `make_P_rest`: production callers zero (grepped
  `make_P_precond|make_P_rest` in src/ excluding tests — no hits). They belong to
  the "Stage-2c Schur split" MINRES design that production abandoned along with
  minres.py. `make_P_val` similarly test-only.

**Weird code.**
- `make_Q_kminq` is an explicitly-documented duplicate of
  `make_P_rest(U, None)` — an alias kept "so driver code spells the operator 1:1
  with the math"; two names for one function.
- `@jax.jit` on `_apply_P_U` with U as a traced arg: docstring claims
  "closure-captured ... static JAX arrays" but U is passed positionally, so the
  factory-closure framing is cosmetic.

**Redundancy suspects.** `make_Q_kminq` ≡ `make_P_rest(·, None)` (self-admitted).

---

## src/solvers/quadrature.py (61 loc)

**Purpose.** Single function producing complex quadrature nodes/weights on the
upper half of an ellipse for FEAST-style contour integration of the spectral
projector (1/2πi)∮(zI−H)⁻¹dz. Pure NumPy.

**Category.** generic spectral method (FEAST contour support).

**Entry points.**
- `feast_ellipse_quadrature` <- bse/bse_feast.py:20 (imported as
  `_feast_ellipse_quadrature_generic`, wrapped at :965 by a WindowSpec adapter of
  the SAME NAME), bse/bse_pseudopoles.py:17, bse/feast_ellipse_mixed_sweep.py:24,
  solvers/__init__.py.

**Function table.**
| function | role |
|---|---|
| `feast_ellipse_quadrature(center, half_width, n_quad, gamma)` | midpoint-rule nodes on upper half-ellipse + weights incl. 1/(2πi)·dz factor |

**I/O.** None.

**Dead suspects.** None.

**Weird code.**
- Docstring header says "Ellipse trapezoid-rule quadrature" but lines 55-57
  implement the *midpoint* rule (θ_j = π(2j−1)/2N) — doc/code mismatch.
- Default `gamma=0.2` aspect ratio magic; bse_pseudopoles hard-codes gamma=0.2
  at call site while bse_feast pins `ELLIPSE_GAMMA_FIXED` separately.

**Redundancy suspects.**
- bse/bse_feast_dense_debug.py:26 defines a second, independent
  `feast_ellipse_quadrature(window, ...)` (debug copy), and bse_feast.py:965
  shadows the generic name with a WindowSpec wrapper — three spellings of one
  quadrature.

---

## src/solvers/dos.py (358 loc)

**Purpose.** Matrix-free KPM density of states end-to-end: Lanczos spectrum-bound
estimation, Chebyshev moments, Jackson damping, DOS reconstruction
(`compute_dos` → `DOSResult`), plus DOS-based spectral window partitioning for
pseudobands (`dos_weighted_windows`, `geometric_windows`,
`compute_window_partition`).

**Category.** generic spectral method (KPM DOS) + pseudobands windowing support.

**Entry points.**
- `compute_dos` <- solvers/pseudobands.py:308, pseudobands_v2.py:512.
- `estimate_spectrum` <- internal (compute_dos); exported via __init__ (no
  external caller found: grepped src/tests/tools/scripts).
- `dos_weighted_windows` <- solvers/pseudobands.py:328, pseudobands_v2.py.
- `geometric_windows` <- solvers/pseudobands.py:332.
- `compute_window_partition` <- solvers/pseudobands.py:417, pseudobands_v2.py.
- `DOSResult`, `WindowPartition` <- pseudobands.py / pseudobands_v2.py type hints.

**Function table.**
| function | role |
|---|---|
| `DOSResult` | dataclass: grid, ρ, bounds, moments |
| `estimate_spectrum(apply_H, dim)` | short Lanczos for E_min; Lanczos on −H for E_max; pad 2% |
| `compute_dos(apply_H, dim, ...)` | bounds → rescaled matvec → moments → Jackson → reconstruct |
| `dos_weighted_windows(E, ρ, E_cross, E_max, ...)` | Altman-SI equal-error window placement over DOS CDF; tau or target-count bisection |
| `WindowPartition` | dataclass: boundaries, n_eff, E_mean |
| `compute_window_partition(dos, boundaries)` | per-window ∫ρ and mean energy |
| `geometric_windows(E_cross, E_max, F)` | ε_j = (1+F)^j geometric ladder (free-electron heuristic) |

**I/O.** None. Verbose prints (bounds, DOS integral).

**Dead suspects.**
- `estimate_spectrum` has no external callers (only compute_dos internally +
  __init__ re-export); grepped `estimate_spectrum` in src/tests/tools/scripts.
  Borderline — reasonable public API, but currently internal-only.

**Weird code.**
- Line 262: `eps_bar = 0.5 * (trial + e_lo) - E_cross + E_cross` — subtract-then-
  add of E_cross is a literal no-op, surrounded by three comment lines of E_F
  hand-wringing ("center from E_F=0 proxy"). Either the intended formula was
  `− E_cross + eps_cross` (distance from Fermi level) and this is a real
  windowing-metric bug, or it's leftover experimentation; the window metric
  currently uses the *absolute* window-center energy.
- Lines 256-268: per-boundary brute-force scan over `np.linspace(..., 500)`
  trial points with early break — O(500·N_S) heuristic; comment says "binary
  search" but it's a linear scan.
- Line 286: bisection over tau via geometric mean in [1e-30, 1e10], accepting
  |n_w − target| ≤ 1 — loose tolerance, and `_place_windows(tau_mid)` is
  re-called after the loop with the last midpoint even when the loop never broke.
- `estimate_spectrum` runs full `simple_lanczos_eig` twice (H and −H) instead of
  one Lanczos read off both ends of the tridiagonal spectrum.

**Redundancy suspects.**
- Window partitioning triplicated: `partition_windows` (chebyshev.py, equal
  mass), `dos_weighted_windows` (equal error), `geometric_windows` (geometric) —
  plus consumers psp/kpm_dos.py and bse/bse_kpm.py wrapping the same
  moments→damp→reconstruct flow that `compute_dos` already packages.

---

## Cross-file summary for the refactor

**Keep (production-load-bearing):** davidson.py, lanczos.py (jit block variants +
simple), chebyshev.py, dos.py, quadrature.py, projectors.make_Q_kminq.

**Delete/merge candidates:**
1. cg_posdef.py — zero callers, superseded by sternheimer_solve inline CG.
2. minres.py — test-only, superseded twice over.
3. projectors.make_P_precond / make_P_rest / (make_P_val?) — test-only, tied to
   the abandoned Schur-MINRES design; make_Q_kminq is the one production entry.
4. lanczos.block_lanczos_eig (Python-loop) — legacy path behind bse method switch.
5. `__init__.py` facade — unused and stale; either fix exports or drop.
6. Deduplicate `_batched_dot`/`_batched_real_norm`/`_identity` (3 copies) and the
   Lanczos `step` body (2 copies), `_orthonormalise_batch` vs `_ortho_expand`.

**Bug-check before refactor:** dos.py:262 no-op `− E_cross + E_cross` in the
window-error metric (possible real windowing bug affecting pseudoband quality).
