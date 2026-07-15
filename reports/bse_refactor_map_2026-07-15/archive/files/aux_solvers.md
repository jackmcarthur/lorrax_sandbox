# src/solvers/{minres,projectors,quadrature,sternheimer_precond,sternheimer_solve}.py — deep-read notes (294+116+61+138+385 = 994 LOC)

Audit date: 2026-07-15, lorrax_D checkout. Stated base agent/slate-linalg-ffi
(e18d0e5); actual HEAD is adc2197 on agent/ppm-fit-conditioning, but
`git diff e18d0e5..HEAD -- src/solvers/` is **empty** — the five audited files
are byte-identical to the stated base.

## Purpose

Five auxiliary-solver files in the physics-free `solvers/` package. They split
into **two unrelated programs**:

1. **Sternheimer linear-response primitives** (minres, projectors,
   sternheimer_precond, sternheimer_solve) — serve `psp/run_sternheimer.py`,
   the insulating DFPT driver for the χ_{G'0}(q, ω=0) head/wing column
   (psp/orbital-mag/W0 program). **Not on the BSE pipeline at all**, and not
   called by gw/ either. The production physics, signs as written in code:

   ```
   Sternheimer system (sternheimer_solve.py:59, QE cgsolve_all convention):
       A_v(x) = H_{k-q}·x − ε_{v,k}·x + α_pv·P_val^{k-q}(x)
       solve   A_v · δu = −b,     b = Q_{k-q} · V_pert(r) · u_{v,k}   (caller does NOT negate)
       α_pv = 2·(E_max − E_min) of occupied spectrum  ⇒ A positive-definite everywhere
       ⇒ plain level-shifted CG, per-band freeze, dead-band mask ‖b‖ ≤ 1e-14 → x=0.

   Schur / truncated-SoS warm start (sternheimer_solve.py:200):
       x^T_v = −Σ_m ⟨U_m|b_v⟩ / (eps_extra[m] − eps_v[v]) · U_m     (k·p sum-over-states
       over the M loaded conduction Ritz vectors; A_TT diagonal since U_extra ⟂ U_val)

   TPA preconditioner (sternheimer_precond.py:21):
       P⁻¹_{v,G} = TPA(x_{v,G}),  x_{v,G} = T_G / K̄²_v,
       TPA(x) = (27+18x+12x²+8x³) / (27+18x+12x²+8x³+16x⁴)
       T_G = |k−q+G|² (kinetic diag at k−q),  K̄²_v = ⟨ψ_{v,k}|T_k|ψ_{v,k}⟩ (source k)

   Projectors (projectors.py:8-11):
       P_U x[b,s,G] = Σ_m U[m,s,G]·(Σ_{s',G'} conj(U[m,s',G'])·x[b,s',G'])
       Q_{k−q} = 1 − P_val^{k−q};  P_rest = 1 − P_val − P_precond

   MINRES (minres.py, Paige–Saunders 1975, superseded in production):
       A_eff = Π ∘ apply_A ∘ Π,  b_eff = Π b;  rolling 2-step Lanczos+Givens,
       fixed max_iter fori_loop, no early exit.
   ```

2. **FEAST contour quadrature** (quadrature.py) — the only file on the BSE
   path. Host-side numpy nodes/weights for the FEAST spectral projector used
   by `bse/bse_feast.py`:

   ```
   P = (1/2πi) ∮ (zI − H)⁻¹ dz     over an ellipse [c−r_x, c+r_x], r_y = γ·r_x
   θ_j = π(2j−1)/(2N), j=1..N      (midpoint rule, upper half only)
   z_j = c + r_x·cosθ_j + i·r_y·sinθ_j
   w_j = (1/2N)·(r_y·cosθ_j + i·r_x·sinθ_j)
   ```
   Verified: w_j ≡ (1/2πi)·(dz/dθ)|_{θ_j}·Δθ with dz/dθ = −r_x sinθ + i r_y cosθ,
   Δθ = π/N — multiply by −i/−i: (π/N)(−r_x sinθ + i r_y cosθ)/(2πi)
   = (1/2N)(r_y cosθ + i r_x sinθ). Exact, counterclockwise (+2πi) orientation.
   Docstring "trapezoid-rule" (quadrature.py:23) is a mislabel; the nodes are
   midpoint (correctly noted at :55).

Category: **infra: iterative-solver library** — quadrature.py = BSE
eigensolver support; the four Sternheimer files = psp linear-response
primitives; minres.py + most of projectors.py = superseded/test-only.

## Entry points (grep over src/, tests/, tools/, scripts/, docs/, sandbox runs/skills/scripts)

| symbol | callers (grep evidence) |
|---|---|
| `minres` | **NONE in production.** Only `tests/test_sternheimer_solvers.py:170,192,234` (file is `pytestmark = pytest.mark.extra`, :23 — deselected by default) and frozen sandbox diagnostics `runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/stern_diag{3,4,5}.py`. Production abandoned it: `run_sternheimer.py:1174-1176` "CG avoids MINRES' pseudo-convergence-NaN pitfall". |
| `MinresInfo` | minres.py only. |
| `make_P_val` | tests/test_sternheimer_solvers.py:68-108; sandbox stern_check_H.py:16, stern_trace_cg.py:16, stern_diag_nopre.py:17. No src/ callers. |
| `make_P_precond` | tests/test_sternheimer_solvers.py:107-109 **only**. |
| `make_P_rest` | tests/test_sternheimer_solvers.py:100-110 **only**. |
| `make_Q_kminq` | `src/psp/run_sternheimer.py:73` — **imported but never called in that file** (only match); `psp/tests/test_sternheimer_jvp.py:61,128`; tests/test_sternheimer_solvers.py:87-102,193-198; sandbox stern_trace_cg.py, stern_diag_nopre.py. Production inlines the Q-einsums instead (run_sternheimer.py:973-980 `_q_apply`, :330-333/:645 `Q_of`). |
| `feast_ellipse_quadrature` | re-exported `src/solvers/__init__.py:14,35`; `src/bse/bse_feast.py:20` (import as `_feast_ellipse_quadrature_generic`, wrapped :965-973, used :592, :1243); `src/bse/bse_pseudopoles.py:17` and `src/bse/feast_ellipse_mixed_sweep.py:24` both import the **bse_feast wrapper**, not this module directly. |
| `compute_per_band_kinetic` | run_sternheimer.py:75→:1239; test_sternheimer_jvp.py:62,131; test_sternheimer_solvers.py:137. |
| `tpa_preconditioner_diag` | run_sternheimer.py:76→:979 (`jax.vmap(tpa_preconditioner_diag)` over k), :1386; test_sternheimer_jvp.py:62,132. |
| `make_tpa_preconditioner` | tests/test_sternheimer_solvers.py:148,235 **only** in repo; sandbox stern_diag{4,5,cg2}.py. Production uses the array form `tpa_preconditioner_diag`. |
| `SternheimerOp` | run_sternheimer.py:78,314,352,415,571,653-654 (incl. rebuilt-with-tiled-fields copies `op_3`/`op_b3`). |
| `sternheimer_solve` | run_sternheimer.py:78,363,465,553,583,730,779; test_sternheimer_jvp.py:243. |
| `cond_subspace_sos_solve` | run_sternheimer.py:776-777 (`sos_only` mode) — single caller. |
| `_apply_A_inline` (private) | reach-ins from run_sternheimer.py:315,343,414-469 (k·p RHS construction + residual diagnostics). |

No `python -m solvers.X` invocations exist (no `__main__` in any of the five
files; grep of runs/skills/scripts found none). `solvers/__init__.py` re-exports
**only** `feast_ellipse_quadrature` of these files — minres/projectors/
sternheimer_* must be imported by module path.

## Function tables

### minres.py (294 LOC)

| symbol | lines | role |
|---|---|---|
| `MinresInfo` | 49-53 | NamedTuple: `res_norms (batch,)`, `iters_used:int` (always = max_iter, :141), `converged (batch,) bool`. Comment ":52 final ‖b−Ax‖" is imprecise — actual residual computed :137 is the **projected** ‖Π(b − A x)‖. |
| `_batched_dot` | 60-62 | `<a_v,b_v> = Σ_{s,G} conj(a[v,s,G])·b[v,s,G]` via `einsum('vsG,vsG->v')`. |
| `minres` | 78-141 | driver: projects b, calls core, recomputes residual, builds info. `tol` is informational only (no early exit, :113-114). |
| `_minres_core` | 148-291 | `@jax.jit(static_argnames=('apply_A','precond','project','max_iter'))` — **callables are static args**, so every fresh closure retraces; exactly the retrace disease sternheimer_solve.py:8-14 documents for the old cg_posdef. Paige–Saunders recurrence verified index-by-index: ε_k = s_{k-2}β_k, δ̄_k = c_{k-2}β_k (:252-253); δ_k = c_{k-1}δ̄_k + s_{k-1}α_k, γ̄_k = −s_{k-1}δ̄_k + c_{k-1}α_k (:258-259); γ_k = √(γ̄²+β²), w_k = (z_k − δ_k w_{k-1} − ε_k w_{k-2})/γ_k, x_k = x_{k-1} + c_kη_k·w_k. Constrained preconditioner M̃ = Π·M·Π (:191, :241, Gould–Hribar–Nocedal). β₀ dead-band 1e-14 with per-band alive mask (:201-209, :282). |

### projectors.py (116 LOC)

| symbol | lines | role |
|---|---|---|
| `_apply_P_U` | 37-47 | shared jitted kernel. Per-element: `coefs[b,m] = Σ_{s,G} conj(U[m,s,G])·x[b,s,G]`; `out[b,s,G] = Σ_m coefs[b,m]·U[m,s,G]` ⇒ P = U·U†. Correct iff U orthonormal (stated assumption :16-19, unchecked). |
| `make_P_val` | 54-61 | P_val = U_val·U_val†, U_val `(nv, nspinor, nG)`. |
| `make_P_precond` | 64-73 | same kernel on U_extra (Schur T-block Ritz vectors). |
| `make_P_rest` | 76-93 | 1 − P_val [− P_precond]; `U_extra=None` branch = Q_val alias. |
| `make_Q_kminq` | 96-108 | 1 − P_val^{k−q}; math-notation alias of `make_P_rest(U, None)` (self-declared, :99). |

### quadrature.py (61 LOC)

| symbol | lines | role |
|---|---|---|
| `feast_ellipse_quadrature` | 17-61 | pure-numpy host function; returns `(z_nodes, w_weights)` complex128 `(n_quad,)` each, **upper half-ellipse only** (Hermitian conjugate symmetry left to callers). Formula verified exact (see Purpose). `gamma` = aspect r_y/r_x, default 0.2. |

### sternheimer_precond.py (138 LOC)

| symbol | lines | role |
|---|---|---|
| `_tpa` | 48-53 | Horner rational form; num = 27+x(18+x(12+8x)), den = num+16x⁴. Tests pin TPA(0)=1, TPA(1)=65/81, TPA(x)≈1/(2x) asymptote (test_sternheimer_solvers.py:121-134). |
| `compute_per_band_kinetic` | 60-76 | `K̄²_v = Σ_{s,G} T_G·|ψ_{v,s,G}|²`: `psi_abs_sq = Σ_s |U|²` (axis=1 = spinor, :75), then `einsum('nG,G->n')`. Note: evaluated at **source k** (T_diag at k), per the docstring rationale :24-27. |
| `tpa_preconditioner_diag` | 83-95 | array form `(nv, 1, nG_p)`: `x[v,G] = T_diag_kminq[G]/K̄²_v` (broadcast `[None,:]/[:,None]`), K̄²≤0 guarded →1. **The production form** (JIT-stable elementwise multiply). |
| `make_tpa_preconditioner` | 98-131 | callable-factory twin of the above, `precond(R) = R*weights`. Test-only survivor of the closure-based era. |

### sternheimer_solve.py (385 LOC)

| symbol | lines | role |
|---|---|---|
| `_STERN_DEBUG` | 44 | `bool(int(os.environ.get('STERN_DEBUG','0')))` — module-import-time read; gates a `jax.debug.print` per-solve convergence line (:301-310). |
| `SternheimerOp` | 53-126 | registered pytree class (`__slots__`), all-array children + aux `(fft_grid, U_extra is None)`; flows through jit/jvp without static retrace. `fft_grid` must stay a hashable tuple. Schur fields None-preserving unflatten (:119-126). |
| `_apply_A_inline` | 133-151 | A_v(x) = `apply_H_k_from_G(x,…)` − `eps_v[:,None,None]·x` + `alpha_pv·U_val(U_val†x)` (einsums 'msG,bsG->bm' / 'bm,msG->bsG' = P_val, same per-element math as projectors._apply_P_U). **Batch row b of x is band v** — requires `len(op.eps_v) == batch`, hence the explicit `jnp.tile(eps_v, 3)` when batching 3 cartesian directions (run_sternheimer.py:356-360, 578-581). |
| `_precond_inline` | 154-156 | `r * op.precond_diag` (TPA array). |
| `cond_subspace_sos_solve` | 171-185 | public alias — body is exactly `return _schur_initial_guess(op, b)`. |
| `_schur_initial_guess` | 188-217 | `coefs[v,m] = Σ_{s,G} conj(U_extra[m,s,G])·b[v,s,G]`; `denom[v,m] = eps_extra[m] − eps_v[v]`; `y = −coefs/denom` (|denom|>1e-8 clamp, else 0); `x0[v,s,G] = Σ_m y[v,m]·U_extra[m,s,G]`. Verified = closed-form T-block of A·x = −b given U_extra ⟂ U_val (so α_pv·P_val·U_m = 0) and H U_m = eps_extra[m] U_m. |
| `_sternheimer_core` | 220-311 | `@jax.jit(static_argnames=('max_iter','use_schur'))`; `rhs = −b`; PCG `while_loop` with per-band freeze (`still_alive = alive & (‖r‖ > tol·‖b‖)`, :280) and any-alive/i<max_iter continue condition (:291-297); dead-band ‖b‖≤1e-14 rows stay 0 (:262-263, alpha/beta zeroed via `jnp.where(alive,…)`). |
| `sternheimer_solve` | 318-342 | `@jax.custom_jvp(nondiff_argnums=(2,3,4))` public primitive; contract "Solve A_v·δu = −b" (:322). |
| `_sternheimer_solve_jvp` | 345-379 | implicit-differentiation JVP: primal solve, `A_dot_x = Ȧ·x` via `jax.jvp` of `_apply_A_inline` in op at fixed x, then tangent solve. **b-channel sign bug — see Suspects.** |

## Flags / config keys / env / CLI

Directly read by these five files: **`STERN_DEBUG`** (env, sternheimer_solve.py:44,
default '0', read once at import) — nothing else. No LorraxConfig keys, no CLI.
All knobs arrive as function args; callers translate:
`run_sternheimer.py` argparse `--tol` (default 1e-6) / `--max-iter` (default 80)
→ `tol`/`max_iter`; `--n-cond-bands` > 0 → `use_schur` (run_sternheimer.py:1212);
`--sos-only` → `cond_subspace_sos_solve` routing (:776-779, :1523). FEAST side:
`bse_feast.py` maps WindowSpec + `n_quad` (per-iteration `quad_schedule`) +
`ELLIPSE_GAMMA_FIXED = 0.2` (bse_feast.py:31) into `feast_ellipse_quadrature`.

## Sharding / residency / spin / TDA

- **Sharding: none.** No PartitionSpec, Mesh, or shard_map anywhere in the five
  files. The consuming Sternheimer driver is explicitly single-GPU
  ("Multi-GPU sharding will land in a later commit; for now the entire
  wavefunction buffer lives on device 0", run_sternheimer.py:32-34). Batched-k
  is via `jax.vmap` (run_sternheimer.py:906-929), not devices.
- **Residency:** all solver arrays are plain device jnp arrays; SternheimerOp
  children (T_diag, V_scf box `(nx,ny,nz)`, G-lists, vnl_Z/vnl_E, U_val,
  precond_diag) live on device and are vmapped per k. quadrature.py is
  host-only numpy (nodes/weights later `jnp.asarray`'d by bse_feast:600-601).
  No io_callback / host caches here.
- **Spin/nspinor:** every kernel is shaped `(batch, nspinor, nG)`; inner
  products sum both s and G ('vsG,vsG->v', 'msG,bsG->bm'), so nspinor ∈ {1,2}
  is transparent. TPA collapses spinor in `Σ_s |ψ|²` (sternheimer_precond.py:75).
  No collinear nspin=2 axis anywhere (consistent with LORRAX spinor convention).
  Spin degeneracy factor (spin_factor = 2 if nspinor==1) is applied by the
  driver's χ prefactor, not here (run_sternheimer test:166-170 convention).
- **TDA vs full BSE:** only quadrature.py touches BSE and it is TDA-agnostic —
  it returns upper-half nodes; **full-BSE handling lives in the caller**:
  `bse_feast.py:595-597` conjugate-augments both nodes and weights
  (`z_nodes = concat([z, conj(z)]); w_weights = concat([w, conj(w)])`) when
  `not use_tda`; same pattern in bse_pseudopoles.py:279-281 and
  feast_ellipse_mixed_sweep.py.
- **Coupling to gw/ and isdf/: none.** Only cross-package imports:
  `psp.dft_operators.apply_H_k_from_G` (sternheimer_solve.py:46) and the
  bse/ consumers of quadrature listed above. tests/test_sternheimer_solvers.py:19-21
  states outright: "Sternheimer/W0 solver tooling (psp/orbital-mag program),
  not the GW pipeline".

## I/O

None in any of the five files. All pure compute; the driver owns sternheimer.h5.

## Suspects

### Bug (CONFIRMED numerically)

- **`_sternheimer_solve_jvp` b-tangent sign flip** — sternheimer_solve.py:376.
  The primitive's contract is A·x = −b (`rhs = -b`, :249). Implicit
  differentiation of A(θ)x(θ) = −b(θ) gives `Ȧx + Aẋ = −ḃ ⇒ Aẋ = −ḃ − Ȧx`.
  The rule instead derives from the wrong premise "Let A x = b" (:349) and
  computes `rhs_tangent = -(b_dot - A_dot_x)` (:376), which after the core's
  internal negation solves `Aẋ = +ḃ − Ȧx`: the Ȧ (op) channel is correct, the
  ḃ (b) channel has the **wrong sign** (returns +A⁻¹ḃ instead of −A⁻¹ḃ).
  Numerical proof (degenerate op: V_scf=0, vnl=0, U_val=0, α_pv=0 ⇒ A=diag(T),
  CPU, this audit):
  ```
  primal   max|x − (−A⁻¹b)|      = 1.1e-16      (contract holds)
  jvp      max|x_dot − (−A⁻¹db)| = 3.6           (wrong)
  jvp-flip max|x_dot − (+A⁻¹db)| = 2.3e-16       (exactly sign-flipped)
  fd       max|x_fd  − (−A⁻¹db)| = 1.3e-10       (FD ground truth)
  ```
  Fix: `rhs_tangent = b_dot + A_dot_x` (and correct the :349-355/:372-375
  docstrings). Why nothing caught it: every in-repo jvp path has b_dot ≡ 0 —
  `psp/tests/test_sternheimer_jvp.py` passes a frozen precomputed `b`
  (:129→:194) so only the op channel is FD-validated; the production
  `chi_col_contrib_at_kvec_traced` frozen-source path uses constant `b_stack`
  (run_sternheimer.py:975-980); and the k·p/S-tensor paths build tangent RHS
  manually with the correct sign, bypassing the custom JVP b-channel
  (run_sternheimer.py:559-563: "passing b = grad_i[v] yields x = −A⁻¹·grad_i[v]
  … matches physical ẋ_i = −A⁻¹·∂b/∂q_i"). The channel fires only via the
  documented "unfrozen source" path `Vu_G_preQ` + `U_val_grad_kp`
  (run_sternheimer.py:698-701: "Pass this … to get the correct q-dependence of
  the source"), used today only by the sandbox bench
  `runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/s_tensor_fd_bench.py:124-125`
  — whose S-tensor values are therefore suspect (its q=0 first-derivative
  checks are 0 by symmetry in either sign and cannot detect this).

### Dead / test-only

- **minres.py — entire module (294 LOC) has zero production callers.** Grepped
  all mechanisms at HEAD: src (none), package `__init__` (not re-exported),
  tools/scripts/docs (none), `python -m` (no `__main__`), string dispatch
  (none), sandbox runs/skills/scripts (only frozen stern_diag{3,4,5}.py
  diagnostics). Only live references: `tests/test_sternheimer_solvers.py`
  (marker `extra`, deselected by default). Superseded by the level-shifted-CG
  primitive on purpose (run_sternheimer.py:1168-1176 explains the α_pv trick
  makes A PD so "CG avoids MINRES' pseudo-convergence-NaN pitfall"); its
  static-callable jit signature also carries the per-(k,q) retrace disease
  the CG primitive was written to kill (sternheimer_solve.py:8-14).
- **projectors.py: `make_P_val`/`make_P_precond`/`make_P_rest` test-only**;
  `make_Q_kminq`'s only src/ reference is an **unused import**
  (run_sternheimer.py:73 — no call in the file; production inlines the two
  einsums at :973-980 and :330-333, and the callable-consuming
  `build_sternheimer_source` (run_sternheimer.py:119) is itself only called by
  psp/tests/test_sternheimer_jvp.py:129). The whole module is effectively a
  test/diagnostic vocabulary at HEAD.
- **`make_tpa_preconditioner`** (sternheimer_precond.py:98-131): callable twin
  of `tpa_preconditioner_diag`; production only uses the array form
  (run_sternheimer.py:979,1386). Repo callers: tests only. Parallel old/new
  pair — the docstring of the array form even says it "skips the
  Python-callable wrapper" (:89-91). Candidate for the no-redundancy rule.

### Redundancy / weird

- `cond_subspace_sos_solve` (sternheimer_solve.py:171-185) is a one-line alias
  of `_schur_initial_guess` with a second docstring — public/private duplicate
  naming for one formula; single caller (run_sternheimer.py:777).
- `src/bse/bse_feast_dense_debug.py:26-35` re-implements
  `feast_ellipse_quadrature` inline (numpy copy, WindowSpec signature) instead
  of importing solvers.quadrature — second source of truth for the node/weight
  formula inside the repo.
- Layered wrapper: bse callers reach the generic quadrature only through
  `bse_feast.feast_ellipse_quadrature` (bse_feast.py:965-973, WindowSpec →
  center/half_width), and the `solvers/__init__.py:14` re-export has no direct
  importers — the generic symbol is consumed solely via the wrapper.
- Stale MINRES strings in the consumer: run_sternheimer.py:1076 "tol, max_iter :
  MINRES knobs" and :1445 CLI description "via projected MINRES" — the driver
  runs level-shifted CG.
- quadrature.py:23 "trapezoid-rule" vs :55 "Midpoint quadrature" — nodes are
  midpoint; weights verified correct either way.
- `MinresInfo.res_norms` comment (minres.py:52) says ‖b − Ax‖ but the value is
  the Π-projected residual (:137); `iters_used` is always `int(max_iter)` (:141).
- `minres.tol` is documented "informational only" (:113-114) yet is a jit
  closure constant via the outer non-jit driver — harmless, but `converged` is
  the only thing it feeds.
- tests/test_sternheimer_solvers.py is `pytest.mark.extra` (:23) — the entire
  unit coverage for these primitives is deselected in the default suite.
