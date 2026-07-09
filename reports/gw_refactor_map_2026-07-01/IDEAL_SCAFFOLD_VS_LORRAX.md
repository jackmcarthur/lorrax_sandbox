# The idealized GW scaffold vs. LORRAX — what physics forces the difference

_2026-07-09. For the author, deciding whether to revise the top-level driver for physics
transparency. Written from a 4-track code-anchored research pass (call-sequence map, quadrature
story, hidden analytic structure, revision draft); file:line refs are `src/gw/` at HEAD.
Companions: MAP.md (stage taxonomy), SIGMA_PPM_MAP.md (the Σ flow in depth)._

## 0. The scaffold, and the verdict in one paragraph

The reference point, quoted exactly:

```
G(t)               = G[wfns, energies, (occupations)]
V                  = simple math (ignoring isdf basis complications)
Chi(omega)         = Chi[G(t)]
W(omega)           = W[Chi(omega), V]
(Plasmon model)    B, Omega = fit[W(omega)];   W(t) = W[B, Omega, t]
Sigma(omega)       = Sigma[G(t), W(t)]
wfns, energies     = update_H[Sigma(omega)]
```

The headline, before any taxonomy: **this scaffold already exists in the tree.** The
self-consistency path — `screening_requests_for` → `compute_sigma_xc`
(`sigma_dispatch.py:88`) → `update_H` inside `sc_iteration.gw_iteration_map`
(`sc_iteration.py:247`) — *is* this sequence, nearly line for line, and a one-shot G0W0 is
provably its first iteration. The ~900-line `gw_jax.main()` is an older, inlined second copy
of the same sequence (its own comment at `:417` concedes SC "would re-do this work on
iter 1"). So the question is not whether the physics *can* be written as your seven lines —
it already is, once — but which deviations from those lines are **missing physics** (the
scaffold must grow a line), which are **representation** (bundle and hide), and which are
**history** (delete the duplicate). Roughly: three real physics omissions (§2), one deep
representation truth (§3), two benign retypings (§4), and a body of history that a gated
two-phase revision removes (§5).

## 1. The map

One G0W0 run (GN-PPM), each actual call tagged against the ideal. `main()` has ~10 physics
call sites; more than half its lines sit downstream of all Σ physics (writers, restart,
debug tables).

| # | Ideal line | Actual (file:line) | Tag | Why it deviates |
|---|---|---|---|---|
| 0 | — (absent) | ζ-fit + V build: `prepare_isdf_and_wavefunctions` (`gw_jax.py:173` → `gw_init.py:561`) | REPR | the ISDF basis must exist before any two-point object; V = V[ζ[ψ]] |
| 0′ | — (absent) | `HeadResolver` (`:164`) | **PHYS** | v(q→0) divergence excised from all body tensors; carried as a scalar track (§2b) |
| 1 | `G(t) = G[wfns, E, occ]` | `build_G_tau(ψ_xn, ψ_yr, enk, t, mask)` (`greens_function_kernel.py:34`) — called *inside* every consumer, never passed | REPR | G is virtual: rank-structured ψψ*, materialized only as phases at ~10 τ-nodes (§3) |
| 2 | `V = simple math` | `compute_V_q` (`gw_init.py:298`) | REPR | bilinear in ζ over the G-sphere; the "simple math" survives inside |
| 3 | `Chi(ω) = Chi[G(t)]` | `build_static_quadrature` + `compute_chi0` (`w_isdf.py:488, 637`) | REPR | the quadrature IS the ω/t axis, solved on G's spectral range (§3) |
| 4 | `W(ω) = W[Chi, V]` | `solve_w` per-q LU (`w_isdf.py:396`) + IBZ slice/unfold around it (`gw_jax.py:203-293`) | REPR/HIST | Dyson solve is verbatim; the 90-line IBZ choreography in the driver is movable bookkeeping |
| 5 | `B,Ω = fit[W(ω)]` | `fit_ppm(W0, W_probe, V)` (`ppm_sigma.py:194`, called inside `ppm_pipeline:388`) | **PHYS** | fit runs on W−V (the pole model must never see bare V); χ is evaluated at exactly the two frequencies a one-pole ansatz is determined by; invalid poles get a policy (§2a) |
| 6 | `W(t) = W[B,Ω,t]` | `_build_W_t_q` inside the τ-kernel (`ppm_tau_kernel.py:204`) | REPR | analytic per-node evaluation; no W(t) object exists between fit and Σ |
| 7 | `Sigma(ω) = Sigma[G,W]` | `compute_cohsex_sigma` (`:364`) ⊕ `compute_sigma_c_ppm_omega_grid` (`ppm_pipeline:399`) ⊕ head injection (`:410-420`) | **PHYS** | Σ is three constructors, not one (§2a); the branch/window structure is Σ's own analytic structure (§3) |
| 8 | `update_H[Sigma]` | qp-solver dispatch (`gw_jax.py:481-769`): one_shot_dft / fixed_point / self_consistent + QSGW Hermitization + `kin_ion` double-counting | **PHYS** | three inequivalent definitions of "the QP energy" (§2c) |
| 9 | — (absent) | degenerate-set averaging (`:746`), writers, restart | NUM/HIST | numerical symmetry restoration; movable |

Two argument-audit exemplars stand for the rest. `solve_w(V_q, chi0_q)` is essentially at
the ideal signature — two physics arguments, everything else layout. At the other pole,
`compute_sigma_c_ppm_omega_grid(wfns, ppm, meta, mesh, ppm_cfg, quad, ω_grid, …)` takes no
G and no W at all: it takes **spectra and poles as data** (E_nk, B_q, Ω_q, masks) and
derives its own discretization — the deep reason is §3, not sloppiness. The remaining
argument noise (mesh, memory, print_fn) is scaffolding already headed into bundles.

## 2. Physics the ideal is missing

### 2a. Σ is three constructors, and the fit line is coupled to the Σ line

`Sigma[G(t), W(t)]` under-specifies three ways.

**Σ = Σ_x ⊕ Σ_c ⊕ head.** Bare exchange Σ_x = −G_occ·V is exact and static — it must be
computed from V directly, never through the pole model. The correlation part is built from
**W − V** (the pole ansatz fits the *screening correction*, which is pole-like; bare V has
completely different analytic structure). And the q→0 head is a third, diagonal-in-band
channel (§2b). One ideal line hides two exact evaluations and one model evaluation with
different inputs each.

**The four branches are Σ's analytic structure, not bookkeeping.** With E_F splitting the
spectrum, each pole of Σ_c sits at S = E_A + Ω with E_A measured from E_F — occupied and
empty intermediate states put their poles on *opposite sides* of the real axis
(time-ordering of ⟨Tψψ†⟩). Evaluating Σ_c(ω) on the real axis therefore splits into
{occ, empty} × {±ω} branches, where per branch the denominator ω̃ ∓ S either can cross zero
(real scattering — a crossing quadrature) or is sign-definite (a Laplace quadrature)
(`ppm_windows.py:16-28`). A corollary the code documents and honors: the tempting global
identity Σ_c(−ω) = −Σ_c(ω)* is **false** (it holds per pole term only), so the −ω half is
computed explicitly, not conjugated.

**Invalid poles couple `fit` to `Sigma`.** Where the two-point fit yields Ω² < 0 the pole
ansatz has failed for that (q,μ,ν) mode, and something must be *decided*: drop it, pin it at
2 Ry, or demote it to its static-COHSEX limit (−½W_c0 per mode — the BGW default, now ours).
That policy is a per-mode verdict on analytic structure that the Σ constructor must receive
alongside (B, Ω). The ideal's clean handoff `fit → W(t) → Sigma` has no slot for it, and this
exact gap is where a real bug lived (three layers each assuming a different policy).

### 2b. The head is a miniature GW pipeline running beside the body

`V = simple math` hides that v(q+G) = 4π/|q+G|² is sampled at q=0, G=0 — an **infinite
sample with finite integration weight** (the mini-BZ average of an integrable divergence).
Worse, W(q→0) = ε⁻¹₀₀(q̂,ω)·v(q) depends on the *direction* of approach in an anisotropic
crystal: "W at q=0" is not a number until an angular average is specified. LORRAX does the
only clean thing: the divergent element is zeroed out of every (q,μ,ν) body tensor, and the
finite weight travels as a parallel scalar track — with its own source resolution
(`HeadResolver`), **its own plasmon-pole fit** (`fit_head_ppm`), and **its own Σ**, which is
closed-form and band-diagonal because M_nm(q→0, G=0) = δ_nm (`head_correction.py:611-686`).
Order ~1 eV/band on Si; it has had its own physics bug (the fixed sign flip in the
negative-Ω² branch). It is the compensating measure of a distributional limit — it can never
become a *stage* of the body pipeline, only a channel threaded through every line:

```
V_body, v_head    = V[ζ]                        # head excised analytically
(B,Ω), (B_h,Ω_h)  = fit[W_body−V], fit[w_head]
Sigma(ω)          = Σ_x + Σ_c[B,Ω] + Σ_head[B_h,Ω_h,occ]   # head: diagonal in band
```

### 2c. `update_H` names a physics choice, not a step

Σ(ω) is non-Hermitian and ω-dependent; "new wfns, energies" requires an extraction recipe,
and the three recipes are **inequivalent physics**: `one_shot_dft` (everything at E_DFT,
Z-linearized — textbook G0W0, the default), `fixed_point` (diagonal on-shell solve
E = h₀ + ReΣ(E)), `self_consistent` (QSGW: Kotani–van Schilfgaarde Hermitization of Σ(ω) →
static Σ_xc, iterate). Beneath all three sits double-counting bookkeeping the ideal omits:
H_QP = (H_DFT − V_xc) + V_H[ρ] + Σ_xc — the V_xc subtraction and Hartree re-add are as much
part of `update_H` as the eigh. This axis was, until this week, smeared across a bool and an
orphaned flag; it is now the `qp_solver` enum, and it belongs *visibly* in the scaffold
because it changes what "quasiparticle energy" means.

## 3. The representation truth: the quadrature IS the object

The deepest single reason `Sigma[G_callable, W_callable]` cannot exist here: **G(t), χ(ω),
and W(t) are never evaluatable functions.** Each exists only as a finite minimax node set
{τ_ℓ, α_ℓ} *solved against the object's own pole structure*. `build_static_quadrature` reads
x_min = the gap and x_max = the full transition bandwidth **off the actual E_nk** before
anything is evaluated, then solves a minimax exponential-sum approximation of 1/x on
[1, R = x_max/x_min] (`w_isdf.py:512`, `minimax_screening.py:532`). χ₀ is then literally
`χ₀ = −2 Σ_ℓ α_ℓ Σ_vc |M|² e^{−τ_ℓ(E_c−E_v)}` — the Laplace identity 1/x = ∫e^{−xτ}dτ,
discretized on nodes placed where G's poles actually are. A node set solved on
[gap, bandwidth] to tolerance ε *is* a discretization of χ's branch cut (the particle-hole
continuum); choosing the nodes is resolving the analytic structure. That is why the Σ
constructor takes spectra and poles as data and derives its own discretization: the node
solver must see the pole positions before any evaluation exists to wrap in a callable.

The compression this buys is the whole method: a dense t/ω grid able to resolve the same
structure would need O(R) ≈ 10³–10⁵ evaluations of nk·μ²-sized objects; minimax needs
**~12–20**, with certified error ε ≈ 0.31·exp[−N(3.55/ln R + 0.68)].

Two honesty notes the ideal should absorb. First, `Chi(omega)` is computed at **exactly two
frequencies** — ω = 0 and one probe iω_p — which is precisely the information content a
one-pole model is determined by. The code is more honest than the ideal here; the true line
is `W₀, W_p = W[Chi[G], V] at the fit's two nodes`. Second, G(t) is the one ideal line that
can **never** be restored as a passed intermediate: G is rank-structured through ψψ* pair
densities, and materializing G(r,r′,τ) — even in the μ basis, per node — is exactly the
O(N²) object the low-scaling method exists to avoid. But its *signature* exists verbatim as
the one shared primitive `build_G_tau(ψ, E, t, mask)` that every consumer (χ₀ kernel, Σ
τ-kernel, COHSEX) calls internally. The ideal's first line is real; it is just a subroutine,
not a value.

## 4. What is benign: two structure-preserving retypings

**ISDF.** Every object in Hedin's equations is a two-point function O(r,r′); ISDF retypes
them all, uniformly, to (q, μ, ν) tensors — *once, before the scaffold starts*. After that,
every equation keeps its algebraic form: χ₀ = G_c·G_v* stays an elementwise product, the
Dyson equation stays a μ×μ linear solve, the PPM fit stays elementwise, W(τ) = B·e^{−iΩτ}
verbatim. The scaffold needs exactly one new first line — `ζ = ISDF_fit[ψ]` — and V becomes
V[ζ]. This is why "ignoring isdf basis complications" was the right instinct: it is a basis
choice, not a physics deformation.

**IBZ symmetry.** One symmetry-table object and one canonical unfold action shared by
ψ/ζ/V/W make the group action a wrapper around constructors, not a change to them. Benign
*iff* the single-action discipline holds — the one cautionary tale being the TRS-blind
unfold bug, which came precisely from a second, parallel sym path.

## 5. Verdict: the corrected scaffold, and whether to revise the driver

First the target, honestly amended. The minimal scaffold that carries the real physics is
twelve lines, not seven — and every line exists as a current function:

```
ζ, V_body, v_head   = ISDF_V[ψ]                                   # basis + Coulomb, head excised
quad                = minimax[E_nk]                               # the t-axis, solved on G's spectrum
χ0, χp              = Chi[G(τ), quad] at ω ∈ {0, iω_p}            # G virtual (build_G_tau inside)
W0, Wp              = W[χ0, V_body], W[χp, V_body]                # per-q Dyson solve
B, Ω                = fit[W0−V, Wp−V]  (+ invalid-pole policy)    # pole model on the screening part
B_h, Ω_h            = fit[w_head(0), w_head(iω_p)]                # the head's own fit
Σ(ω)                = Σ_x[G_occ, V_body]                          # exact, static
                    + Σ_c[E_nk, occ, B, Ω, quad]                  # 4-branch τ-integration
                    + Σ_head[B_h, Ω_h, occ]                       # band-diagonal
E, U                = update_H[Σ; qp_solver]                      # one_shot_dft | fixed_point | self_consistent
```

**The decision** is whether `gw_jax.main()` should read as those lines. Three options:

- **A. Document only** (this memo is the map). Against it: the duplication is live — one-shot
  main() re-implements what `sigma_dispatch.compute_sigma_xc` already does for the SC path,
  and this opacity has already produced defects (the qp_solver trichotomy was found smeared
  across a bool + an orphaned flag; the invalid-pole policy across three layers).
- **B. Phase 1 — pure moves, ~350 lines out of main()**: the 90-line IBZ slice/unfold block
  into a `solve_w` wrapper; restart flush into `persist_restart`; the ~100-line freq-debug
  table into `gw_output`; degeneracy-averaging into a helper. Zero seam changes; the existing
  e2e gates (incl. IBZ-equivalence) pin bit-identity.
- **C. Phase 2 — the real unification**: one-shot main() consumes the SC path's
  `screening_requests_for` + `compute_sigma_xc` dispatch; `fit_ppm` lifts to top level
  (small seam — its inputs already come from main); the three QP branches collapse into
  `solve_qp(...)`. Gate: SC-iteration-1 ≡ one-shot on a fixture, plus a basis-rotation
  regression (the one documented silent-breakage risk in the QP block).

**Recommendation: B then C, each gated** — because C is *deletion of a duplicate pipeline*,
which is the standing no-redundancy rule, not new architecture; the transparency is a
by-product of removing the second copy. With three physics-integrity guarantees as the
don't-list: **don't materialize G(t)** (it is the method's central absence); **don't promote
the head to a stage** (thread the channel); **don't evaluate W on an ω-grid to match the
ideal** (amend the ideal — two frequencies is the honest input of a one-pole model). Done
this way, the driver's top level reads as the twelve lines above, one level down reads as
representation (ISDF, minimax, IBZ), and the physics choices that used to hide in the
plumbing — invalid-pole policy, QP definition, head channel — are the *visible* arguments.
