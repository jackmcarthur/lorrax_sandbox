# Minimax Crossing-Solver Audit and Speedup

**Date**: 2026-07-21  
**Branch**: `agent/minimax-solver-speed`  
**Commit**: `4d71e2d`  
**Scope**: nonlinear generation of real-axis crossing-regularization quadratures

## Summary

The existing solver was not globally optimal, but it already used the right broad
numerical family: variable projection for separable nonlinear least squares, SVD
for the exponential basis, Levenberg-Marquardt node updates, Lawson reweighting,
and a final minimax LP. The dominant avoidable cost was an LP/backward-elimination
initializer that took most of the runtime and lost to the analytic Chebyshev start
on every profiled shipped HGL case.

The safe patch removes that initializer, tries the productive Chebyshev start before
the uniform fallback, and stops Lawson passes or additional starts once the requested
dense-grid tolerance is certified. It retains the current approximate projected
Jacobian and linear scan over node count because more ambitious changes exposed
serious coefficient-conditioning failures.

## Solver assessment

| Component | Assessment |
|---|---|
| Variable projection | Appropriate and literature-standard for linear weights plus nonlinear nodes. |
| SVD linear solve | Appropriate for the near-collinear exponential basis. |
| LM update | Reasonable, but custom and based on the Kaufman approximate projected Jacobian rather than a modern constrained trust-region implementation. |
| Lawson reweighting | Useful; some hard cases require all five passes, while easy cases can be certified earlier. |
| Final dense LP | Appropriate for optimizing the infinity norm at fixed nodes. |
| LP/backward-elimination start | Expensive and consistently discarded for the profiled HGL family; removed. |
| Initial nodes | Chebyshev nodes are the strongest current start; uniform support remains a fallback. |
| Search over `N` | Linear search is conservative but currently necessary because local minima make achieved error nonmonotone in `N`. |

The implementation is therefore well motivated, but it is not an implementation of
a globally convergent Remez method or an optimally parameterized nonlinear solver.
O'Leary and Rust describe the SVD/variable-projection basis and the exact residual
Jacobian; Hackbusch emphasizes transformed variables and continuation because
exponential-sum Remez iterations are sensitive.

## Performance

All timings are end-to-end `crossing_grids` generation for HGL at `eps = 1e-6`.
The baseline is the untouched pre-patch solver. Errors are the solver's dense-grid
maximum errors.

| A | N (old/new) | Error (old/new) | Time old | Time new | Speedup |
|---:|---:|---:|---:|---:|---:|
| 30 | 32 / 32 | 3.702e-7 / 3.702e-7 | 11.49 s | 2.68 s | 4.3x |
| 40 | 48 / 48 | 1.528e-7 / 1.533e-7 | 66.07 s | 28.97 s | 2.28x |
| 60 | 66 / 66 | 9.435e-7 / 9.435e-7 | 149.10 s | 7.65 s | 19.5x |

The new A=20/N=25 regression has reported error `4.463e-7`, independently
evaluated 20,001-point error `4.468e-7`, and weight L1 norm `0.629`.

## Rejected experiments

| Experiment | Result | Decision |
|---|---|---|
| Exact Golub-Pereyra Jacobian | Lower residuals/node counts in places, but weight L1 norms grew to about 2.3e4 at A=40 and 4.6e7 at A=60. | Reject until node parameterization controls collisions and cancellation. |
| Explicit minimum node separation | Prevented useful target convergence for tested separation. | Reject fixed separation heuristic. |
| N-to-N+1 continuation | Found a passing A=20/N=24 fit with L1 norm about 149 rather than about 0.63. | Reject unconditioned continuation. |
| Binary/bracket search over N | Skipped valid N=32 at A=30 because solver error is nonmonotone across local minima; returned a pathological N=39 fit with L1 norm about 1.9e6. | Retain linear scan. |

These are not evidence against exact variable projection or continuation in general.
They show that those techniques are unsafe with the current clip-and-sort node
coordinates and no explicit conditioning objective.

## Remaining work

1. Reparameterize nodes as ordered positive gaps, ideally with bandwidth-scaled
   bounds, before retrying exact Jacobians or trust-region updates.
2. Add a conditioning guard or secondary objective based on weight L1 norm and/or
   basis singular values; maximum approximation error alone accepts destructive
   cancellation.
3. Treat the Fermi target separately. A preliminary A=10 run took 37.5 s,
   returned N=46, and had weight L1 norm about 7.9e7.
4. Only after those controls exist, revisit continuation in A and N and a proper
   exchange/Remez update.

## Verification

- Focused: `15 passed` in `tests/test_minimax_quadrature.py`.
- Full GPU suite: `208 passed, 24 deselected, 4 warnings` in 247.66 s.
- No existing user changes in `AGENTS.md`, `docs/ENVIRONMENT_COMPREHENSIVE.md`,
  or `test_jvp_verify.py` were staged or modified.

## References

- O'Leary and Rust, [Variable Projection for Nonlinear Least Squares Problems](https://www.nist.gov/publications/variable-projection-nonlinear-least-squares-problems)
- Hackbusch, [Computation of best L-infinity exponential sums for 1/x by Remez' algorithm](https://link.springer.com/article/10.1007/s00791-018-00308-4)

## Follow-on: Lorentzian and complex noncrossing resolvents

The existing imaginary-frequency solver fits the dispersive component

$$
\frac{x}{x^2+\bar\omega^2}.
$$

For off-axis full-frequency sampling, a noncrossing branch also needs the
absorptive Lorentzian $\bar\omega/(x^2+\bar\omega^2)$. Commit `5bb668c` now:

- generalizes the solver to either real component;
- combines independently certified component fits into
  $1/(x+i\bar\omega)$ with real decay nodes and complex weights;
- preserves complex weights through the disk cache, `MinimaxNodes`, and both
  chi0 host-side prefactor folds;
- stops Lawson polishing at the requested dense-grid tolerance;
- selects the lower-L1 start when both starts pass, and rejects automatic-grid
  fits with dimensionless weight L1 norm above $10^4$.

Representative generated complex fits at scaled tolerance $10^{-5}$:

| R | omega_hat | Total nodes | Dense complex error | Generation time | Weight L1 |
|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 10 | 1.89e-6 | 0.90 s | 5.01e3 |
| 52 | 16.3 | 14 | 5.38e-7 | 2.13 s | 1.70e3 |
| 200 | 20 | 13 | 4.88e-6 | 3.77 s | 2.63e3 |

A physical-scale smoke test for $1/(x+i)$ on $x\in[0.5,5]$ used 9 nodes,
generated in 0.85 s, and had 20,001-point absolute error `2.523e-5` under the
existing scaled-tolerance convention.

An analytic Kaufman projected Jacobian reduced one documented fixed fit from
about 0.30 s to 0.13 s, but was rejected as the default after a broad sweep:
some small-R cases entered worse attraction basins, took about 20x longer, or
required extra nodes. Finite-difference TRF remains the conservative solver.

This follow-on covers only **noncrossing** off-axis denominators. A sampling
frequency whose real part lies inside the transition continuum still requires
the global crossing/complex-time chi0 kernel described in
`reports/multipole/MPA_IMPLEMENTATION_PLAN.md` section 9.4.

Verification status for the follow-on:

- Focused minimax suite: `19 passed`.
- CPU-compatible suite: `176 passed, 22 skipped`; three unrelated config tests
  fail because the CPU backend intentionally forces distributed linear algebra
  to `off` while those tests expect `auto/slate/cusolvermp`.
- The full GPU suite was not rerun for this follow-on at the user's direction;
  the preceding crossing-solver commit passed `208` GPU tests.
