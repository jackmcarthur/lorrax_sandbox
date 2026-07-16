# Si BSE degeneracy vs symmetry-obeying ISDF centroids — 2026-07-16

**Question:** do the symmetry-obeying (orbit-closed) centroids from the main
`kmeans_cli` path restore exact BSE eigenvalue degeneracies on Si?

**Answer: No.** Symmetry-closed centroids leave every intra-manifold splitting
essentially unchanged (old/sym ratio **1.004–1.018**), nowhere near the ~2 μeV
BGW exactness. The historical attribution (src/bse/STATUS.md 2026-04-28:
"likely ISDF centroid placement not perfectly symmetric") is **refuted**.

## Setup

`tests/regression/si_cohsex_debug` fixture (BGW-anchored Si 4×4×4, no-SOC,
8 IBZ / 64 full-BZ k, nband=60, nval=8, ntran=48 Fd-3m). Two arms, matched
target N_c=960, seed=42, `--oversample 1.0`:

- **old (literal):** `kmeans_cli 960 --no-orbit` → orbit off, **960** grid points.
- **sym (symmetry-obeying):** `kmeans_cli 960` (default) → orbit on, n_sym=48,
  20 representatives unfolded to **792** orbit-closed centroids (union of
  complete orbits; the count mismatch is inherent to the orbit unfold).

Each arm: fresh `gw.gw_jax` COHSEX (`do_screened`, minimax, ~2 min/GPU) → ISDF
restart with W0 + q=0 head. BSE = dense Q=0 TDA H = diag(D)+Kx−Kd from
`bse_io._load_ring_subset`, `eigvalsh`; manifolds = eigenvalue clusters with
gap < 1 meV.

## Splitting table (μeV), lowest manifolds, per band window

| window | manifold | size | **sym split** | **old split** | old/sym | BGW ref |
|---|---|---|---|---|---|---|
| 4v4c | 0 (doublet) | 2 | **518.5** | **520.7** | 1.004 | ~2 |
| 4v4c | 2 (doublet) | 2 | 978.1 | 995.7 | 1.018 | ~2 |
| 4v4c | 4 (triplet) | 3 | 1663.2 | 1666.6 | 1.002 | ~2 |
| 6v6c | 0 | 8 | 2409.5 | 2422.0 | 1.005 | ~2 |
| 8v8c | 0 | 8 | **2030.9** | **2054.3** | 1.012 | ~2 |
| 8v8c | 1 (doublet) | 2 | 972.6 | 986.6 | 1.014 | ~2 |

(2v2c is too small to hold a multi-D manifold — its 4 lowest states are
non-degenerate 1-D irreps ~3.4 meV apart in both arms.) The 4v4c doublet
(518–521 μeV) reproduces the historical ~485 μeV LORRAX datum; the symmetric
arm does **not** collapse it toward BGW's ~2 μeV. The sym arm even has *fewer*
centroids (792 vs 960, lower ISDF rank) yet near-identical splitting — count is
not the lever either.

## Interpretation

Orbit-closed centroids are necessary but not sufficient. The symmetry violation
enters **downstream of centroid placement** — the ψ full-BZ unfold from 8 IBZ
k-points and/or the ζ-fit are not themselves symmetry-covariant. This lines up
with the deferred ψ-side symmetry unification (TRS-blind-sym-bug Phase 2 /
unified-sym-action). The fix belongs there, not in centroid generation.

`glide_symmetry_lloyd_exact.py` (untracked in lorrax_A): a 2-D proof-of-concept
of exact symmetry-adapted weighted Lloyd for a hard-coded glide group, synthetic
density, no WFN input/centroid output — a prototype of the idea the production
`kmeans_cli --orbit` path already ships. Not used here.

## Notes

- `tests/test_bse_dense_reference.py::_build_dense_H` has an O(nk²)=4096-step
  Python (k,k′) loop — infeasible beyond 2v2c at 4×4×4. `analyze_fast_all.py::fast_H`
  is a vectorised builder (batched-k′ einsums, identical formula), **validated
  bit-equal at 2v2c (rel-err 4.8e-17)**, used for the larger windows. Worth
  folding back into the gate builder.
- Artifacts: `work_{old,sym}/` (centroids, inputs, gw logs, restarts),
  `results_*.json` per window, `summary_windows.json`, `comparison.json`,
  analysis scripts, `run_experiment.sh`/`run_fast.sh`. Restart h5s (2.0/1.4 GB)
  kept on disk, not committed.
