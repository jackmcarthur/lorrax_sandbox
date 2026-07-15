# BSE refactor map — program report (fresh-session entry point)

_2026-07-15. The BSE analogue of `reports/gw_refactor_map_2026-07-01/`. Produced by
a 45-agent workflow: 18 deep-readers over `src/bse` (~7.7 kL) + `src/solvers`
(~3.9 kL), adversarial verification of all 207 reader claims, two independent
teleological sortings, and 5 feature-design agents for the missing physics.
No source was modified — this is a map + design program only._

## Read order

1. This file → 2. **`MAP.md`** (the spine: taxonomy, dataflow, ranked targets,
attack order) → 3. `archive/DEAD_CODE.md` (§1 bugs FIRST) →
4. `archive/files/kernel_dataflow_trace.md` (the full dataflow + boundary arrays) →
5. `archive/designs/*.md` (the five feature designs) → 6. `archive/FEATURES.md`,
`archive/files/*.md` per-file detail as needed.

## Audit base

`sources/lorrax_D` @ `e18d0e5` (`agent/slate-linalg-ffi`). The BSE + solvers trees
are byte-identical to current HEAD — untouched by the whole GW refactor program.
Verdict tally (207 claims, each adversarially attacked): 88 confirmed-bug ·
36 convention · 30 benign-cruft · 23 refactor-target · 20 confirmed-dead ·
8 test-only · 1 unverifiable · 1 alive. (The 88 counts every sub-claim; the
structural ones are B1-B7 + the solver-layer items below.)

## Headline findings

1. **The BSE package cannot load restarts written by current gw_jax.** Three
   independent loader bugs (B3: V_q0 reader is 8-D-legacy-only; B4: flat-q W
   reader demands a `kgrid` attr no writer sets; B5: 1-device head injection
   indexes the legacy layout before the shim). Sharded and 1-device paths both
   die on any fresh `do_screened` restart — every path only works against
   legacy-era h5 files. This, concretely, is why BSE is "less finished".
2. **The exchange kernel is k-block-diagonal in all four matvec implementations**
   (B1) where the physical K^x is dense in (k,k′). Triple-confirmed: Henneke
   Eq. 4-5 re-derivation, a 2-k toy numerically diffed against a hand-built dense
   reference, and BGW `kernel_main.f90` cross-check. It survived the Si-vs-BGW
   3 meV validation because Si's lowest manifold is exchange-insensitive and the
   only "correctness check" compares ring-vs-serial — the same formula on both
   sides. No independent dense reference exists in the tree.
3. **Full (non-TDA) BSE with W has never executed** — the B-block T-encode einsum
   has an unbound output index and crashes at first apply (B2); non-TDA only ever
   ran as RPA.
4. **Zero pytest-collected BSE coverage.** `src/bse/test_*.py` are argparse smoke
   scripts outside testpaths; test_bse's own loader is stale vs current writers.
5. Solver layer: KPM is broken at HEAD (phantom kwarg TypeError); FEAST (the
   silent default route — with the **RPA** kernel unless `--bse` is passed) uses a
   fixed n_ritz=4 with no count-in-window estimate, which is exactly the
   clustered-eigenvalue failure; `block_lanczos_eig` has a transposed-β bug;
   bse_pseudopoles is un-importable **lost wiring** (roadmap feature, not dead).
6. Structure: bse_ring_comm straddles five tiers (the package's sharding
   vocabulary should be extracted); bse_jax carries a dead matvec trio beside the
   live dispatch; bse_io mixes ingest/writers/private-config-parsing and hosts
   the loader bugs; ~20 confirmed-dead symbols; heavy duplication (pair-amplitude
   ×3, eqp-window block ×3 with 2 buggy copies, head-injection ×2, eigvec
   writer ×2).

## The five feature designs (`archive/designs/`)

| Design | One line | LOC |
|---|---|---|
| fine_grid_interpolation | Skip BGW's dcc/dvv for ψ — sample fine-NSCF ψ at the k-independent centroids (exact); Fourier-interpolate W_μν via its real-space image; analytic head/wing re-add per fine q | ~510-570 |
| coulomb_sr_lr | Gaussian range separation at the one v(q+G) seam; W split by singularity (W^LR = ε₀₀⁻¹V^LR); Phase 1 is bit-identical scaffolding | ~430 net |
| w_head_wings_interp | Promote experimental head_wing_schur to THE single head/wing/body path (deletes the scalar rank-1 injection as its special case); anisotropic S_cart reused; wings become load-bearing only on fine grids | ~700-750 |
| solver_program | FEAST matured (DOS-sized subspace + tr(P) count + locking) as interior engine; one Lanczos; Haydock as the fine-grid-scale primary; archive 3 sweep harnesses | **−800…−1000** |
| finite_q_bse | On-grid Q = conduction k-axis remap + umklapp phase + exchange tile swap (V_qmunu[Q] incl. G=0); matvec/solvers untouched; S(Q,ω) via existing g0 + Haydock; arbitrary Q deferred to fine-grid layer | ~450-600 |

Shared: the B1 dense-exchange fix is Phase 0 for all of them; sr_lr P1 and
head_wings P1 are behaviour-preserving and can land first.

## Recommended next steps (MAP.md §7)

Gates → loaders → physics (B1 own PR) → deletes → single-source → solvers →
features. Nothing moves before the dense-reference kernel gate and a
fresh-restart e2e pytest gate exist.

## Decisions needed from Jack (consolidated from the designs)

1. **Fine-grid target dimensionality**: 3D Si first, or 2D MoS2 near-term? (2D
   screening is not band-limited in R → needs clustered subsampling; much larger.)
2. **Fine energy source**: is one fine NSCF per system acceptable, or must QP
   energies be interpolated from coarse eqp (dcc/dvv-style)?
3. **B1 fix priority**: land the dense-exchange fix as its own gated PR first?
   (Recommended by three designs; it is also an independent coarse-grid
   correctness bug.)
4. **SR/LR split**: commit to Gaussian range separation (BGW's head-only
   exclusion = its α→∞ limit), or is bit-identical BGW head/wing storage needed?
   α policy: fixed c·Δk default vs exposed convergence knob?
5. **Anisotropic screening in the LR channel**: scalar ε₀₀ head sufficient, or
   fold the S_cart tensor for 2D/layered magnets (VI3/CrI3)?
6. **Clustered band-edge states (~100/0.5 eV)**: do they need converged
   eigenvectors (FEAST maturation P3), or is the Haydock/KPM spectrum enough
   there (much cheaper)?
7. **Non-TDA**: near-term target or defer? (B-block is broken; deferring makes
   TDA the only supported surface and simplifies FEAST/KPM.)
8. **Finite-Q first target**: commensurate on-grid Q (no new DFT, cheap) vs
   arbitrary Q (needs the interpolation layer)? Exchange head convention:
   BGW energy_loss default ok?
9. **Default-routing repair** (C1): make bare `bse_jax` = TDA screened solve
   instead of FEAST+RPA? (Behaviour change to a documented-but-surprising default.)
10. **Zolotarev as default FEAST contour** (API already defaults to it; CLI
    validated runs used ellipse)?

## Artifacts

- `MAP.md` — the spine (this program's main deliverable)
- `archive/DEAD_CODE.md` — 88 bugs / 20 dead / 23 refactor targets, evidence-linked
- `archive/FEATURES.md` — tiered feature catalog (merged 2-sorter taxonomy)
- `archive/files/*.md` — 18 per-file/topic deep-read notes (incl.
  `kernel_dataflow_trace.md`, `flags_surface.md`, `bgw_fine_grid_reference.md`,
  `tests_gates.md`, `context_docs.md`)
- `archive/designs/*.md` — 5 feature designs
- `archive/_raw_verdicts.json` (416 KB), `_raw_sorts.json`, `_digest.json` — raw
  agent outputs
