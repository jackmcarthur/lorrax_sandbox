# Bispinor rebase — session 2 progress (2026-05-04)

Continuation of `report.md`.  Branch in this session: `agent-B/bispinor-on-main`,
fresh off `origin/main` (LORRAX repo).  4 commits landed; Phases 1, 2, and
4a from the original handoff are done.  Phases 3 (driver wiring + WFN lift +
centroid file layer) and 4b (budget-split chunker) remain.

## Commits landed

| Sha | Title |
|---|---|
| 0d9ea78 | common/isdf_fitting: vertex-weighted pair density + LU branch for bispinor |
| 3a7000e | runtime: gate_print_to_rank0 + tee_stdout_to_file helpers |
| 9eb5098 | gw: re-calibrate chunker memory estimators against measured peaks |
| c771bdc | gw: bispinor V_q tile kernel + Lorentz 7-tile driver |
| e80034b | gw/compute_vcoul: flatten get_v_per_G_and_phase output when sphere_idx is None (smoke-test-driven fix) |
| 93ab4ca | common/phdf5_wfn_reader: bispinor=True kinetic-balance lift (Phase 3a) |

## GPU smoke test (Phase 2 validation)

Ran `runs/MoS2/B_v_q_lorentz_smoke/smoke_v_q_lorentz.py` on a single
A100 in the existing pool allocation (`lxrun`).  Synthetic 4-channel ζ,
2×2×1 q-grid, n_rmu=64, fft_grid=8³, sys_dim=2 slab, sphere_idx=None.
**13/13 checks pass** in ~2.3 s (after the FFT compile):

* Key set: 10 expected tiles (7 unique + 3 hermitian fill).
* Hermitian fill V[(j,i)] == V[(i,j)]† exact (zero error) for (1,2), (1,3), (2,3).
* Diagonal tiles V[(i,i)] hermitian to f64 precision (~2e-16 rel).
* Off-diagonal V[(i,j)] non-zero (transverse projector working).
* G0 head shape + finite (only populated by (0,0) tile).
* (0,0) shape correct, finite, max ~6e6 in synthetic units.

Bug found and fixed during this validation: when ``sphere_idx is None``
the new ``get_v_per_G_and_phase`` helper returned a box-shaped
``(nx, ny, nz)`` array instead of flat ``(n_G_sph,)`` — broadcasting
against the (Q, μ, n_rtot) ζ slab failed.  Fix: reshape ``v_per_G`` to
``(n_sph,)`` in that branch.  Patch in commit e80034b.

All four pass syntax + import-smoke checks on `origin/main`.  The pytest
suite has 71 passing + 5 skipped + 3 pre-existing kmeans-sharded failures
(documented in CHANGELOG; unrelated to bispinor); my changes don't add new
failures.  The full GW regression test (`tests/test_gw_jax_regression.py`)
fails on this Perlmutter login node because `liblapackpp.so.2` isn't on
the loader path — pre-existing env problem, not a regression from these
commits.

## What's done vs the original Phase plan

| Phase | Done | Notes |
|---|---|---|
| 1 (bispinor V_q on new interface, in-spirit) | partial | v_q_tile + v_q_lorentz scaffolding land; driver wiring TBD |
| 2 (LU branch + vertex pair density) | yes | commit 0d9ea78; back-compat default `vertex_mu_L=0` |
| 3 (4-channel zeta fit thread) | partial | `vertex_mu_L` thread is in `fit_zeta_chunked_to_h5`; 4-channel loop in gw_init not yet |
| 4 (chunker recalib) | yes | commit 9eb5098; ZCT_ADDITIONAL_COEF=1, RTOT_FFT_COEF=7, +0.8GB runtime overhead |
| 5 (bc-loop scan-ification) | not started | independent of rebase; CrI3 OOM blocker; out of scope here |

## What's left for the next session

### ~~Phase 3a~~ — DONE (commit 93ab4ca)

`PhdfWfnReader.coeffs_gspace(bispinor=True)` lifts 2-spinor source ψ to
4-spinor (large+small) via ψ_S = (α/2)(σ·(k+G)) ψ_L on the FFT box.
``self.alat`` + ``self._kfrac_dev`` + ``_make_bispinor_lift_kernel``
factory.  Default ``bispinor=False`` is bit-identical to prior
behavior — no production callers change.  Lift not yet exercised
end-to-end (the gw_init driver doesn't pass ``bispinor=True`` yet —
that's part of Phase 3c).

### Phase 3b — Two-centroid-file architecture

* `src/centroid/centroid_io.py` (new, ~55 lines): read/write the density
  header tag (`scalar` for charge density, `current` for Gordon-decomposed
  Pauli current).
* `src/centroid/current_density.py` (new, ~178 lines): Gordon current
  helper for the bispinor centroids.
* `src/centroid/kmeans_cli.py`: `--density-mode {scalar,current}` flag.
* `src/common/bispinor_init.py`: read both centroid files (`centroids_file`
  for μ_L=0, `centroids_file_current` for μ_L=1,2,3).
* `LorraxConfig`: new input keys `centroids_file_current` and
  `density_mode`.

Total ~300 lines across 4 modules + 1 config field.  Ref commit 9397e35.

### Phase 3c — gw_init / gw_jax driver loop

Wire up the 4-channel zeta fit + bispinor V_q dispatch.  Concretely:

1. `gw_init.fit_zeta`: when `cfg.bispinor` is true, open 4 zeta_io's
   (`zeta_q.h5`, `zeta_q_mu1.h5`, `zeta_q_mu2.h5`, `zeta_q_mu3.h5`),
   call `fit_zeta_chunked_to_h5` with `vertex_mu_L=mu_L` for each, and
   reset the JAX cache between channels.

2. `gw_init.compute_V_q`: when bispinor, dispatch to
   `compute_all_V_q_lorentz_sharded(zeta_io_by_channel=…, …)` instead of
   the existing `compute_all_V_q`.  Returns `V_blocks: dict[(μ_L,ν_L),
   Array]` instead of a single `V_qmunu`.

3. `gw_jax.main`: convert the `V_blocks` dict into a (4, 4, …) Lorentz-
   tiled tensor (or rebuild the Σ-projection with the bispinor
   structure).  This step needs careful attention — the downstream
   `build_G` and `Σ_μν → Σ_ij` projection assume rank-3 V_qmunu.

Roughly ~200 lines but tightly coupled.  Ref commit 9397e35 in
`src/gw/gw_init.py` and `src/gw/gw_jax.py`.

### Phase 4b — Budget-split chunker

Re-apply commit `agent-B/wip-budget-split @ 17bfb5e` (`WFN_WORKSPACE_FRAC =
0.30` gate before `_find_r_chunk`'s retry loop) against the new
`compute_optimal_chunks` which uses `cfg.memory.per_device_gb` and
`cfg.memory.chunk_target_utilization`.  ~20 lines of context-shift.

Note: `cfg.memory.target_utilization` was lowered 0.97 → 0.80 in the WIP
branch and `runtime overhead` was raised to `max(0.8 GB, 5%×budget)` —
re-check whether those tweaks are still needed after Phase 4a's
recalibration (it may have closed enough of the gap that they're moot).

### Phase 5 — bc-loop scan-ification (independent of rebase)

Per-iteration donation of `psi_bc_Y` only works at jit boundaries.
The current Python `for bc_idx, ... in enumerate(bc_classify)` Python-
unrolls the bc-loop; XLA sees ~10 separate `io_callback + FFT + reshard +
accumulate` traces and holds 10× ψ buffers concurrent.  Real fix:
`jax.lax.scan` over the bc-loop so each iteration uses the same compiled
body.  Estimated ~2-3 hours.  See `report.md` Phase 5 for the design.

This is the *real* fix for the CrI3 60Ry bispinor 8/16-GPU OOM and is
independent of the rebase mechanics — can land before, during, or after
the bispinor port lands.

## Smoke testability

Without Phase 3 the bispinor V_q driver can't be exercised end-to-end.
The unit-style smoke is currently:

```python
from gw.v_q_lorentz import compute_all_V_q_lorentz_sharded
from gw.compute_vcoul import make_v_munu_chunked_kernel
# ... mock 4 SlabIO handles + a coulomb_kernels bundle, call the driver,
# assert (a) the (i,j) tile equals (j,i)†, (b) the (0,0) tile equals
# what compute_all_V_q would have produced for the same zeta.
```

Recommended: a tiny pytest in `tests/active/` that mocks 4 channels with
random ζ and checks the tile structure (Hermitian fill + 6 zeros).  Not
written this session.

## File pointers

* Two ports landed mostly verbatim from the bispinor branch:
  - `sources/lorrax_B/src/gw/v_q_tile.py` ← cleaned-up port of
    `agent-B/refactor-compute-vcoul:src/gw/v_q_tile.py` (post-recalib at
    c3c1c26), with the local `_choose_v_q_chunks` re-imported from
    compute_vcoul instead of duplicated.
  - `sources/lorrax_B/src/gw/v_q_lorentz.py` ← verbatim port of
    `agent-B/refactor-compute-vcoul:src/gw/v_q_lorentz.py @ 9397e35`.
* `sources/lorrax_B/src/gw/compute_vcoul.py`: chunker recalib applied;
  `make_v_munu_chunked_kernel` extended with `get_v_per_G_and_phase` +
  `get_K_cart_on_sphere` for bispinor consumers.
* Original branches (now stale, but kept as references):
  `agent-B/refactor-compute-vcoul`, `agent-B/wip-budget-split`.
