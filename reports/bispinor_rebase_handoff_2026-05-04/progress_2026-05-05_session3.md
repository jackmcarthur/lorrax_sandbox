# Bispinor rebase — session 3 (2026-05-05)

Continuation of `progress_2026-05-04_session2.md`.  Branch
`agent-B/bispinor-on-main` on `lorrax_B`, **9 commits ahead of
`origin/main`**.  End-to-end MoS2 60Ry bispinor smoke test on 4 A100
**runs cleanly through fit_zeta (4 channels) + V_q lorentz (10 tiles)
+ bare-X Σ → eqp0.dat**.

## Commits landed this session (Phase 3b/3c + 3 runtime fixes)

| Sha | Title |
|---|---|
| aedc74d | gw + centroid: two-centroid-file architecture for bispinor |
| 74c5ab3 | gw_init: bispinor 4-channel zeta loop + V_q lorentz dispatch |
| 2d1df04 | Fix 3 runtime gates exposed by bispinor end-to-end run |

## Phase status

| Phase | Status |
|---|---|
| 1a (LU branch + vertex pair density) | committed (session 2) |
| 1b (runtime tee + gate_print) | committed (session 2) |
| 4a (chunker recalibration) | committed (session 2) |
| 2 (v_q_tile + v_q_lorentz + kernel helpers) | committed (session 2) |
| 3a (WFN bispinor lift in PhdfWfnReader) | committed (session 2) |
| **3b (two-centroid-file architecture)** | **committed (this session)** |
| **3c-1 (gw_init 4-channel zeta loop)** | **committed (this session)** |
| **3c-2 (gw_init V_q lorentz dispatch)** | **committed (this session)** |
| **3c-3 (Σ-projection threading)** | **STUB ONLY** — V_blocks[(0,0)] forwarded, the 9 transverse tiles are computed and logged but DISCARDED.  Loud warning at the forwarding site. |
| MoS2 60Ry bispinor end-to-end smoke | **PASSED** (4 A100, ~90 s wall) |
| 4b (budget-split chunker) | not needed — Phase 4a closed the gap on this system |

## End-to-end smoke run

* Allocation: JID 52472138 (4 nodes, 4 h, urgent_gp).
* Run dir: [`runs/MoS2/D_60Ry_bispinor/`](runs/MoS2/D_60Ry_bispinor/).
* System: MoS2 3×3, 32 bands, 60 Ry, sys_dim=2 slab, bispinor=true,
  x_only=true, do_screened=false; 640 charge centroids + 668 current
  centroids; ``memory_per_device_gb = 30.0``.
* Wall (4 A100):
  - fit_zeta μ_L=0: ~13 s
  - fit_zeta μ_L=1,2,3: ~13 s each
  - compute_all_V_q_lorentz_sharded total: 36.5 s
    (0,0) 4.8 s · (i,i) 4.2–4.8 s · (i,j) off-diag 5.6–6.7 s
  - sigma (bare X via V_blocks[(0,0)] stub): ~1 s
* Outputs: ``eqp0.dat``, ``eqp1.dat``, ``sigma_diag.dat``,
  ``zeta_q.h5``, ``zeta_q_mu1.h5``, ``zeta_q_mu2.h5``,
  ``zeta_q_mu3.h5``, ``isdf_tensors_640.h5``.  Snapshot copies
  preserved as ``eqp0_first_e2e_smoke.dat`` etc.

## What this proves

1. **Phase 1+2 V_q lorentz infrastructure is wired correctly on real
   data.**  All 10 (μ_L, ν_L) tiles compute to f64 precision; trace
   diagnostics print at run time.
2. **WFN bispinor lift in PhdfWfnReader works end-to-end.**  Host tiles
   populate as 4-spinor; pair density + CCT + Cholesky run cleanly on
   ns=4 wavefunctions.
3. **Two-centroid-file architecture loads correctly** — the
   `n_rmu_by_channel` dict comes back ``{0: 640, 1: 668, 2: 668, 3: 668}``,
   matching the centroid file headers.
4. **The chunker recalibration (Phase 4a) is tight enough** for this
   system that no extra budget-split (Phase 4b) is needed.

## Stub: Σ-projection still over-simplified

The bispinor compute_V_q forwards only V_blocks[(0,0)] (the charge
channel) to the canonical ``(1, npol, npol, nkx, nky, nkz, μ, μ)`` V
output.  A loud warning prints:

```
[bispinor] WARNING: forwarding only V_blocks[(0,0)] to downstream Σ —
the 9 transverse tiles are computed and logged but DISCARDED until
Σ-projection lands.
```

Output Σ is therefore identical to a μ_L=0-only non-bispinor run on the
bispinor centroid set + lifted ψ.  **Not yet a physical bispinor result.**
The Σ^B walk over all 10 (μ_L, ν_L) tiles + tile-aware band projection
is the next, separable piece of work.

## Three runtime fixes landed in 2d1df04

These were latent issues in the existing code paths exposed when the
bispinor 4-channel loop hit them:

1. **Tracer-capture in `get_sharded_wfns_rchunk_slice`** — closure-
   captured ``jnp.asarray(kvecs_frac)`` becomes a tracer when the cache
   builder runs inside an outer jit; after the trace closes the tracer
   is dead; re-using the cached jit raises ``UnexpectedTracerError``.
   Fix: ``np.asarray`` for closure-captured kvecs + fx/fy/fz grids.
   Pre-existing latent bug; non-bispinor runs survive only because
   they never re-use the cache.
2. **`HostPsiGStore` doesn't pass `bispinor=` to the phdf5 reader** —
   the reader returns 2-spinor data into a 4-spinor host tile (shape
   mismatch).  Fix: detect ``meta.nspinor==4 and reader.nspinor==2``
   and pass ``bispinor=True``.
3. **`compute_V_q` bispinor reshape** — the lorentz driver returns
   blocks shaped ``(nq, μ, μ)`` but the downstream broadcast wants
   ``(nkx, nky, nkz, μ, μ)``.  Reshape before broadcasting.

## File pointers

* ``runs/MoS2/D_60Ry_bispinor/cohsex.in`` — added ``vhead = 0.0`` +
  ``whead_0freq = 0.0`` overrides to bypass dipole.h5 / eps0mat.h5
  (head correction is not meaningful for x_only and shouldn't be
  required, but the head_resolver currently requires either an eps0mat
  or a dipole.h5; explicit overrides are the documented escape hatch).
* ``runs/MoS2/B_v_q_lorentz_smoke/smoke_v_q_lorentz.py`` — synthetic
  4-channel smoke from session 2 (still passes 13/13).
* The stale ``isdf_zeta_mode_test.py`` / ``test_gw_jax_regression.py``
  failure modes are still pre-existing env / path issues unrelated to
  this branch.

## What's left for a real bispinor result

* Σ^B-projection: walk all 10 (μ_L, ν_L) V_blocks tiles in
  ``compute_cohsex_sigma``.  The current consumer site assumes a
  single rank-3 V_qmunu; needs a Lorentz-aware version that picks up
  the bispinor pair-density (γ̃^{μ_L} ψ̄ ψ) and contracts against the
  matching V tile.  Estimated 100-200 lines, mostly in
  ``src/gw/projection_kernel.py`` + the cohsex/x_only kernels.
* Comparison check: Σ^B against an independent reference (e.g. a
  full-relativistic BGW run with full ψ_S, or analytic limits where
  the transverse contributions vanish).
