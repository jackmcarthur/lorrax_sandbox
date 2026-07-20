# Mini-BZ head averaging + BSE integration branch — design & provenance

Session 2026-07-20. Worktree: sources/worktrees/lorrax_A_bse_integration
Branch: agent/bse-integration (off agent/bse-bands-perf @ 2e90edb).

## STEP 1 — consolidation (DONE, commit b65401b)
Merge agent/bse-bands-80 (a39b3ba full-band htransform + 5120fe4 gate-tightening)
into bands-perf. Both are siblings off 9fca293 (NOT off exciton-bands — bands-perf
forked ONE commit before a39b3ba, so it lacked full-band htransform; merge brings
it in). Conflict: bse_setup.py comment (kept both); exciton_bands.py auto-merged.

## STEP 2 — mini-BZ head averaging (flag-gated, default FALSE = bit-identical)
Physics (bgw_minibz_coulomb_averaging.md + arbitrary_q_bse.md §16/§9.5/§9.8):
- Single-source averaging routine: gw/coulomb/base.minibz_average, BARE units
  (8π·[trunc]·[gauss]/|K|², NO 1/celvol — callers apply their volume conv).
- Two BGW branches on |shift|² (shift = Q+G*):
    |shift|²<TOL & analytic_sphere(3D only): MC over δq OUTSIDE inscribed sphere
      + analytic Baldereschi 4·√q0sph2·celvol·N_k/π  (= 32π²√q0sph2/V_mBZ).
    else: adaptive MC N_Q=clamp(round(N_coarse·4·q0sph2/|shift|²),1,N).
- 2D slab: pure in-plane MC (head is |Q| cusp, NO analytic sphere); slab_lr kernel.
- eval_vq (BSE): per-Q, G*=argmin_G|Q+G|; replace v_LR[G*] POINT value with
  <v_LR(Q+G*)>_mBZ/celvol. zt (winding e^-i2θ) and V_SR (body) UNTOUCHED — §16
  no-double-count: cell-avg carries |Q| magnitude, phase-factored zt carries angle.
- nmax 1→3 Voronoi wrap ONLY on the flag-on path (nmax=1 default preserves bit-id).

## STEP 3 — bse_io loader head-skip guard
load_bse_data_from_restart_sharded: if G0_mu_nu present & inject_head but vhead/whead
both resolve None -> inject block silently skipped. Add LOUD warning. Recompute is
NON-TRIVIAL (loader has no wfn/meta/sym/S_cart) -> warn-only (per §16.5).

## Volume-factor bookkeeping (CRITICAL for bit-identity)
- q0_average vc0_mean (bulk_3d/slab_2d): BARE <8π.../|q|²> (NO /celvol); /celvol at
  injection (apply_q0_head_rank1 -> _head_rank1_scalars: head/cell_volume).
- eval_vq v[G] (vq_interp:806): HAS /celvol.  So eval_vq head = minibz_average(bare)/celvol.
