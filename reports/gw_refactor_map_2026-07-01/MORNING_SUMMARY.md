# Morning summary — overnight loop 2026-07-08

Branch `agent/memplanner-cleanup`, all pushed. Suite at last full run: **257 passed / 9 skipped / 0 failed**.
Every landing gated; no golden number moved except where explicitly re-frozen/designed.

## The big one — device-invariance bug FOUND and FIXED
Your padding suspicion was right; accumulation-order was wrong (measured chunk-order effects
≤6e-5 eV vs the 3.8 eV signature).
- **Root cause** (`reports/device_invariance_2026-07-08/ROOT_CAUSE.md`): computations ran on the
  PADDED μ extent, which is P-dependent (1204→1216 pad rows only at P=16; transverse 668→672).
  Two manifestations: the transverse ζ LU solved at padded extent (proven by reproducing the full
  16-GPU signature at fixed P=4 with forced pad rows), and the PPM mode census/window stats over
  padded μ².
- **Fixed** (083d209..7801d46): solves at logical extent, census on logical modes.
  **Bispinor 4g↔16g eqp: 2.535 eV → 1.45e-7 eV.** Charge census now P-invariant; remaining on-pole
  residual is near-threshold mode-flip noise amplified by GN-PPM ill-posedness (documented as a
  robustness item — fixing it changes golden numbers, deferred deliberately).
- **New gates**: Tier-1 pad-flip bit-identity (in suite, 1 GPU), Tier-2 cross-P script (P=1 vs P=4,
  passing), restart pad-roundtrip gate.

## Padding audit + consolidation (your "is padding clean?" question)
- 4-agent audit (`PADDING_AUDIT.md`): design right (one birth site, heavy ops structurally
  neutral), implementation was ~2× needed size with 3 latent bugs.
- Consolidation (6c850bd..620b501, net **−553 lines**): restart files now store LOGICAL μ extent
  (P-portable restarts + roundtrip gate), allgather write unified on SlabIO, census masking moved
  to fit birth (pad modes born dead — consumers structurally safe), dead PadAxis API deleted
  (−757), one `round_up`, one `solve_at_logical` (was 6 hand copies), sym-table pad baked.
  Left documented: ψ/enk band-axis padded restart (band-pad convention, out of μ-scope).
- Bonus: the new pad gates UNBLOCK the deferred zeta_loader/zeta_reader merge (#8).

## qp_solver toggle (your G0W0-vs-SC requirement)
(4e3ecfd..509e492) `qp_solver = one_shot_dft | fixed_point | self_consistent`, **default
one_shot_dft** = textbook G0W0 (Σ(E_DFT), DFT energies, no iteration). Absorbs the orphaned
`sigma_at_dft_energies`; `SCConfig` replaces the 5 LORRAX_SC_* envs (NEXT_TARGETS #11 closed);
invalid combos error loudly. `fixed_point` proven a pure re-labeling (frozen-reference gate).
**Single-jit check**: qsgw nested-jit hoist → SC steady-state = **0 compiles/0 retraces per
iteration** for both COHSEX and GN-PPM (was 2/iter for GN-PPM). SC verified working for GN-PPM
(stale "COHSEX only" comments fixed).

## BGW invalid-pole reference set + validation
- BGW refs (`reports/bgw_invalid_mode_refs_2026-07-08/`): Si 4×4×4, GN+HL lines, modes 0/2/3
  (+default==3 verified empirically), 8.63% invalid poles, full mode-diff tables.
- LORRAX `zero`/`2ry` validated: wiring PASS (statics bit-identical, toggle-only delta, sign
  pattern matches BGW's strong features; magnitudes scale with the invalid-population ratio).
- **static_limit: DONE, and it is now the DEFAULT** (fdf89c2 impl, 9925d43 default+re-freeze).
  Correction en route: my research note's "SX pole → 0" was WRONG — BGW mode 3 keeps BOTH the
  static SEX and CH terms (read from mtxel_cor.f90 directly); implemented as
  Σ_static = sigma_sx(G_occ, Wc0·inv_mask) + sigma_coh(Wc0·inv_mask), reusing the two
  cohsex_sigma kernels verbatim; Wc0 retained on PPMBuildResult (replacing two never-read fields).
  **Three-way BGW parity PASS** (Si: sign pattern matches on all edge features, magnitudes scale
  with the invalid-population ratio; lorrax_mode_table3.dat). Re-freeze: gnppm ref moved mean
  40 meV / max 0.322 eV (band-1 deep valence; in-family — corr 0.89 with the validated 2ry−zero
  delta; this 2D fixture has stronger invalid-pole weight than the ~meV Si-based guess). Statics,
  Si-3D, bispinor, IBZ gates unmoved, as required. Per-q n_invalid print added.

## Decisions/notes for you
1. **On-pole charge residual** (P-dependent node-count flips on near-threshold divergent-Ω modes,
   amplified by on-pole Σ evaluation): real robustness item, needs a physics decision (mode-census
   hysteresis / census at fixed reference / document as ill-posed). `ROOT_CAUSE.md` AS-FIXED §.
2. **static_limit re-freeze magnitudes for your review**: the default flip LANDED; the gnppm
   golden moved mean 40 meV / max 0.322 eV (2D fixture, stronger invalid-pole weight than Si).
   In-family with the validated mode deltas, but eyeball the new sigma_diag_gnppm_ref if you want.
3. Speed policy applied going forward: regression-subset per commit + one full suite per series,
   blocking waits (no polling), persistent allocation, tighter agent scopes.

## Where everything is
- Plan/log: `archive/OVERNIGHT_PLAN.md` (this dir). Root cause + padding: `reports/device_invariance_2026-07-08/`.
- BGW refs + validations: `reports/bgw_invalid_mode_refs_2026-07-08/`.
- Toggle design + impl evidence: `G0W0_SC_TOGGLE_DESIGN.md`, `archive/g0w0_sc_toggle_impl/` (this dir).
- NEXT_TARGETS.md updated (TIER-0★A partially done; #11 closed; #8 unblocked).

**Final verification:** full suite after the static_limit default flip: **257 passed / 9 skipped / 0 failed**. Loop closed.
