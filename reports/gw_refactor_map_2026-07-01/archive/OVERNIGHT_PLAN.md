# Overnight loop — 2026-07-08 → morning

Autonomous run on `agent/memplanner-cleanup`. Priorities in order. Every code change
gated (12 Σ_PPM/golden gates); feature branches, never main; grep-verify dead before delete.

## P1 (MOST SERIOUS) — device-invariance root cause (4-GPU vs 16-GPU eqp up to 3.8 eV)
**Lead directive: this is NOT accumulation-order roundoff. 3.8 eV at P=4 vs P=16 is a real bug.**
Quantitative refutation of the roundoff theory: measured chunk-order effects are ≤6e-5 eV (the
gnppm re-freeze); 3.8 eV is 5 orders larger.

**Hard leads (computed 2026-07-08):**
- **Padding arithmetic — P-dependent pad rows exist in BOTH diverging systems:** n_rmu_padded =
  round_up(n_rmu, P) (meta.py:130-133, product of mesh axes). Charge run: 1204 centroids → P=4:
  1204%4=0, ZERO pad rows; P=16: 1204→**1216, 12 pad rows only at 16 GPU**. Bispinor: charge 640
  divisible by both, but transverse 668 → P=4 clean; P=16 → **672, 4 pad rows only at 16 GPU**.
  If pad rows leak into ANY bilinear form / unfold / solve, the result is P-dependent. Prime suspect.
- **Second divisibility suspect:** the all-P q/k shards (W-solve `q_shard=P(('x','y'))`, SC/QP k
  paths) with nq/nk (=16 here) vs P=4/16 — different shard layouts, any uneven-shard or pad-row
  handling there is also P-dependent.
- **Discriminator observed:** Σ reported bit-identical at k=0 (bispinor) while eqp diverges 2.3-3.8 eV
  → FIRST measurement must be ΔΣ_nnk at ALL k (4g vs 16g), not eqp. Splits the space:
  (a) Σ identical everywhere → bug lives between Σ and eqp (QP-solve path, near-pole fixed-point);
  (b) Σ differs at k≠0 → trace upstream (ζ → V_q → W → Σ_μν), stage by stage, to the first divergence.
- **Existing artifacts** (start here, don't rerun blindly): runs/MoS2/Z_memplanner_validation_2026-07-06/
  {A_charge,B_bispinor}/{head,base}_{4g,16g} — full logs + outputs for both device counts.

Squad: (A) stage-bisect via ΔΣ/Δstage dumps 4g-vs-16g; (B) padding math-neutrality audit across P
(Cholesky √1 pad block, μ-unfold divisibility, every pad-row consumer; targeted repro: same system
at a P where padding differs); (C) FFI/all-P-shard P-dependence audit (cuSolverMp 2D, reshards,
phdf5 valid_shape, W-solve q-shard). Synthesis → root cause + fix plan + a multi-device
eqp-invariance gate design (4-GPU eqp == 16-GPU eqp within tol). Fix lands on next wake, gated.

## P2 — Σ_PPM WS6: static_limit (default) + BGW validation
- Recon: a subagent reads prior reports (invalid-pole details in the BGW research + SIGMA_PPM_MAP)
  and figures out how to run **BGW GW references compatible with the sandbox run instructions +
  environment** (skills/execute_workflow) with `invalid_gpp_mode` = static_limit (3), 2 Ry (2),
  and skip (0) — to validate the new modes.
- Implement `static_limit`: analytic −½·Wc0=B/Ω Coulomb-hole term for Ω²<0 poles; retain Wc0 on
  PPMBuildResult (data-seam). Make it the default. Gate re-freeze (~meV, MoS2 has invalid poles)
  + BGW parity for all three modes.

## P3 — G0W0 vs self-consistent toggle (clarity + single-jit)
- Make a CLEAR in-code toggle: **G0W0** = single-shot, Σ_nnk(E_nk^DFT), DFT energies as QP (no SC
  iteration); **self-consistent** = later iterations use previous-iteration energies (per the SC
  algorithm). This overlaps `sigma_at_dft_energies` — unify.
- **Check carefully**: the entire Σ pipeline looped in the SC loop should be ~a single jit, so one
  compile for a single-shot covers ~everything SC needs. Verify the jit boundary; report if the SC
  path recompiles vs reuses the G0W0 compile.

## P4 — consolidation for maintainable scientific-community software
General: continue removing dead/duplicated code, clarify names→physics, doc-map conventions. Only
after P1–P3 or in parallel where safe.

## Loop discipline
Checkpoint each landing (pytest + push + note here). Leave a MORNING_SUMMARY.md with what landed,
what's blocked, and decisions needed. Prefer correctness (P1) over features. Ask nothing that can
be defaulted; surface only genuine forks in the morning summary.

## Progress log
- [night] P1 squad dispatched (3 diagnosers + synthesis → reports/device_invariance_2026-07-08/).
- [night] P2 DONE: BGW invalid_gpp_mode refs on Si 4×4×4 (GN + HL lines, modes 0/2/3; default==3
  verified empirically; 8.63% invalid poles; mode-diff table). reports/bgw_invalid_mode_refs_2026-07-08/.
  Also fixed a compare-skill parser gap (freq_dep=1 logs) → KNOWN_SANDBOX_ERRORS.
- [night] P2b dispatched: LORRAX zero/2ry runs on the same Si system, delta-vs-delta validation
  against the BGW refs (no source edits; scaffold for the static_limit 3-way comparison).
- [pending chain] P1 synthesis → fix (gated) → multi-device invariance gate. P2b + P1 done →
  static_limit implementation (Wc0/B÷Ω analytic term, default, gate re-freeze, validate vs 01c).
  P3a design → toggle implementation once the tree is free of running diagnosis jobs.
- [night] P3a DONE: G0W0_SC_TOGGLE_DESIGN.md. eqp0/eqp1 already = G0W0 at-DFT; toggle -> qp_solver
  enum (one_shot_dft default | fixed_point | self_consistent) + SCConfig (closes #11). Jit audit:
  SC-COHSEX 0 retraces/iter; SC-GN-PPM exactly 2 (qsgw_utils.py:165/:262 nested-scope jits — hoist).
  SC works for GN-PPM (stale comments). Implementation P3b HELD until P1/P2b runs finish (shared tree).
- [night] P1 DONE — ROOT CAUSE FOUND (reports/device_invariance_2026-07-08/ROOT_CAUSE.md): the
  lead's padding suspicion CONFIRMED. Computations run on the PADDED mu extent (P-dependent):
  (1) bispinor transverse zeta LU solved at padded extent — PROVEN: full 16g signature reproduced
  at fixed P=4 with +4 forced pad rows; (2) charge PPM mode census + minimax node selection over
  padded mu^2 — node counts flip with P. eV scale = on-pole amplification of a real deterministic
  leak. NOT accumulation order. Fixes: slice LU to logical (~30 L isdf/core.py), logical-mask the
  census (~60 L minimax_screening/ppm_windows), + 1-GPU pad-flip bit-identity gate. One charge
  confirm run pending (EXTRA_MU_PAD=12 @ P=4).
- [pending] Fix implementation HELD until P2b's runs finish (shared tree), then: pad knob + charge
  confirm + Fix 1 + Fix 2 + Tier-1/2 invariance gates + full gate suite, one gated commit series.
- [night] P2b DONE: LORRAX zero/2ry wiring PASS vs BGW refs (statics bit-identical, toggle-only
  delta, sign pattern matches; magnitudes scale with the ISDF invalid-pole population 1.13% vs
  BGW 8.63% — expected). reports/bgw_invalid_mode_refs_2026-07-08/lorrax_zero_2ry_validation.md.
- [night] P1-FIX dispatched (tree quiet): pad knob + charge confirm (PAD=12@P=4) + Fix 1 (bispinor
  LU logical extent) + Fix 2 (census logical mask) + Tier-1 suite gate + Tier-2 script + post-fix
  4g/16g revalidation. Order: P1 fixes -> P3b (qp_solver) -> static_limit (re-freeze LAST).
- [night] PADDING AUDIT DONE (reports/device_invariance_2026-07-08/PADDING_AUDIT.md): design right,
  implementation ~2x needed size (~1150 live + ~660 DEAD pad lines, 3 divisor conventions, solve
  skeleton x6). 3 latent bugs remain at HEAD: restart writers persist PADDED P-dependent extents
  (disk-contract violation), isdf_fitting allgather branch crash, opt-out-by-omission mask defaults.
  Census fix should move to FIT BIRTH (~3 lines, deletes the consumer arm, bit-identical). Net
  achievable ~-900 lines, bugs 3->0. #8 zeta merge UNBLOCKED by the new pad-invariance gate.
- [pending chain] After P1-fix reports: padding consolidation pass (latent bugs + fit-birth rework +
  dead PadAxis delete + _solve_at_logical helper) -> P3b qp_solver -> static_limit (re-freeze last).
- [night] P1-FIX DONE (083d209..7801d46, 249 passed): bispinor 4g/16g eqp 2.535 eV -> 1.45e-7 eV
  (FIXED); charge census P-invariant; on-pole residual = near-threshold mode flips x GN-PPM
  ill-posedness (Fix-3, documented, would change golden numbers -> deferred). Tier-1 pad-flip gate
  in suite; Tier-2 cross-P script passing. AS-FIXED tables in ROOT_CAUSE.md.
- [night] PADDING CONSOLIDATION dispatched (audit items re-verified at HEAD): restart-writer clip,
  allgather-crash branch, explicit masks, fit-birth census rework, dead PadAxis delete (~-760),
  round_up + _solve_at_logical helpers. Then P3b qp_solver -> static_limit (re-freeze last).
- [night] PADDING CONSOLIDATION DONE (6c850bd..620b501, 7 commits, net -553): all 3 latent bugs
  fixed (restart logical-on-disk + roundtrip gate, allgather unified on SlabIO, no opt-out masks),
  census moved to fit birth (pad modes born dead), dead PadAxis deleted (-757), one round_up, one
  solve_at_logical, sym-table pad baked. Goldens bit-identical; Tier-2 cross-P re-passed unchanged.
  Left alone (documented): psi/enk band-axis padded restart, g0 conventions.
- [night] P3b dispatched: qp_solver enum (one_shot_dft default) + SCConfig (closes #11) + qsgw
  nested-jit hoist (2 retraces/iter -> 0) + fixed_point-relabeling proof. static_limit LAST.
- [night] P3b DONE (4e3ecfd/caf1f9a/509e492): qp_solver = one_shot_dft(default)|fixed_point|
  self_consistent; SCConfig replaces LORRAX_SC_* envs (#11 closed); sigma_at_dft_energies absorbed;
  qsgw nested-jit hoist -> SC steady-state 0 compiles/iter (was 2); fixed_point pinned to frozen
  qp_rotations ref. 257 passed; goldens bit-identical (only the eigh-family outputs move, by design).
- [night] STATIC_LIMIT dispatched (the last WS6 item): analytic -1/2*Wc0 static CH for invalid
  poles via the COHSEX contraction template, BGW parity on the Si three-way scaffold FIRST, then
  default flip + golden re-freeze (statics/bispinor must not move).
