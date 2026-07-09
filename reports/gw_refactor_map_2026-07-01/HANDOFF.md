# HANDOFF — GW refactor program state (fresh-session entry point)

_2026-07-09. READ THIS FIRST. One page: what's done, what's live, what's next.
Everything else in this directory is either one of the 8 live docs indexed below or
archived history under `archive/`._

## State of the world
- Branch: **`agent/memplanner-cleanup`** on `sources/lorrax_D`, all pushed. Never commit to main.
- Test suite: **257 passed / 9 skipped / 0 failed**. Terminology used throughout these docs:
  a **"gate"** = a regression test that runs a small end-to-end GW calculation and compares
  against a frozen reference output. There are 12 of these, covering: BGW-anchored Si 3D COHSEX,
  MoS2 COHSEX + GN-PPM, bispinor, IBZ-vs-full-BZ equivalence, host-vs-streamed accumulator
  parity, padding invariance, and restart round-trip.
- Working rules that have earned their keep: (1) every commit runs at least the regression
  subset, every commit series runs the full suite; (2) before deleting anything "dead",
  re-verify zero callers with grep at current HEAD — documentation claims of deadness have
  been wrong four separate times in this program; (3) tests must never require 16 GPUs
  (the code ships to users on arbitrary device counts); (4) LORRAX source changes go on
  feature branches, never main; (5) keep communication with the user concise and plain.

## What landed (the program so far, one line each)
1. **Σ_PPM tightening** (planned in SIGMA_PPM_MAP.md + the 4-agent consensus at
   reports/sigma_ppm_tighten_2026-07-04/consensus_draft.md, then executed workstream by
   workstream): the 1632-line ppm_sigma.py monolith split into 4 single-concern files; the
   drifted config-mirror dataclass deleted (config read directly); the ±1 sign-carrying
   variables (kernel_sign/scale) removed — each branch's sign now written inline where the
   physics dictates it, with the S = E_A+Ω convention; two duplicated accumulators unified
   into one (+38% streamed-path speedup); 3 real bugs fixed (head sign-flip, streamed runs
   silently dropping the q→0 head, a shard-0-only assumption wrong on multi-GPU);
   ~1000 lines of dead code removed.
2. **Memory model redesigned**: one ISDF planner (persistent floor + max stage-transients +
   rank floor), validated ≤0.1% vs BFC peaks on 4+16 GPU. MEMORY_MODEL_DESIGN.md.
3. **Device-invariance bug found + fixed**: computations ran on the PADDED (P-dependent) μ
   extent — bispinor 4↔16 GPU eqp 2.535 eV → 1.4e-7. Padding then consolidated (−553 L, one
   convention, 3 latent bugs incl. P-portable restarts). reports/device_invariance_2026-07-08/.
4. **qp_solver axis**: one_shot_dft (default, textbook G0W0) | fixed_point | self_consistent;
   SCConfig replaces env knobs; SC steady-state = 0 recompiles/iter. G0W0_SC_TOGGLE_DESIGN.md.
5. **ppm_invalid_mode complete**: zero/2ry/static_limit all BGW-validated three-way on Si;
   **static_limit is now the default** (= BGW's default). reports/bgw_invalid_mode_refs_2026-07-08/.

## Live docs (read order for a fresh session)
1. `HANDOFF.md` (this) → 2. `STATUS.md` + `NEXT_TARGETS.md` (ledger; trust dates ≥ 07-09)
3. `MAP.md` — the code map (stage taxonomy A1–A8/B/C axes)
4. `SHARDING_RULES.md` — the SPMD/padding contract (permanent reference)
5. `SIGMA_PPM_MAP.md` — the Σ_PPM flow + ledger (permanent reference)
6. `IDEAL_SCAFFOLD_VS_LORRAX.md` — the pending driver-revision decision (see Next)
7. `MEMORY_MODEL_DESIGN.md`, `G0W0_SC_TOGGLE_DESIGN.md` — as-built subsystem references
8. `MORNING_SUMMARY.md` — the 2026-07-08 overnight record (most recent detailed session log)

## Remaining work, ranked
1. **Driver transparency revision** — the user's active interest: make gw_jax.main() read
   as the 12-line physics scaffold. Spec: IDEAL_SCAFFOLD_VS_LORRAX.md §5, two phases.
   Phase B = pure code motion, ~350 lines out of main() (the IBZ slice/unfold block into a
   solve_w wrapper; restart flush, debug writers, degeneracy-averaging into helpers) —
   output must be bit-identical, the existing tests verify that. Phase C = the real
   unification: the one-shot path starts consuming the same screening/sigma dispatch the
   self-consistent loop already uses (deleting a duplicate pipeline); fit_ppm lifts to top
   level; the three QP-solver branches collapse into one solve_qp(). Acceptance for C:
   self-consistency iteration 1 must equal the one-shot result on a fixture. Three things
   deliberately NOT to do (physics reasons in the spec): don't materialize G(t), don't turn
   the q→0 head into a pipeline stage, don't evaluate W on a frequency grid.
2. **On-pole PPM census robustness — needs a physics decision from the user** (called
   "Fix-3" in ROOT_CAUSE.md): bands sitting on a PPM pole are still device-count-sensitive,
   because a handful of PPM modes sit near the validity threshold (Ω² ≈ 0) and flip
   valid↔invalid with floating-point noise, which then changes the adaptive quadrature's
   node counts. Candidate fixes: hysteresis on the validity threshold, a fixed reference
   for the census, or documenting on-pole Σ(E_dft) as ill-posed. See
   reports/device_invariance_2026-07-08/ROOT_CAUSE.md, "AS-FIXED" section.
3. **zeta_loader/zeta_reader merge** — long-deferred, NOW UNBLOCKED (the padded-μ gate exists:
   tests/test_mu_pad_invariance.py). ~−350 L.
4. Small hygiene: derive the Σ τ-kernel's projection code from the quadrature-node kind
   instead of a free string (last leftover of the Σ_PPM plan); the restart file still stores
   the ψ/energy band axis at a padded, device-count-dependent size (documented limitation,
   PADDING_AUDIT.md "AS-CONSOLIDATED"); NEXT_TARGETS #12/#13 (split the driver-helpers
   grab-bag file; split get_DFT_mtxels.py).

## Sibling report dirs (evidence, keep)
`reports/device_invariance_2026-07-08/` (root cause + padding audit + as-fixed),
`reports/bgw_invalid_mode_refs_2026-07-08/` (BGW mode refs + three-way tables),
`reports/sigma_ppm_tighten_2026-07-04/` (the 4-agent consensus round),
`reports/memory_model_refit_2026-07-03/` (planner validation data).
