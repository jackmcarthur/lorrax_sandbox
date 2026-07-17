# BSE deletion ledger + current inventory verdicts (2026-07-17)

Owner-requested file-by-file record: what has been deleted/archived/consolidated
in src/bse since base c7a30ff, and verdicts on everything that remains.
Rules applied (owner): gate-supplanted old test scripts can go; BGW-comparison
efforts and purposeful stubs are KEPT (consolidated at most); the pseudopoles
thread is FLAGGED (alternative W(ω) model, owner workshopping) — untouched.

## 1. Retrospective ledger (phase-1 cleanup, merged to main at 6bd4dc9)

| Commit | File / symbol | Action | LOC | Supplanted by / reason |
|---|---|---|---|---|
| 6bd4dc9 | feast_sweep.py, feast_zolo_sweep.py, feast_ellipse_mixed_sweep.py | Archived → archive/experiments/ | −1287 | absorbed into bse_feast.main --quadrature/--n-quad-schedule; all 3 broken at HEAD; zero importers |
| bd3ff6d | bse_io.BSEData | Deleted | −6 | never instantiated |
| bd3ff6d | bse_preconditioner: 7 of 8 exports | Deleted | −141 | no external entry point; energy_diff_cv_k kept (live) |
| ce724ab | bse_jax matvec trio (apply_bse_hamiltonian/apply_D/apply_V/apply_W) | Deleted | −102 | zero callers; live path = serial/ring builders |
| ce724ab | bse_ring_comm: apply_bse_hamiltonian_ring + apply_W_ring | Deleted | −74 | zero callers |
| ce724ab | bse_serial.symmetrize_W_q | Deleted | −15 | imported, never called |
| 3f30db3 | write_eigenvectors.py | Deleted | −236 | superseded + buggy (Ry-not-eV, no valence flip); replaced by bse_io.write_eigenvectors_stream |
| eb316f4 | test_bse.py stale loader + find_restart_file | Deleted (consolidated) | −171 | read datasets no restart carries; repointed to bse_io |
| eb316f4 | generate_kpts_grid | Consolidated → bse_io._generate_kpts_grid | — | removed last production dep on write_eigenvectors |
| 41db774 | phantom v_couples_k kwarg (bse_kpm, bse_pseudopoles) | Removed | −6 | TypeError on every entry; never existed in ring history |
| 0495a44 | density drive/readout probe ops | RESTORED (+32) | +32 | lost wiring for pseudopoles thread, not dead |
| adfeb9a | use_nohead kwarg + bare-V-fallback warning | Restored | — | W0-not-ready now warns loudly |
| 0a6d407…402e1cc | B3–B7 loader/einsum/pad/eqp/doc repairs | Fixed/consolidated | — | no deletions |

Deferred consolidations (surfaced, not actioned — CLEANUP_LOG): BSE config →
gw_config; mesh-builder unification; gw_jax.main setup-prefix extraction;
bse_feast_dense_debug quadrature dedup; fold bse_preconditioner into bse_serial;
apply_q0_head_rank1 now-dead (GW scope).

## 2. Current inventory verdicts (src/bse @ f19136e-era HEAD)

| File | Verdict | Key evidence |
|---|---|---|
| test_bse.py | DELETE after carve-out (HELD for tree) | no asserts; correctness fully supplanted by test_bse_dense_reference (@1e-9, strictly stronger); zero importers. UNIQUE remainder: only e2e exerciser of write_eigenvectors_stream (BGW-spec writer, un-gated) + 1-device bench harness → lift writer exercise into a pytest/tools script, update context/README + docs, then delete |
| test_davidson_bse.py | KEEP-CONSOLIDATE | BGW-comparison effort: sharded Davidson vs BGW eigenvectors.h5 cosine-sim + meV — IS the future BGW anchor gate; promote when the BGW h5 fixture is wired |
| bse_feast_dense_debug.py | KEEP (documented intent) | zero callers, but CLEANUP_LOG deliberately kept it as the seed of a future FEAST pytest fixture; its contour-filter+RR validation is NOT gate-supplanted. Revisit at solver-program P2 |
| pseudopoles_sweep.py, pseudopoles_eval.py, bse_pseudopoles.py | KEEP (FLAGGED thread) | alternative pseudopole W(ω) model, owner workshopping; OVERLAPS w_omega_chain.py — decision deferred to the W(ω) physics workshop |
| w_omega_chain.py | KEEP (MVP, physics frozen pending owner) | validated vs oracle; no consumers wired |
| bse_w_exact.py | KEEP | rewritten as the single-source resolvent oracle; no pre-resolvent remnants found |
| ring probe ops (density drive/readout) | KEEP | live pseudopoles-thread infra |
| davidson_absorption.py | KEEP-CONSOLIDATE | MAP §4 duplicate-driver → collapse into bse_jax dispatch (consolidation, not deletion) |
| eigvals_to_eps2, absorption_{haydock,eigvecs,common} | KEEP | live BGW-comparison ε₂ paths |
| bse_kpm.py | KEEP | live KPM-DOS |
| bse_jax, bse_io, bse_serial, bse_simple, bse_stack_matvec, bse_lanczos, bse_feast, bse_davidson_helpers | KEEP | live pipeline (some MAP consolidation targets) |
| bse_preconditioner.py | KEEP-CONSOLIDATE (minor) | single function; fold into bse_serial eventually |
| BGW_COMPARE.md, eigenvectors.h5.spec, STATUS.md, context/ | KEEP by rule | comparison docs/specs |

src/solvers: no BSE-only dead code (lanczos/davidson/chebyshev/dos/quadrature
are shared; pseudobands*/sternheimer*/minres/projectors serve psp/).

## 3. Ready-to-execute queue (when the tree quiesces)

1. test_bse.py: lift the write_eigenvectors_stream exercise into a small pytest
   gate (or tools/ script), update context/README.md + docs/architecture/
   codebase.md references, delete the rest. Provably suite-invariant.
2. (deferred to solver-program P2) bse_feast_dense_debug → pytest fixture per
   its documented intent, or archive then.
