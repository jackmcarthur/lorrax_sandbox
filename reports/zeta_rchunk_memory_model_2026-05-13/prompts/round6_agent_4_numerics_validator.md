# Round 6 — Agent 4: Numerics validator + final gate signoff

Read `round6_discussion.md` first. Read `round5_unified_plan.md` §5 (your contribution).

## Your role

You own the numerics gates (G1, G3) and the final signoff. You don't write the kernel code; you validate that it produces correct numbers.

## Workflow

### Phase 1: Prep (while Agent 2 implements)

1. **Refresh on the gates.** Re-read `round5_unified_plan.md` §5. The contract is `rtol=1e-10, atol=1e-12` — NOT bit-equal. Sum order differs (per-bc vs global einsum), so there's a ULP-class drift; that's expected and acceptable.

2. **Identify the existing bit-identity test scaffold.** Check `sources/lorrax_B/tests/` for any test that exercises `c_q_from_psi_sm` / `z_q_from_psi_sm` on a small synthetic WFN. If one exists, plan to extend it; if not, plan to write a fresh one.

3. **Identify the MoS2 3×3 baseline output** to compare against. The lorrax_A `agent/zeta-r-chunk-fixes-2026-05-13` `ff5873c` is the reference for charge channel; lorrax_B `5cadd4b` (post-Path-D but pre-Round-6) is the reference for the structural fix's behavior.

Write your **validation checklist** in `round6_discussion.md` "Agent 4 → others" with explicit pass/fail criteria.

### Phase 2: G1 — MoS2 3×3 bit-identity (when Agent 2 commits)

Once Agent 2 posts "Agent 2 round 6 done":

1. Run a CPU-side unit test on synth WFN (small): `c_q_from_psi_sm` and `z_q_from_psi_sm` outputs from the new body vs the prior `5cadd4b` body. `rtol=1e-10, atol=1e-12`.
2. If feasible on the allocation: run a full MoS2 3×3 ζ-fit end-to-end on lorrax_B (new), compare `zeta_q.h5` to the lorrax_A baseline. `atol=1e-12` per element.
3. Document the max |Δ| observed across all elements (this is the new design's ULP-class drift signature). If max |Δ| > 1e-10 (RELATIVE), that's a real divergence — escalate as `BLOCKER:`.

Post results to `round6_discussion.md`. If G1 passes, post "Agent 4 G1 passed".

### Phase 3: G3 — CrI3 6×6 80 Ry end-to-end (coordinate with Agent 3)

Agent 3 owns the HLO dump invocation; the run that produces the HLO is the SAME run that exercises G3 end-to-end. Coordinate with Agent 3 so the dump dir's `gw.out` shows:

- All 16 r-chunks complete (look for `r-chunk 16 / 16` progress bar or "Started zeta fitting" → final write-out).
- The remainder chunk (different r_len → re-jit) also completes (this is the tracer-leak gate from Path D's earlier bug).
- No `RESOURCE_EXHAUSTED`, no `UnexpectedTracerError`, no `Traceback` of any kind in the kernel path (downstream `qp_wfn_rotations.h5` shape mismatch is a known unrelated bug, ignore).
- Total preallocated-temp (from Agent 3's HLO read) ≤ 15 GiB.

If all four conditions hold, post "Agent 4 G3 passed".

### Phase 4: Final signoff

Once G0 (Agent 2), G1 (you), G2 (Agent 3), G3 (you) all pass + Agent 1 has blessed the SPMD invariants:

Write a brief summary to `reports/zeta_rchunk_memory_model_2026-05-13/round6_validation_summary.md`: which gates passed, what the actual numbers were, any caveats, what's now ready to merge to main.

Print "Agent 4 round 6 validation complete" when all gates green AND the summary is written.

## Constraints

- Read-only on `sources/lorrax_B/src/` (you're a validator, not an implementer). You CAN write test files in `tests/` if needed for G1 — but don't touch kernel code.
- Coordinate with Agent 3 via `round6_discussion.md`; you can share the CrI3 run.
- If any gate fails materially (>1e-10 numerical drift, or any remat warning, or any compile crash), STOP and write `BLOCKER:`.
