# Round 4 — Agent 1: What improved (HLO before/after)

Read `round4_discussion.md` first (orchestrator-maintained status snapshot, plus the file-polling protocol). Specifically the "Headline numbers" table and "The new defect" section.

## Your task

Produce a rigorous before/after HLO comparison between:
- **Before**: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/xla_dump/module_0408.jit__kernel.*memory-usage-report.txt` (lorrax_A morning baseline, OOMing at 200 GiB).
- **After**: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_path_d_hlo_2026-05-13/xla_dump/module_0394.jit__kernel.*memory-usage-report.txt` (lorrax_B Path D fix, 48 GiB, progressing). Plus modules 0293, 0295 if useful (same kernel, different compile slots).

For both reports, extract:

1. **Total bytes** + **preallocated-temp pool size**.
2. **FFT-box-class slot count** — count distinct slots holding `c128[k_chunk, bc, ns, nx, ny, nz]`-shape or related (variants Agent 1 enumerated in this morning's `hlo_findings.md` §1).
3. **Pair-density slot count** — count distinct slots holding the rank-5 / rank-7 pair-density shape (`c128[ns, ns, μ_local·r_local, nk]` and its reshape variants).
4. **Per-side band slab presence** — is there now a dedicated `c128[nk, nb_local, ns, r_chunk]` slot (the new helper's output)? If so, what size?
5. **Anything new that wasn't in the baseline.**

Then quantify what the structural fix achieved:
- Memory savings (200 GiB → ? GiB; FFT-box slot collapse 58 → ?).
- What's NOT in the new HLO that was in the baseline (defect mechanism gone).
- What's NEW in the after HLO (new defect — the remat warnings, the per-side band slab, anything else).

**Don't speculate; cite line numbers in the HLO reports.**

## Output

Write to `reports/zeta_rchunk_memory_model_2026-05-13/round4_improvements.md`. Sections:
1. **Before/after numbers** (table).
2. **What's no longer in the HLO** (the eliminated defect, with evidence).
3. **What's new in the HLO** (esp. anything around the remat warnings).
4. **Net assessment**: how much of the principle is satisfied? Score it: was the goal 200 → ~3 GiB, and we landed at 48 GiB — what fraction of the savings came from the structural fix vs is still pending the remat fix?

Communicate with the other 3 agents via `round4_discussion.md` (poll before each work cycle, respond to peer messages). Print "Agent 1 round 4 done" when finished.

Read-only on `sources/`. No compute. ~10–20 min target.
