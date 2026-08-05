# Harness re-architecture investigation — index

Session artifacts, 2026-08-04/05. Read order for a newcomer:

| File | What | Status |
|---|---|---|
| `report.md` | The harness proposal grounded in Weng's "Harness Engineering for Self-Improvement" (2026-07-04): scorecard of what the sandbox already implements, 7 prioritized proposals, anti-goals | proposal |
| `CONSOLIDATION.md` | Ordered runbook for the Frontera push + truth repair — the owner's stated next step; phases 2-5 reference the other docs | runbook |
| `RULES_v2.md` | Curated design-rule set with owner verdicts and a placement map (DESIGN.md / scoped AGENTS.md / gates / runtime refusals / TASTE.md) | owner-reviewed |
| `DESIGN_draft.md` | Draft of the proposed lorrax docs/architecture/DESIGN.md — the three-layer spec, the O(N³) identity, JAX discipline ("the traced graph is the bill"), mesh/process rules | draft, needs owner edit |
| `EVIDENCE_DESIGN.md` | Synthesis of three research sweeps (ML tracking, scientific provenance, HPC benchmarking) on evidence-record design; validates the CLAIMS pattern, prescribes the harness-written JSONL layer | synthesis |
| `RULES_seed.md` | Raw mined rules from lorrax origin @ 2026-07-22 (~140 rules with file:line + enforcement status) — evidence base for RULES_v2; re-sweep after consolidation | raw evidence |
| `rules_gate.py` + `rules_gate_allowlist.json` | Working banned-pattern gate (raw jnp.fft, device_put) with ratcheting allowlist; python 3.7 stdlib; tested against origin (203 sites frozen) | draft tool, tested |

Caveat on scope: everything code-facing here was derived from the GitHub
origin of lorrax (2026-07-22), which is ~150 commits behind the certified
Frontera tree. CONSOLIDATION.md phase 1 closes that gap; RULES_seed and
the gate allowlist must be regenerated afterward.
