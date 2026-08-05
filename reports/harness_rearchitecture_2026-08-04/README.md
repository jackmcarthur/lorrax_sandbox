# Harness re-architecture investigation — index

Session artifacts, 2026-08-04/05. LIVING documents (read these):
TARGET_ARCHITECTURE.md (what), PRIORITIES.md (what actually gets built,
in what order -- supersedes every other adoption-order list),
CONSOLIDATION.md (the immediate runbook), RULES_v2.md + DESIGN_draft.md
(the rule content). Everything else is the RESEARCH RECORD -- read only
when questioning why a decision was made. Full index:

| File | What | Status |
|---|---|---|
| `TARGET_ARCHITECTURE.md` | Capstone: the intended end-state in five functions (orientation, enforcement, verification, evidence, execution) with per-piece contents, spec pointers, and the five compounding loops | capstone summary |
| `PRIORITIES.md` | Ruthless triage of every piece against the measured-cost bar: tiers 0-3 plus explicit cuts with reinstatement conditions; the owner's design-against-objections rule; SUPERSEDES all other adoption-order lists | authoritative priority list |
| `report.md` | The harness proposal grounded in Weng's "Harness Engineering for Self-Improvement" (2026-07-04): scorecard of what the sandbox already implements, 7 prioritized proposals, anti-goals | proposal |
| `CONSOLIDATION.md` | Ordered runbook for the Frontera push + truth repair — the owner's stated next step; phases 2-5 reference the other docs | runbook |
| `RULES_v2.md` | Curated design-rule set with owner verdicts and a placement map (DESIGN.md / scoped AGENTS.md / gates / runtime refusals / TASTE.md) | owner-reviewed |
| `DESIGN_draft.md` | Draft of the proposed lorrax docs/architecture/DESIGN.md — the three-layer spec, the O(N³) identity, JAX discipline ("the traced graph is the bill"), mesh/process rules | draft, needs owner edit |
| `EVIDENCE_DESIGN.md` | Synthesis of three research sweeps (ML tracking, scientific provenance, HPC benchmarking) on evidence-record design; validates the CLAIMS pattern, prescribes the harness-written JSONL layer | synthesis |
| `RULES_seed.md` | Raw mined rules from lorrax origin @ 2026-07-22 (~140 rules with file:line + enforcement status) — evidence base for RULES_v2; re-sweep after consolidation | raw evidence |
| `ANTIPATTERN_ENFORCEMENT.md` | Idiom-propagation diagnosis (3 failure classes, from an 18-idiom parity audit of gw/psp vs lagging modules) + the 8-rung enforcement stack: scaffolds, routed context, edit-time hooks, ast-grep corpus, import contracts, extractions, jaxtyping, behavioral gates | synthesis |
| `rules_gate.py` + `rules_gate_allowlist.json` | Working banned-pattern gate (raw jnp.fft, device_put) with ratcheting allowlist; python 3.7 stdlib; tested against origin (203 sites frozen) | draft tool, tested |

Caveat on scope: everything code-facing here was derived from the GitHub
origin of lorrax (2026-07-22), which is ~150 commits behind the certified
Frontera tree. CONSOLIDATION.md phase 1 closes that gap; RULES_seed and
the gate allowlist must be regenerated afterward.
