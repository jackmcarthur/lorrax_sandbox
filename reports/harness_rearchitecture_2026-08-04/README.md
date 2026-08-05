# Harness re-architecture investigation — index

Session artifacts, 2026-08-04/05.

**Read PLAN.md first — it is the whole plan on one page, as a tree.**
Then CONSOLIDATION.md (the step-by-step checklist for the GitHub push),
and RULES_v2.md + DESIGN_draft.md for the actual rule content. Everything
else is the research record behind those decisions; read it only when
questioning why a decision was made.

| File | What | Status |
|---|---|---|
| `PLAN.md` | The plan in plain language: one tree, the order of work, what was cut and why. Replaces the earlier TARGET_ARCHITECTURE.md and PRIORITIES.md | current |
| `CONSOLIDATION.md` | Step-by-step checklist for pushing the Frontera work to GitHub and fixing the stale docs afterward | runbook |
| `RULES_v2.md` | The design rules with Jack's rule-by-rule verdicts applied, and where each rule should live (design doc / per-directory notes / lint / startup check) | owner-reviewed |
| `DESIGN_draft.md` | Draft of the proposed 2-page lorrax design doc: the three code layers, the O(N^3) rule, the JAX rules, mesh/process rules | draft, needs Jack's edit |
| `rules_gate.py` + allowlist | Working lint for banned patterns (raw jnp.fft, device_put); existing violations frozen, count can only decrease; runs on login-node python 3.7 | working, 2 rules |
| `report.md` | Original proposal based on Lilian Weng's harness-engineering post | research record |
| `RULES_seed.md` | Raw rule mining from the repo (~140 rules with file:line) — the evidence behind RULES_v2 | research record |
| `EVIDENCE_DESIGN.md` | How other fields (ML tracking, materials-science provenance, HPC benchmarking) keep experiment records; why we keep the ledger but add machine-written job logs | research record |
| `ANTIPATTERN_ENFORCEMENT.md` | The audit of why good patterns in gw/psp don't spread to other directories, and the enforcement options considered | research record |

Caveat: everything code-facing was derived from the GitHub copy of
lorrax (2026-07-22), which is ~150 commits behind the real Frontera
tree. After the push, re-run the rule mining and regenerate the lint
allowlist.
