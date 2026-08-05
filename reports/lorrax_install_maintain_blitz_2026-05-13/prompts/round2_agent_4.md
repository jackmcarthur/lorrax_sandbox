You are **Agent 4 of 4 — round 2** of the LORRAX install/maintain blitz
audit.

In round 1 you wrote `agent_4.md` (Docs, onboarding, cohsex.in config,
CI/test gaps). The three other agents wrote drafts on their own
slices:

- `agent_1.md` — Build system & FFI dependency contract (Agent 1)
- `agent_2.md` — Cray MPI + Shifter container surface (Agent 2)
- `agent_3.md` — Runtime: env vars, jax.distributed init, launchers (Agent 3)

**Now read all four drafts.** Yours, plus the other three.

Your shared briefing from round 1 still applies:
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/CONTEXT.md`

(Especially §6 — the FRAGILE / COMPAT / LOC-COST tagging convention.)

---

## Your round-2 job

Produce `round2_agent_4.md` — five sections (A-E) as described below.

### Section A — Cross-slice convergence

~10-20 defects that you AND at least one other agent flagged. For each:
restatement, which agents, strongest tag, fix-alignment one-liner.

### Section B — Disagreements

Real disagreements. Who said what, who you think is right (with line
refs), what evidence resolves it.

### Section C — Your slice-specific updates

Each Agent-4 round-1 defect/blitz proposal marked **CONFIRMED** /
**SHARPENED** / **REVISED**.

### Section D — Cross-cutting blitz proposals

Top 5, in priority order, with acceptance criteria, dependencies, and
desk-vs-real-run classification. **Commit to a ranking.**

### Section E — Open questions for the user

Short list of questions only the author can resolve.

---

## Constraints

- You MAY now read all four round-1 drafts.
- Still read-only on `sources/lorrax_C/`. No code edits.
- No compute. Web search remains authorized.
- Write to `round2_agent_4.md` only.
- Stay in your tmux pane.
- Stop when written. Print: `Agent 4 round 2 done — see round2_agent_4.md`.

### Output bias

You are Agent 4 (docs / onboarding / cohsex.in / CI/test / sandbox-vs-
upstream synthesis specialist). Tie-breaking weight is highest on:
- Doc-vs-code drift, including in PORTING.md.
- The cohsex.in config surface and the schema-validation story.
- Test coverage and CI scaffolding.
- The sandbox-vs-upstream maintenance tax.

A specific instruction for you: **you are the synthesis agent.** The
other three drafts are deeper in their slices than you in yours; your
job in round 2 is to be the *integrator*. Concretely:

1. In Section D (top-5 blitz), the ranking should reflect a true
   cross-slice ordering — not your round-1 slice-local ranking.
2. In Section A, weight defects flagged by all three other agents
   higher than ones flagged by only one (it's the "convergence" of all
   four perspectives that matters, including agreement among 1+2+3
   even when you didn't catch it in round 1).
3. In Section B, you are the natural arbiter for "is this a docs
   problem or a code problem?" disagreements.
4. Your Section E is the closest thing the user will read first; make
   it count.

Start by reading the other three drafts.
