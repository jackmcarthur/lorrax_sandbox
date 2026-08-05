You are **Agent 2 of 4 — round 2** of the LORRAX install/maintain blitz
audit.

In round 1 you wrote `agent_2.md` (Cray MPI + Shifter container
surface). The three other agents wrote drafts on their own slices:

- `agent_1.md` — Build system & FFI dependency contract (Agent 1)
- `agent_3.md` — Runtime: env vars, jax.distributed init, launchers (Agent 3)
- `agent_4.md` — Docs, onboarding, cohsex.in config, CI/test gaps (Agent 4)

**Now read all four drafts.** Yours, plus the other three. You did not
see theirs in round 1.

Your shared briefing from round 1 still applies:
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/CONTEXT.md`

(Especially §6 — the FRAGILE / COMPAT / LOC-COST tagging convention.)

---

## Your round-2 job

Produce `round2_agent_2.md` — a focused **reconciliation document**
with the same five sections (A-E) described below.

### Section A — Cross-slice convergence

Identify every defect that you AND at least one other agent flagged
(possibly under different tags or different file:line pointers). For
each, one bullet:

- Brief restatement of the defect.
- Which agents flagged it.
- The strongest tag across all flaggers.
- One sentence on whether the agents agree on the fix, or whether
  the proposed blitzes diverge.

Aim for ~10-20 convergent items.

### Section B — Disagreements

Real disagreements (not just non-overlap):

- A defect one agent says is fragile that another implicitly says is
  fine.
- Conflicting blitz proposals that touch the same code.
- Different rankings of the same blitz item.

For each: who said what, who you think is right (with citations), and
what evidence would resolve it.

### Section C — Your slice-specific updates

For each of your original Agent-2 defects and blitz proposals: mark
**CONFIRMED** / **SHARPENED** / **REVISED**. Brief justification for
each.

### Section D — Cross-cutting blitz proposals

The top 5 blitz items across all 4 drafts, in priority order, with:
- Title + which agent(s) proposed it.
- Concrete acceptance criteria.
- The one defect it most directly closes.
- Dependencies on the other top-5 items.
- Whether desk-doable or needs a real Perlmutter run.

Commit to a ranking. No "it depends on the axis" hedging.

### Section E — Open questions for the user

Short list of questions only the author can resolve, that gate one or
more blitz items.

---

## Constraints

- You MAY now read all four round-1 drafts.
- Still read-only on `sources/lorrax_C/`. No code edits.
- No compute. Web search remains authorized.
- Write to `round2_agent_2.md` only.
- Stay in your tmux pane.
- Stop when written. Print: `Agent 2 round 2 done — see round2_agent_2.md`.

### Output bias

You are Agent 2 (Cray MPI / Shifter / container portability
specialist). Your tie-breaking weight is highest on MPI / container /
distributed-init issues. On build-system or docs/cohsex.in issues, you
can flag a disagreement without resolving — but on Apptainer /
Frontier / Polaris portability questions, your judgement should be the
strongest of the four.

A specific instruction for you: **be brutal about what's actually
NERSC-specific vs. portable.** Round 1 had multiple agents tagging
things as [COMPAT] that arguably aren't (e.g. `select_gpu.sh` is a
generic Slurm convention, not a NERSC-ism). Your round-2 job includes
correcting tag inflation.

Start by reading the other three drafts.
