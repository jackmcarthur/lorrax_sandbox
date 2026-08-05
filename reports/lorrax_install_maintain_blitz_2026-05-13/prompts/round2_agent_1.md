You are **Agent 1 of 4 — round 2** of the LORRAX install/maintain blitz
audit.

In round 1 you wrote `agent_1.md` (Build system & FFI dependency
contract). The three other agents wrote drafts on their own slices:

- `agent_2.md` — Cray MPI + Shifter container surface (Agent 2)
- `agent_3.md` — Runtime: env vars, jax.distributed init, launchers (Agent 3)
- `agent_4.md` — Docs, onboarding, cohsex.in config, CI/test gaps (Agent 4)

**Now read all four drafts.** Yours, plus the other three. You did not
see theirs in round 1.

Your shared briefing from round 1 still applies:
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/CONTEXT.md`

(Especially §6 — the FRAGILE / COMPAT / LOC-COST tagging convention.)

---

## Your round-2 job

Produce `round2_agent_1.md` — a focused **reconciliation document** that:

### Section A — Cross-slice convergence

Identify every defect that you AND at least one other agent flagged
(possibly under different tags or different file:line pointers). For
each, one bullet:

- Brief restatement of the defect.
- Which agents flagged it (e.g. "Agent 1 D#13 + Agent 2 D4").
- The strongest tag across all flaggers (FRAGILE > COMPAT > LOC-COST
  priority is fine — but record disagreements about tagging).
- One sentence on whether the agents agree on the fix, or whether
  the proposed blitzes diverge.

Aim for ~10-20 such convergent items. **This list is the single most
load-bearing output of round 2** — it's what the consensus document
will quote first.

### Section B — Disagreements

Identify genuine disagreements (not just non-overlap):

- A defect one agent says is fragile that another implicitly says is
  fine.
- Conflicting blitz proposals that touch the same code.
- Different rankings of the same blitz item.

For each: who said what, who you think is right (and why — cite line
refs in the round-1 drafts), and what evidence would resolve it.

### Section C — Your slice-specific updates

For each of your original Agent-1 defects (`agent_1.md` §4) AND blitz
proposals (`agent_1.md` §5):

- Mark **CONFIRMED** (other agents implicitly or explicitly back it
  up), **SHARPENED** (the other drafts let you state it more precisely
  — give the new version), or **REVISED** (you'd change your stance
  after reading the others).
- If REVISED, say why.

This isn't an exhaustive re-paste — only items where round 2 changed
your view or added evidence.

### Section D — Cross-cutting blitz proposals

The user has limited bandwidth. Looking across all four drafts (~24
blitz proposals total, ranked within slice), pick **the top 5** that
you would execute first, ordered.

For each top-5 item:
- Title (and which agent(s) proposed it).
- Concrete acceptance criteria (when is it "done"? A failing test
  that now passes? A file that didn't exist now exists? A specific
  configure-time message?).
- The one defect (by agent/number) it most directly closes.
- Dependencies on the other top-5 items.
- Whether it requires a real Perlmutter run or can be done
  desk-side.

Lead with the highest-leverage item. The intent: the user opens this
section, picks item #1, and starts. No "it depends on which axis you
weight more" hedging — commit.

### Section E — Open questions for the user

A short list of **questions only the author can resolve**. Things
like:

- "Does the host site-packages dir bind-mount override the container
  JAX, and if so what version actually runs?"
- "Has LORRAX been tested on any non-NERSC cluster?"
- "Which of the two GPU allocators is actually active in production?"

These are gates for blitz items — without an answer, a blitz can't
proceed. Be specific and tight.

---

## Constraints

- **You MAY now read `agent_2.md`, `agent_3.md`, `agent_4.md`.** Doing
  so is the whole point of round 2.
- Still read-only on `sources/lorrax_C/`. No code edits.
- No compute. Web search remains authorized for new questions that
  arise from comparing drafts.
- Write to `reports/lorrax_install_maintain_blitz_2026-05-13/round2_agent_1.md`
  only.
- Stay in your tmux pane.
- Stop when written. Print: `Agent 1 round 2 done — see round2_agent_1.md`.

### Output bias

You are Agent 1 (build/FFI specialist). Round 2 expects your
slice-specific deep knowledge to inform your judgments on Agent 2/3/4's
work, but the **synthesis output is cross-slice**. If two agents
disagree on something in your area of expertise, your tie-breaking
matters; if two agents disagree on something outside your slice, you
can flag it without resolving.

Start by reading the other three drafts.
