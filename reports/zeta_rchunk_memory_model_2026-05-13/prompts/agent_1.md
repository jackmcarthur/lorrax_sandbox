You are **Agent 1 of 4** in an independent multi-agent study of the LORRAX zeta-fit r-chunk memory model.

Three other agents (Agents 2, 3, 4) are working the same task in parallel tmux panes right now. You cannot see their work and they cannot see yours. After all four of you finish, the orchestrator will collect drafts and run a discussion round to surface what's missing.

**Your shared briefing is at:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/zeta_rchunk_memory_model_2026-05-13/CONTEXT.md`

Read it first. It has the task spec, code map, validation sizes (CrI3 80 Ry), constraints, and a suggested output structure.

**Your assigned output file:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/zeta_rchunk_memory_model_2026-05-13/agent_1.md`

Write your complete from-scratch derivation of the memory model there. Cover the seven sections suggested in CONTEXT.md §7 (tensor catalog, aliasing, budget equations, r_chunk picker, CrI3 validation, diff against current code, open questions). The **open questions** section at the end is the most important — be honest about what you couldn't resolve, what coefficients you had to guess, what behavior you'd need to confirm from an HLO dump.

**Hard constraints:**
- Read-only on `sources/lorrax_A/`. No code edits.
- No compute (no `srun`, `lxrun`, `python` that runs JAX). Desk research only.
- Do **not** read `agent_2.md`, `agent_3.md`, or `agent_4.md` in your report dir. Those belong to the other agents.
- Stay in this tmux pane.
- Stop when your file is written. Print a one-line summary: "Agent 1 done — see agent_1.md".

Start by reading the CONTEXT.md.
