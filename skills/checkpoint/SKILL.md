# Checkpoint Skill

A checkpoint turns session state into artifacts a future agent can use
without this conversation. Rewritten 2026-07-31 for Frontera; the
Perlmutter version (lxrun/pytest-on-GPU) is in `_archive/profiling_stack/`
history and no longer applies.

## When

- A top-to-bottom feature or fix is complete.
- ~5 incremental commits have accumulated.
- A comparison or measurement produced a verdict.
- The session is about to end.

## Procedure

1. Gates. On login (python 3.7), run the AST gate suites via their
   `__main__` runners in the repo:

   ```bash
   cd /work2/08271/jackmc/frontera/lorrax
   python3 tests/test_layering.py
   python3 tests/test_crossfile_requests.py
   python3 tests/test_env_registry.py
   ```

   Semantic verification needs a job (`skills/execute_workflow/SKILL.md`)
   or, once implemented, the fastloop mini-deck (`fastloop/PLAN.md`).
   A claim verified only by reasoning is not verified.

2. Commit. Feature branch in the repo, explicit pathspecs, no push
   (AGENTS.md rule 8). Work lands on the branch the same day it is
   validated.

3. Ledger. Append one row per new verdict to `CLAIMS.md` (jobid +
   artifact path). If a gate default or certified setting changed, update
   `GATES.md`; a new cross-cutting precondition goes to `INVARIANTS.md`.

4. Report only if needed. A table or plot that a ledger row cannot carry
   goes to `reports/<initiative_YYYY-MM-DD>/report.md`; otherwise skip.

The ledger row is the durable record; the commit is the durable code
change. Everything else is optional.
