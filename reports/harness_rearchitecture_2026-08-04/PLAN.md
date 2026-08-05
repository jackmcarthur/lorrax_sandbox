# The plan, in plain language

2026-08-05. This replaces TARGET_ARCHITECTURE.md and PRIORITIES.md (in
git history if needed). One page; the tree is the whole idea.

We develop LORRAX with AI agents on a SLURM cluster. Three recurring
problems: agents don't reliably follow the project's rules or reuse its
best code; checking a change requires slow batch jobs; and measurements
and decisions are recorded in prose that is getting too long to read.
The plan:

```
Make agent development of LORRAX fast and reliable
│
├── 1. One source of truth  (FIRST — everything else waits on it)
│   ├── Push the Frontera branch to GitHub. It is ~150 commits ahead;
│   │   every agent working off GitHub sees a stale, misleading repo.
│   ├── Merge (or at least push) the two certified side branches:
│   │   fix/slab-io-audit and fix/bispinor-zeta-reuse.
│   └── Rewrite the repo's AGENTS.md — it still describes Perlmutter,
│       which no longer exists.
│
├── 2. Rules: write each once, enforce with code, not prose
│   ├── A 2-page design doc (DESIGN_draft.md): the three code layers
│   │   (physics drivers / algorithms / plumbing), the O(N^3) rule
│   │   (never sum over valence-conduction pairs), and the JAX rules
│   │   (jit the outer loop, no oversized replicated arrays, one FFT
│   │   helper, declare shardings and verify them from the compiled HLO).
│   ├── Short per-directory notes (src/common/AGENTS.md etc.), each
│   │   pointing at the best existing code to copy. An audit showed the
│   │   good patterns in gw/ and psp/ don't spread because agents in
│   │   other directories never see them: some shared helpers have zero
│   │   or two callers while agents rebuilt worse versions by hand.
│   ├── A lint script for banned patterns (rules_gate.py — works today;
│   │   bans raw jnp.fft and device_put; existing violations are frozen
│   │   in an allowlist that can only shrink). Add ~5 more rules that
│   │   each correspond to a bug we actually had. Error messages name
│   │   the helper to use instead.
│   ├── Run those lints automatically after every file an agent edits
│   │   (a Claude Code hook), so the agent sees the error immediately
│   │   and fixes it in the same session instead of a later one.
│   └── Startup checks inside LORRAX itself for physics preconditions:
│       centroid window covers the sigma window, dipole.h5 matches the
│       band window, htransform gets all valence bands, band subsets
│       don't split degeneracies. Refuse with a message that says how
│       to fix it.
│
├── 3. Testing: cheapest test first, always
│   ├── Seconds (login node): syntax + lint + the AST gates, one command.
│   ├── Minutes (1 node): fastloop — already exists and works. Finish
│   │   its three open items: make it the required pre-commit check, add
│   │   a BSE stage, add the check that no stage gathers a full N_mu^2
│   │   array (scan the compiled HLO).
│   └── Hours (many nodes): full-size A/B runs, only after the cheap
│       rungs pass.
│
├── 4. Records: machines log the facts, humans write the verdicts
│   ├── The sbatch template appends one JSON line per job: jobid, git
│   │   hash, input deck, machine, timings per stage, peak memory, exit
│   │   code, output paths. Agents can forget to log; the template can't.
│   ├── CLAIMS.md rows shrink to one line each (claim, verdict, jobid);
│   │   details move to one small file per claim, including the key
│   │   numbers copied in when the job finishes — before scratch purges
│   │   them. (CLAIMS.md is currently ~30k tokens in 68 lines; it no
│   │   longer fits in a single read.)
│   └── This design was checked against how numpy/scipy, materials-
│       science provenance tools, and HPC centers do it (EVIDENCE_
│       DESIGN.md). Verdict: keep the human-written ledger, move the
│       machine-recordable facts out of it. No databases, no servers.
│
└── 5. Jobs: use the queue well, don't collide
    ├── Pack several test variants into one SLURM allocation — queue
    │   slots, not node-hours, are the scarce resource.
    └── One shared file listing in-flight jobs (who, what, which scratch
        dirs) so the ~3 concurrent agents don't step on each other.
```

## One process rule (Jack's)

Before any major design change, write down the answer to: **"what about
this would Jack most likely object to?"** and address those points before
building. Keep a running file of past objections and their resolutions
(TASTE.md) so the question gets easier to answer over time. This is now
in DESIGN_draft.md §5.

## Deliberately dropped

Cut because no actual incident justifies them yet (each can come back if
one occurs): injecting fake bugs to test whether agents find them; a cron
job re-running fastloop on a schedule; a test-fixture suite for the
sandbox's own scripts; adopting import-linter (the existing AST layering
gates already work); dashboards and statistics over the job records
(premature until the records exist); and three extra registry files that
duplicated existing ones.

## Order of work

1. The push and AGENTS.md rewrite (item 1). Nothing else matters until
   this is done.
2. Same week: CLAIMS split, the per-job JSON logging, the design doc,
   the one-command login check, fastloop-as-required-precommit.
3. Same month, attached to work that touches the area anyway: the edit
   hook, the extra lint rules, the startup checks, the per-directory
   notes, timing instrumentation for sigma^B (currently unmeasurable),
   the job-packing script, the in-flight-jobs file.
4. Later, when adjacent work makes them free: a template for new driver
   stages, promoting the async device-to-host accumulator and the kernel
   cache into common/, shape annotations (jaxtyping) on the main array
   functions.
```
