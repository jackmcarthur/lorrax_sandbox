You are **Agent 2 of 4** on the LORRAX docs paring team.

**Read first:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_docs_pare_2026-05-22/CONTEXT.md`

Especially the "What's already done by the integration branch" section
— that's the floor of what should be assumed-current.

## Your slice: **Install / environment / porting / overview docs**

| Doc | Lines | Notes |
|---|---:|---|
| `README.md` | 39 | Top-level overview. Quick start. Doc index. |
| `AGENTS.md` | 112 | Read-order, key documentation table, agent conventions, AGENTS-style code map. |
| `docs/CODEBASE_COMPREHENSIVE.md` | 641 | Module map, data flow, key classes, entry points. |
| `docs/ENVIRONMENT_COMPREHENSIVE.md` | 455 | Dependencies, installation, cluster usage, CUDA memory. Updated by Blitz #5 (LORRAX_NGPU=8 doc lie fix). |
| `config/README.md` | 269 | Modulefile + lxrun/lxalloc/lxshell/lxpre usage. Updated by Blitz #5 (multi-user concurrency section added). |
| `src/ffi/PORTING.md` | 225 | FFI porting reference. Updated by Blitz #5 (LORRAX_CONTAINER_*_PATH + multi-user callouts). |

Total: ~1700 lines. Your slice is the install/porting surface — what
a second user reads before running a single calculation.

## What to look for

### A. Post-blitz drift

Many statements in these docs are now stale because the integration
branch landed. Check at least:

- `docs/ENVIRONMENT_COMPREHENSIVE.md`: does it still claim "must run
  `bash build.sh`"? (Blitz #1 wired `pip install -e .`.) Does it
  document `XLA_PYTHON_CLIENT_ALLOCATOR=platform`? (Blitz #0 changed
  to `cuda_async`.) Does it claim `LORRAX_NNODES` doesn't work?
  (Blitz #5 wired it.) Does it document env vars that migrated /
  are about to migrate to `cohsex.in` (the deferred Blitz #2)?
- `src/ffi/PORTING.md`: does it still describe MPI selection as a
  two-source-of-truth problem? (Blitz #3 unified into
  `config/mpi_stacks/`.) Does the JAX-version table say
  `jax[cuda13]>=0.9.0`? (Blitz #1 loosened to `jax>=0.5`.)
- `config/README.md`: any references to "the base module is
  single-node only" that Blitz #5 should have updated?
- `README.md`: is the doc-index it lists still accurate? Does it
  point at the QUICKSTART that doesn't exist (Blitz #4-original
  proposed one)?
- `AGENTS.md`: is the "Key documentation" table aligned with what
  currently exists in `docs/`?

### B. Two indices of truth

`README.md` and `AGENTS.md` both list "key docs" but the lists don't
agree. Recommend: which one should canonicalize? Or should they
become a single `docs/_index.md` (one of the install-blitz
recommendations)?

### C. Layered redundancy: PORTING vs ENVIRONMENT vs config/README

These three docs all describe "how to install LORRAX on Perlmutter"
to varying degrees. Identify overlap. Recommend a layering:
- `README.md` → minimum-viable "what is this, how do I run on Perlmutter."
- `config/README.md` → modulefile + Perlmutter specifics.
- `PORTING.md` → non-Perlmutter target.
- `ENVIRONMENT_COMPREHENSIVE.md` → ???

Is `ENVIRONMENT_COMPREHENSIVE.md` redundant with the others? If yes,
recommend what to keep, what to merge into PORTING or config/README.

### D. CODEBASE_COMPREHENSIVE.md

641 lines of module map + data flow. Likely mostly accurate but check
for:
- Stale paths (the `src/isdf/` → `src/` rename per agent_4 round-1).
- References to env vars that no longer exist.
- References to functions / modules that got renamed in the May 17-19
  memory-model refit (e.g., `gflat_to_rchunk`, the new sphere-idx
  accessor canonicalization).

If most is current, KEEP with a list of spot-fixes. If it's >30%
drifted, recommend a more aggressive overhaul or auto-generation
strategy.

### E. Quickstart gap

There's no `docs/QUICKSTART.md` and no `examples/` dir. The
install-blitz consensus said a quickstart was deferred. Decide: is
the right artifact for a "first calculation on Perlmutter in 10
minutes" path *inside* one of the existing docs (`README.md`,
`config/README.md`), or a new file? Be concrete — show the proposed
outline / target line count.

### F. Web search latitude

Compare to exemplar scientific-Python install docs (JAX, scipy,
PyTorch's HPC-deployment docs, BerkeleyGW's install docs).
Cite URLs.

## Output

Write to:
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_docs_pare_2026-05-22/agent_2.md`

Cover the six sections from CONTEXT §"Output structure". Per-doc
verdict table is the headline.

## Constraints (from CONTEXT)

- Read-only on `sources/lorrax_D/`.
- No compute.
- Stay in your own report file. Do NOT read other agents' drafts.
- Stop when written. Print: `Agent 2 done — see agent_2.md`.

Start by reading the six docs in your slice.
