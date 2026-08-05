You are **Agent 3 of 4** in an independent multi-agent audit of LORRAX's
install/maintain surface.

Three other agents (Agents 1, 2, 4) are working the same overall task in
parallel tmux panes right now. You cannot see their work and they cannot
see yours. After all four of you finish, the orchestrator will collect
drafts and run a discussion round.

**Your shared briefing is at:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/CONTEXT.md`

**Read it first.** It has the overall task, code/path map, NERSC-isms
checklist, constraints, and the suggested 6-section output structure.

---

## Your slice: **Runtime — env vars, jax.distributed init, launchers**

You own the audit of what happens *between* `module load lorrax_C` and
the first jit running on rank 0 — i.e. the env-var surface, the JAX
distributed bootstrap, and the lxrun/lxalloc/lxshell/lxpre launcher
contract. The deliverable answers: *"What is the runtime contract
between LORRAX and its environment? Which env vars are required vs.
optional vs. silently consequential? What does the distributed-init
sequence assume about its caller? Where is the launcher surface
brittle, and where is it actually clean?"*

Tag every finding in your defect catalog with **[FRAGILE]**,
**[COMPAT]**, or **[LOC-COST]** per CONTEXT.md §6. The runtime slice
is **especially [FRAGILE]-heavy** — silent env-var consequences,
sentinel files, double-import / double-module-load failure modes — and
secondarily [LOC-COST] (the sandbox `lorrax_agent` overlay is ~hundreds
of lines of pool coordination that exists only because A/B/C/D parallel
agents share an allocation; ask whether any of that complexity should
live upstream or whether a simpler upstream story is possible).

**Web search is encouraged.** Use `WebSearch` / `WebFetch` for:
`jax.distributed.initialize()` API + version stability (esp. across
JAX 0.4 → 0.5 → current); `JAX_COMPILATION_CACHE_DIR` semantics
(multi-user safety, cross-host invalidation, stale-cache pitfalls);
`HDF5_USE_FILE_LOCKING=FALSE` data-corruption history on non-Lustre
filesystems; `XLA_PYTHON_CLIENT_ALLOCATOR=platform` /
`TF_GPU_ALLOCATOR=cuda_malloc_async` semantics & known bugs; SLURM
`SLURM_LOCALID` ↔ `CUDA_VISIBLE_DEVICES` patterns on Perlmutter vs.
elsewhere. Cite URLs.

### Primary read targets

1. `src/runtime/__init__.py` (214 lines) — **`init_jax_distributed()`,
   `set_default_env()`, `fallback_to_cpu_if_no_gpu_backend()`. Your
   most important target.** Trace what env vars it reads, what env vars
   it sets, what fails closed vs. fails open.
2. `config/modulefiles/lorrax/0.1.0.lua` lines that `setenv` /
   `pushenv` / export env vars (there are many). Make a complete
   table.
3. `config/modulefiles/lorrax/0.1.0.lua` lines that define the shell
   functions `lxrun`, `lxalloc`, `lxshell`, `lxpre` (near bottom).
4. `/pscratch/sd/j/jackm/lorrax_sandbox/modulefiles/lorrax_agent/1.0.lua`
   — the sandbox-level pool-aware overlay. **Note this is sandbox-only
   tooling, not part of upstream LORRAX** — its existence is itself a
   data point about what's missing from upstream.
5. `/pscratch/sd/j/jackm/lorrax_sandbox/modulefiles/lorrax_agent/lx_pool.py`
   — pool coordination logic (free-node search, lxstatus / lxattach /
   lxreap).
6. Recent env-var-cleanup commits (in `git log --oneline -- src/gw/gw_config.py`):
   - `488e870` — three buffer-affecting chunk sizes (`r_chunk_size`,
     `gflat_chunk_size`, `psig_k_chunk_size`) moved from env vars to
     `cohsex.in`.
   - `9fe5fde` — four algorithmic toggles moved from env vars to
     `cohsex.in`.
   - `40a4cca` — dropped `LORRAX_GSPACE_MODE` env override.
   Look at the diffs to understand what *used* to be env-controlled.
7. **All remaining env-var reads in the source.** Run:
   `cd /global/homes/j/jackm/software/lorrax_C && grep -rnE "os\.(environ\.get|getenv|environ\[)" src/`
   and audit the full list.

### Specific questions you must answer

- **Env-var taxonomy.** Classify every env var LORRAX reads into:
  (a) **required** — code fails or misbehaves if missing,
  (b) **optional** — falls back to a sensible default,
  (c) **silently consequential** — changes behavior without logging.
  Note: the modulefile *sets* many of these, so they're effectively
  required-on-NERSC but absent on a fresh machine.
- **`init_jax_distributed()` contract.** What does it require from its
  caller? What does the sentinel `_LORRAX_JAX_DISTRIBUTED_DONE` guard
  against — re-import? double `module load`? Are there hidden failure
  modes (single-rank run with stale env, multi-rank with
  `JAX_PROCESS_INDEX` unset)?
- **Modulefile env-var inventory.** Complete table of every
  `setenv` / `pushenv` / `prepend_path` in `0.1.0.lua`. For each:
  what is it for, who reads it, what happens without it.
- **Cleanup commits — assessment.** Were `488e870` / `9fe5fde` /
  `40a4cca` the right calls? Are there *more* env vars that should
  similarly migrate to `cohsex.in`? Conversely, are there things
  currently in `cohsex.in` that arguably belong as env vars (e.g.,
  cluster-specific paths)?
- **Launcher contract.** `lxrun <cmd>` does what exactly? Trace the
  full `srun` command line it constructs. What does it assume about
  the caller's environment? (E.g., does it require `LORRAX_NGPU` to be
  set? Does it require to be run from inside `lxalloc`?)
- **Agent overlay vs. upstream.** The
  `/pscratch/sd/j/jackm/lorrax_sandbox/modulefiles/lorrax_agent/`
  overlay adds pool-aware lxrun + lxstatus + lxattach + lxreap. **Why
  isn't this upstream?** Should some of it be? What's sandbox-specific
  (multi-agent A/B/C/D coordination) vs. universally useful
  (lxstatus, lxattach)?
- **JAX compilation cache.** `JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache`
  is in the modulefile. Multi-user safety? Multi-host coherency? Stale
  invalidation when the LORRAX source changes?
- **`HDF5_USE_FILE_LOCKING=FALSE`.** Required on Lustre, but it's a
  silent data-corruption hazard on other filesystems. Documented
  anywhere?

### Blitz proposals — be concrete

For each proposal: file(s) it touches, what changes, what test or CI
check locks it in. ~1-day scope. Examples (not prescriptive):

- A `lorrax doctor` CLI command that audits env vars + JAX
  distributed state + GPU visibility and prints a green/yellow/red
  report.
- A unit test for `init_jax_distributed()` with mocked env vars
  covering single-rank / multi-rank / re-entry scenarios.
- Upstream the universally-useful parts of the
  `lorrax_agent` overlay (e.g., `lxstatus`) into
  `config/modulefiles/lorrax/0.1.0.lua`.
- An `ENV_VARS.md` reference doc that tables every env var with
  required/optional/consequential classification, defaults, and
  consumer.
- Continue the env-var → cohsex.in migration for any remaining
  env-controlled toggles you find.

### Your assigned output file

`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/agent_3.md`

Write the complete audit there. Cover the six sections from CONTEXT.md
§6 (scope, current state, NERSC-isms, defect catalog, blitz proposals,
open questions).

### Hard constraints (from CONTEXT.md §5, restated)

- Read-only on `sources/lorrax_C/`. No code edits.
- No compute (no `srun`, `lxrun`, `python` that runs JAX). Desk research
  only.
- Do **not** read `agent_1.md`, `agent_2.md`, or `agent_4.md` in your
  report dir. Those belong to the other agents.
- Stay in this tmux pane.
- Stop when your file is written. Print a one-line summary:
  `Agent 3 done — see agent_3.md`.

Start by reading CONTEXT.md, then `src/runtime/__init__.py`, then the
modulefile Lua.
