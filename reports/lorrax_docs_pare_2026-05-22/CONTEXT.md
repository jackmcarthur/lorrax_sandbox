# Docs paring team — shared context

You are one of four agents auditing LORRAX's documentation for pare-down.
Each of you owns a slice; you do not see each other's drafts in round 1.

## The problem

LORRAX `docs/` plus top-level docs total **~10,800 lines** of Markdown.
Much of it is honest, but the corpus has accumulated:
- Multiple "_COMPREHENSIVE.md" docs whose sections have been superseded
  in-place (`docs/PHYSICS_COMPREHENSIVE.md` §11 supersedes §3-5 explicitly).
- Plans / progress / "REVISED" docs that may now be finished work
  (`docs/FREQ_INTEGRATION_REWRITE_PLAN.md`, `docs/PLAN_zeta_g_flat_migration.md`,
  `docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`, `docs/NEW_WINDOW_MINIMAX_GUIDELINES.md`).
- `docs/AGENT_TODO.md` (292 lines) that explicitly disowns its own contents
  (header says "NOT user priorities").
- `docs/index.md` (58 lines) with stale `src/isdf/...` paths from a
  rename that happened months ago; references a non-existent `examples/`
  directory.
- A canonical `COHSEX_INPUT.md` (411 lines) living in the **sandbox** at
  `/pscratch/sd/j/jackm/lorrax_sandbox/docs/docs_gwjax/COHSEX_INPUT.md`
  — not in the lorrax_C repo at all. A second user `git clone`-ing
  LORRAX doesn't get the cohsex.in reference.
- Two top-level doc indices (one in `AGENTS.md`, one in `README.md`)
  that don't agree.
- Recent (May 17-19) memory-model refit added two big new docs
  (`SYMMETRY_COMPREHENSIVE.md`, `ZETA_V_Q_ALGORITHMS.md`) plus 1030
  lines added to `MEMORY_MODEL.md`. These are excellent but may overlap
  with PHYSICS_COMPREHENSIVE.

## What "pare down" means

The goal is **reducing the corpus to what's actually current and useful**.
Concrete actions per doc/section, in your reports:

- **KEEP** — the section is accurate for current code and someone will
  read it.
- **MERGE INTO X** — the content belongs in another doc that already
  exists (or could exist); the current location is redundant.
- **ARCHIVE** — move to `docs/archive/` as historical record; not
  actively maintained but preserved for context.
- **DELETE** — wrong, stale, or never useful; can be removed without
  archiving (commit history preserves it).
- **REGENERATE** — content should be auto-generated from code, not
  hand-maintained (the canonical example is `COHSEX_INPUT.md` should
  come from `_cohsex_schema.py` per the install-blitz consensus, even
  though Blitz #2 was deferred).

## Read order

1. **This file** (CONTEXT.md).
2. **Your slice prompt** at `prompts/agent_<N>.md`.
3. **`/pscratch/sd/j/jackm/lorrax_sandbox/AGENTS.md`** — sandbox
   conventions.
4. **`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D` on branch
   `agent/install-blitz-integration`** — the integration head with
   all six recovered blitzes applied. Use this as the canonical
   current-state checkout. (HEAD: `3079a1f` blitz #4 fix.)
5. **`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/consensus.md`** — the recent install/maintain
   blitz consensus document, especially Decision 1 (allocator) and
   Decision 2 (Cray HPC canonical, Apptainer aspirational, AMD/ROCm
   future). The doc world should reflect the same decisions: Cray
   MPICH + Shifter today, Apptainer + AMD as planned, NOT
   "portability-is-not-a-goal".
6. The actual docs in your slice (see `prompts/agent_<N>.md`).

## Current state — what's already done by the integration branch

These shifts already landed on `agent/install-blitz-integration` and
the docs need to *reflect* them (or be consolidated with them):

- **Allocator unified to `cuda_async`** (Blitz #0). Docs that reference
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform` or `TF_GPU_ALLOCATOR=cuda_malloc_async`
  are now stale.
- **`pyproject.toml` declares `[build-system]=scikit-build-core`**
  (Blitz #1). Docs that say "`pip install -e .` doesn't build the FFI"
  or "must run `bash build.sh`" are stale.
- **JAX pin loosened to `jax>=0.5`** (Blitz #1). Docs claiming
  `jax[cuda13]>=0.9.0` is needed are stale.
- **`config/mpi_stacks/cray_mpich.{cmake,sh}` is the single MPI-stack
  source-of-truth** (Blitz #3). Docs describing the build-time and
  runtime MPI selection as two independent strings are stale.
- **`select_gpu.sh` has PBS/PMIx fallback chain** (Blitz #3). Docs
  claiming it's Slurm-only are wrong.
- **`runtime.init_jax_distributed()` post-init asserts
  `jax.process_count() == SLURM_NTASKS`** (Blitz #3). Docs describing
  the distributed-init failure modes can reference this.
- **Multi-node `lxrun` works via `LORRAX_NNODES` in the base modulefile**
  (Blitz #5). Previous doc lies (`ENVIRONMENT_COMPREHENSIVE.md:322`)
  are fixed.
- **`LORRAX_CONTAINER_{NVHPC,PHDF5,SLATE}_PATH` env vars parametrize
  bind-mount paths**; `INSTALL_RPATH` uses them so the `.so` is no
  longer pinned to `/lorrax_*` literals (Blitz #5). Docs about a
  future Apptainer/Frontier port can reference this mechanism.
- **CI smoke trio + GitHub Actions workflow** (Blitz #4). Docs
  describing "no CI exists" are stale.
- **15 dist-init duplicate copies in `src/common/*_test.py` collapsed
  to imports of `runtime.init_jax_distributed`**, with 6 SLATE
  exemptions (Blitz #4). Doc references to "every test file
  re-implements dist init" are stale.
- **`batched_potrf`/`batched_potrs` cuSolverMp workspace lives in the
  JAX pool via `ffi::ScratchAllocator`** (Blitz #6). Docs claiming
  cuSolverMp workspace lives in a separate cudaMalloc'd Ctx-owned
  buffer are stale.

## What's deferred / NOT done

The install-blitz **Blitz #2** (cohsex.in schema + autodoc + remaining
env-var migration) was deferred because the May 17-19 memory-model
refit churned every consuming file. This means:

- `docs/docs_gwjax/COHSEX_INPUT.md` (in sandbox) is still
  hand-maintained and probably drifted vs. the parser's `_DEFAULTS`.
- Some env vars (`LORRAX_SC_*`, some `LORRAX_V_Q_*`, `ISDF_*`) are
  still read from `os.environ` in code; docs may reference them
  either as env vars or (incorrectly) as cohsex.in keys.

Your recommendations should note where Blitz #2 would close gaps, but
shouldn't depend on it landing first.

## Constraints

- **Read-only on `sources/lorrax_D/`.** This is the canonical checkout
  for the audit. No edits.
- **No compute.** No `srun`, no `lxrun`, no Python that runs JAX.
- **Stay in your own report file.** Write to
  `reports/lorrax_docs_pare_2026-05-22/agent_<N>.md` only.
- **Do NOT read sibling drafts** (`agent_<M>.md` for M ≠ your N).
- **Web search is authorized** — useful for "what does an exemplary
  scientific-Python docs structure look like?" (Sphinx vs MkDocs,
  numpy / jax / scipy approaches). Cite URLs in your draft.

## Output structure

`agent_<N>.md` should cover roughly:

1. **Scope.** What slice you're auditing; what's out of scope.
2. **Per-doc verdict.** A table of every doc in your slice with a
   one-word action (KEEP / MERGE / ARCHIVE / DELETE / REGENERATE)
   and a one-sentence justification. Be exhaustive within your slice.
3. **Per-section verdicts where finer granularity matters.** Some
   docs (especially PHYSICS_COMPREHENSIVE, MEMORY_MODEL) have
   internal supersession patterns where you'd keep §X and archive
   §Y. Call those out.
4. **Cross-cutting recommendations.** Patterns you see that span your
   slice: doc-vs-code drift, missing canonical sources, deduplication
   opportunities, places where regeneration from code would be
   higher-leverage than maintaining prose.
5. **What you'd write fresh.** If your slice has clear gaps (no
   QUICKSTART; missing API reference for FFI; etc.), name them — with
   a target line count and an outline. Be ruthless about what's
   actually needed vs. what's "would be nice."
6. **Open questions.** Things you genuinely couldn't resolve. Most
   important: any place where "is this still relevant?" needs the
   author to answer.

## Tone

You're writing for the author of LORRAX (jackmcarthur@berkeley.edu)
who is maintaining the project alone and has limited bandwidth. The
goal is **less prose to maintain**, not more. A recommendation to
delete or archive is more valuable than one to "expand and clarify."
Don't pad your draft; terse + specific.
