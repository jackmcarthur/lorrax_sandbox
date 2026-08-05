You are **Agent 4 of 4** in an independent multi-agent audit of LORRAX's
install/maintain surface.

Three other agents (Agents 1, 2, 3) are working the same overall task in
parallel tmux panes right now. You cannot see their work and they cannot
see yours. After all four of you finish, the orchestrator will collect
drafts and run a discussion round.

**Your shared briefing is at:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/CONTEXT.md`

**Read it first.** It has the overall task, code/path map, NERSC-isms
checklist, constraints, and the suggested 6-section output structure.

---

## Your slice: **Docs, onboarding, runtime config surface, CI/test gaps**

You own the audit of what a *second user* encounters when they try to
install + run LORRAX, plus the long-term maintainability story for the
author. The deliverable answers two paired questions:

  *"If a competent new user — a PhD student in a peer group — was
  handed the LORRAX repo and pointed at the docs, with no synchronous
  help from the author, how far would they get? Where would they
  stall? What would they have to give up on?"*

and

  *"For the author maintaining LORRAX over the next year, what hidden
  maintenance cost is being paid right now — undocumented invariants,
  parallel implementations of the same thing, drift between docs and
  code, missing regression tests — and what blitz items would buy back
  the most time?"*

You're the **synthesis** agent on this team: less code-deep than
Agents 1/2/3, but expected to look broadly across docs, CI, the
`cohsex.in` config surface that's been growing, and the relationship
between the canonical checkout and the sandbox.

Tag every finding in your defect catalog with **[FRAGILE]**,
**[COMPAT]**, or **[LOC-COST]** per CONTEXT.md §6. Your slice tends to
be **[FRAGILE]-heavy on doc/code drift** and **[LOC-COST]-heavy on
parallel implementations / sandbox-vs-upstream redundancy**. Pure
[COMPAT] items in your slice are rarer — they're mostly downstream of
the build/MPI/runtime decisions Agents 1/2/3 are auditing — but watch
for cohsex.in flags whose defaults are NERSC-shaped.

**Web search is encouraged.** Use `WebSearch` / `WebFetch` for things
like: what does an exemplary modern HPC-Python package's docs look
like (e.g., Cython-built / nanobind-built scientific packages with
multi-cluster install stories)? `jax.experimental` API stability
warnings; conventions for `pyproject.toml`-driven CMake builds
(scikit-build-core best practices); standards-of-the-art for input-file
schema (Pydantic, dataclasses-json, configparser pitfalls). Cite URLs
when you reference an external benchmark or standard.

### Primary read targets

1. `README.md` — the entry point. Read as if you've never seen LORRAX
   before. Where does it leave you?
2. `docs/CODEBASE_COMPREHENSIVE.md`, `docs/ENVIRONMENT_COMPREHENSIVE.md`,
   `docs/PHYSICS_COMPREHENSIVE.md`, `docs/MEMORY_MODEL.md` — the main
   docs (the "_COMPREHENSIVE" naming pattern suggests they're meant to
   be the single source of truth for each area).
3. `src/ffi/PORTING.md` — the FFI porting doc. (Agent 1 is auditing
   this from the build angle; you're auditing it from the
   docs-coherency angle.)
4. `config/README.md` — modulefile/launcher docs.
5. `/pscratch/sd/j/jackm/lorrax_sandbox/docs/docs_gwjax/COHSEX_INPUT.md`
   — the cohsex.in reference, which is now the primary runtime-config
   surface after commits `488e870` / `9fe5fde` / `40a4cca` migrated
   chunk sizes + algorithmic toggles out of env vars. **Is this doc
   in sync with the parser?** (Find the parser at
   `src/gw/gw_config.py` or similar; grep for `cohsex` in `src/`.)
6. `templates/cohsex.in` (sandbox) — the canonical input template.
7. `pyproject.toml` and any `tests/` tree — what's actually tested?
   What's not? Run `find tests -type f -name '*.py' | head -30` to get
   a sense.
8. The sandbox `AGENTS.md` and `KNOWN_SANDBOX_ERRORS.md` —
   sandbox-level conventions and known issues that point to fragile
   spots.
9. `git log --oneline --stat -20` — recent activity pattern. What's
   actively changing? What hasn't been touched in a while (potential
   bit-rot)?

### Specific questions you must answer

- **Onboarding walkthrough.** Pretend you're a new user. Trace
  literally what they would read, in what order, from the front page.
  Where does the trail break? Where do paths diverge into
  "Perlmutter" / "your-own-cluster" without enough guidance?
- **Doc-vs-code coherency.** Spot-check: does
  `ENVIRONMENT_COMPREHENSIVE.md` mention every env var the modulefile
  sets? Does `COHSEX_INPUT.md` list every flag the parser accepts?
  Does `MEMORY_MODEL.md` match the actual chunker code (note: a
  parallel audit team is finding this drift — see
  `reports/zeta_rchunk_memory_model_2026-05-13/`)?
- **The `_COMPREHENSIVE` problem.** Four ~500-line docs named
  `*_COMPREHENSIVE.md` is a *lot* for a new user to ingest before
  doing anything. Is the layering right? Should there be a
  `QUICKSTART.md`? A `FAQ.md`?
- **Parallel implementations / dead code.** Spot any places where two
  modules do the same thing (e.g., `gflat_memory_model.py` vs.
  `aot_memory_model/` per the zeta-rchunk team's findings). Each
  parallel implementation is a maintenance tax — list every one you
  spot.
- **Test coverage gaps.** What does CI run? What doesn't it cover that
  it should? Specifically: is there a smoke test that catches a
  broken Shifter image? A broken FFI build? A broken jax.distributed
  init? An ABI shift in a vendor dep?
- **Sandbox vs. upstream split.** A lot of useful tooling lives in the
  sandbox (`/pscratch/sd/j/jackm/lorrax_sandbox/`) — the agent
  overlay, the lx-* scripts, the reports. What of this should
  graduate upstream? What is intentionally sandbox-only?
- **`cohsex.in` as the new config surface.** The recent migration
  (commits `488e870` / `9fe5fde` / `40a4cca`) puts more runtime
  knobs in `cohsex.in`. Is the input-file format keeping up — schema,
  validation, sensible defaults, error messages on typos? Is there a
  schema doc the parser can be tested against?
- **Doc bit-rot risk.** Which docs are most likely to drift out of
  sync with the code? `git log -- docs/` will tell you which haven't
  been touched.

### Blitz proposals — be concrete

For each proposal: file(s) it touches, what changes, what test or CI
check locks it in. ~1-day scope. Examples (not prescriptive):

- A `QUICKSTART.md` (≤ 100 lines) covering the "I just want a
  9-band MoS2 GW calc on Perlmutter" path.
- A `cohsex.in` schema (e.g., a Pydantic model or a simple JSON
  schema) that the parser uses, with a generated `COHSEX_INPUT.md` so
  doc and code can't drift.
- A `tests/integration/test_smoke_cohsex.py` that runs a tiny
  end-to-end calc in CI and catches catastrophic regressions
  (modulefile broken, FFI broken, parser broken).
- Consolidate `gflat_memory_model.py` + `aot_memory_model/` into one
  model (the zeta-rchunk team is converging on this — your blitz
  could be to *act* on their report).
- A `MAINTENANCE_TODO.md` at the repo root that tracks the slow-burn
  defects (sentinel files, undocumented invariants, ABI fragility
  callouts) so future-author can chip away at them.
- A `docs/_index.md` that explicitly says "if you're a new user, read
  X then Y then Z; if you're porting, read PORTING.md; if you're
  hacking on physics, read PHYSICS_COMPREHENSIVE.md."

### Your assigned output file

`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/agent_4.md`

Write the complete audit there. Cover the six sections from CONTEXT.md
§6 (scope, current state, NERSC-isms, defect catalog, blitz proposals,
open questions).

### Hard constraints (from CONTEXT.md §5, restated)

- Read-only on `sources/lorrax_C/`. No code edits.
- No compute (no `srun`, `lxrun`, `python` that runs JAX). Desk research
  only.
- Do **not** read `agent_1.md`, `agent_2.md`, or `agent_3.md` in your
  report dir. Those belong to the other agents.
- Stay in this tmux pane.
- Stop when your file is written. Print a one-line summary:
  `Agent 4 done — see agent_4.md`.

Start by reading CONTEXT.md, then the top-level `README.md`, then walk
through the docs from the new-user perspective.
