You are **Agent 3 of 4** on the LORRAX docs paring team.

**Read first:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_docs_pare_2026-05-22/CONTEXT.md`

## Your slice: **Plans, work-in-progress, status, drift**

The misc-but-accumulating layer:

| Doc | Lines | Notes |
|---|---:|---|
| `docs/index.md` | 58 | Stale paths (`src/isdf/...`) + broken `examples/` link per round-1 of the install-blitz. |
| `docs/AGENT_TODO.md` | 292 | Header says contents are "NOT user priorities" — self-disowned. Why is it still in main? |
| `docs/PLAN_zeta_g_flat_migration.md` | 373 | Migration plan. Migration likely landed in the May refit; verify. |
| `docs/FREQ_INTEGRATION_REWRITE_PLAN.md` | 651 | Rewrite plan. Done? In progress? |
| `docs/FREQ_INTEGRATION_PROGRESS.md` | 71 | Companion progress doc. |
| `docs/SIGMA_FREQ_AUDIT_STATUS.md` | 121 | Audit status. Done? |
| `docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` | 459 | "_REVISED" suffix is a tell — where's the un-revised version, did it get archived? |
| `docs/NEW_WINDOW_MINIMAX_GUIDELINES.md` | 478 | Guidelines. Still applicable? |
| `docs/PROFILING_SUGGESTIONS.md` | 342 | Suggestions. Acted on? |
| `docs/plans/phdf5_cray_mpich_migration.md` | 197 | Migration plan for phdf5. Per install-blitz consensus this was done (`project_phdf5_mpich_default` memory says 2026-04-20 default stack landed). |
| `docs/plans/unified_slab_io.md` | 291 | Migration plan. Done? |
| `docs/archive/` (8 files, ~1640 lines) | varies | Already archived. Verify nothing has un-archived itself by being referenced from a current doc. |
| `docs/advanced/` (dir) | ? | Inspect. May be old Docker-based setup superseded by Shifter (per install-blitz round-1). |

Total: ~3300+ lines in your slice. **This is the highest-pare-down
slice — most of it is finished work or status docs that should be in
git history, not the doc corpus.**

## What to look for

### A. "Plan" docs whose work is done

For each `PLAN_*.md`, `*_REWRITE_PLAN.md`, `plans/*.md`:
- `git log --oneline -- src/<area>` to see if the planned changes
  landed. The git-log dates relative to the plan's modification time
  are a strong signal.
- The integration branch (`agent/install-blitz-integration` on
  `sources/lorrax_D`) is the canonical-current state.
- For phdf5_cray_mpich_migration and unified_slab_io: the user's
  memory has these as landed (2026-04-20 phdf5 default; unified
  slab I/O was done in early May per zeta-rchunk reports).

Action: if the plan's work has landed → **ARCHIVE** (move to
`docs/archive/`). If half-landed → keep with a status header. If
genuinely pending → keep but consolidate into a single
`MAINTENANCE_TODO.md` at the repo root (the install-blitz consensus
already proposed this).

### B. AGENT_TODO.md — should it exist?

It explicitly disowns its contents. Either:
- Delete entirely (its contents are also in commit history if needed).
- Re-promote it to "actively-maintained backlog" with a clear scope.
- Archive it.

Round-1 of the install-blitz audit said this doc is half-alive — the
worst state. Pick one extreme.

### C. "_REVISED" docs

`GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` has the "_REVISED" suffix.
Where's the un-revised version? If in archive, fine. If not, the
suffix is just noise — the doc IS the guide. Recommend renaming.

### D. docs/index.md repair

Per round-1 of the install-blitz, this doc has stale `src/isdf/...`
paths and references `examples/` which doesn't exist. Decide:
- Repair in place (still useful as a router).
- Delete (it's only 58 lines and AGENTS.md / README.md already
  provide indices).
- Replace with a new `docs/_index.md` that's a proper router (the
  install-blitz proposed this).

### E. docs/advanced/

Inspect. If it's the old Docker-based pre-Shifter setup, archive or
delete — Shifter is canonical today; the old setup is preserved in
git history.

### F. SIGMA_FREQ_AUDIT_STATUS.md / FREQ_INTEGRATION_PROGRESS.md

These are status / progress docs. Their content has a half-life
measured in weeks. If the audit / progress is now stable / finished,
archive them. If active, recommend a single
`docs/PROJECT_STATUS.md` that consolidates *current* status (and is
expected to be edited every week or two), so individual status docs
don't accumulate.

### G. Web search latitude

Compare to how exemplar scientific-Python projects handle
"plans/in-progress" docs: do they live in the repo? In a wiki? In
GitHub Discussions? On the LORRAX scale (one author, no upstream
documentation site yet), what's the right answer? Cite URLs if
relevant.

## Output

Write to:
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_docs_pare_2026-05-22/agent_3.md`

Per CONTEXT §"Output structure". Be especially aggressive about
DELETE / ARCHIVE recommendations — this slice has the most "weight"
to shed.

## Constraints (from CONTEXT)

- Read-only on `sources/lorrax_D/`.
- No compute.
- Stay in your own report file. Do NOT read other agents' drafts.
- Stop when written. Print: `Agent 3 done — see agent_3.md`.

Start by `git log`-ing the plan docs against current code, then
verdict each.
