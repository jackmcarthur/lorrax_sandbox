# Checkpoint Skill

A checkpoint captures the current state of LORRAX development and sandbox testing
into durable artifacts that any future agent can read without conversation context.

## When to checkpoint

A major checkpoint is required when any of these occur:
- You complete a top-to-bottom code feature (new module, new interface, bug fix)
- You accumulate ~5 incremental commits without a report update
- You produce a meaningful comparison result (new pass/fail, table, or plot)
- You are about to end a session

## Procedure

Execute these steps in order. Skip steps that don't apply (e.g. skip git steps if
you haven't modified LORRAX source). See AGENTS.md "Git discipline" for branching
requirements — you must be on an `agent/<initiative>` branch before reaching this point.

### 1. Run the test suite (if source was modified)

```bash
cd /home/jackm/projects/lorrax && uv run python -m pytest -q
```

If tests fail, fix them before proceeding. If you cannot fix them, note the failure
in the report and CHANGELOG and do NOT commit broken code.

### 2. Commit to the feature branch (if source was modified)

You should already be on an `agent/<initiative>` branch (required by AGENTS.md).
Stage only the files you changed and write a descriptive commit message.

```bash
cd /home/jackm/projects/lorrax
git add <changed files>
git commit -m "Concise description of what changed and why"
```

Do not push without explicit instruction. Do not commit to `main`.

### 3. Update or create the report

Reports live in `reports/<initiative_YYYY-MM-DD>/report.md`. A report should be
readable by someone with zero conversation context. Include:

- **Summary**: 2-3 sentences on what was done and the key result.
- **Code changes**: Table of modified files and what changed.
- **Results**: Comparison tables with numbers. Use the parsers from `PARSE_OUTPUTS.md`.
- **Plots**: Embed with `![description](filename.png)`. Generate with matplotlib
  using `MPLBACKEND=Agg`. Not every checkpoint needs a plot — only when there are
  meaningful outputs to compare.
- **Status checklist**: What's done, what's next.
- **Open questions**: Anything unresolved that the next agent should know.

If a report already exists for this initiative, update it in place rather than
creating a new one.

### 4. Update CHANGELOG.md

Add a concise entry under the current date section. Focus on results and decisions,
not implementation details. The report has the details; the CHANGELOG is an index.

### 5. Update manifest.yaml

If any run directory step states changed during this work (e.g. a run went from
`pending` to `complete` or `failed`), update the manifest.

## Quick reference

```
pytest → commit → report → CHANGELOG → manifest
```

That's it. Five steps, in order, skip what doesn't apply.
