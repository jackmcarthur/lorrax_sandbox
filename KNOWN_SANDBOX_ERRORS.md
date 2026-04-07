# Known Sandbox Errors

This file tracks documentation errors, broken paths, incorrect instructions, and
other issues agents encounter while using this sandbox. It exists so that problems
are recorded immediately rather than rediscovered by the next agent.

**If you are an agent and you encounter an error caused by incorrect documentation,
a missing file, a wrong path, a stale reference, or any other sandbox infrastructure
problem: record it here immediately before continuing your work.** Use the format below.

Do not fix the documentation yourself unless the fix is trivial and obvious. Record
the issue here for human review. Fixing incorrect physics or code is your job; fixing
incorrect sandbox scaffolding is the human's job.

---

## Format

```
### YYYY-MM-DD: Short description
- **Where**: which file/doc had the error
- **What happened**: what you tried and what went wrong
- **Expected**: what should have happened
- **Workaround**: what you did instead (if anything)
```

---

## Issues

### ~~2026-04-04: kin_ion.h5 not mentioned in BUILD_INPUTS or run setup docs~~ FIXED
- **Fix**: Added `kin_ion_file = kin_ion.h5` documentation to `skills/build_inputs/SKILL.md` Step 6 (GWJAX input), describing the file, its shape, how to generate it, and its dependencies.

### ~~2026-04-05: centroid.kmeans_isdf does not accept `-i cohsex.in`~~ FIXED
- **Fix**: Removed `-i cohsex.in` from both invocations in `skills/execute_workflow/SKILL.md` (Perlmutter step 5a and local step 5). The module reads `WFN.h5` from the CWD; correct invocation is `python3 -m centroid.kmeans_isdf $N_CENTROIDS --no-plot --seed 42`.

### ~~2026-04-05: `uv run` from sandbox fails because editable LORRAX path points to missing `/pscratch` location~~ FIXED
- **Fix**: Changed `pyproject.toml` editable path from `"../lorrax"` to `"./sources/lorrax"`. Ran `uv sync` to rebuild the venv. LORRAX submodules (`gw`, `centroid`, `psp`) now import correctly via `uv run`.
