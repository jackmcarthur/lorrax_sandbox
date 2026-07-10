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

Run the full suite on GPUs (never the login node).  **Fast path (4-GPU
parallel, ~3-7 min):** pytest-xdist distributes tests across workers and
`tests/conftest.py` pins each worker to its own GPU.  Do NOT use plain
`lxrun` for this — it couples srun task count to `LORRAX_NGPU`, so 4 GPUs
means 4 tasks each running the whole suite.  Use one srun task with 4
GPUs visible:

```bash
cd sources/lorrax_<X>
srun --jobid=$SLURM_JOBID --mpi=cray_shasta -N1 -n1 --gres=gpu:4 --overlap \
  --immediate=10 --job-name=lx-pytest $LORRAX_SHIFTER \
  src/ffi/common/cpp/in_container.sh env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python3 -m pytest -q -n 4 --dist load tests
```

Serial fallback (1 GPU, ~8-13 min — use if xdist is unavailable or the
parallel run shows worker-crosstalk suspicion):

```bash
cd sources/lorrax_<X> && LORRAX_NGPU=1 lxrun python3 -m pytest -q tests
```

For quick mid-series iteration only (NOT a checkpoint gate):
`-m "not regression"` runs the ~245 unit tests (~1-2 min) without the
13 e2e gates.  A checkpoint always runs the full suite.

**Golden gates — these MUST pass.** A checkpoint is not valid unless all three are green:

- `tests/test_gw_jax_regression.py::test_gw_jax_matches_reference[cohsex]`
- `tests/test_gw_jax_regression.py::test_gw_jax_matches_reference[gnppm]`
- `tests/test_gw_jax_regression.py::test_ibz_full_bz_equivalence`

A clean run reports **0 failed**. Four tests are expected to **skip**, not fail —
`tests/active/test_reshard_all_to_all.py` (one) and `tests/test_aot_memory.py`
(three GPU cuFFT tests) are conditionally skipped on the Shifter container's JAX
0.5.3 (pyproject pins ~0.9); their skipif conditions auto-clear once the env
matches, so they still catch regressions. Any other failure — or a skip of a
golden gate — is real: fix it before proceeding, or note it in the report and
CHANGELOG and do NOT commit broken code.

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
