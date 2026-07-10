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

Run the full suite on GPUs (never the login node).  The standard
invocation is PLAIN — serial, 1 GPU, ~4 min (2026-07-09 suite redesign;
see `reports/test_suite_redesign_2026-07-09/`):

```bash
cd sources/lorrax_<X> && LORRAX_NGPU=1 lxrun python3 -m pytest -q tests
```

Optional fast path (4-GPU parallel via pytest-xdist; never required):
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

Note: under xdist the session-scoped e2e state (gnppm/bispinor) rebuilds
per worker — correct but partially redundant; serial is the contract.

For quick mid-series iteration only (NOT a checkpoint gate):
`-m "not regression"` runs the unit tests (~1 min) without the e2e
gates.  A checkpoint always runs the full suite.

**Golden gates — these MUST pass.** A checkpoint is not valid unless all four are green:

- `tests/test_gw_jax_regression.py::test_gw_jax_matches_reference[cohsex]`
- `tests/test_gw_jax_regression.py::test_gw_jax_matches_reference[si_cohsex_3d]`
- `tests/test_gw_jax_regression.py::test_gnppm_matches_reference`
- `tests/test_invariance_gates.py::test_ibz_equals_full_bz`

A clean run reports **0 failed** (a handful of `deselected` is normal —
the `extra`-marked tooling suites are excluded by pyproject addopts; run
them with `-m extra` when working on those tools). Any failure — or a
skip of a golden gate — is real: fix it before proceeding, or note it in
the report and CHANGELOG and do NOT commit broken code.

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
