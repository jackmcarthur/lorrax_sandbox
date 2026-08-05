# Audit context — review all changes from the dep-migration + docs-overhaul efforts

Adversarial cleanliness/correctness audit of everything changed across two efforts this
session. You are reviewers: hunt for what is broken, sloppy, inconsistent, or half-done.
READ-ONLY — do not edit any repo.

## What changed, and where to see it

### A. LORRAX source repo: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`
Full change set is the 5 commits `main..agent/docs-tier1` (branch `agent/docs-tier1` is
checked out). See it with:
`git -C /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D diff main..agent/docs-tier1`
and per-commit with `git -C … show <sha>`. The commits:
- `dc5f7f7` deps: relocate FFI staging defaults `$SCRATCH` → `$HOME/software`
  (`config/perlmutter/site_config.sh`, `src/ffi/common/cpp/run_shifter.sh`).
- `5e8d970` docs: Tier-1 P0 + small doc fixes (README, ENVIRONMENT_COMPREHENSIVE,
  CODEBASE_COMPREHENSIVE, config/README, docs/index, etc.).
- `ca840ec` build: S1 packaging metadata — `pyproject.toml` JAX pin
  `jax[cuda13]>=0.9.0` → `jax[cuda12]>=0.5.3,<0.6`, mkdocs→`[project.optional-dependencies]
  docs` extra, new `[cuda12]` extra, dropped `pybind11`/`pdoc`/`pydoc-markdown`; stripped
  personal `jackm` paths from `run_shifter.sh`, `CMakeLists.txt`, `site_config.sh`.
- `fa9f98b` docs: Tier-2 scaffold — new `mkdocs.yml`; `git mv` of dev notes into out-of-nav
  `docs/dev/{plans,progress,notes,archive}`; KEEP refs promoted into `docs/theory/` +
  `docs/architecture/`; new `docs/installation/{index,ffi-native-libs,perlmutter}.md`;
  rewritten `docs/index.md` + `docs/quickstart.md`; CUT `pydoc-markdown.yml` +
  `tools/gen_api_docs.sh`.
- `ea1ea3c` build: regenerate `uv.lock` for the JAX 0.5.x / CUDA-12 pin (cu13→cu12 stack).
- `290de0a` chore: gitignore `site/`.

### B. Sandbox repo: `/pscratch/sd/j/jackm/lorrax_sandbox` (changes UNCOMMITTED)
Two clusters of edits (ignore other unrelated uncommitted files like *.dat):
- `skills/execute_workflow/SKILL.md` — the GWJAX Shifter prefix now bind-mounts the
  `$HOME/software` deps; added `$SEL`(`select_gpu.sh`)/`$INC`(`in_container.sh`) wrappers to
  all Step 5/6 srun commands; rewrote the "Never set CUDA_VISIBLE_DEVICES" pitfall.
  See: `git -C /pscratch/sd/j/jackm/lorrax_sandbox diff -- skills/execute_workflow/SKILL.md`
- The `runs/**` production scripts (`run*.sh`, `*.sbatch`) where `--volume` SOURCE paths
  were repathed `/pscratch/sd/j/jackm/lorrax_*` → `/global/homes/j/jackm/software/lorrax_*`.
  See: `git -C /pscratch/sd/j/jackm/lorrax_sandbox diff -- runs/` (focus on `.sh`/`.sbatch`).

## Intent of each change (judge whether it was achieved WITHOUT collateral damage)
- **Migration:** move the 4 FFI dep dirs (nvhpc cuSolverMp, phdf5_cray, phdf5_openmpi,
  slate_cray) off purged `$SCRATCH` to `$HOME/software`; only the `--volume` SOURCE and the
  `$LORRAX_FFI_*_DIR` defaults should change — container mount targets (`/lorrax_nvhpc`,
  `/lorrax_phdf5`, `/lorrax_slate`) and all `LD_*` paths must be UNCHANGED.
- **SKILL fixes:** make the documented GWJAX recipe actually runnable (per-rank GPU
  binding via select_gpu.sh; MPICH_GPU re-assert via in_container.sh).
- **Tier-1/2/S1 docs+packaging:** professionalize for a non-NERSC user — fix stale paths,
  de-"For AI agents", split user-docs from dev-notes, wire mkdocs, make `pip install`
  metadata coherent, remove personal/site paths from shipped files.

## VALIDATED FACTS — do NOT re-flag these as broken (they were tested this session)
- The relocated `$HOME/software` deps work: a 4-GPU `gw.gw_jax` calc ran end-to-end using
  the documented (fixed) SKILL recipe — cuSolverMp 0.7.2 drove the distributed Cholesky,
  `eqp0.dat` produced, no NCCL crash.
- `uv lock` resolves cleanly to `jax==0.5.3`; `uv sync --locked --dry-run` is consistent.
- `mkdocs build` (non-strict) exits 0 and renders all nav pages; `pytest --collect-only`
  collects 275 tests with no errors (in the repo `.venv`).
- nvhpc `0.7.0`/`0.8.0` were intentionally NOT copied to `$HOME` (LU-bug regression only;
  production uses 0.7.2 + 25.5). This is a known, accepted tradeoff — not a bug.

## Output contract (each audit agent)
Write your findings to `reports/audit_2026-06-29/audit_<lens>.md` and return them. Each
finding: **severity** (BLOCKER / SHOULD-FIX / NIT), location (`file:line` or commit), what
is wrong, and the concrete fix. Be adversarial and specific; verify before asserting (read
the actual lines, follow the actual links). Distinguish a real defect from a pre-existing
issue you merely surfaced. End with a one-line verdict for your lens.
