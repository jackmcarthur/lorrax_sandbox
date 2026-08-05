# Shared context — migrate lorrax runtime deps from $SCRATCH to $HOME/software

**Goal of this campaign:** produce the complete, exact list of everything in the
sandbox that must change to relocate four runtime-dependency directories off
Perlmutter `$SCRATCH` (Lustre, purge-exposed) and onto `$HOME`-class storage,
without breaking any run. Three agents investigate in parallel, each a different
lens, each writes `agent_<N>.md` in this directory. The orchestrator synthesizes.

## Already-established facts (do NOT re-derive — build on these)

The sandbox is `/pscratch/sd/j/jackm/lorrax_sandbox` (a uv project, own `.venv`).
`/global/homes/j/jackm/scratchperl` is a symlink → `/pscratch/sd/j/jackm`, so paths
written either way resolve to the same scratch tree.

Four sibling dirs on `$SCRATCH` are **active runtime deps**, bind-mounted into NERSC
`shifter` containers by the run scripts (NOT installed into the `.venv`):

| Scratch dir | Size | Mounted in container as | Provides |
|---|---|---|---|
| `/pscratch/sd/j/jackm/lorrax_nvhpc/` | 422M | `/lorrax_nvhpc` | NVHPC CUDA math/comm libs (cuSolverMp etc). Versioned subdirs: `0.7.2_cuda12.9` (current June runs use this), `25.5_cuda12.9` (older), `0.7.0`, `0.8.0`. |
| `/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage` | 169M | `/lorrax_phdf5` | parallel HDF5 built vs Cray MPICH (current CrI3/VI3 runs). |
| `/pscratch/sd/j/jackm/lorrax_slate_cray/stage` | 56M | `/lorrax_slate` | Cray MPI GPU-transport (`libmpi_gtl_cuda.so.0`, LD_PRELOAD'd) + libsci. |
| `/pscratch/sd/j/jackm/lorrax_phdf5_openmpi/stage` | 12M | `/lorrax_phdf5` | parallel HDF5 built vs OpenMPI/hpcx (the `Si/C_vqsph_*` runs). |

Total ~660 MB — fits easily in the `$HOME` 40 GB quota.

**Proposed target:** `/global/homes/j/jackm/software/` (i.e. `$HOME/software/lorrax_nvhpc`,
`.../lorrax_phdf5_cray`, etc.), or `/global/common/software/<project>` if an allocation
exists. Note `$HOME` = `/global/homes/j/jackm`, and Shifter is known to NOT reliably
expand `$HOME` — it needs the literal `/global/homes/j/jackm` (or the `/global/u2/j/jackm`
equivalent that `execute_workflow/SKILL.md` already uses for PYTHONPATH).

**Known wrinkle to chase:** `skills/execute_workflow/SKILL.md` documents a *simpler*
shifter prefix (JAX image + `PYTHONPATH=/global/u2/j/jackm/software/lorrax/src:$SITE`,
no nvhpc/phdf5/slate mounts at all), while the actual `runs/**/run*.sh` scripts use a
*different, heavier* shifter invocation WITH the three bind mounts. Doc and reality
diverge — both surfaces may need updating, and the divergence itself is worth flagging.

## Where to look (starting pointers, not exhaustive)
- `runs/**/run*.sh`, `runs/**/run_one.sh`, `run_sweep_*.sh` — the `--volume`,
  `LD_LIBRARY_PATH`, `LD_PRELOAD`, `LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR` lines.
- `AGENTS.md`, `skills/*/SKILL.md`, `docs/`, `CHANGELOG.md`, `KNOWN_SANDBOX_ERRORS.md`,
  `modulefiles/`, `templates/`.
- `reports/lorrax_install_maintain_blitz_2026-05-13/` — documents how these dep dirs
  were originally built/staged (relevant to whether a plain copy reproduces them).

## Output contract (every agent)
Write `reports/home_migration_2026-06-24/agent_<N>.md`. Be exact: cite `file:line`.
For every change, give: (a) the current text, (b) the proposed replacement, (c) why.
End with an "Open questions / risks" section. Read-only on the repo — touch nothing
except your own `agent_<N>.md`.
