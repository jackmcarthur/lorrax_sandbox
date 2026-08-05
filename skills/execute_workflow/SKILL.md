# Executing LORRAX on Frontera

Rewritten 2026-07-31 for the Frontera CPU target. The repo
(`/work2/08271/jackmc/frontera/lorrax`) is the authority for the
environment; this skill is the operational glue. The Perlmutter/local-GPU
version of this skill is in `_archive/` and is not runnable.

## The certified launch path

Jobs run the 9-layer stack described in repo `docs/environment/overview.md`
(container `py312.sif` -> venv -> mpi4py overlay -> MPIwrapper thread
patch -> host FFI `.so` -> PMI2 glue -> source bundle -> node staging ->
sbatch template). Do not improvise a launch; copy the certified template.

1. Freeze the source. Jobs never read the live tree:

   ```bash
   cd /work2/08271/jackmc/frontera/lorrax
   config/frontera/build_cpu_runtime_bundle.sh     # prints the bundle path
   ```

2. Copy `config/frontera/templates/gw_dev.sbatch` next to the deck, point
   it at the bundle and the deck's `cohsex.in`. It sources
   `config/frontera/mpi_transport_env.sh` (collectives impl `mpi`,
   provider `mlx`, thread-main patch — do not hand-tune these; GATES.md).

3. `sbatch` it. Dev queue limits: 2 jobs, 40 nodes. Login-node rules
   (AGENTS.md rules 3-6) apply to everything you do while waiting:
   no srun/containers on login, RLIMIT_NPROC 300, python3 is 3.7.

4. Read results from disk (`.out`, `eqp*.dat`, `sigma_mnk.h5`) and
   append verdicts to `CLAIMS.md` with the jobid. Never predict a job's
   output. `.h5` inspection needs a small in-container job — login h5
   tools are HDF5 1.8 and cannot open 1.14 files.

Interactive checks that need jax but not scale: 1-node dev job, then
in-container `python3` with `XLA_FLAGS=--xla_force_host_platform_device_count=N`
(see `docs/HLO_HOWTO.md` and `fastloop/PLAN.md`).

## Decks

Current decks live under `/scratch2/08271/jackmc/`:
`mos2_4x4_test` (pinned baselines + active agent — treat read-only),
`lorrax_mos2_12x12`, `mos2_80ry_12x12`. A run directory needs a
`manifest.yaml` (`templates/manifest.yaml`) and follows the variant rules
in `skills/build_inputs/SKILL.md`.

Building a NEW deck requires the QE -> pw2bgw -> WFN.h5 leg. That leg was
last exercised on Perlmutter and is NOT certified on Frontera; treat the
first Frontera QE run as new work (evidence rules apply), not a recipe
replay. The input-file construction rules in `build_inputs` remain valid.

## Dependency chain and invalidation (unchanged)

`SCF -> NSCF -> pw2bgw -> epsilon -> sigma`, and `NSCF -> LORRAX`.
Each NSCF overwrites `<prefix>.save/`, so `pw2bgw.x` + `wfn2hdf.x` must
run immediately after each NSCF. SCF or k-grid/nbnd changes: new
top-level run. Input-file changes: new variant directory. Never mutate a
completed run. `dipole.h5` regenerates on any band-window change
(INVARIANTS row 3).
