# Shared context — professionalize LORRAX's environment / build / dependency docs

## Goal
Make **free-standing LORRAX** something another researcher can build and run **on a
different cluster with different dependencies** (other MPI, other CUDA, no Shifter,
maybe Apptainer/Singularity or bare-metal), following docs of the quality and tone you'd
expect from a Google-grade OSS project like **JAX** or a well-maintained quantum-chem
package (PySCF, ASE, QuantumESPRESSO, GPAW). The current docs are written *"For AI
agents"* and assume Jack's NERSC Perlmutter account — that is the core thing to fix.

Two tiers of output are wanted:
1. **Small-scale, immediately-actionable** fixes to the current environment-related docs.
2. **Big-picture** ideas, explicitly invited: restructuring the docs overall, cutting or
   rewriting whole sections, and — most valuable — **redesigning how dependencies are
   declared/wired in LORRAX to remove antipatterns** (see below).

## The repo under review
`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D` (live checkout, branch
`agent/dep-home-migration`). This is the LORRAX **source package**, NOT the sandbox.
Key surfaces:
- `README.md`, `AGENTS.md`, `docs/index.md`
- `docs/ENVIRONMENT_COMPREHENSIVE.md` (the env/install/deps doc — has a "Generic SLURM
  clusters" §7 and an "FFI stack" §5), `docs/CODEBASE_COMPREHENSIVE.md`
- `config/` — `perlmutter/install.sh`, `perlmutter/site_config.sh`, `config/README.md`,
  and the Lmod modulefiles that define `module load lorrax` / `lxrun` / `lxpre`
- `pyproject.toml` (declares Python deps + a `build` group: scikit-build-core, cmake,
  ninja, pybind11, nanobind — but the native FFI libs are NOT packaged there)
- The FFI build/run system: `src/ffi/common/cpp/{build.sh,run_shifter.sh,select_gpu.sh,
  in_container.sh}`, `src/ffi/{phdf5,slate}/scripts/stage_*.sh`
- `docs/plans/phdf5_cray_mpich_migration.md`, `docs/advanced/jax_multihost.md`
- Lots of `docs/*_COMPREHENSIVE.md` / `*_PLAN.md` / `*_AUDIT_STATUS.md` / `*_PROGRESS.md`
  — dev notes intermixed with user docs.

## Validated facts from THIS session (build on these; do not re-derive)
- The three FFI runtime deps (NVHPC cuSolverMp, parallel HDF5, Cray SLATE) are **prebuilt
  staged trees**, bind-mounted into a Shifter container at stable targets `/lorrax_nvhpc`,
  `/lorrax_phdf5`, `/lorrax_slate`. They were just **relocated from `$SCRATCH` to
  `$HOME/software`** (scratch is purged). Paths are resolved via `$LORRAX_FFI_{NVHPC,PHDF5,
  SLATE}_DIR` defaults in `config/perlmutter/site_config.sh`.
- Per-rank GPU binding is done by `src/ffi/common/cpp/select_gpu.sh` (sets
  `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`; required by SLATE/cuSolverMp's 1-device-per-process
  model). `in_container.sh` re-asserts `MPICH_GPU_SUPPORT_ENABLED=1`. Every srun is
  wrapped `$SEL $SHIFTER $INC python3 -m ...`.
- **A fresh `git clone` does NOT contain `liblorrax_ffi.so`** (gitignored build artifact);
  a newcomer hits `FileNotFoundError … Build with: bash src/ffi/common/cpp/build.sh`. The
  FFI build step is the single biggest from-scratch onboarding cliff.
- Known inconsistencies to verify and flag: `pyproject.toml` pins `jax[cuda13]>=0.9.0` but
  the production Shifter image runs JAX ~0.5.3 (image `jax:25.04-py3`); README references
  `gw_isdf/gw_jax.py` while `[project.scripts]` uses `gw.gw_jax:main` (`gw_isdf` vs `gw`);
  docs are addressed "For AI agents" rather than users.

## The professionalism bar (what "good" looks like)
Compare against JAX / PySCF / ASE docs: a clear landing page; an **Installation** page with
a support matrix (OS / CUDA / MPI / cluster) and copy-paste paths for pip/uv, container,
and from-source; a **Quickstart**; a **User Guide**; **API reference** (mkdocs-material is
already a dep); **Architecture/internals**; **Contributing**; a separation of *user docs*
from *developer/agent notes*; consistent admonitions, versioned commands, and no
person-specific or site-specific absolute paths in user-facing docs.

## Output contract (per analysis agent)
Be concrete: cite `file:line` or `file §section`. For every finding give (current state →
problem → proposed fix). Separate "small immediate doc fix" from "structural/architectural
proposal". **READ + PROPOSE ONLY — do not edit any file in the lorrax repo.** End with the
single highest-leverage change you'd make in your lens.
