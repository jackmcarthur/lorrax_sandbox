You are **Agent 2 of 4** in an independent multi-agent audit of LORRAX's
install/maintain surface.

Three other agents (Agents 1, 3, 4) are working the same overall task in
parallel tmux panes right now. You cannot see their work and they cannot
see yours. After all four of you finish, the orchestrator will collect
drafts and run a discussion round.

**Your shared briefing is at:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/CONTEXT.md`

**Read it first.** It has the overall task, code/path map, NERSC-isms
checklist, constraints, and the suggested 6-section output structure.

---

## Your slice: **Cray MPI + Shifter container surface**

You own the audit of how LORRAX gets *containerized and MPI-linked* at
runtime. The deliverable answers: *"What does it take to run LORRAX
inside a different container runtime (Apptainer / Singularity / Enroot /
plain Docker) with a different MPI implementation (OpenMPI, MPICH, or a
non-Cray MPICH) on a non-NERSC system, and where are the load-bearing
Cray/Shifter/Perlmutter assumptions?"*

Tag every finding in your defect catalog with **[FRAGILE]**,
**[COMPAT]**, or **[LOC-COST]** per CONTEXT.md §6. The MPI/container
slice will be **the most [COMPAT]-heavy** of the four agents — almost
everything in `0.1.0.lua` is Shifter/Cray-shaped, and you should
explicitly answer "on cluster X with Y, what breaks?" for each
NERSC-ism. Pick at least two concrete alternative clusters to reason
against (e.g., OLCF Frontier with Apptainer + Cray MPICH; ALCF Polaris
with NVIDIA HPC SDK MPI; a university Slurm cluster with Singularity +
OpenMPI).

**Web search is encouraged and expected.** Use `WebSearch` / `WebFetch`
liberally. Concrete prompts: Shifter vs. Apptainer feature parity
(MPI bind-mount, GPU passthrough, image registry); Cray MPICH GTL
(`libmpi_gtl_cuda`) — what's the OpenMPI equivalent (UCX-CUDA), the
MPICH-OFI equivalent? What's the actual ABI contract of
`/opt/udiImage/modules/mpich`? `--mpi=cray_shasta` vs. `--mpi=pmix`
vs. `--mpi=pmi2` — when does each apply? How does
`MPICH_GPU_SUPPORT_ENABLED` interact with non-Cray MPICH installs?
Cite URLs.

### Primary read targets

1. `config/modulefiles/lorrax/0.1.0.lua` (337 lines) — **the single
   Lua file that wires Shifter + Cray MPICH + bind-mounts + env vars +
   lxrun. Your single most important target.** Read every line. The
   shell functions (`lxrun`, `lxalloc`, `lxshell`, `lxpre`) are
   defined here, near the bottom.
2. `src/ffi/common/cpp/in_container.sh` — the script Shifter re-execs
   inside the container; re-asserts `MPICH_GPU_SUPPORT_ENABLED=1`.
3. `src/ffi/common/cpp/select_gpu.sh` — `SLURM_LOCALID` →
   `CUDA_VISIBLE_DEVICES` per rank.
4. `src/ffi/PORTING.md` §"Container/MPI" subsection — and any section
   mentioning Shifter / Cray MPICH / `libmpi_gtl_cuda` / `--mpi=cray_shasta`.
5. `config/perlmutter/site_config.sh` — Perlmutter-specific values.
6. `config/perlmutter/install.sh` — modulefile installation. What does
   it patch into the Lua template?
7. `src/ffi/phdf5/context.py`, `src/ffi/cusolvermp/context.py`,
   `src/ffi/slate/context.py` — how the FFI Python side obtains its MPI
   comm. Pay attention to whether they reach into `MPI_COMM_WORLD`
   directly or get a comm passed from runtime init.
8. `git log --oneline --all -- config/modulefiles/ src/ffi/common/cpp/in_container.sh` — recent activity.

### Specific questions you must answer

- **The bind-mount layout.** The modulefile bind-mounts three host
  roots into the Shifter container: NVHPC, parallel HDF5, SLATE.
  Document the exact paths on both sides (host → container), every
  env var that points to either side, and the assumption each makes.
- **`/opt/udiImage/modules/mpich`.** This is Shifter's MPICH ABI
  passthrough — Cray-specific. Trace every file that references it.
  What would have to replace it on Apptainer? On Enroot? On vanilla
  Docker?
- **`libmpi_gtl_cuda.so.0` LD_PRELOAD.** This is the Cray
  GPU-aware-MPI translation layer. List every file that LD_PRELOADs
  it. What's the OpenMPI equivalent (UCX-CUDA)? The MPICH-OFI
  equivalent? What breaks if this is missing or version-mismatched?
- **GPU-aware MPI assumptions.** Does LORRAX assume the device pointer
  can be passed straight to MPI (`MPICH_GPU_SUPPORT_ENABLED=1`)? Where
  are the buffers passed device→device? What's the host-staging
  fallback?
- **`--mpi=cray_shasta`.** Hardcoded in `lxrun`. Is this the only
  place? What's the portable alternative — `--mpi=pmi2`, `--mpi=pmix`,
  `srun` without `--mpi=`?
- **Image registry.** The container image
  (`nvcr.io/nvidia/jax:25.04-py3`?) — is it pinned anywhere? What
  installs (apt / pip) happen *inside* the image at build time vs. at
  module-load time?
- **Multi-node coordination.** When `lxrun` launches across N nodes,
  how is the JAX coordinator discovered? Is there an
  `srun --multi-prog`-style setup, or does it use `JAX_COORDINATOR_ADDRESS`?
  How does this interact with the `--immediate` flag in the agent
  overlay?
- **Slurm-specific assumptions.** `lxalloc`, `lxrun` shell out to
  `salloc`, `srun`. On a PBS or LSF cluster, what changes? On a
  cluster without job arrays?

### Blitz proposals — be concrete

For each proposal: file(s) it touches, what changes, what test or CI
check locks it in. ~1-day scope. Examples (not prescriptive):

- An `Apptainer.def` companion to the Shifter image so non-NERSC
  clusters can build the same container.
- A `config/generic/site_config.sh` template with the
  Perlmutter-specific knobs called out, plus a `make-site-config`
  helper.
- A `config/modulefiles/lorrax/portable.lua` variant that uses no
  Shifter — for clusters with Apptainer or bare-metal Python envs.
- An `lxrun --dry-run` that prints the full `srun` command with all
  env vars + bind-mounts, so a user can debug their cluster's
  variant.

### Your assigned output file

`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/agent_2.md`

Write the complete audit there. Cover the six sections from CONTEXT.md
§6 (scope, current state, NERSC-isms, defect catalog, blitz proposals,
open questions).

### Hard constraints (from CONTEXT.md §5, restated)

- Read-only on `sources/lorrax_C/`. No code edits.
- No compute (no `srun`, `lxrun`, `python` that runs JAX). Desk research
  only.
- Do **not** read `agent_1.md`, `agent_3.md`, or `agent_4.md` in your
  report dir. Those belong to the other agents.
- Stay in this tmux pane.
- Stop when your file is written. Print a one-line summary:
  `Agent 2 done — see agent_2.md`.

Start by reading CONTEXT.md, then the modulefile Lua, then PORTING.md.
