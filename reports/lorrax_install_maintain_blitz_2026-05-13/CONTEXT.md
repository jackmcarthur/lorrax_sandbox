# Shared context — LORRAX install/maintain blitz (4-agent study)

You are one of four agents working independently on the same problem. The
user-orchestrator has set this up so each of you produces an honest,
from-scratch audit, then we compare drafts and surface what's actually
brittle vs. what's just under-documented.

This file is your shared briefing. Read it once, then read what it points
at.

---

## 1. The problem in one paragraph

LORRAX (GWJAX) is a JAX-based GW package developed by a single author on
NERSC Perlmutter, with several FFI shared libraries that wrap distributed
GPU primitives: **cuSOLVERMp** (distributed Cholesky / LU / eigh),
**parallel HDF5** (wavefunction slabs), **SLATE** (Cholesky+Trsm), and
**cuBLASMp** (batched GEMM). The whole thing runs inside a NERSC Shifter
container that bind-mounts host NVHPC + Cray MPICH + SLATE installs, and
is driven by an Lmod modulefile (`lorrax/0.1.0.lua`) plus a sandbox-level
agent overlay (`lorrax_agent/1.0.lua`) that adds pool-aware launchers
(lxrun / lxalloc / lxshell / lxpre / lxstatus / lxattach / lxreap).
Recent commits (488e870, 9fe5fde, 40a4cca) have been **moving runtime
toggles out of env vars into `cohsex.in`** — narrowing the env-var
surface — which is the natural moment to ask: what would a *blitz* look
like to make LORRAX installable by a second user on a non-NERSC cluster
without the author's hand-holding, and maintainable by the author with
fewer one-off hacks?

Your deliverable: an honest audit of one slice of the install/maintain
surface, with specific defects identified, NERSC-isms called out
explicitly, and concrete blitz proposals (each scoped to ~1 day of work)
ranked by leverage. The reader is the author himself — assume deep
knowledge of LORRAX internals, JAX, GW, Slurm. Don't pad with
background.

---

## 2. Read order

1. **`/pscratch/sd/j/jackm/lorrax_sandbox/AGENTS.md`** — sandbox conventions (read-only on sources, no compute, etc.).
2. **`/global/homes/j/jackm/software/lorrax_C/README.md`** — top-level package overview, entry points, quick start. **(All four of you use lorrax_C as the canonical checkout.)**
3. **`/global/homes/j/jackm/software/lorrax_C/src/ffi/PORTING.md`** — the FFI porting doc (196 lines). This is the one document that already tries to answer "how do I install this elsewhere?" — and is therefore one of your primary critique targets.
4. **`/global/homes/j/jackm/software/lorrax_C/docs/ENVIRONMENT_COMPREHENSIVE.md`** (455 lines) — full environment doc; dependencies, CUDA memory, cluster usage.
5. **`/global/homes/j/jackm/software/lorrax_C/config/README.md`** — modulefile + site_config + lxrun/lxalloc/lxshell/lxpre usage.
6. Code paths in §3 below, scoped to your agent.

Your **assigned slice** is in `prompts/agent_<N>.md`. Read that *after*
this CONTEXT.

---

## 3. Code & path map

All paths below are relative to `/global/homes/j/jackm/software/lorrax_C/`
unless noted as sandbox-level (`/pscratch/sd/j/jackm/lorrax_sandbox/`).

### Build system / FFI source

- `src/ffi/PORTING.md` — porting guide (196 lines).
- `src/ffi/common/cpp/CMakeLists.txt` — unified FFI build; autodetects NVHPC, CUDA, MPI, HDF5, NCCL, SLATE.
- `src/ffi/common/cpp/build.sh` — build driver script (cmake + ninja).
- `src/ffi/common/cpp/select_gpu.sh` — SLURM_LOCALID → CUDA_VISIBLE_DEVICES per rank.
- `src/ffi/common/cpp/in_container.sh` — re-asserts `MPICH_GPU_SUPPORT_ENABLED=1` inside Shifter.
- `src/ffi/__init__.py` and `src/ffi/common/ffi_loader.py` — Python side: how `liblorrax_ffi.so` is dlopened and dispatched.
- FFI backends, each with its own `cpp/` and Python wrappers:
  - `src/ffi/cusolvermp/` — distributed cuSOLVERMp (Cholesky / LU / eigh); `scripts/stage_pypi.sh` stages NVHPC.
  - `src/ffi/phdf5/` — parallel HDF5 slab I/O; `ARCHITECTURE.md` (read-only) documents sync vs async callbacks.
  - `src/ffi/slate/` — SLATE Cholesky + Trsm; `README.md` + `scripts/` for staging.
  - `src/ffi/cublasmp/` — cuBLASMp batched GEMM (less actively used).
- `pyproject.toml` — scikit-build-core + cmake + nanobind; defines entry points (`lorrax-gw`, `gw_jax`, `lorrax-centroids`, `lorrax-bse`).

### Container / MPI / modulefile

- `config/modulefiles/lorrax/0.1.0.lua` (337 lines) — **the single Lua file that wires Shifter + Cray MPICH + bind-mounts + env vars + lxrun/lxalloc/lxshell/lxpre shell functions.** Read carefully.
- `config/perlmutter/site_config.sh` — NERSC-specific placeholders (SLURM account / QOS / constraint / NVHPC subpath / 4 GPUs/node).
- `config/perlmutter/install.sh` — patches modulefile, installs to `~/.local/modulefiles/lorrax`.
- `config/perlmutter/run_gw.slurm` — batch submission template.
- `config/README.md` — module setup, site config, porting checklist.

### Runtime / distributed init / env vars

- `src/runtime/__init__.py` (214 lines) — **`init_jax_distributed()`, `set_default_env()`, `fallback_to_cpu_if_no_gpu_backend()`.** Sentinel `_LORRAX_JAX_DISTRIBUTED_DONE` guards re-entry.
- `src/runtime/padding.py`, `src/runtime/aot_memory.py` — adjacent runtime utilities.
- Env vars exported by the modulefile (Lua, lines vary):
  `HDF5_USE_FILE_LOCKING`, `XLA_PYTHON_CLIENT_PREALLOCATE`,
  `XLA_PYTHON_CLIENT_ALLOCATOR`, `TF_GPU_ALLOCATOR`,
  `JAX_COMPILATION_CACHE_DIR`, `MPICH_GPU_SUPPORT_ENABLED`,
  `LD_PRELOAD libmpi_gtl_cuda.so.0`, `LORRAX_ROOT`, `LORRAX_SRC`,
  `LORRAX_SITE`, `LORRAX_IMAGE`, `LORRAX_SHIFTER`,
  `LORRAX_FFI_NVHPC_HOST`, `LORRAX_FFI_PHDF5_HOST`,
  `LORRAX_FFI_SLATE_HOST`, `LORRAX_SLATE_INSTALL_DIR`,
  `LORRAX_MPICH_LIB_DIR`, `LORRAX_MPI_INCLUDE_DIR`, plus
  `SHIFTER_MODULES="gpu,mpich"`.
- Env vars read by Python: `ISDF_JAX_PROFILE_DIR`, `JAX_PROCESS_COUNT`,
  `JAX_PROCESS_INDEX`, `JAX_COORDINATOR_ADDRESS`, the standard SLURM
  ones (`SLURM_NTASKS` / `SLURM_PROCID` / `SLURM_NODELIST` /
  `SLURM_JOBID` / `SLURM_LOCALID`).
- Recent env-var **shrinkage** (good signal for the blitz audit):
  - `488e870` — three buffer-affecting chunk sizes (`r_chunk_size`, `gflat_chunk_size`, `psig_k_chunk_size`) moved from env vars to `cohsex.in`.
  - `9fe5fde` — four algorithmic toggles moved from env vars to `cohsex.in`.
  - `40a4cca` — dropped `LORRAX_GSPACE_MODE` env override.
- `docs/docs_gwjax/COHSEX_INPUT.md` (at sandbox top-level, **not** inside lorrax checkout) — the canonical cohsex.in reference. The agents 3/4 will rely on this.

### Sandbox-level agent overlay

- `/pscratch/sd/j/jackm/lorrax_sandbox/modulefiles/lorrax_agent/1.0.lua` — pool-aware overlay (loaded *on top of* `lorrax_X` base module).
- `/pscratch/sd/j/jackm/lorrax_sandbox/modulefiles/lorrax_agent/lx_pool.py` — pool coordination Python (free-node selection, lxstatus / lxattach / lxreap).
- `/pscratch/sd/j/jackm/lorrax_sandbox/AGENTS.md` — sandbox conventions.
- `/pscratch/sd/j/jackm/lorrax_sandbox/KNOWN_SANDBOX_ERRORS.md` — known infrastructure issues.

### Comprehensive docs (in `docs/`)

- `docs/CODEBASE_COMPREHENSIVE.md` — module map, data flow.
- `docs/ENVIRONMENT_COMPREHENSIVE.md` — dependencies + cluster usage.
- `docs/PHYSICS_COMPREHENSIVE.md` — ISDF + GW theory.
- `docs/MEMORY_MODEL.md` — chunked-op memory budget.

### Recent FFI / build / MPI / container commits (orientation)

```
488e870  gw_config: move 3 buffer-affecting chunk sizes from env to cohsex.in
9fe5fde  gw_config: move 4 algorithmic toggles from env vars to cohsex.in
bd06879  slab_io_ffi: hardwire write-queue depth to 2
cb2ae6e  Add cuSOLVERMp FFI profiling harness
704c99f  ζ-fit: default-on cuSolverMp Cholesky+LU for true 2D meshes
24bfa5f  ffi/cusolvermp: namespace NCCL unique-id KV key by mesh signature
334f87c  isdf_fitting: opt-in cuSolverMp Cholesky+potrs for ζ-fit charge channel
929be9a  ffi/build: require explicit MPI env to avoid silent HPC-X fallback
65e2f59  ffi/cusolvermp: default to 0.7.2; add stage_pypi.sh
c52fbd2  ffi/cusolvermp: dispatch CAL vs NCCL comm at runtime (0.7+ ABI shift)
3e71cf5  ffi/phdf5: drop redundant MPI_Comm_dup in open_ctx
2c5e5b5  ffi/slate: auto-detect SLATE_INSTALL_DIR from modulefile env
1e3c8e3  ffi/phdf5: native async read callbacks (replaces prior queue wrapper)
```

---

## 4. NERSC-isms to be alert for

Specific things that are **likely** NERSC-only and would not transfer to
another cluster. Don't take any of these on faith — verify in the code —
but use them as a starting list:

- Shifter (NERSC's containerd-equivalent; **not** standard Docker / Singularity / Apptainer / Enroot).
- Bind-mount path `/opt/udiImage/modules/mpich` — Shifter-specific.
- Cray MPICH GTL preload `libmpi_gtl_cuda.so.0` — Cray-specific naming and ABI.
- `--mpi=cray_shasta` passed to srun.
- `--constraint="gpu&hbm80g"` and similar Perlmutter SLURM constraints.
- `--account=` and `--qos=` set in `site_config.sh`.
- 4 GPUs/node default (`LORRAX_NGPU=4`) — Perlmutter A100 nodes.
- `$SCRATCH` and `$CFS` references — NERSC filesystem env vars.
- NVHPC subpath under `/opt/nvidia/hpc_sdk/...` — vendor-installed at NERSC.
- `module load` invocations to system Lmod with NERSC-shipped modules.

---

## 5. Constraints

- **Read-only on `sources/lorrax_C/`** (the canonical checkout for this study) and read-only on everything under `sources/`. No code edits. If you find a bug, note it in your report — don't fix it.
- **No compute.** No `srun`, no `lxrun`, no Python that runs JAX. Reading code, reading docs, running `git log`/`grep`/`find` — fine.
- **Stay in your own report file.** Write to `reports/lorrax_install_maintain_blitz_2026-05-13/agent_<N>.md` only.
- **Do NOT read `agent_<M>.md` for M ≠ your N.** You'll see the other drafts in the discussion phase after everyone finishes.
- **Stay in your own pane.** No tmux escapes, no talking to the other agents directly.
- **Web search is encouraged.** Use `WebSearch` / `WebFetch` whenever you need authoritative info on:
  - JAX APIs (`jax.distributed`, `jax.ffi`, `jax.experimental.shard_map`, compilation cache semantics) and their stability across versions.
  - Vendor library version history & ABI changes (cuSOLVERMp, NCCL, NVHPC SDK, CUDA toolkit, Cray MPICH, OpenMPI, parallel HDF5, SLATE/BLAS++/LAPACK++).
  - Container runtimes (Shifter / Apptainer / Singularity / Enroot / Podman / Docker) — feature parity, MPI-passthrough mechanisms, common pitfalls.
  - Cluster scheduler conventions (`srun --mpi=*`, PMI vs PMIx, GPU binding) on systems other than Perlmutter.
  - Whether something the code does is a known idiom or an outlier — quote authoritative sources (vendor docs, JAX docs, official examples) when calling out fragility.
  Cite URLs in your report when you do.

---

## 6. The audit lens — fragility, compatibility, LoC cost

Every observation in your report should fall under one of three buckets.
Tag findings in the defect catalog explicitly with **[FRAGILE]**,
**[COMPAT]**, or **[LOC-COST]** so the synthesis round can group them.

- **[FRAGILE]** — Something that *works today on Perlmutter* but would
  silently break on a vendor version bump, container rebuild, or
  modulefile drift. Examples: pinned ABI assumptions, autodetect that
  picks the wrong dep, hardcoded `.so.0` suffixes, sentinel files,
  undocumented invariants between two modules. The test is: *would the
  author notice if this broke between two installs of the same code on
  the same machine?*
- **[COMPAT]** — Something that *will not work as written* on a
  non-NERSC cluster (different MPI, different container runtime,
  different scheduler, different filesystem). Each entry should
  identify the assumption AND name at least one concrete alternative
  cluster setup where it breaks (e.g., "Frontier with Cray MPICH but
  no Shifter → Apptainer", "Polaris with OpenMPI", "a university
  Slurm+Docker cluster"). Use web search to back up cluster-specifics
  if needed.
- **[LOC-COST]** — Code or config that *only exists because of
  Perlmutter peculiarities*, and that another installer would either
  ignore, rewrite, or have to delete. This is the maintainability
  side: every LoC of NERSC-specific glue is a LoC the author has to
  keep working and a LoC a porter has to read. Examples: the
  bind-mount triad for NVHPC/phdf5/SLATE, `select_gpu.sh`,
  `in_container.sh`, Lua functions whose body is half SLURM-specific.

Be explicit about **what is fundamentally needed by any GPU+MPI+JAX
project** vs. **what is LORRAX's own NERSC-coping mechanism**. A finding
in the second bucket is much more interesting than the first.

## 7. Output structure (suggested)

Your `agent_<N>.md` should cover roughly:

1. **Scope.** One paragraph: what slice you're auditing, what you're explicitly out-of-scope.
2. **Current state.** What exists today, with file paths and line refs. Be specific — point at the actual lines in `0.1.0.lua` / `CMakeLists.txt` / `runtime/__init__.py` that you're critiquing.
3. **NERSC-isms.** What is implicitly Perlmutter-specific. For each: would it break, degrade gracefully, or work as-is on a different cluster (e.g., Frontier / Polaris / Leonardo / a generic Slurm+OpenMPI+Docker cluster)?
4. **Defect catalog.** Concrete bugs, hidden assumptions, silent fallbacks, undocumented requirements. Each one a short bullet with file:line **and a `[FRAGILE]` / `[COMPAT]` / `[LOC-COST]` tag** (multiple tags ok).
5. **Blitz proposals.** A ranked list of concrete ~1-day blitz tasks. For each:
   - What it changes (one sentence).
   - Which defect-catalog entries it addresses (by tag).
   - Why it's high-leverage (installability or maintainability win — pick a primary axis).
   - Risk / downside.
   - Whether it requires a new test / CI to lock in the win.
6. **Open questions.** **The most important section.** Things you genuinely could not resolve. Assumptions you had to make about how a non-NERSC cluster works — even after web search. Code paths you'd need to actually run on a different cluster to verify. Be specific. Honest "I don't know" beats confident-but-wrong.

---

## 8. Tone

You're writing for the author of LORRAX. Don't pad with background. Be
terse where you can, exhaustive where you must (the defect catalog
should be exhaustive — no "etc."). Disagreements with PORTING.md are
welcome and expected — that doc is partly what we're auditing.

Now go read your `prompts/agent_<N>.md`, then PORTING.md and the relevant
code, and write your draft.
