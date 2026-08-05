# Agent 3 — Runtime: env vars, jax.distributed init, launchers

## 1. Scope

What happens between `module load lorrax_C` and the first `jit` running on
rank 0: env-var surface (set + read), `jax.distributed.initialize()`
contract, and the lxrun / lxalloc / lxshell / lxpre launcher contract,
including the sandbox-local `lorrax_agent` overlay.

Explicitly **out of scope**: FFI build/link (Agents 1/2), `cohsex.in`
parsing semantics (Agent 4), physics, performance tuning, memory model.

Read targets:
`src/runtime/__init__.py` (214 lines),
`config/modulefiles/lorrax/0.1.0.lua` (337 lines),
`config/perlmutter/site_config.sh`,
`config/perlmutter/install.sh`,
`src/ffi/common/cpp/select_gpu.sh` (5 effective lines),
`src/ffi/common/cpp/in_container.sh` (2 effective lines),
sandbox overlay `modulefiles/lorrax_agent/{1.0.lua,lx_pool.py}` (343 +
645 lines),
and `git log` on `src/gw/gw_config.py` for the env→cohsex.in migration
commits (488e870, 9fe5fde, 40a4cca).

---

## 2. Current state

### 2a. `set_default_env()` / `init_jax_distributed()` — what the code does

`src/runtime/__init__.py` owns three entry points. `set_default_env()`
(L47–62) calls `os.environ.setdefault` for `JAX_ENABLE_X64=1` and, when
`platform="gpu"`, `JAX_PLATFORMS="cuda,cpu"`. `setdefault` means a
caller-set value wins — this is the right pattern but only works if the
caller `import runtime; set_default_env()` *before* any other import
that pulls in `jax`. There is no enforcement; any sibling module that
imports `jax` at top-level (and is itself imported before
`set_default_env` runs) silently locks in JAX's own defaults.

`init_jax_distributed()` (L109–152) is the interesting one. Sequence:

1. Sentinel guard: if `os.environ["_LORRAX_JAX_DISTRIBUTED_DONE"]` is
   set, return.
2. Resolve `proc_count = JAX_PROCESS_COUNT or JAX_NUM_PROCESSES or
   SLURM_NTASKS or 1`. If ≤1 → mark sentinel, return (no distributed
   init).
3. Count local GPUs from `CUDA_VISIBLE_DEVICES`. Build
   `init_kwargs = {"local_device_ids": list(range(n_local))}` if any are
   visible, else `{}`.
4. Try `jax.distributed.initialize(**init_kwargs)`. On any exception,
   fall through.
5. Fallback: `jax.distributed.initialize(coordinator_address=...,
   num_processes=..., process_id=...)`. Coordinator address resolution
   (L81–106): `JAX_COORDINATOR_ADDRESS` env → first host of
   `SLURM_NODELIST` resolved via `scontrol show hostnames` →
   `SLURMD_NODENAME` or `HOSTNAME` or `localhost`; port hard-coded to
   `12355`.

`nccl_warmup(mesh_xy)` (L155–195) is a separate, mesh-aware function
that pre-fires three NCCL communicator patterns; documented as paying
the 1–2 s `ncclCommInitRank` cost up front. Not part of the bootstrap
contract — only called from `gw_init` after distributed init succeeds.

`fallback_to_cpu_if_no_gpu_backend()` (L198–214) catches the literal
string `"Unknown backend: 'gpu'"` in a `RuntimeError` from
`jax.devices()`, then sets `JAX_PLATFORMS=cpu`. All other exceptions
re-raise.

### 2b. Modulefile env-var inventory (`config/modulefiles/lorrax/0.1.0.lua`)

Complete table of every `setenv` in `0.1.0.lua`. (`pushenv` is not used;
`prepend_path` is used only for `PATH` in the agent overlay.)

| Line | Env var                          | Value (effective)                                    | Purpose / who reads it                                                          |
|------|----------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------|
| 123  | `HDF5_USE_FILE_LOCKING`          | `FALSE`                                              | Required on Lustre; **silent corruption hazard on NFS / local FS** (see §3,§4)  |
| 129  | `XLA_PYTHON_CLIENT_PREALLOCATE`  | `false`                                              | Disables BFC preallocation                                                      |
| 130  | `XLA_PYTHON_CLIENT_ALLOCATOR`    | `platform`                                           | **Conflicts with line 131** — see [FRAGILE-2]                                   |
| 131  | `TF_GPU_ALLOCATOR`               | `cuda_malloc_async`                                  | Picks CUDA async pool allocator. Conflicts with 130                             |
| 205  | `LORRAX_ROOT`                    | (derived from modulefile path or @-patched fallback) | Read by `lorrax_agent` overlay to derive agent letter, by FFI loader indirectly |
| 206  | `LORRAX_SRC`                     | `$LORRAX_ROOT/src`                                   | Read by overlay `select_gpu_sh` / `in_container_sh` paths                       |
| 207  | `LORRAX_SITE`                    | site_packages path (h5py/scipy/matplotlib)           | Spliced into PYTHONPATH for the container                                       |
| 208  | `LORRAX_IMAGE`                   | `nvcr.io/nvidia/jax:25.04-py3`                       | Recorded but not re-read by the shell functions (they use `shifter_args`)       |
| 209  | `LORRAX_SHIFTER`                 | full `shifter --image=... --module=... --volume=...` | Spliced into the overlay's `lxrun`                                              |
| 210  | `LORRAX_FFI_NVHPC_HOST`          | host bind-mount source path                          | Diagnostic; not read by runtime                                                 |
| 211  | `LORRAX_FFI_PHDF5_HOST`          | host bind-mount source path                          | Diagnostic; not read by runtime                                                 |
| 212  | `LORRAX_FFI_SLATE_HOST`          | host bind-mount source path                          | Diagnostic; not read by runtime                                                 |
| 213  | `LORRAX_SLATE_INSTALL_DIR`       | host SLATE install                                   | **Read by FFI CMake at build time**, not by runtime                             |
| 214  | `JAX_COMPILATION_CACHE_DIR`      | `$SCRATCH/.jax_cache`                                | Inherited by JAX inside the container via `--env=` (L187)                       |

Additional env vars exported only into the container via `shifter --env`
(L171–190): `PYTHONPATH`, all four XLA/HDF5 vars above,
`LD_LIBRARY_PATH` (the six-segment chain), `LD_PRELOAD` of
`/lorrax_slate/lib/libmpi_gtl_cuda.so.0`, `MPICH_GPU_SUPPORT_ENABLED=1`,
`LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include`,
`LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich`.

The last two are **build-time** env vars consumed by
`src/ffi/common/cpp/CMakeLists.txt` (L283, L298), not by the running
Python — they exist in `lxrun`'s env so `python -m ... ffi.cpp.build`
can rebuild from inside the container. They are dead env vars during a
normal `lxrun python -m gw.gw_jax` execution.

### 2c. Code-side env-var reads (from
`grep -rnE "os\.(environ\.get|getenv|environ\[)" src/`)

After the recent 488e870 / 9fe5fde / 40a4cca migrations, the remaining
**production** env-var reads (excluding `*_test.py` / `*_bench.py` /
`*_sweep.py` boilerplate which is a pattern unto itself, see [FRAGILE-9])
are:

| Env var                                 | Read at                                       | Class      | Notes                                                                  |
|-----------------------------------------|-----------------------------------------------|------------|------------------------------------------------------------------------|
| `JAX_ENABLE_X64`, `JAX_PLATFORMS`       | `runtime/__init__.py` (write via setdefault)  | required   | Set by `set_default_env()`                                             |
| `JAX_PROCESS_COUNT`, `JAX_NUM_PROCESSES`| `runtime/__init__.py` _resolve_proc_count     | optional   | Falls back to `SLURM_NTASKS`                                           |
| `JAX_PROCESS_INDEX`, `SLURM_PROCID`     | _resolve_proc_id                              | optional   | Defaults to 0                                                          |
| `JAX_COORDINATOR_ADDRESS`               | _resolve_coordinator_address                  | optional   | Falls back to `SLURM_NODELIST` + scontrol                              |
| `SLURM_NODELIST`, `SLURMD_NODENAME`, `HOSTNAME` | coordinator fallback chain            | optional   | Final fallback `localhost`                                             |
| `CUDA_VISIBLE_DEVICES`                  | init_jax_distributed                          | required-ish | Determines `local_device_ids`; empty → no kwarg; controlled by select_gpu.sh |
| `_LORRAX_JAX_DISTRIBUTED_DONE`          | init_jax_distributed                          | internal   | Sentinel                                                               |
| `HDF5_USE_FILE_LOCKING`                 | (HDF5 library directly, not LORRAX code)      | required-on-Lustre | Silently consequential if absent                                       |
| `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION` | `gw/gw_output.py` L128-129 (logged only) | optional | Logged for run reproducibility but not enforced                        |
| `LORRAX_FFI_SO`                         | `ffi/common/ffi_loader.py` L65                | optional   | Override path to liblorrax_ffi.so; falls back to glob search           |
| `LORRAX_PHDF5_STRIPE_COUNT`, `LORRAX_PHDF5_STRIPE_SIZE_FS` | `file_io/_slab_io_ffi.py` L353-354 + `isdf_fitting.py` L1916-1917 | optional | **Duplicated read in two modules** — drift risk [FRAGILE-7] |
| `LORRAX_LUSTRE_STRIPE_COUNT`, `LORRAX_LUSTRE_STRIPE_SIZE`, `LORRAX_NO_PRESTRIPE` | `0.1.0.lua` `lxrun` body                       | optional   | Shell-side; different name from the Python-side stripe vars **[FRAGILE-7]** |
| `LORRAX_FFI_DEBUG_SHARDS`               | `file_io/_slab_io_ffi.py` L583                | optional   | Debug-only                                                             |
| `LORRAX_WRITE_NO_JIT`                   | `file_io/_slab_io_ffi.py` L608                | optional   | Debug-only                                                             |
| `LORRAX_PHDF5_CLOSE_VERBOSE`            | `file_io/_slab_io_ffi.py` L744                | optional   | Defaults to `"1"` (on) — set `"0"` to silence; **on-by-default** is unusual |
| `LORRAX_MEM_PROFILE`                    | `common/isdf_fitting.py` L20                  | optional   | Profile mode toggle                                                    |
| `LORRAX_V_Q_FFT_COEF`, `LORRAX_V_Q_AOT_VERBOSE`, `LORRAX_V_Q_Q_CHUNK`, `LORRAX_V_Q_TIME_STAGES`, `LORRAX_V_Q_MU_CHUNK` | `gw/v_q_tile.py`, `gw/compute_vcoul.py` | optional+silently-consequential | **5 remaining V_q tunables not yet migrated to cohsex.in** [LOC-COST-3] |
| `LORRAX_SC_MAX_ITER`, `LORRAX_SC_TOL_EV`, `LORRAX_SC_ACCEL`, `LORRAX_SC_DEPTH`, `LORRAX_SC_MIXING` | `gw/gw_jax.py` L477-481 | silently-consequential | **5 self-consistency knobs in env vars** — exactly the discoverability hole 488e870/9fe5fde fixed elsewhere [LOC-COST-1] |
| `LORRAX_SC_DUMP_DIR`                    | `gw/sc_iteration.py` L430                     | optional   | Debug path                                                             |
| `LORRAX_DISABLE_MINIMAX_DISK_CACHE`, `LORRAX_MINIMAX_CACHE_DIR` | `gw/minimax_screening.py` L52-54   | optional   | Disk-cache controls                                                    |
| `LORRAX_LU_DEBUG_DUMP`                  | `common/cusolvermp_solve_lu_test.py` L157     | optional   | Test-side                                                              |
| `ISDF_CHUNK_TARGET_UTILIZATION`, `ISDF_ZCT_STAGE_CAP_GB`, `ISDF_ZCT_STAGE_CAP_FRAC` | `gw/gw_config.py` L865, L873-874 | silently-consequential | **3 ISDF planner knobs still in env vars** — survives in `gw_config.py` itself [LOC-COST-2] |
| `ISDF_JAX_CACHE_DIR`, `JAX_COMPILATION_CACHE_DIR` | `common/jax_compile_cache.py` L82-84 | optional | Per-session cache subdir layer; **alternate cache var that overrides the modulefile's** [FRAGILE-8] |
| `ISDF_JAX_PROFILE_DIR`                  | `bse/test_bse.py` L333-334, `common/jax_profile.py` L17 | optional | Profile output dir                                                     |
| `PF_ARTIFACTS_DIR`                      | `gw/gw_driver_helpers.py` L81                 | optional   | Profile dir; defaults to `"profile"`                                   |
| `STERN_DEBUG`, `KP2_DEBUG`              | `solvers/sternheimer_solve.py`, `psp/run_sternheimer.py` | optional | Debug                                                                  |

That's **~25 production env-var read sites still active** post-cleanup,
roughly half of them in the runtime / config layer (deliberate) and
roughly half scattered through `gw/`, `common/`, `file_io/` (continuing
the discoverability problem the cleanup commits attacked).

### 2d. Launcher contract — what `lxrun <cmd>` actually does

From `0.1.0.lua` L257–284, the base lxrun expands to:

```
# Pre-stripe $PWD/tmp via host-side `lfs setstripe -c 16 -S 4M`
# (unless LORRAX_NO_PRESTRIPE=1; silently no-op if lfs not on PATH)
ngpu=${LORRAX_NGPU:-4}
mpitype=${LORRAX_MPI_TYPE:-cray_shasta}
jobflag=$([[ -n $SLURM_JOBID && -z $SLURM_STEP_ID ]] && echo --jobid=$SLURM_JOBID)
srun $jobflag --mpi=$mpitype --gres=gpu:$ngpu -N 1 -n $ngpu \
     $LORRAX_SRC/ffi/common/cpp/select_gpu.sh \
     shifter --image=... --module=gpu,mpich --volume=...:/lorrax_nvhpc \
             --volume=...:/lorrax_phdf5 --volume=...:/lorrax_slate \
             --env=PYTHONPATH=... --env=HDF5_USE_FILE_LOCKING=FALSE \
             --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
             --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
             --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
             --env=LD_LIBRARY_PATH=<6-segment chain> \
             --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
             --env=MPICH_GPU_SUPPORT_ENABLED=1 \
             --env=JAX_COMPILATION_CACHE_DIR=... \
             --env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include \
             --env=LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich \
     $LORRAX_SRC/ffi/common/cpp/in_container.sh \
     "$@"
```

Required caller environment:
- `SLURM_JOBID` set (else srun creates a new step against whatever
  default partition the user has — **no fail-fast**).
- `LORRAX_NGPU` may be set; defaults to `gpus_per_node` (=4 on
  Perlmutter).
- The host `lfs` binary on PATH if user wants Lustre striping
  (silent no-op otherwise — could be a problem on non-Lustre clusters
  where someone copies this verbatim).

Not required: nothing else. The shell function is self-contained and
self-documenting modulo the `select_gpu.sh` / `in_container.sh` paths
being hard-coded under `$LORRAX_SRC/ffi/common/cpp/`.

### 2e. Agent overlay — sandbox layer

`/pscratch/sd/j/jackm/lorrax_sandbox/modulefiles/lorrax_agent/1.0.lua`
(343 lines) + `lx_pool.py` (645 lines) sit *on top of* a base
`lorrax_X` module. They:

1. Derive `LORRAX_AGENT={A,B,C,D}` from the trailing component of
   `$LORRAX_ROOT` (path-match, brittle on non-`lorrax_<LETTER>` install
   names).
2. Override `lxalloc` to add `-J lx-alloc-$USER` so other shells can
   find the allocation via `lxattach`.
3. Override `lxrun` to call `lx_pool.py prelaunch <N>` for free-node
   selection, render a 4-line banner to stderr, build
   `--nodelist=nidA,nidB`, add `--immediate=10`, and set
   `--job-name=lx-${LORRAX_AGENT}-${HHMMSS}-${BASHPID}-${RANDOM}`.
4. Override `lxshell` to add `--overlap` (so an interactive shell
   doesn't block other agents' lxrun on the same node — non-trivial,
   see [COMPAT-3]).
5. Add `lxstatus` / `lxstatus --agents` / `lxreap` / `lxattach`
   subcommands by shelling out to `lx_pool.py`.
6. Touch `~/.lorrax/agents/<LETTER>.heartbeat` on every call.

The overlay rebuilds the srun line by splicing
`$LORRAX_SHIFTER` (exported by the base module) verbatim — i.e. the
contract is "base module sets `LORRAX_SHIFTER` as a self-contained
shifter command-line". The overlay does **not** wrap the base
`lxrun`; it shadows it. Fragile if the base module's srun shape
changes; intentional decoupling per the comment at L107–111.

`lx_pool.py` is a 645-line Python tool. Of those 645 lines, ~150 are
banner rendering (ANSI colors, runtime formatting, heartbeat-age
formatting), ~120 are subprocess-wrappers around `squeue` /
`scontrol show hostnames` / `scancel`, ~150 are subcommand handlers
(`prelaunch` / `status` / `reap` / `attach` / `heartbeat`
/ `other-allocs`), and the rest is dataclass scaffolding and the
"insufficient capacity" decision-tree.

### 2f. Env→cohsex.in migration — what was moved

| Commit  | Var removed                                | Replacement (cohsex.in key)                                    |
|---------|--------------------------------------------|----------------------------------------------------------------|
| 488e870 | `LORRAX_PSIG_KCHUNK`                       | `psig_k_chunk_size`                                            |
| 488e870 | `LORRAX_GFLAT_CHUNK_SIZE`                  | `gflat_chunk_size`                                             |
| 488e870 | (was unwriteable)                          | `vq_g_chunk_size` (new)                                        |
| 9fe5fde | `LORRAX_USE_CUSOLVERMP_CHARGE_FACTOR`      | `cusolvermp_charge` (auto/on/off)                              |
| 9fe5fde | `LORRAX_USE_CUSOLVERMP_LU`                 | `cusolvermp_lu` (auto/on/off)                                  |
| 9fe5fde | `LORRAX_GAMMA_CONTRACT_MODE`               | `gamma_contract_mode` (take/einsum/scan)                       |
| 9fe5fde | `LORRAX_CHOOSER_MODE`                      | `chunk_chooser_mode` (heuristic/analytic)                      |
| 40a4cca | `LORRAX_GSPACE_MODE`                       | `gspace_mode` (host_cache/file_reread)                         |

All three commit messages identify the same rationale: **"discoverability
hole — non-default behavior set by an env var that no one would find
without reading source"**. Yes. The work is incomplete (see §4 defect
catalog [LOC-COST-1..3]).

---

## 3. NERSC-isms

Things in this slice that are implicitly Perlmutter-specific:

| Item                                         | Where                          | On a non-NERSC cluster                                                                                                                                       |
|----------------------------------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HDF5_USE_FILE_LOCKING=FALSE`                | 0.1.0.lua L123                 | **Required-on-Lustre, dangerous-on-other**. On Apptainer/Singularity over NFS or BeeGFS, this silently masks data corruption ([HDF5 group][1])               |
| `--mpi=cray_shasta`                          | 0.1.0.lua L270, L273           | Wrong choice on non-Cray. Polaris (PBS+pmix), Leonardo (slurm+pmix), generic clusters (`pmix`/`pmi2`) need a different value. SLURM "MPI Guide" warns of silent singleton-init ([SchedMD][2]) |
| `MPICH_GPU_SUPPORT_ENABLED=1` + `libmpi_gtl_cuda.so.0` LD_PRELOAD | 0.1.0.lua L181, L186, `in_container.sh` | **Cray-only**. Upstream MPICH and Open MPI use entirely different mechanisms; this var literally does nothing under upstream MPICH ([NERSC docs][3])         |
| Shifter `--module=gpu,mpich` + bind-mount triad `/opt/udiImage/modules/mpich` | 0.1.0.lua L75, L156 | Shifter-only. Apptainer/Enroot need `--nv` + a fully different bind-mount strategy                                                                            |
| `JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache` | 0.1.0.lua L115-116, L187, L214 | `$SCRATCH` is NERSC-specific. Falls back to `$HOME` if unset — OK in principle, but a non-NERSC user on a system with a separate fast scratch (Polaris: `/eagle`; Frontier: `/lustre/orion`) will silently get a slow location |
| `select_gpu.sh` pattern (`SLURM_LOCALID` → `CUDA_VISIBLE_DEVICES`) | `src/ffi/common/cpp/select_gpu.sh` | **Generic SLURM convention, not NERSC-only.** Works on any cluster with SLURM task plugin. Modern alternative `--gpus-per-task` documented to break MPI launchers historically ([SLURM GRES][4]) — fine to keep |
| 4-GPU/node default (`LORRAX_NGPU=4`)         | 0.1.0.lua L41, L237, L269       | Hard-coded for A100 Perlmutter; needs override on H100 (8/node), MI250X Frontier (8 GCDs/node), GH200 nodes (4 but big-VRAM)                                  |
| `slurm_account`, `slurm_qos`, `slurm_constraint` | 0.1.0.lua L66-68 patched from `site_config.sh` | Cluster-specific. The template-patching layer handles this correctly; the issue is `lxalloc` body bakes them into the function body, so changing account mid-session requires reloading the module |
| Lustre `lfs setstripe` in `lxrun` body       | 0.1.0.lua L263-268             | Silent no-op on non-Lustre clusters (fine), but on GPFS / BeeGFS / VAST a wholly different tuning step would apply                                            |

**Specifically NOT NERSC-only** in this slice:
- The `init_jax_distributed()` SLURM-aware bootstrap. JAX
  `jax.distributed.initialize()` auto-detects SLURM via env vars on all
  clusters; the `local_device_ids` override is the right move
  everywhere with 1-GPU-per-rank ([JAX docs][5]).
- `select_gpu.sh`.
- The `family("lorrax")` Lmod pattern.
- The `_LORRAX_JAX_DISTRIBUTED_DONE` re-entry sentinel concept (every
  multi-rank JAX project needs this in some form).

---

## 4. Defect catalog

Tagged **[FRAGILE]** (works today, silent break on drift), **[COMPAT]**
(doesn't work as written on non-NERSC), **[LOC-COST]** (exists only to
cope with Perlmutter or sandbox).

### [FRAGILE-1] `set_default_env()` is order-dependent and unenforced

`src/runtime/__init__.py:47-62` uses `os.environ.setdefault`. JAX reads
these at module import time. The contract "call `set_default_env()`
BEFORE `import jax`" is documented in the docstring (L17) but nothing
enforces it. Any module that's importable from the LORRAX `src/` PATH
and that has `import jax` at top-level can break this contract if
imported (directly or transitively) before `set_default_env()` runs.
**Fix**: add an assertion `assert "jax" not in sys.modules` at the top
of `set_default_env`, with a clear error message.

### [FRAGILE-2] `XLA_PYTHON_CLIENT_ALLOCATOR=platform` and `TF_GPU_ALLOCATOR=cuda_malloc_async` are mutually exclusive

`0.1.0.lua:130-131` sets both. Per [JAX GPU memory docs][6], these are
two different selectors for XLA's GPU allocator and **only one wins**
(precedence in current XLA: the JAX-prefixed name).
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` means "use cudaMalloc per
allocation, deallocate eagerly — for OOM debugging, VERY slow".
`cuda_malloc_async` is the modern async-pool allocator. The intent in
the code comment at L127-128 ("grow on demand from the CUDA async
mempool") matches the latter. **It is plausible the platform allocator
is the one that's actually active**, with cuSOLVERMp/NCCL VRAM sharing
working by accident. Worth confirming with a `jax.debug.visualize_array`
or by reading the JAX-startup log lines that record the chosen
allocator.

### [FRAGILE-3] coordinator port `12355` hard-coded, no collision handling

`src/runtime/__init__.py:100, 106` returns `f"{first_host}:12355"` with
no override. If two LORRAX runs land on the same first-node of an
allocation (shared-allocation scenarios — exactly what the agent
overlay enables) and both fall into the explicit-coordinator branch,
they will fight for port 12355. The fast path (no-args
`jax.distributed.initialize()`) probably masks this on Perlmutter
because Cray PMI hands jax.distributed everything it needs, so the
fallback rarely fires; but the failure mode if it does is silent
hang, not loud failure.

### [FRAGILE-4] `init_jax_distributed()` swallows the first-attempt exception

`src/runtime/__init__.py:144` catches `Exception` and falls through
silently to the fallback path. If the first attempt fails for a reason
unrelated to "no-args auto-detection failed on Cray MPICH" (e.g.,
coordinator port already bound, mismatched NCCL versions, a JAX bug
that throws after partial init), the fallback runs and either succeeds
(in which case we now have a partly-initialized JAX) or fails with a
different error that doesn't carry the original cause. **Fix**: log the
caught exception via `warnings.warn` before falling through, with a
clear marker that we're trying the explicit path.

### [FRAGILE-5] Sentinel only protects against re-import, not against double `module load`

`_LORRAX_JAX_DISTRIBUTED_DONE` is in `os.environ`, so it persists across
`os.execv` / re-import within the same Python process. But if a user
does `module unload lorrax_C && module load lorrax_C` mid-session and
then re-runs Python, the sentinel survives in the parent shell's env
and propagates into the next `lxrun`. The new Python process would then
**skip distributed init entirely**, leading to single-process JAX with
no error. Mitigation: the sentinel only blocks distributed init when
`proc_count > 1`, and a re-`module load` is rare; but it's a footgun.
**Fix**: name the sentinel with a process-scoped suffix (e.g. include
`SLURM_STEP_ID`) so it doesn't carry across srun-steps.

### [FRAGILE-6] `--immediate=10` in the agent overlay's lxrun

`modulefiles/lorrax_agent/1.0.lua:217`. The 10-second budget is fine
when the allocation is healthy; if Slurm is sluggish or the node
selection raced with another agent, the failure is a misleading "srun:
job did not start within 10 seconds" rather than "the node you picked
is now busy". Acceptable but worth a louder error message on
`--immediate` timeout.

### [FRAGILE-7] Two parallel naming schemes for Lustre stripe vars

`lxrun` (Lua, L265-266) reads `LORRAX_LUSTRE_STRIPE_COUNT` and
`LORRAX_LUSTRE_STRIPE_SIZE`. `_slab_io_ffi.py` and `isdf_fitting.py`
read `LORRAX_PHDF5_STRIPE_COUNT` and `LORRAX_PHDF5_STRIPE_SIZE_FS`. Two
schemes for the same concept (Lustre striping on the same `$PWD/tmp`
tree). **Fix**: standardize on one prefix, or document the split
intentionally (lxrun pre-stripe vs Python-side `H5Pset_fapl_mpio`).
The two reads in `_slab_io_ffi.py:353` and `isdf_fitting.py:1916` are
also literal duplicates.

### [FRAGILE-8] `ISDF_JAX_CACHE_DIR` overrides `JAX_COMPILATION_CACHE_DIR` undocumented

`common/jax_compile_cache.py:82-84` reads `ISDF_JAX_CACHE_DIR` first and
only falls back to `JAX_COMPILATION_CACHE_DIR`. The modulefile sets the
latter (L214) but not the former. So a user who looks at the modulefile
and assumes the cache var is `JAX_COMPILATION_CACHE_DIR` may be
confused when this layer creates per-session subdirs under a different
parent. **Fix**: either drop the `ISDF_*` variant entirely, or document
both in `lorrax help`.

### [FRAGILE-9] ~15 test/bench files each re-implement distributed init

Every file matching `src/common/*_test.py` and `*_bench.py` has a
~10-line copy of the distributed-init dance with its own sentinel name
(`_DIST`, `_DIST_FLAG`, `_DIST_SENTINEL`). Examples:
`chol_natural_test.py:13-20`, `cusolvermp_eigh_test.py:45-64`,
`slate_vs_cusolvermp_bench.py:40-51` (this last one explicitly uses
`_LORRAX_JAX_DISTRIBUTED_DONE` — the canonical sentinel name — but
inlined, not via `runtime.init_jax_distributed()`). The whole **point**
of `src/runtime/__init__.py` per its docstring (L24-30) was to retire
these copies. The copies survive. Drift risk: as `runtime/__init__.py`
evolves (e.g., to handle a new JAX cluster-detection API), 15 test
files don't.

### [FRAGILE-10] `nccl_warmup` couples to mesh detail

`runtime/__init__.py:155-195` hard-codes three communicator patterns
(full-mesh, axis-x, axis-y) tied to a 2D mesh assumption. If a future
LORRAX run uses a 1D or 3D mesh the warmup either over-warms or
under-warms. Not a bug today but a latent foot-gun.

### [COMPAT-1] `--mpi=cray_shasta` is not portable

`0.1.0.lua:270, 273`. Overridable via `LORRAX_MPI_TYPE`, default in
site_config.sh. The wrong value silently gives each rank its own
singleton `MPI_COMM_WORLD` — JAX collectives over MPI then no-op
locally with wrong answers. See SLURM MPI guide warning ([SchedMD][2]).
On Polaris (PBS), the whole salloc/srun layer doesn't apply — `mpiexec`
+ `palsd` is used. On Leonardo (CINECA, slurm + pmix), the value would
be `pmix`. On a generic Slurm + OpenMPI cluster, `pmi2` or `pmix`.
**Fix**: the site_config.sh story is right; what's missing is a startup
sanity check ("after MPI_Init_thread, assert size==SLURM_NTASKS or fail
loudly").

### [COMPAT-2] Shifter is unique to NERSC

`0.1.0.lua:75, L181, L196-197, L209` and effectively the entire bottom
half of the modulefile. Other clusters use Apptainer/Singularity
(Frontier, Polaris), Enroot+Pyxis (Leonardo, a few university
clusters), or no container at all. The bind-mount layout
`/lorrax_nvhpc`, `/lorrax_phdf5`, `/lorrax_slate` and the
`/opt/udiImage/modules/mpich` path are Shifter-internal — Apptainer's
mount syntax is `--bind`, MPI passthrough is `--nv` for CUDA and
explicit `--bind /opt/cray/pe:/opt/cray/pe` for Cray MPICH, plus
different LD_LIBRARY_PATH wiring. **Not a defect to fix** — Shifter is
the right tool on Perlmutter — but it means "porting the modulefile"
is "rewriting the modulefile".

### [COMPAT-3] `--overlap` on `srun --pty` is Slurm-version-specific

`modulefiles/lorrax_agent/1.0.lua:260`. `--overlap` was added in
Slurm 20.11. The agent overlay assumes the cluster has it. Older
clusters (some HPCs are still on 20.02/19.05) lack the flag, and
lxshell would error. Not relevant to NERSC today but if the overlay
ever upstreamed, this becomes a portability constraint.

### [COMPAT-4] `LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich`

Hard-coded in `0.1.0.lua:189` from `site_config.sh`. This is the
Shifter-mounted Cray MPICH path. Build-time only (CMake reads it), but
it's an in-container path that doesn't exist on the host, so anyone
trying to debug-build outside Shifter has to override it. Not a
runtime defect.

### [COMPAT-5] `lxalloc` body bakes `--account=m2651` and `--qos=interactive`

In the agent overlay (`1.0.lua:147-149`), unlike the base lxalloc which
patches via site_config.sh, the **overlay** hard-codes `m2651` and
`interactive` literally. If the overlay ever gets shared across users
or charges accounts, this is wrong. Not a base-LORRAX issue but
flagging because it shows up if the overlay upstreams.

### [LOC-COST-1] 5 self-consistency env vars not yet migrated to cohsex.in

`src/gw/gw_jax.py:477-481` reads `LORRAX_SC_MAX_ITER`,
`LORRAX_SC_TOL_EV`, `LORRAX_SC_ACCEL`, `LORRAX_SC_DEPTH`,
`LORRAX_SC_MIXING`. Exact same discoverability problem 488e870 /
9fe5fde fixed. Should be `scf_max_iter` / `scf_tol_ev` / `scf_accel`
/ `scf_depth` / `scf_mixing` in cohsex.in. Plus
`LORRAX_SC_DUMP_DIR` in `sc_iteration.py:430` (debug-only — could stay
as env). ~5 keys, ~1-hour migration.

### [LOC-COST-2] 3 ISDF planner env vars in `gw_config.py` itself

The config-resolver reads `ISDF_CHUNK_TARGET_UTILIZATION`,
`ISDF_ZCT_STAGE_CAP_GB`, `ISDF_ZCT_STAGE_CAP_FRAC` at
`gw_config.py:865, 873-874`. The whole point of `cohsex.in` is that
config-side decisions are visible in the input file. These three are
literally in the resolver — so they should be in the schema next to
the things they shadow. ~30-minute migration.

### [LOC-COST-3] 5 V_q tunables in env vars

`gw/v_q_tile.py` and `gw/compute_vcoul.py` read `LORRAX_V_Q_FFT_COEF`,
`LORRAX_V_Q_AOT_VERBOSE`, `LORRAX_V_Q_Q_CHUNK`, `LORRAX_V_Q_TIME_STAGES`,
`LORRAX_V_Q_MU_CHUNK`. AOT_VERBOSE and TIME_STAGES are reasonably
debug-flag-like (could stay); the chunkers and the FFT-coef tunable
are the same kind of buffer-affecting knobs that 488e870 just moved
out. Should follow.

### [LOC-COST-4] `lx_pool.py` is 645 lines of multi-agent coordination

Sandbox-only tooling — by the comment at `1.0.lua:4-7` it is
explicitly not part of upstream LORRAX. But within that scope, ~150
lines are ANSI banner rendering and ~200 lines are squeue parsing.
This is high LoC-cost for what amounts to "pick a free node from the
allocation and tag the step". A leaner implementation: a single
`squeue -j $JID -s -o '%i|%j|%N|%M'` shell snippet would handle 80%
of the value (free-node selection + duplicate-name avoidance) in <50
lines of Lua/bash. Banner+heartbeat are nice-to-haves; reap and
attach are useful but each is <20 lines if it's not class-based.
Worth considering whether the multi-agent A/B/C/D coordination should
exist at all or whether each agent should just run its own salloc.

### [LOC-COST-5] `select_gpu.sh` and `in_container.sh` are 2 lines each, of "things shifter unsets"

`select_gpu.sh` (8 lines, 1 effective: `export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec "$@"`)
and `in_container.sh` (5 lines, 1 effective: `export MPICH_GPU_SUPPORT_ENABLED=1; exec "$@"`)
exist as standalone shell scripts purely so srun invokes them as
single-process images and PMI env vars survive. This is a fine pattern.
The LoC cost is the comment / explanation footprint: the comments
correctly identify *why* they exist (Shifter's `--module=mpich` unsets
`MPICH_GPU_SUPPORT_ENABLED`; bash `-c` strips PMI). On a non-Shifter
cluster, `in_container.sh` is dead weight (Apptainer doesn't unset
anything). `select_gpu.sh` is portable.

### [LOC-COST-6] Two unused LORRAX_FFI_*_HOST exports

`0.1.0.lua:210-212` export `LORRAX_FFI_NVHPC_HOST`,
`LORRAX_FFI_PHDF5_HOST`, `LORRAX_FFI_SLATE_HOST`. Grepping the LORRAX
tree shows no Python or shell reads. They're for human / debug
inspection only ("which dir did the bind-mount come from?"). Fine to
keep, but tag as diagnostic-only.

### [LOC-COST-7] Inconsistent prefix: `LORRAX_*` vs `ISDF_*` vs `JAX_*` vs `TF_*`

Counting the production env-var reads (§2c table): 3 `ISDF_*`, 1
`STERN_*`, 1 `KP2_*`, 1 `PF_*`, ~12 `LORRAX_*`. The `ISDF_*` and
non-LORRAX prefixes are historical; standardizing on `LORRAX_*` (or
`GWJAX_*`) would make `env | grep LORRAX_` a complete inventory.

---

## 5. Blitz proposals

Ranked by leverage (installability or maintainability, axis explicit).
Each is scoped to ~1 day.

### B1. `lorrax doctor` CLI command — green/yellow/red env audit
**Axis:** installability **AND** maintainability.
**Touches:** new file `src/cli/doctor.py`, hook into pyproject.toml
entry point (similar to `lorrax-gw`).
**What it does:** Auditable startup check. Reads every env var the
runtime consumes; verifies `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cuda,cpu`,
`CUDA_VISIBLE_DEVICES` is set sanely, exactly one of
`{XLA_PYTHON_CLIENT_ALLOCATOR, TF_GPU_ALLOCATOR}` is set, the
LD_LIBRARY_PATH contains the expected six segments in order,
`MPICH_GPU_SUPPORT_ENABLED=1`, `LD_PRELOAD` has
`libmpi_gtl_cuda.so.0`, `HDF5_USE_FILE_LOCKING=FALSE` *and* the FS
under `$PWD` is Lustre (else flag yellow). Final block: tries
`jax.distributed.initialize()` if `SLURM_NTASKS>1`, then prints
`jax.devices()`, `jax.process_count()`, `jax.process_index()`. Each
finding green/yellow/red with a one-line remediation hint.
**Addresses:** [FRAGILE-1], [FRAGILE-2], [FRAGILE-5], [FRAGILE-7],
[FRAGILE-8], [COMPAT-1].
**Risk:** Low. Read-only.
**Locked in by:** none required; one smoke test that the command
exists and runs with `--help` is enough.

### B2. Migrate the remaining env vars to cohsex.in
**Axis:** maintainability.
**Touches:** `src/gw/gw_config.py`, `src/gw/gw_jax.py:477-481`,
`src/gw/v_q_tile.py`, `src/gw/compute_vcoul.py`,
`docs/docs_gwjax/COHSEX_INPUT.md`.
**What it changes:** 5 SC knobs ([LOC-COST-1]), 3 ISDF planner knobs
([LOC-COST-2]), 3 V_q chunk/coef knobs ([LOC-COST-3]) → cohsex.in. Keep
the debug-only ones (AOT_VERBOSE, TIME_STAGES, DEBUG dump dirs) as
env vars and label them accordingly.
**Addresses:** [LOC-COST-1], [LOC-COST-2], [LOC-COST-3].
**Risk:** Medium — touches the hot GW driver. Migration commits are
straightforward (precedent: 488e870, 9fe5fde) and the LORRAX tests
already exercise the planner.
**Locked in by:** extend the existing pytest `test_cohsex_in_parse`
fixture to assert defaults for the new keys, plus a new regression
test that runs the GW driver end-to-end with non-default values
threaded through cohsex.in and confirms they reach the planner.

### B3. Unit-test `init_jax_distributed()` with mocked SLURM env
**Axis:** maintainability.
**Touches:** new `src/runtime/test_init.py`. CPU-only.
**What it tests:** Single-rank (no init called); multi-rank with
`CUDA_VISIBLE_DEVICES="0"` (path 1 should succeed with
`local_device_ids=[0]`); multi-rank fallback with `CUDA_VISIBLE_DEVICES`
unset (mock `jax.distributed.initialize` first call to raise, verify
second call gets explicit coord); re-entry (sentinel set → noop). Mock
`jax.distributed.initialize` and assert kwargs. Mock `subprocess.run`
for the scontrol path.
**Addresses:** [FRAGILE-3], [FRAGILE-4], [FRAGILE-5], [FRAGILE-9]
(the test would also serve as the reference implementation that
boilerplate sites should imitate).
**Risk:** Low. JAX import is heavy though — use `unittest.mock` for
the `jax` module surface.
**Locked in by:** new pytest, runs in the existing CPU test sweep.

### B4. ENV_VARS.md reference doc
**Axis:** installability **AND** maintainability.
**Touches:** new `docs/ENV_VARS.md`; cross-link from
`docs/ENVIRONMENT_COMPREHENSIVE.md` and `src/ffi/PORTING.md`.
**What it contains:** Table per the §2c structure above:
{name, where-read, who-sets, required/optional/consequential, default,
prefix family, "remove plan if any"}. Generated semi-automatically by
a grep script that lives next to the doc (so it stays in sync —
periodically run as part of release-prep).
**Addresses:** [LOC-COST-7] and is the documentation predicate for B1
to be useful.
**Risk:** Zero. Doc only.
**Locked in by:** a tiny CI script that runs the grep and diffs the
table — fails CI if a new `os.environ.get` lands without a doc entry.

### B5. Kill the 15× duplicated distributed-init boilerplate in tests/benches
**Axis:** maintainability.
**Touches:** all `src/common/*_test.py`, `*_bench.py`, `*_sweep.py`
that currently inline the dance. Replace each with
`from runtime import set_default_env, init_jax_distributed,
fallback_to_cpu_if_no_gpu_backend; set_default_env(); ...;
init_jax_distributed()`.
**Addresses:** [FRAGILE-9].
**Risk:** Mechanical edit. The inline sentinels (`_DIST`,
`_DIST_FLAG`, etc.) all have different names — confirm dropping them
doesn't break some test's "called twice via pytest" property
(likely a non-issue since `_LORRAX_JAX_DISTRIBUTED_DONE` does the
same job globally).
**Locked in by:** the pytest sweep itself, plus a `grep -L
'from runtime import init_jax_distributed' src/common/*_test.py` CI
check that fails when a new test file reintroduces the pattern.

### B6. Upstream `lxstatus` (only) into the base modulefile
**Axis:** maintainability.
**Touches:** `config/modulefiles/lorrax/0.1.0.lua` (add a 20-line
`lxstatus` function reading `squeue -j $SLURM_JOBID -s`), DROP the
heartbeat / multi-agent table parts.
**What it changes:** Universally useful (any user wants "what's
running in my allocation"). Keeps the multi-agent A/B/C/D pool logic
in the sandbox overlay where it belongs.
**Addresses:** half of [LOC-COST-4] by promoting the one piece that
isn't sandbox-specific.
**Risk:** Low.
**Locked in by:** none — it's a diagnostic command. Manual smoke test.

### B7. Fail-fast wrappers on the runtime contract assumptions
**Axis:** installability.
**Touches:** `src/runtime/__init__.py`.
**What it changes:** (a) `set_default_env` raises if `jax` is already
in `sys.modules` ([FRAGILE-1]). (b) `init_jax_distributed` logs the
swallowed exception via `warnings.warn` before falling back
([FRAGILE-4]). (c) The sentinel includes `SLURM_STEP_ID` so it doesn't
leak across srun steps ([FRAGILE-5]). (d) After init, assert
`jax.process_count() == proc_count` and fail loud if not (catches the
`--mpi=` wrong-value singleton-init footgun, [COMPAT-1]).
**Addresses:** [FRAGILE-1], [FRAGILE-4], [FRAGILE-5], [COMPAT-1].
**Risk:** Medium — (d) in particular is paranoia that might fire
during legitimate driver work. Make it a warning before promoting to
assert.
**Locked in by:** B3's unit tests cover (a)-(c). (d) needs an
integration test on real SLURM (deferred to CI on Perlmutter).

### B8. Resolve `XLA_PYTHON_CLIENT_ALLOCATOR` vs `TF_GPU_ALLOCATOR`
**Axis:** maintainability + correctness.
**Touches:** `0.1.0.lua:130-131`.
**What it changes:** Drop one. Per the JAX docs, the intent comment
matches `cuda_malloc_async`; verify by reading the JAX-startup log
line on a real run (which allocator did XLA actually pick?), then
remove the loser.
**Addresses:** [FRAGILE-2].
**Risk:** Low — but **must verify which is currently active** before
choosing. There's a chance the platform allocator is silently in
charge and removing it changes performance / OOM behavior.
**Locked in by:** smoke test of `lxrun python3 -c "import jax;
print(jax.devices())"` plus a startup-log scrape that asserts which
allocator is reported.

**Ranking (high → low leverage):**
B1, B4, B2, B3, B5, B7, B8, B6.

B1 + B4 together (doctor + ENV_VARS.md) are the installability
keystone — a second user can `module load` and `lorrax doctor` and
know whether they're green before reading any LORRAX code. B2 is the
maintainability keystone — continuing the trend the author has
already validated. B3+B5 retire ~150 lines of boilerplate. B7 is the
defensive-programming pass. B8 resolves a conflict whose current
resolution is "unknown". B6 is small + nice.

---

## 6. Open questions

**Honest "I don't know"s; named so the synthesis round can decide.**

1. **Is `XLA_PYTHON_CLIENT_ALLOCATOR=platform` actually active on
   Perlmutter today, or is `TF_GPU_ALLOCATOR=cuda_malloc_async` winning?**
   I cannot tell from the docs alone — the JAX docs say "platform"
   when set is selected, but `TF_GPU_ALLOCATOR` is a legacy XLA env that
   may override under some code paths. Needs a real run + log inspection.
   Hangs on [FRAGILE-2] / B8.

2. **Does Cray PMI's auto-detection in modern `jax.distributed.initialize()`
   (no-args) work when `local_device_ids` is wrong?** JAX docs say
   no-args defaults to "one device per process" on SLURM; LORRAX
   explicitly passes `local_device_ids=list(range(n_local))` because of
   "the no-args default hangs in topology exchange". But on JAX 0.5+
   that default supposedly matches what LORRAX computes — so the
   explicit pass may be unnecessary on current JAX. Worth retesting
   no-args on a clean JAX 0.6 + Cray MPICH allocation.

3. **What does Cray PMI actually export?** I've been assuming
   `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_NODELIST` are present from
   `srun --mpi=cray_shasta`. If Cray PMI strips or remangles any of
   them, the explicit-coordinator fallback in `init_jax_distributed`
   could silently degrade. Needs a single `env | grep -E
   "SLURM_|PMI_"` run inside an `lxrun` to confirm.

4. **Is `JAX_COMPILATION_CACHE_DIR` safe with multiple LORRAX
   variants (A/B/C/D) hitting it simultaneously?** JAX docs say
   shared-FS is supported and only rank 0 writes. But with four agent
   processes each being process-0 in their own `jax.distributed` group,
   you have four global rank 0's writing to the same cache directory
   with overlapping but-not-identical XLA flags / mesh sizes. JAX's
   cache key includes the device topology, so in principle they don't
   collide — but stale entries from a moved-on code state are never
   evicted, and the cache grows without bound. Worth a `du -sh
   $SCRATCH/.jax_cache` periodically and maybe a cleanup blitz I haven't
   proposed.

5. **Does the `LORRAX_FFI_NVHPC_HOST` / `_PHDF5_HOST` / `_SLATE_HOST`
   diagnostic export actually serve anyone?** I grep'd the entire
   `lorrax_C` tree and find no reads. They might be read by external
   scripts in `runs/`, sandbox tooling, or by humans inspecting
   `env`. If unused they should be dropped from the modulefile to
   shrink the surface. Question for the author.

6. **Should the `lorrax_agent` overlay survive at all?** Looking at
   `lx_pool.py`'s 645 lines, the multi-agent A/B/C/D pattern is the
   reason the LoC exists. The alternative is: each agent gets its own
   salloc (4 nodes vs 1 node × 4 allocations). Per `lxalloc 1`'s
   2-hour default, that's 8 node-hours/day per agent — same total
   budget if there were 4 agents on one 4-node alloc. The argument
   *for* sharing is convenience (one allocation to manage); the
   argument *against* is 645 lines of pool-coordination Python and a
   class of bugs (races, stale steps, attach/reap) that wouldn't exist
   otherwise. I'd want to read `KNOWN_SANDBOX_ERRORS.md` and the
   reports under `reports/lorrax_install_maintain_blitz_*` to see if
   pool bugs have actually cost the author hours. Question for the
   author. (Out of scope to chase here — that's a sandbox-design
   call, not a LORRAX-install one.)

7. **`HDF5_USE_FILE_LOCKING=FALSE` is a silent corruption risk on
   non-Lustre. Should the modulefile detect filesystem type and only
   set it when Lustre?** Possible with `stat -f -c %T` (Linux) but
   adds a chunk of Lua / shell. Probably yes for porting; on
   Perlmutter today, `$SCRATCH` is always Lustre, so it's safe — but
   if a user has a config that runs from `$HOME` (NFS-like) the
   silent risk is real.

8. **Why is `set_default_env(platform="gpu")` the only valid call
   path?** The function takes a `platform` kwarg but no caller in the
   tree passes `platform="cpu"`. If the only valid platform is gpu,
   the parameter is dead weight. (Maybe used by the doctor command in
   B1.)

9. **Does `nccl_warmup`'s assumption that all eventual collectives
   match one of its three patterns hold?** The function fires
   full-mesh psum + per-axis psum on a 2D mesh. If LORRAX ever uses
   a different sharding (1D mesh, or psum_scatter / all_to_all with
   different replica groups), the warmup misses and the cost is paid
   live. Not a defect per se — but worth a comment that the warmup
   patterns must be updated when the mesh changes.

10. **What's the test plan for the modulefile itself?** There's an
    `install.sh` that patches and writes the .lua, but no test that
    `module load lorrax` actually exports the expected env-var set.
    Could be a shell-script CI step: in a fresh Lmod env, `module load
    lorrax`, then `env | grep -E '^(LORRAX_|LD_|JAX_|XLA_|HDF5_|MPICH_)' |
    sort > observed.txt`, diff against a checked-in `expected.txt`.
    Catches both modulefile drift and `setenv` regressions. Would
    require a containerized Lmod, but doable.

---

[1]: https://support.hdfgroup.org/documentation/hdf5/latest/_file_lock.html
[2]: https://slurm.schedmd.com/mpi_guide.html
[3]: https://docs.nersc.gov/development/programming-models/mpi/cray-mpich/
[4]: https://slurm.schedmd.com/gres.html
[5]: https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html
[6]: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
