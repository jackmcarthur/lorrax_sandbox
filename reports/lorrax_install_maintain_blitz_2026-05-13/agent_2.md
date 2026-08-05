# Agent 2 — Cray MPI + Shifter container surface

> Slice: container runtime + MPI wiring. What survives if Shifter, Cray MPICH,
> or Cray-flavoured Slurm goes away.

## 1. Scope

This audit covers the layer that gets a JAX process inside a container,
talking GPU-aware MPI to its peers, with the right `LD_LIBRARY_PATH`/PMI
glue: the modulefile `config/modulefiles/lorrax/0.1.0.lua`, the two
in-container shims (`in_container.sh`, `select_gpu.sh`), the
`run_shifter.sh` build wrapper, the bind-mount triad
(`/lorrax_nvhpc`, `/lorrax_phdf5`, `/lorrax_slate`), and the FFI Python
context constructors that consume those mounts at runtime
(`src/ffi/{phdf5,slate,cusolvermp}/context.py`).

Explicitly out of scope: the GW algorithm, anything inside `cohsex.in`,
the FFI C++ kernels themselves, CMake autodetect for cuSOLVERMp (agent
on the build slice owns that), and the agent overlay's pool semantics.
The overlay is touched only where it affects multi-node behaviour the
base module cannot reach.

Two reference clusters are used throughout when calling out portability
breaks:

- **OLCF Frontier** — Apptainer + Cray MPICH (no Shifter; AMD GPUs but
  the runtime contract is what we care about). Frontier already has the
  hybrid bind-mount-CPE-into-container playbook used everywhere below.
- **ALCF Polaris / a generic university Slurm + Apptainer + OpenMPI/UCX
  cluster** — no Cray MPICH, no Slingshot, no `cray_shasta` PMI.

## 2. Current state

The runtime path is one Lua file and three shell scripts:

1. `config/modulefiles/lorrax/0.1.0.lua:60-95` reads site placeholders
   and the LORRAX checkout root.
2. Lines 122-131: sets `HDF5_USE_FILE_LOCKING=FALSE`,
   `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
   `XLA_PYTHON_CLIENT_ALLOCATOR=platform`,
   `TF_GPU_ALLOCATOR=cuda_malloc_async` — all general defaults, none
   NERSC-specific.
3. Lines 150-161 assemble container `LD_LIBRARY_PATH` from a fixed
   five-element ladder: `slate_install/lib64`, `/lorrax_slate/lib`,
   `/lorrax_phdf5/lib`, `/lorrax_nvhpc/<subpath>`, `mpich_container_dir`,
   `mpich_container_dir/dep`, optional darshan. The container mount
   names `/lorrax_*` are **load-bearing strings**: they appear in the
   modulefile, in `run_shifter.sh`, in CMake's `find_package(HDF5)`
   fallback (`CMakeLists.txt:258-263`), and in the FFI's
   `INSTALL_RPATH` (`CMakeLists.txt:418`).
4. Lines 171-199 build the canonical `shifter --image=… --module=… …`
   argv and stuff it into `$LORRAX_SHIFTER`. Every shell function below
   pastes this same string in.
5. Lines 234-243 (`lxalloc`): `salloc --qos=… --constraint=…
   --account=… bash -c "sleep 100000"`.
6. Lines 257-284 (`lxrun`): `srun $jobflag $mpiflag --gres=gpu:N
   -N 1 -n N select_gpu.sh shifter <args> in_container.sh "$@"`.
   **Hardcoded `-N 1`** — the base modulefile cannot launch multi-node
   from lxrun. The agent overlay's lxrun adds `LORRAX_NNODES` but the
   base does not.
7. Lines 296-306 (`lxshell`): same shape, `--pty`, one rank.
8. Lines 312-337 (`lxpre`): three single-rank invocations of the same
   srun-shifter-in_container pattern.
9. `src/ffi/common/cpp/in_container.sh` (5 LoC of code): `export
   MPICH_GPU_SUPPORT_ENABLED=1 ; exec "$@"`. Comment cites
   `/etc/shifter/udiRoot.conf`'s `module_mpich_siteEnvUnset` line as
   the reason — Shifter strips this env at module activation, so we
   restore it inside.
10. `src/ffi/common/cpp/select_gpu.sh` (2 LoC): `export
    CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0} ; exec "$@"`.
11. `src/ffi/common/cpp/run_shifter.sh:42-78`: build-time analog of
    lxrun. Switches on `LORRAX_PHDF5_MPI_STACK={mpich,openmpi}`. The
    mpich branch sets `MPI_TYPE_DEFAULT=cray_shasta`, the openmpi
    branch sets it to `pmix`. **This is the only file that knows how
    to build the FFI against a non-Cray MPI**, and it's the file
    `lxrun` does not consult — runtime and build-time MPI selection
    are split across two files.

FFI Python context objects (`phdf5/context.py`, `slate/context.py`,
`cusolvermp/context.py`) never touch MPI directly from Python — they
hand `jax.process_index()` / `jax.process_count()` to a C++ side that
calls `MPI_Init_thread(MPI_THREAD_MULTIPLE)` and dups
`MPI_COMM_WORLD`. cuSOLVERMp bootstraps over JAX's existing KV store
(no MPI). So the Python side is MPI-implementation-agnostic; the only
MPI dependency lives in the C++ libs the FFI links against, selected
at build time by `run_shifter.sh`'s `LORRAX_PHDF5_MPI_STACK`.

`src/runtime/__init__.py:81-106` resolves the JAX coordinator via
`scontrol show hostnames $SLURM_NODELIST | head -1` + port 12355,
falling back to `SLURMD_NODENAME` / `HOSTNAME`. Cray-agnostic but
Slurm-only.

## 3. NERSC-isms — verdict per cluster

Each row: assumption, where it lives, and what happens on Frontier
(Apptainer + Cray MPICH) and Polaris-class (Apptainer + OpenMPI/UCX).

| # | Assumption                                    | Where                              | Frontier            | Polaris/generic     |
|---|-----------------------------------------------|------------------------------------|---------------------|---------------------|
| 1 | `shifter --image= --module= --volume=` binary | `0.1.0.lua:192-199`, every shell fn| **breaks** (no shifter) | **breaks**       |
| 2 | `shifter --module=gpu,mpich`                  | `site_config.sh:64`, `0.1.0.lua:194`| **breaks** (no module concept) | **breaks**  |
| 3 | `/opt/udiImage/modules/mpich` bind path       | `site_config.sh:91`, `0.1.0.lua:75,154-156`, `run_shifter.sh:63` | replace w/ Apptainer host-MPI bind | replace |
| 4 | `--mpi=cray_shasta`                           | `0.1.0.lua:270-273`, `run_shifter.sh:72` | works (Frontier is HPE Cray) | **breaks** — pmix or pmi2 |
| 5 | `libmpi_gtl_cuda.so.0` GTL preload            | `0.1.0.lua:181`, `slate/scripts/stage_cray.sh:50`, `run_shifter.sh:128-133` | works (also Cray GTL) | **breaks** — no GTL; UCX-CUDA / GDRCopy instead |
| 6 | `MPICH_GPU_SUPPORT_ENABLED=1`                 | `0.1.0.lua:186`, `in_container.sh:13`, `run_shifter.sh:149` | works | **silently no-op** under OpenMPI (env var unknown) |
| 7 | `select_gpu.sh` reads `SLURM_LOCALID`         | `select_gpu.sh:12`, called by every lxrun | works | works on Slurm; **breaks** on PBS/LSF (Polaris uses PBS) |
| 8 | `lfs setstripe` pre-stripe in `lxrun`         | `0.1.0.lua:263-268`                | works (Lustre) | depends — GPFS no-op, but `lfs` simply missing → guarded `command -v` |
| 9 | NVHPC subpath `0.7.2_cuda12.9/...` baked in   | `site_config.sh:87`, `build.sh:24` | needs re-stage | needs re-stage |
|10 | `--constraint=gpu` / `interactive` QOS        | `site_config.sh:51-54`, `0.1.0.lua:240-241` | constraint name differs | site-specific |
|11 | `--account=m2651`                             | `site_config.sh:48`                | site-specific       | site-specific       |
|12 | 4 GPUs/node default `LORRAX_NGPU=4`           | `site_config.sh:57`                | Frontier: 8 GCDs → wrong default | Polaris: 4 ok |
|13 | `$SCRATCH` exists                             | `site_config.sh:100-102`           | works ($MEMBERWORK) — but env var name differs; falls back to `$HOME` in Lua | likely set |
|14 | `--volume=` source restricted to `/pscratch`  | `phdf5/scripts/stage_cray.sh:5-11`, `slate/scripts/stage_cray.sh:8-12` | not relevant (no Shifter) | not relevant |
|15 | Container image `nvcr.io/nvidia/jax:25.04-py3`| `site_config.sh:32`                | needs ROCm-side image | works (NVIDIA) |
|16 | `/global/u2` siteFs for `select_gpu.sh` path  | `0.1.0.lua:221-222`, `run_shifter.sh:181` | breaks (different home FS) | breaks |
|17 | NCCL: `NCCL_NET_PLUGIN=ofi` etc not set       | absent — TODO in PORTING.md gotchas| needed on Frontier (libfabric AWS-OFI plugin for HPE Slingshot via libfabric/ROCm version) | UCX_NET_DEVICES etc |
|18 | `-N 1` hardcoded in lxrun                     | `0.1.0.lua:279`                    | breaks any multi-node usage | same |
|19 | JAX coordinator: `scontrol show hostnames`    | `runtime/__init__.py:95-100`       | works | works on Slurm |
|20 | `LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0` is *unconditional* even when `lxshell`'s single rank doesn't need it | `0.1.0.lua:181` | works | **breaks**: the preload file doesn't exist on a non-Cray slate stage → loader prints `cannot preload` and continues, but dl warnings clutter every invocation |

The interesting picture: ~20 of these are Perlmutter/Shifter glue, but
roughly half (1, 2, 3, 5, 6, 10-12, 14-16, 20) actively *break* on a
non-NERSC site; the rest (4, 7, 9, 13, 17, 19) work but with caveats.
Items 4 and 18 are the cheapest wins.

## 4. Defect catalog

Each entry: `file:line` + tag(s) + one-paragraph diagnosis.

### D1. `lxrun` hardcodes single node — **[COMPAT][FRAGILE]**
`0.1.0.lua:279` — `srun … -N 1 -n ${ngpu} …`. The base modulefile
**cannot** launch a multi-node job. The agent overlay
(`modulefiles/lorrax_agent/1.0.lua`) silently fixes this with
`LORRAX_NNODES`, so anyone using the sandbox overlay sees it work and
infers the base supports it too. A second user installing only the
upstream module gets a one-node tool. PORTING.md doesn't mention
this. Tagged COMPAT because installing on Frontier/Polaris without
realising you also need the overlay is a high-probability foot-gun.

### D2. `--mpi=cray_shasta` hardwired into the build wrapper — **[COMPAT]**
`run_shifter.sh:72`, `0.1.0.lua:73`. `site_config.sh:68` carries the
default but the openmpi branch in `run_shifter.sh:58` is the only
place a non-Cray value gets written. Worse, `run_shifter.sh` and
`0.1.0.lua` make this choice **independently** — one is the build's
view, the other the runtime view, and PORTING.md doesn't reconcile
them. On Polaris with OpenMPI, you'd need both files set to `pmix` or
unset `--mpi=` entirely.

### D3. Hardcoded container mount paths `/lorrax_{nvhpc,phdf5,slate}` — **[LOC-COST]**
`0.1.0.lua:154-156`, `0.1.0.lua:195-197`, `CMakeLists.txt:258-263`,
`CMakeLists.txt:418`, `run_shifter.sh:100-104`, `phdf5/cpp/ctx.h`
defaults, every `stage_*.sh` script. The names are arbitrary but the
fact that every component independently knows them is the cost. An
Apptainer port that picks `/opt/lorrax_nvhpc` for clarity has to
edit at least four files.

### D4. `LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0` set unconditionally — **[COMPAT][FRAGILE]**
`0.1.0.lua:181`. The preload is needed because Shifter's
`--module=mpich` injects a CUDA-11-built copy that we mask. On
Apptainer/Singularity there is no Shifter mpich module and no
masking — the LD_PRELOAD path either does not exist (loader prints a
warning) or, if a stage_cray.sh was run, refers to a Cray GTL that
isn't appropriate on a non-Cray fabric. On Polaris with OpenMPI/UCX
the file shouldn't exist at all. There is no env-controlled disable;
to remove it you have to edit the modulefile.

### D5. `MPICH_GPU_SUPPORT_ENABLED=1` set with no guard for non-MPICH MPIs — **[COMPAT]**
`0.1.0.lua:186`, `in_container.sh:13`, `run_shifter.sh:149`. Harmless
no-op under OpenMPI but it signals (a) the runtime assumes Cray
MPICH semantics and (b) the symmetric OpenMPI knob
(`OMPI_MCA_pml=ucx`, `OMPI_MCA_btl=^openib`,
`UCX_TLS=…,cuda_copy,gdr_copy,cuda_ipc`) is absent. CUDA-aware
OpenMPI on UCX needs those env vars set; today the module ignores
that case.

### D6. `in_container.sh` exists *only* to undo a Shifter quirk — **[LOC-COST][COMPAT]**
`in_container.sh:1-14`. The script's whole job is to re-export
`MPICH_GPU_SUPPORT_ENABLED=1` because `/etc/shifter/udiRoot.conf`'s
`module_mpich_siteEnvUnset` strips it. On Apptainer/Enroot there is
no such stripping, so this script is dead code; but lxrun still
invokes it. Tagged LOC-COST because every porter has to read this
file and realise it's NERSC-specific magic, not a generic "set up
container env" hook. COMPAT because if a porter copies it verbatim
into an Apptainer pipeline they get a confusing extra fork between
srun and the user command for no reason.

### D7. `select_gpu.sh` reads `SLURM_LOCALID` — **[COMPAT][FRAGILE]**
`select_gpu.sh:12`. Polaris uses PBS, where the equivalent is
`PMI_LOCAL_RANK`/`PMIX_LOCAL_RANK`. Anyone porting to PBS gets all
ranks pinned to GPU 0 (since `SLURM_LOCALID` is unset → fallback to
0). The script doesn't fail loudly; it silently breaks GPU
distribution.

### D8. `--volume=` source restriction → `stage_*.sh` exists at all — **[LOC-COST]**
`phdf5/scripts/stage_cray.sh:5-11`, `slate/scripts/stage_cray.sh`,
`cusolvermp/scripts/stage_pypi.sh`. Three "copy 80 MB of vendor libs
into `$SCRATCH`" scripts because Shifter's `udiRoot.conf` blocks
`--volume` from `/opt/cray`. Apptainer/Enroot have no such
restriction — they will happily bind-mount `/opt/cray/pe` directly.
The three stage scripts are pure NERSC-coping mechanism and add
~250 LoC + a 12-MB copy step to every new install. A porter on
Frontier would skip them entirely; one on a non-Cray cluster would
rewrite them. PORTING.md should make that explicit in step (2)/(3)
of the checklist.

### D9. `stage_cray.sh` libreadline.so.7 → libreadline.so.8 ABI shim — **[FRAGILE]**
`slate/scripts/stage_cray.sh:62-66`. Cray libsci's
`liblustreapi.so.1` transitively NEEDs `libreadline.so.7` which the
JAX container has only as `.so.8`. A symlink works today (readline 8
is binary-compatible with the surface lustreapi actually touches),
but this is precisely the kind of "works on this Cray Driver
release" assumption that silently breaks on the next vendor bump.
Tagged FRAGILE because nothing tests it.

### D10. `/opt/hpcx/ompi/lib` fallback in CMakeLists — **[FRAGILE]**
`CMakeLists.txt:301-316`. If `LORRAX_MPICH_LIB_DIR` is unset, CMake
silently falls back to `/opt/hpcx/ompi/lib` (HPC-X OpenMPI inside the
container) and the .so ends up with `DT_NEEDED libmpi.so.40` —
incompatible with `--module=mpich`. `build.sh:33-45` adds a guard
that aborts when the var is unset, but the guard is bypassable
(`LORRAX_FFI_ALLOW_DEFAULT_MPI=1`) and only protects the build.sh
entry point — anyone running `cmake … && ninja` by hand walks right
into it. The KNOWN_SANDBOX_ERRORS.md 2026-05-10 entry referenced in
the comment is institutional memory, not a test.

### D11. Image `nvcr.io/nvidia/jax:25.04-py3` not digest-pinned — **[FRAGILE]**
`site_config.sh:32`. The tag is mutable in the NVIDIA registry. A
re-pull six months from now might give a different image; reproducing
a 2026-05 run after a 2027-05 image rebuild requires guessing the
old digest. Shifter has a per-cluster image cache that papers over
this on Perlmutter; a porter using Apptainer pulls fresh.

### D12. Supplemental site-packages dir built once, by hand — **[FRAGILE][LOC-COST]**
`site_config.sh:25` →
`$HOME/scratchperl/.isdf/isdf_venvs/isdf_site`. The list of pip
packages (`h5py scipy matplotlib contourpy …`) lives in a comment
inside `site_config.sh:21-24`. No `requirements.txt`, no version
pins, no rebuild script. A second user has to manually `pip install
--target=…` the list and hope dependency resolution gives them the
same set.

### D13. `lxshell` keeps `LD_PRELOAD` set even though it never goes through `in_container.sh` — **[FRAGILE]**
`0.1.0.lua:296-306`. Comment on line 184 acknowledges that lxshell
doesn't run `in_container.sh` and routes `MPICH_GPU_SUPPORT_ENABLED`
through `--env=` instead. But the same comment doesn't explain why
the GTL `LD_PRELOAD` is *also* via `--env=`. On a single-rank
lxshell the preload is unnecessary; on a non-Cray site it's harmful
(D4). Tagged FRAGILE — the lxshell path has its own implicit MPI
assumptions invisible to the user.

### D14. `family("lorrax")` swap is single-shell, not single-machine — **[FRAGILE]**
`0.1.0.lua:56`. Comment lines 52-55 say the swap protects against
mixed state. But two shells can each load `lorrax_A` and `lorrax_B`
respectively and share an allocation via `lxattach`; if the two
checkouts have diverged FFI .so layouts, the resulting `srun
shifter` invocations use different `LD_LIBRARY_PATH` strings against
the same compute node. Tagged FRAGILE not COMPAT because it doesn't
break a port — it's a latent bug in NERSC usage.

### D15. JAX coordinator port `12355` hardcoded — **[FRAGILE]**
`runtime/__init__.py:100,106`. Single port, no fallback. Two
LORRAX processes started on the same login/compute node without
proper `SLURM_NODELIST` (e.g. inside `lxshell`-then-manually
launched second process) collide. On a multi-tenant compute node
without per-job network namespace isolation, collision is also
possible across users.

### D16. No `--dry-run` for `lxrun` — **[LOC-COST]**
The shell function is a multi-line srun pipeline with bind-mounts,
env injection, LD_PRELOAD, in-container fork. Debugging "why does my
rank see the wrong GTL?" requires reading
`set_shell_function("lxrun", […])` in Lua and mentally interpolating
local variables. No way to print the materialised command. Every
porter rediscovers this.

### D17. The agent-overlay `LORRAX_NNODES` is the multi-node path, but PORTING.md doesn't mention it — **[COMPAT]**
`reports/lorrax_install_maintain_blitz_2026-05-13/CONTEXT.md` itself
points out the overlay exists; `PORTING.md:54-60` claims `module
load lorrax` is all you need. The base module is single-node only
(D1). A reader follows PORTING.md and finds out the hard way.
Tagged COMPAT because a porter sets up the upstream module and then
hits the single-node ceiling without warning.

### D18. `LORRAX_DARSHAN_LIB_DIR` is empty on non-NERSC but the code path still appends to `LD_LIBRARY_PATH` — **[FRAGILE]**
`0.1.0.lua:158-160`. The guard `if darshan_lib_dir ~= ""` works
today. If a porter sets a wrong path (Darshan exists but at a
non-standard layout), the path silently joins `LD_LIBRARY_PATH` even
if Darshan isn't usable. No probing for `libdarshan.so` presence.

### D19. `salloc … bash -c "sleep 100000"` to keep allocation alive — **[LOC-COST]**
`0.1.0.lua:242`. A NERSC habit (interactive allocation kept warm
between agent sessions). Fine on Slurm; meaningless on PBS where
`qsub -I` is the interactive path. A porter staring at this is
likely to delete it ("why does my module sleep for 27 hours?")
unaware that lxrun/lxshell silently depend on `$SLURM_JOBID` being
exported by salloc itself.

### D20. `MPI_TYPE` default split between two files — **[LOC-COST][FRAGILE]**
`run_shifter.sh:72` (build-time) and `0.1.0.lua:73` (runtime). On
Frontier you want both to read `cray_shasta` from one source;
today they're two independent strings tied together by code review.

### D21. NCCL multi-node env not set anywhere — **[COMPAT]**
PORTING.md `Gotchas` line 185-187 acknowledges it ("Multi-node NCCL
needs cluster-specific NCCL env e.g. `NCCL_NET_PLUGIN=ofi`. Not
validated here.") Note that `nccl_warmup` in
`runtime/__init__.py:155-195` *does* fire dummy psums per axis at
init, which would surface multi-node NCCL bring-up failures eagerly
— good for diagnosis but doesn't paper over the missing env. No
`config/<site>` knob exists for NCCL net plugin.

### D22. `mpich_container_dir + "/dep"` is only appended on the mpich branch — **[FRAGILE]**
`0.1.0.lua:156`. The modulefile *always* appends `"/dep"`, but the
existence of that subdir is a property of Shifter's mpich module.
On a non-NERSC port with `LORRAX_SHIFTER_MODULES="gpu"` (no mpich
sub-module) the `/dep` directory doesn't exist; loader just skips
it. Harmless until a stale path in `LD_LIBRARY_PATH` happens to
contain a `libfabric.so.1` that conflicts with the active stack.

### D23. SLATE context's MPI_Comm_dup blanket-assumes `MPI_COMM_WORLD` — **[FRAGILE]**
`slate/context.py:74-80` + `cpp/context.cc`. The Python side passes
only `(rank, world, p, q)`; the C++ side fishes
`MPI_COMM_WORLD` out and dups it. Hardcoded to the single-comm-per-
process model. If a future hybrid run launches multiple JAX
processes inside one MPI world (e.g. sharing a comm with a non-JAX
analyser), this can't accommodate it. Not blocking today; flagged.

### D24. cuSOLVERMp NCCL UID broadcast key namespaced by mesh only — **[FRAGILE]**
`cusolvermp/context.py:83-85`. The unique-id KV key is
`lorrax_ffi/cusolvermp/nccl_unique_id/v0/{p}x{q}/{layout}`. If two
LORRAX-driven processes in the same JAX coordinator (multi-program
on the same allocation) both ask for the same `(p,q,layout)`, they
collide on the KV broadcast. Today there's one driver per
allocation so it's fine; flagged for the future.

## 5. Blitz proposals (ranked by leverage)

For each: file(s) touched, one-day scope, what locks it in.

### B1. Unify the MPI-type knob into one file
**Touches** `site_config.sh`, `0.1.0.lua`, `run_shifter.sh`,
`build.sh`. **Change**: introduce a single
`LORRAX_MPI_TYPE_DEFAULT` (already in `site_config.sh:68`) and have
`run_shifter.sh:51-78` *read it* instead of branching on
`LORRAX_PHDF5_MPI_STACK={mpich,openmpi}` with its own opinion. Map
`mpich→cray_shasta`, `openmpi→pmix` as defaults *inside one
table*. **Addresses** D2, D20. **Why**: the install-elsewhere
checklist becomes "edit one variable". **Risk**: stale callers of
`LORRAX_PHDF5_MPI_STACK` need to be repointed. **Test**: a 4-line CI
that loads the module, prints `$LORRAX_SHIFTER`, greps for
`--mpi=…`, asserts it matches site_config.

### B2. `lxrun --dry-run` / `lxrun --print`
**Touches** `0.1.0.lua` (lxrun, lxshell, lxpre shell functions; a
new `lxprint` or `--dry-run` prefix). **Change**: emit the full
materialised srun + shifter argv + bind-mounts + env injections to
stdout and exit. **Addresses** D3, D4, D16. **Why**: every porter's
first 30 minutes spent reverse-engineering this Lua becomes 1
minute. **Risk**: none — additive. **Test**: a single shell test
that does `LORRAX_NGPU=2 lxrun --dry-run python3 -V` and greps for
`--mpi=cray_shasta` and `/lorrax_phdf5`.

### B3. `Apptainer.def` companion + `config/apptainer/`
**Touches** new file `config/apptainer/Apptainer.def`, new
`config/apptainer/site_config.sh`, new
`config/apptainer/install.sh`, and a `LORRAX_CONTAINER_RUNTIME` env
that the Lua reads to swap `shifter` for `apptainer exec --nv
--bind …`. **Change**: a single Apptainer image that mirrors the
NVIDIA JAX content minus the bind-mount needs (or with a leaner
contract). Modulefile composes either `shifter <args>` or
`apptainer exec <args>` based on the runtime knob. **Addresses**
D1, D3, D4, D6, D8, D11. **Why**: the most NERSC-specific surface
(Shifter) becomes one of two backends, and a porter on
Frontier/Polaris has a documented path. **Risk**: highest of these
proposals — needs cluster access to validate. **Test**: image
build under `apptainer build` succeeds in CI on a runner with a
recent Apptainer; non-MPI smoke test (just `jax.devices()`).
Hybrid host-MPI bind-mount has to be done at runtime regardless.

### B4. Drop `in_container.sh` and `select_gpu.sh` in favour of `--env=` + Slurm task-prolog
**Touches** `0.1.0.lua` (remove the two file paths from every shell
function), `in_container.sh` (delete), `select_gpu.sh` (delete or
move to a contrib dir). **Change**: pass
`--env=MPICH_GPU_SUPPORT_ENABLED=1` (already done — line 186) and
`--env=CUDA_VISIBLE_DEVICES_PER_RANK=auto` + use a `srun
--gpu-bind=closest` / `--gpus-per-task=1` strategy that doesn't
require a wrapper script. **Addresses** D6, D7. **Why**: removes
two LoC-costing wrappers that are pure NERSC-coping. **Risk**:
`--gpus-per-task` broke JAX topology sync in the past
(`run_shifter.sh:163-167` comment). Needs validation that the
2026-current JAX `local_device_ids` path tolerates per-task cgroups.
If it doesn't, keep `select_gpu.sh` but make it PBS-aware (read
`PMI_LOCAL_RANK` as a fallback).

### B5. Make container mount paths configurable, not hardcoded
**Touches** `0.1.0.lua`, `CMakeLists.txt`, `run_shifter.sh`,
`stage_*.sh`. **Change**: introduce `LORRAX_CONTAINER_NVHPC_PATH`
(default `/lorrax_nvhpc`), `LORRAX_CONTAINER_PHDF5_PATH`,
`LORRAX_CONTAINER_SLATE_PATH`. CMake reads
`LORRAX_CONTAINER_PHDF5_PATH` instead of `/lorrax_phdf5`.
**Addresses** D3. **Why**: a porter doesn't have to grep for
`/lorrax_*` across the tree. **Risk**: CMake `INSTALL_RPATH`
needs the var at build time; document the contract. **Test**:
CMake-time message echoes the resolved paths; PORTING.md adds one
table row.

### B6. Pin image by digest
**Touches** `site_config.sh:32`, `PORTING.md`. **Change**: pin to
`nvcr.io/nvidia/jax:25.04-py3@sha256:…` and document how to
re-pin. **Addresses** D11. **Why**: reproducibility, no risk
beyond mild ugliness in the tag. **Test**: install.sh asserts
the configured tag matches `shifter --image=… --verbose`'s
reported digest after `shifterimg lookup`. **Risk**: Shifter on
Perlmutter may not support digest-pinned image refs (verify;
`shifterimg pull` historically takes tag only).

### B7. `requirements.txt` for the supplemental site-packages
**Touches** new `config/lorrax_site/requirements.txt`,
`site_config.sh:21-25` (replace pip-by-hand comment with a
`make-site` target or one-liner pointing at the requirements).
**Addresses** D12. **Why**: a second user gets the same Python
deps. **Test**: a script that runs `pip install --target=$tmp
-r requirements.txt` and asserts `h5py`, `scipy`, `matplotlib`
import.

### B8. Guard `LD_PRELOAD` and PMI knobs by content of `LORRAX_MPI_TYPE`
**Touches** `0.1.0.lua:181-186`. **Change**: only set the GTL
preload + `MPICH_GPU_SUPPORT_ENABLED` when
`LORRAX_MPI_TYPE` starts with `cray` / `mpich`; emit the
OpenMPI/UCX equivalents (`UCX_TLS=…,cuda_copy,gdr_copy,cuda_ipc`,
`OMPI_MCA_pml=ucx`) when it's `pmix`. **Addresses** D4, D5.
**Why**: every cluster gets a sensible default without rewriting
the modulefile. **Test**: dry-run from B2 + grep.

### B9. `--print-bind-mounts` helper in `config/`
**Touches** new `config/print_mounts.sh` (~30 LoC). **Change**:
print the three host paths (NVHPC, phdf5, SLATE), three container
paths, three SONAME signatures (`readelf -d
$LORRAX_FFI_NVHPC_DIR/.../libcusolverMp.so.0`), and the active
`LORRAX_MPI_TYPE`. **Addresses** D8, D9, D10 (diagnostic surface).
**Why**: makes the implicit contract between stage scripts and
modulefile explicit. **Test**: invoked by `module load`; on
missing stage, print a one-line warning.

### B10. Document multi-node base path or wire it
**Touches** `0.1.0.lua:279` and/or `PORTING.md`. Either: (a) add
`LORRAX_NNODES` to the base lxrun (port over the agent overlay's
logic minus the pool coordination) and document; or (b) document
clearly that the base module is single-node and that multi-node
requires the agent overlay's wrapper. **Addresses** D1, D17.
**Why**: install.sh users on a non-NERSC site don't have the
overlay. **Risk**: the overlay's free-node selection is the right
solution for the shared-allocation sandbox but is overkill for
single-user batch use; option (a) probably wants a stripped-down
multi-node lxrun without pool logic.

Ranking by leverage (installability/maintainability):

1. **B2 (lxrun --dry-run)** — cheapest, unlocks B5 and B8 review.
2. **B1 (unify MPI knob)** — removes one cross-file invariant.
3. **B8 (preload/PMI guard)** — makes the modulefile actually
   honour `LORRAX_MPI_TYPE`.
4. **B10 (multi-node base)** — closes a hidden ceiling.
5. **B5 (config'd container mounts)** — removes hardcoded
   `/lorrax_*` strings.
6. **B7 (requirements.txt)** — small install-elsewhere win.
7. **B3 (Apptainer.def)** — high-leverage but high-risk; needs
   real porting work.
8. **B9 (print_mounts.sh)** — diagnostic, nice-to-have.
9. **B4 (drop in_container.sh / select_gpu.sh)** — depends on JAX
   tolerating `--gpus-per-task`; risk > 1 day if it doesn't.
10. **B6 (digest-pin image)** — small reproducibility win,
    contingent on Shifter feature support.

## 6. Open questions

Honest unknowns, ranked by how much they would change the above.

### Q1. Does Shifter on Perlmutter actually support digest-pinned image references?
B6 depends on it. The `shifterimg lookup` and `shifter --image=`
historically take a tag, not a digest. I did not find authoritative
NERSC docs saying digest pinning works end-to-end through the
udiRoot resolver. **Cannot resolve without a live Shifter shell.**

### Q2. Does `--gpus-per-task=1` plus JAX `local_device_ids=[0]` actually work in 2026?
`run_shifter.sh:163-167` comment says it broke JAX topology sync in
the past; B4 assumes a current JAX (post `jax.distributed`
stabilisation) tolerates it. The comment is undated. The fact that
`select_gpu.sh` still exists and lxrun still routes through it
suggests no one has retested. **Cannot resolve without running
`jax.distributed.initialize()` on a 2-rank node with
`--gpus-per-task=1` and checking `jax.devices()` returns the right
local device.**

### Q3. What is the actual ABI contract of `/opt/udiImage/modules/mpich`?
PORTING.md (line 196) cites the NERSC docs but doesn't enumerate
the contract. Concretely: is the libmpi.so.12 inside Shifter's
mpich module always pinned to a particular Cray MPICH minor? When
NERSC rolls out Cray MPICH 9.1, does the SONAME flip? The stage
scripts include shim files (`libmpi_gnu_91.so.12`,
`libmpi_gnu_110.so.12`, `libmpi_gnu_123.so.12` — see
`phdf5/scripts/stage_cray.sh` SHIM_TARGET note) that suggest the
SONAME has changed at least three times across compiler versions.
**Not resolvable without NERSC release-note tracking; flagged for
a future "library SONAME canary" in the FFI build.**

### Q4. Does Apptainer on Frontier need a separate libfabric provider list?
HPE Slingshot 11 OFI provider configuration on Frontier (cxi
provider) differs from Polaris (cxi or psm3). B3's Apptainer
hybrid model would need site-specific FI_PROVIDER values per
cluster. The current LORRAX runtime sets nothing fabric-related.
**Cannot resolve without Frontier access; should be a future
`config/frontier/site_config.sh` knob.**

### Q5. Does `nccl_warmup` (runtime/__init__.py:155) work on Frontier with AWS-OFI-NCCL plugin?
`NCCL_NET_PLUGIN=ofi` plus `NCCL_DEBUG=INFO` should make the warmup
psum surface a multi-node-OFI bring-up trace. Today this is
untested. **Cannot resolve without a multi-node Slingshot
allocation outside NERSC.**

### Q6. Is `MPICH_GPU_SUPPORT_ENABLED=1` necessary inside Shifter when the FFI sends pointers, given JAX itself doesn't go through libmpi for collectives (NCCL handles that)?
Only SLATE and phdf5 actually send device pointers through
`MPI_Send`/`MPI_File_write`. cuSOLVERMp's NCCL communicator is
distinct. If phdf5 always does independent writes (PORTING.md
line 138-143) and SLATE's GPU traffic could be re-routed through
NCCL too, the entire libmpi_gtl_cuda / `MPICH_GPU_SUPPORT_ENABLED`
machinery could be conditional on "is SLATE in this run". **Worth
profiling; would shrink the bind-mount surface significantly if
true.**

### Q7. Does the JAX coordinator port 12355 ever collide in practice?
D15 is theoretical. The Lua sets nothing port-related; the
`SLURM_NODELIST[0]` resolution implicitly assumes the chosen host
isn't running another LORRAX process. Two `lxshell` users on the
same compute node would collide. **Easy fix is `random.randint`
seeded by `$SLURM_JOBID`; flagged but not in the blitz list above
because I couldn't reproduce the collision.**

### Q8. Is `nvcr.io/nvidia/jax:25.04-py3` the canonical baseline, or has the unwritten contract shifted?
PORTING.md (line 14) says yes. `run_shifter.sh:45` defaults to it.
But the Lua's `image = "@LORRAX_IMAGE@"` reads from site_config,
which today is set to the same. A second user uses whatever's
currently on the registry — and the JAX/CUDA pairing inside that
image determines whether `cusolverMp 0.7.2_cuda12.9` even links.
**The pairing of (image tag, NVHPC subpath, NCCL version) is an
implicit invariant. A pinning manifest would help; covered by B6
and B7 but those don't capture the three-way constraint.**

---

`Agent 2 done — see agent_2.md`

### Sources

- [Slurm MPI Users Guide — `--mpi` plugin types incl. `cray_shasta`, `pmi2`, `pmix`](https://slurm.schedmd.com/mpi_guide.html)
- [Apptainer / MPI applications — host-MPI bind-mount (hybrid) model](https://apptainer.org/docs/user/latest/mpi.html)
- [OLCF Frontier — containers + bind-mounting CPE / Cray MPICH ABI compat](https://docs.olcf.ornl.gov/software/containers_on_frontier.html)
- [HPE Cray MPICH — `MPICH_GPU_SUPPORT_ENABLED`, GTL, GPUDirect on Slingshot](https://cpe.ext.hpe.com/docs/latest/mpt/mpich/intro_mpi.html)
- [NERSC Cray MPICH usage notes](https://docs.nersc.gov/development/programming-models/mpi/cray-mpich/)
- [Open MPI 5.0 — CUDA support via UCX (`UCX_TLS=cuda_copy,gdr_copy,cuda_ipc`)](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html)
- [CIQ — A new approach to MPI in Apptainer (bind-mount libfabric, MPI, host libs)](https://ciq.com/blog/a-new-approach-to-mpi-in-apptainer/)
- [ALCF Polaris — Apptainer container guidance](https://docs.alcf.anl.gov/polaris/containers/containers/)
