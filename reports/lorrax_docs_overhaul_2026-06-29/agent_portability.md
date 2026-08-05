# LORRAX docs review — Portability lens

**Agent:** portability (build LORRAX free-standing on a non-NERSC cluster)
**Date:** 2026-06-29
**Scope:** Every Perlmutter/NERSC/jackm-specific assumption baked into the docs and
build/run scripts that blocks a foreign-cluster build, plus what a proper portable
"Installation / Build from source" guide must cover.
**Mode:** READ + PROPOSE ONLY. No repo files edited.

Files read: `README.md`; `docs/ENVIRONMENT_COMPREHENSIVE.md`;
`config/perlmutter/{install.sh,site_config.sh}`; `config/README.md`;
`config/modulefiles/lorrax/0.1.0.lua`; `src/ffi/common/cpp/{build.sh,run_shifter.sh,
in_container.sh,select_gpu.sh,CMakeLists.txt}`;
`src/ffi/{phdf5/scripts/stage_cray.sh,phdf5/scripts/stage_openmpi.sh,
slate/scripts/stage_cray.sh,cusolvermp/scripts/stage_nvhpc.sh}`;
`src/ffi/{AGENTS.md,PORTING.md}`; `pyproject.toml`.

---

## Executive summary

A researcher at a generic university HPC center cannot build or run LORRAX from the
current docs. The path from `git clone` to a working multi-GPU GW run is gated by
**four undeclared, unpackaged native libraries** (cuSolverMp, parallel HDF5, SLATE,
plus a GPU-aware MPI transport) whose only documented acquisition mechanism is
"copy them out of NERSC's `/opt/cray` and `/opt/nvidia` trees into `$SCRATCH` and
bind-mount them into a NERSC-Shifter container." Every layer of that mechanism is
Perlmutter-specific:

- The container runtime is **Shifter** (NERSC-only); the image is a single hardcoded
  tag `nvcr.io/nvidia/jax:25.04-py3`.
- The MPI is assumed to be **Cray MPICH** reachable at the NERSC-specific container
  path `/opt/udiImage/modules/mpich`, bootstrapped with `--mpi=cray_shasta`.
- The dependency staging scripts `cp` from **NERSC's exact `/opt/cray/pe/...` and
  `/opt/nvidia/hpc_sdk/...` paths** and shim **Cray-PE-specific SONAMEs**
  (`libmpi_gnu_{91,110,123}.so.12`, `libmpi_gtl_cuda.so.0`).
- Defaults assume the **`$SCRATCH` / `$HOME` NERSC filesystem layout** and the
  **`m2651` allocation / `interactive` QOS / `gpu` constraint** SLURM vocabulary.
- Several scripts contain **literal `/global/.../jackm/...` absolute paths** as
  fallbacks.

The docs *acknowledge* portability in three places — `ENVIRONMENT_COMPREHENSIVE.md §7`
("Generic SLURM clusters"), `config/README.md §Porting`, and `src/ffi/PORTING.md` — but
all three describe porting as *"edit `site_config.sh` and re-run a stage script,"* which
silently assumes the foreign cluster has Cray PE modules to copy from. They never tell a
researcher **how to obtain or build cuSolverMp + parallel HDF5 + SLATE from scratch** on a
system that has none of NERSC's vendor trees. That is the real portability gap, and no
current doc fills it.

The single most important structural fix is a **provider-agnostic "Installation / Build
from source" page** that (a) declares the native deps with a support matrix, (b) gives
real acquisition recipes for each (PyPI wheel / spack / conda-forge / build-from-source),
and (c) decouples LORRAX from Shifter by documenting Apptainer and bare-venv runtime
paths as first-class, not as a two-line afterthought.

---

## A. Small immediate doc fixes

Concrete, low-effort corrections to current environment-related docs. (current → problem
→ fix.)

### A1. README quick-start uses a module path that does not exist as written, and a NERSC-only "every session" line with no portable counterpart
- **Location:** `README.md:23` and `README.md:26`.
- **Current:** Quick start shows `uv run python -m gw.gw_jax -i cohsex.in`; line 26 is
  the only "how to actually run it" guidance: *"On NERSC Perlmutter: `module load lorrax`
  then use `lxrun`/`lxpre`."*
- **Problem:** The single run-it instruction in the README is NERSC-Shifter-only. A
  reader on any other cluster has no entry point. (Also note `README.md:14` references
  the driver as `gw_isdf/gw_jax.py`, but the actual module is `src/gw/gw_jax.py` —
  `gw_isdf/` does not exist; the package is `gw`. Confirmed: `src/gw/gw_jax.py` present,
  `src/gw_isdf/` absent.)
- **Fix:** Add a generic non-NERSC run line next to line 26, e.g. *"On a generic SLURM +
  GPU cluster, see [Installation → From source] and use the bare-venv launcher."* Fix the
  `gw_isdf/gw_jax.py` reference at line 14 to `gw/gw_jax.py`.

### A2. `ENVIRONMENT_COMPREHENSIVE.md` opens "For AI agents" and never addresses an end-user installer
- **Location:** `docs/ENVIRONMENT_COMPREHENSIVE.md:3`.
- **Current:** *"**For AI agents**: Dependencies, installation, JAX configuration..."*
- **Problem:** The one doc a foreign-cluster builder would open for "installation" is
  framed for in-house agents, not external users. Tone signals "internal note," and the
  whole document is structured around Perlmutter being authoritative
  (`config/README.md` is "authoritative — this file summarises").
- **Fix:** Re-target the header to human installers; move the agent-orientation note to a
  separate developer doc. Make this page the *generic* installation reference and demote
  Perlmutter specifics to one clearly-labeled cluster appendix.

### A3. §1.1 declares only Python deps as "dependencies"; the four native libs are invisible until §5
- **Location:** `docs/ENVIRONMENT_COMPREHENSIVE.md:20-22` (§1 "Dependencies", "Authoritative
  source: pyproject.toml").
- **Current:** §1 lists Python packages and says `pyproject.toml` is authoritative; the
  native FFI stack (cuSolverMp, parallel HDF5, SLATE, GPU-aware MPI) is not mentioned as a
  dependency until §5, ~200 lines later.
- **Problem:** A reader scanning "Dependencies" concludes LORRAX is a pure-Python+JAX
  package. It is not: distributed `eigh`, sharded HDF5 I/O, and distributed Cholesky/trsm
  all require native libs that `pyproject.toml` does **not** declare. The biggest
  onboarding cliff (no `liblorrax_ffi.so` in a fresh clone) is invisible here.
- **Fix:** Add a "§1.3 Native (FFI) dependencies" subsection up front: a table of
  cuSolverMp / parallel-HDF5 / SLATE / MPI with "required for which feature," a one-line
  "these are NOT in `pyproject.toml` and must be obtained separately (see Installation →
  Native stack)," and an explicit note that a fresh clone has no `liblorrax_ffi.so` and
  must be built.

### A4. §1.1 hard-pins `jax[cuda13]>=0.9.0` but the production image runs JAX ~0.5.3 (CUDA 12)
- **Location:** `docs/ENVIRONMENT_COMPREHENSIVE.md:29` and `pyproject.toml:9`; cross-check
  `src/ffi/PORTING.md:14` ("JAX with `jax.ffi` ≥ 0.5") and the image
  `nvcr.io/nvidia/jax:25.04-py3`.
- **Current:** Runtime deps table and `pyproject.toml` require `jax[cuda13]>=0.9.0` (CUDA
  13), but the only documented working container is `nvcr.io/nvidia/jax:25.04-py3`, which
  ships JAX in the 0.5.x / CUDA-12 line, and the FFI explicitly links **CUDA 12.9**
  (`CUSOLVERMP_*`, `NVHPC_CUDA=12.9`). `PORTING.md:14` says JAX ≥ 0.5 suffices.
- **Problem:** A foreign builder who follows `pyproject.toml` installs a CUDA-13 JAX that
  is ABI-incompatible with the CUDA-12-linked FFI `.so` and the documented container. The
  declared and actual stacks contradict each other.
- **Fix:** Reconcile to one truth. Either (a) state explicitly "the `uv`/PyPI path targets
  CUDA 13 + JAX ≥ 0.9; the container path targets CUDA 12 + JAX 0.5.x; these are different
  supported configurations" in a support matrix, or (b) relax the pin to a range that
  matches the FFI's CUDA-12 link. Today they are simply inconsistent.

### A5. §7 "Generic SLURM clusters" omits the entire native-stack acquisition story
- **Location:** `docs/ENVIRONMENT_COMPREHENSIVE.md:348-378` (§7).
- **Current:** §7 is the "non-NERSC" section. It says: port via `config/<cluster>/`, edit
  `site_config.sh` knobs (table at 352-362), swap the `shifter` invocation for
  Apptainer/Singularity/bare-venv (364), and gives a bare-venv `sbatch` example (366-378).
- **Problem:** The bare-venv example (`module load cuda/12.3 python/3.12`, `pip`, run) only
  works for the **pure-JAX path with no FFI**. It silently omits that on a non-Cray cluster
  you must *build* cuSolverMp/HDF5/SLATE access and a matching `liblorrax_ffi.so` — the
  `site_config.sh` knobs at 352-362 (`LORRAX_NVHPC_SUBPATH`,
  `LORRAX_MPICH_CONTAINER_DIR`, `LORRAX_DARSHAN_LIB_DIR`, the `LORRAX_FFI_*_DIR` stage
  roots) are **meaningless on a cluster without NERSC's `/opt/cray` layout to stage from.**
  The reader is told "just edit the knobs" with no source for the values.
- **Fix:** Either fold §7 into a real "From source on a generic cluster" guide (see B1) or,
  minimally, add a prominent admonition: *"The bare-venv example below runs only the
  pure-JAX code path. The distributed FFI features (cuSolverMp eigh, sharded HDF5, SLATE)
  require building `liblorrax_ffi.so` against native libs you must obtain — see [Native
  stack]. The `site_config.sh` stage knobs assume Cray-PE modules to copy from."*

### A6. §5.2 / config/README bind-mount tables still show `$SCRATCH` defaults that the code has already moved to `$HOME/software`
- **Location:** `docs/ENVIRONMENT_COMPREHENSIVE.md:236-240` (§5.2 table) and
  `config/README.md:107-111`. Contrast with the actual defaults in
  `config/perlmutter/site_config.sh:102-104` (now `$HOME/software/...`) and
  `run_shifter.sh:44,61,107` (now `$HOME/software/...`).
- **Current:** Both doc tables list defaults `$SCRATCH/lorrax_nvhpc`,
  `$SCRATCH/lorrax_phdf5_cray/stage`, `$SCRATCH/lorrax_slate_cray/stage`.
- **Problem:** Stale. `site_config.sh:99-104` documents the 2026-06-24 relocation to
  `$HOME/software` (scratch is purged), and that is what the scripts now default to. The
  docs disagree with the shipped defaults — a porter copying the doc value stages into the
  wrong (purgeable) location.
- **Fix:** Update both tables to `$HOME/software/...`. Better: replace the literal default
  with the override-var name only and state "default is cluster-config-dependent; see your
  `site_config.sh`." (Also note: `$SCRATCH`/`$HOME` are themselves NERSC-isms — see B3.)

### A7. §5.3 / §5.4 / AGENTS cold-start instructions are pure NERSC and mutually inconsistent on which phdf5 stage is "default"
- **Location:** `docs/ENVIRONMENT_COMPREHENSIVE.md:252-261` (§5.3, §5.4);
  `src/ffi/AGENTS.md:17-37` ("Cold start (fresh clone on Perlmutter)"); contrast
  `src/ffi/PORTING.md:133-144` and `run_shifter.sh:42`.
- **Current:** §5.3 lists four `stage_*.sh` scripts with `→ /pscratch` comments. AGENTS.md
  cold-start uses `stage_openmpi.sh` and calls it the default; `run_shifter.sh:42` and
  `PORTING.md:134` say **`mpich` is the default** since 2026-04-20; `stage_openmpi.sh:6`
  still calls itself *"the default / verified stack."*
- **Problem:** Three docs give three different "defaults." A foreign builder cannot tell
  which phdf5 stack to set up. All cold-start text is hard-wired to NERSC (`lxalloc`,
  `/pscratch`, `udiRoot.conf`).
- **Fix:** Pick one canonical default in one place (the new Installation page) and have the
  others link to it. Correct `stage_openmpi.sh:6-9` header (it is now the fallback, not the
  default). State the OpenMPI stack is the **portable** one for non-Cray clusters.

### A8. §5.2 LD_LIBRARY_PATH block and §5.6 GPU-aware MPICH are presented as universal but are Cray-MPICH-only
- **Location:** `docs/ENVIRONMENT_COMPREHENSIVE.md:242-250` (LD_LIBRARY_PATH order),
  `281-290` (§5.6 `LD_PRELOAD=libmpi_gtl_cuda.so.0`, `MPICH_GPU_SUPPORT_ENABLED=1`).
- **Current:** Both are written as "what the module sets," with container paths
  `/opt/udiImage/modules/mpich`, the `libmpi_gtl_cuda.so.0` preload, and
  `MPICH_GPU_SUPPORT_ENABLED=1`.
- **Problem:** `libmpi_gtl_cuda` and `MPICH_GPU_SUPPORT_ENABLED` are **Cray MPICH**
  GPU-Direct mechanisms; they do not exist for OpenMPI/UCX or a different MPICH.
  `/opt/udiImage/...` is a Shifter container path. Presented without an "(Cray MPICH only)"
  qualifier, a porter on OpenMPI will chase these dead ends.
- **Fix:** Label this whole subsection "(Cray MPICH stack)" and add the OpenMPI/UCX
  equivalent (`UCX_*`, `OMPI_MCA_*`, CUDA-aware UCX) or at least state "OpenMPI clusters
  use CUDA-aware UCX; none of the `MPICH_*`/`gtl` vars apply."

### A9. `config/README.md` "Quick Start" leads with Perlmutter and never mentions any other cluster until §Porting at the very bottom
- **Location:** `config/README.md:6-17` (Quick Start = Perlmutter) and `:215-244` (Porting).
- **Current:** First thing a reader sees is "Quick Start (Perlmutter)" with `module load
  lorrax`. The generic story is the last section.
- **Problem:** Reinforces "LORRAX = Perlmutter." The doc is structured so the NERSC path is
  the main road and everything else is an exit ramp.
- **Fix:** Re-title "Quick Start (Perlmutter)" → "Quick Start (NERSC Perlmutter — reference
  cluster)" and add one sentence pointing non-NERSC readers to Porting / the Installation
  page first.

### A10. `config/README.md:129` mislabels `LORRAX_MPI_TYPE=pmix` as "legacy OpenMPI path (not wired up)" while it IS the documented OpenMPI runtime elsewhere
- **Location:** `config/README.md:129`; contrast `run_shifter.sh:31,58` and
  `PORTING.md:144` where `pmix` + HPC-X OpenMPI is an actively-supported fallback stack.
- **Current:** config/README says `pmix` is "legacy OpenMPI path (not wired up)";
  ENVIRONMENT §5.5 (`:276`) says it "has hung non-FFI workloads"; `run_shifter.sh`
  treats `pmix` as the real OpenMPI launch protocol.
- **Problem:** Contradictory status for the exact knob a non-Cray cluster (which is the
  OpenMPI majority) most needs. "Not wired up" tells a porter not to use the only MPI flavor
  that works for them.
- **Fix:** State consistently: `pmix` is the launch protocol for the OpenMPI stack; it is
  not the default because the Cray stack is faster on Perlmutter, and it has hung some
  non-FFI workloads — but it is the correct choice on OpenMPI clusters.

### A11. Hardcoded personal absolute paths as script fallbacks
- **Location:** `src/ffi/common/cpp/run_shifter.sh:95` (`LORRAX_SRC` default
  `/global/u2/j/jackm/software/lorrax/src`), `:96` (`LORRAX_SITE` default
  `/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site`), `:108`
  (`SLATE_INSTALL_HOST` default `/global/homes/j/jackm/software/slate/install`);
  `src/ffi/common/cpp/CMakeLists.txt:348` (`LORRAX_SLATE_INSTALL_DIR` default
  `/global/homes/j/jackm/software/slate/install`); `config/perlmutter/site_config.sh:29`
  (`LORRAX_DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources`).
- **Current:** These literal `jackm` home/scratch paths are the *fallback defaults* when the
  env override is unset.
- **Problem:** On any other account/cluster these resolve to nonexistent directories, and
  the failure mode is a confusing "file not found" deep in the build/run, not a clear "set
  this variable." `CMakeLists.txt:348` in particular bakes a personal path into the build
  system itself; `PORTING.md:32` even documents `/global/homes/<u>/software/slate/install`
  as the SLATE auto-probe — a NERSC home layout.
- **Fix:** Replace personal-path fallbacks with either (a) no default + a clear error
  ("set `LORRAX_SLATE_INSTALL_DIR`"), or (b) a derived default under `$PREFIX`/repo-relative
  location. `LORRAX_DEPS` should default empty (it is a sandbox-specific path, not a LORRAX
  dep) — note `site_config.sh.example` should ship with it blank.

### A12. `run_shifter.sh` NVHPC-stage "create it with" hint hardcodes NERSC's `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/...`
- **Location:** `src/ffi/common/cpp/run_shifter.sh:82-93`; same pattern in
  `cusolvermp/scripts/stage_nvhpc.sh:24` (`NVHPC_ROOT` default
  `/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_VERSION}`).
- **Current:** When the NVHPC stage dir is missing, the script prints copy-paste `cp -a
  /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/...` commands.
- **Problem:** Assumes the NVHPC SDK is installed at NERSC's exact path. A cluster that
  provides cuSolverMp via a PyPI wheel (`nvidia-cusolvermp-cu12`, which `site_config.sh:74`
  itself notes is the *working baseline*!) or spack has a totally different layout. The
  "create it with" guidance points at a tree that doesn't exist for them.
- **Fix:** Make the hint reference `$NVHPC_ROOT` (already a variable in
  `stage_nvhpc.sh`) and add the PyPI-wheel and spack acquisition alternatives. See B2.

### A13. SLATE acquisition is mentioned only in passing; no build recipe
- **Location:** `src/ffi/PORTING.md:87-90` (checklist item 3: *"clone icl-utk-edu/slate,
  build against the target MPI + BLAS, install under `$HOME/software/slate/install`"*);
  `slate/scripts/stage_cray.sh` (only stages Cray *runtime* libs, not SLATE itself).
- **Current:** The single from-source SLATE instruction is one sentence with a personal
  install path; there is no actual cmake invocation, no blaspp/lapackpp note beyond "live
  alongside," no GPU-backend flags, no version pin.
- **Problem:** SLATE-from-source is the hardest of the three native deps to build, and it
  gets one sentence. A foreign builder has no concrete recipe.
- **Fix:** Add a real SLATE build recipe (clone + tag, `cmake -Dgpu_backend=cuda
  -Dblas=...`, install prefix as a variable). See B2.

### A14. NCCL multi-node env is flagged "not validated here" with cluster-specific examples and no general guidance
- **Location:** `src/ffi/PORTING.md:185-187`.
- **Current:** *"Multi-node NCCL: needs cluster-specific NCCL env (e.g. `NCCL_NET_PLUGIN=ofi`
  on Perlmutter, `NCCL_IB_HCA=...` on IB fabrics). Not validated here."*
- **Problem:** This is exactly where a foreign cluster (which is far more likely to be
  InfiniBand than Slingshot) needs guidance, and the doc waves it off.
- **Fix:** Add a short "multi-node interconnect" subsection: for IB fabrics set
  `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME`; point to NCCL's own env docs; note JAX's
  `jax.distributed.initialize` coordinator config (already in ENVIRONMENT §6.4) is the
  portable part.

---

## B. Structural / architectural proposals

These are the high-value changes. They restructure the docs and, where noted, the way
dependencies are declared and wired.

### B1. Create a single provider-agnostic "Installation / Build from source" page (the missing front door)

**Effort: L.**

**Problem:** There is no installation page in the JAX/PySCF sense — a page that takes any
reader from "I have a GPU cluster" to "LORRAX runs," with a support matrix and three
copy-paste tracks (pip/uv, container, from source). Installation guidance is currently
scattered across ENVIRONMENT §2/§4/§5/§7, `config/README.md`, `src/ffi/AGENTS.md`, and
`src/ffi/PORTING.md`, each Perlmutter-centric and partially contradictory (see A7, A10).

**Proposal:** A new top-level `docs/installation.md` structured as:

1. **Support matrix** (the thing JAX/PySCF lead with). Rows = supported configurations;
   columns = OS, CUDA major, JAX version, MPI, parallel-HDF5 source, container runtime,
   tested scale. At minimum:

   | Config | OS | CUDA | JAX | MPI | HDF5 source | Runtime | Tested |
   |---|---|---|---|---|---|---|---|
   | NERSC Perlmutter (reference) | SLES 15 | 12.9 | 0.5.x | Cray MPICH | cray-hdf5-parallel | Shifter | 1–4 nodes ×4 A100 |
   | Generic Cray EX | — | 12.x | 0.5.x | Cray MPICH | cray-hdf5-parallel | Apptainer | untested |
   | Generic SLURM + OpenMPI | Linux x86_64 | 12.x | 0.5.x | OpenMPI 4/5 + UCX | conda-forge `hdf5=*mpi_openmpi*` | Apptainer / bare venv | untested |
   | Pure-JAX (no FFI) | any | 12/13 | ≥0.9 | — | — | bare venv | CI |

   The matrix makes explicit which features each config supports — crucially, that the
   pure-JAX path is the only one that works with **zero native libs**, and that everything
   distributed needs the FFI stack.

2. **Track 1 — pip/uv (pure JAX, no FFI).** The current §2.1 content. State plainly: this
   gives you centroids/load/serial GW but **not** distributed eigh, sharded HDF5, or SLATE.

3. **Track 2 — container.** Generalize beyond Shifter: document the image
   (`nvcr.io/nvidia/jax`), the bind-mount contract (which host dirs map to
   `/lorrax_{nvhpc,phdf5,slate}`), and give the invocation for **Shifter, Apptainer, and
   Singularity** side by side (see B4). Today only Shifter is real.

4. **Track 3 — from source (the native stack).** The currently-missing recipe: how to
   obtain cuSolverMp, parallel HDF5, and SLATE on a system with no Cray PE (see B2), then
   build `liblorrax_ffi.so` and point LORRAX at it (see B5).

5. **Quickstart + verification** that works on any track (the smoke tests from
   `src/ffi/AGENTS.md:84-105`, de-NERSC-ified).

This page becomes the canonical entry; ENVIRONMENT/config/PORTING become deep-dive
references that link to it. It directly fixes A2, A5, A7, A9.

### B2. Write real "obtain the native stack" recipes that do NOT require copying from `/opt/cray`

**Effort: M.**

**Problem:** Every current acquisition path is "copy NERSC's vendor tree":
`stage_nvhpc.sh` copies `/opt/nvidia/hpc_sdk/...`; `phdf5/stage_cray.sh` copies
`/opt/cray/pe/hdf5-parallel/...` and shims Cray SONAMEs; `slate/stage_cray.sh` copies
`/opt/cray/pe/lib64/libsci*` and `libmpi_gtl_cuda`. A cluster without Cray PE has nothing
to copy. Yet the pieces are all obtainable independently — `site_config.sh:74` even notes
cuSolverMp is available as the PyPI wheel `nvidia-cusolvermp-cu12==0.7.2.888`. The docs
just never say so in one place.

**Proposal:** A "Native stack acquisition" section with a concrete recipe per dep,
**provider-agnostic first, Cray as one option**:

- **cuSolverMp + CAL:**
  - *Preferred (any cluster):* `pip install nvidia-cusolvermp-cu12` (the wheel
    `site_config.sh` calls the validated 0.7.2 baseline) + matching `nvidia-cal-cu12`;
    point `CUSOLVERMP_INCLUDE_DIR`/`CUSOLVERMP_LIB_DIR` at the wheel's
    `site-packages/nvidia/.../{include,lib}`. (The CMake autodetect ladder at
    `CMakeLists.txt:42-117` already supports explicit `-DCUSOLVERMP_*_DIR`; document it.)
  - *Alternative:* NVHPC SDK install (spack `nvhpc`, or the tarball from
    developer.nvidia.com) → use `stage_nvhpc.sh` with `NVHPC_ROOT` pointed at it.
- **Parallel HDF5:**
  - *Preferred (non-Cray):* conda-forge `hdf5=*=mpi_openmpi_*` (the
    `stage_openmpi.sh:36` recipe) or spack `hdf5+mpi`. Build the FFI against it directly
    (`-DHDF5_ROOT=...`) — no staging/shimming needed off-container.
  - *Cray:* `cray-hdf5-parallel` module → `stage_cray.sh` (existing).
  - The CMake check at `CMakeLists.txt:265-273` already enforces `HDF5_IS_PARALLEL` and
    prints a good error; surface that requirement in the doc.
- **SLATE + blaspp + lapackpp:**
  - Real recipe: `git clone --recurse-submodules https://github.com/icl-utk-edu/slate`
    (pin a tag), `cmake -Dgpu_backend=cuda -Dblas=<openblas|mkl|libsci>
    -DCMAKE_INSTALL_PREFIX=$PREFIX`, build, install. blaspp/lapackpp build as part of the
    superbuild and land under the same prefix (matching the `slate_DIR`/`blaspp_DIR`/
    `lapackpp_DIR` wiring at `CMakeLists.txt:355-357`). Point
    `LORRAX_SLATE_INSTALL_DIR=$PREFIX`.
  - Note BLAS choice: Cray uses libsci; elsewhere OpenBLAS or MKL is fine — SLATE doesn't
    require libsci, that's a Perlmutter perf choice (the docs currently conflate the two).

This is the single biggest content gap and what actually unblocks a foreign build.

### B3. Decouple defaults from the NERSC `$SCRATCH`/`$HOME` filesystem vocabulary

**Effort: S.**

**Problem:** `$SCRATCH` (a NERSC-set env var) and the `$HOME/software` layout are baked
into defaults across `site_config.sh:102-108`, the modulefile (`0.1.0.lua:116`
`env_or("SCRATCH", HOME)` for the JAX cache), `stage_*.sh` (`/pscratch/sd/${USER:0:1}/...`
defaults), and the doc tables. `$SCRATCH` is undefined on most non-NERSC clusters; the
`/pscratch/sd/<initial>/<user>` path template is pure NERSC Lustre layout.

**Proposal:** Introduce one indirection variable, e.g. `LORRAX_STAGE_ROOT` (default:
`${SCRATCH:-$HOME}/lorrax-stage` or a config value), and derive all stage-dir defaults
from it. Replace `/pscratch/sd/${USER:0:1}/${USER}/...` literals in the four `stage_*.sh`
with `${LORRAX_STAGE_ROOT}/...`. Document `JAX_COMPILATION_CACHE_DIR` as
`${LORRAX_STAGE_ROOT}/.jax_cache`. This removes the assumption that a bind-mountable
scratch filesystem exists at a NERSC path while preserving NERSC behavior when `$SCRATCH`
is set.

### B4. Promote non-Shifter container runtimes (Apptainer/Singularity) to first-class, not a swap-one-line afterthought

**Effort: M (docs) / M–L (modulefile).**

**Problem:** Shifter is NERSC-only. The portability story for the container runtime is
three near-identical hand-waves:
- `ENVIRONMENT_COMPREHENSIVE.md:364`: *"swap the `shifter` invocation in
  `lxrun`/`lxshell`/`lxpre`."*
- `config/README.md:240-244`: *"the Lua modulefile's `shifter_args` composition needs
  adaptation — swap the `shifter` invocation."*
- `src/ffi/PORTING.md:108-112`: *"swap `shifter ...` for `apptainer exec --nv --bind ...
  image.sif ...`."*

None of these is executable. The `shifter_args` are assembled inline in the modulefile
(`0.1.0.lua:192-199`) as a flat `--module/--volume/--env` string with Shifter-specific
flag syntax (`--module=gpu,mpich`, `--volume=host:container`, `--env=K=V`). Apptainer uses
entirely different flag syntax (`--nv`, `--bind host:container`, `--env K=V`, a `.sif`
file instead of a registry tag, and no `--module` concept). "Swap the invocation" hides a
real porting task, and the Shifter-only `--module=mpich` mechanism (which provides the MPI
bind-mount) has no Apptainer analog documented — on Apptainer you'd bind-mount MPI
yourself.

**Proposal:**
1. **Docs:** Replace the three hand-waves with one worked Apptainer example: how to build
   or pull the `.sif`, the exact `apptainer exec --nv --bind <nvhpc>,<phdf5>,<slate>
   --env LD_LIBRARY_PATH=... image.sif python3 -m gw.gw_jax ...` line, and how MPI gets in
   (bind-mount the host MPI lib dir; set `LD_LIBRARY_PATH`). Note CUDA-aware MPI differences
   (UCX vs Cray gtl, A8).
2. **Code (recommended):** Factor the container invocation into a runtime abstraction. Add
   a `LORRAX_CONTAINER_RUNTIME={shifter,apptainer,singularity,none}` knob and a small
   wrapper script (e.g. `src/ffi/common/cpp/run_container.sh`) that maps the abstract
   "image + binds + env + command" onto the chosen runtime's flag syntax. The modulefile
   and `run_shifter.sh` then call the wrapper instead of hardcoding `shifter`. This makes
   "swap the invocation" a config value, not a code edit — and makes the bare-venv path
   (`none`) a real, tested code path rather than a doc snippet (`ENVIRONMENT §7:366-378`).

### B5. Package or first-class-document the `liblorrax_ffi.so` build so a fresh clone isn't a dead end

**Effort: M.**

**Problem (validated fact):** a fresh `git clone` has no `liblorrax_ffi.so` (gitignored
build artifact); a newcomer hits `FileNotFoundError … Build with: bash
src/ffi/common/cpp/build.sh`. And `build.sh` only works inside Shifter with the staged libs
already mounted, with `LORRAX_MPI_INCLUDE_DIR`/`LORRAX_MPICH_LIB_DIR` exported (it `exit 2`s
otherwise, `build.sh:35-48`). The build group in `pyproject.toml:63-69` (scikit-build-core,
cmake, ninja, pybind11, nanobind) exists but is **not wired** to actually build the FFI as
part of `uv sync`/`pip install` — the native libs aren't declared, so scikit-build-core has
nothing to drive.

**Proposal (pick the achievable subset):**
1. **Minimum:** Document the FFI build as a required step in the Installation page with the
   non-Shifter invocation (the CMake `-D` overrides at `CMakeLists.txt:37-40,255-260,
   343-350` let you build outside any container: `cmake -DCUSOLVERMP_INCLUDE_DIR=...
   -DHDF5_ROOT=... -DLORRAX_SLATE_INSTALL_DIR=... -DLORRAX_MPI_INCLUDE_DIR=...`). Add a
   `LORRAX_FFI_ALLOW_DEFAULT_MPI=1` note for the OpenMPI/non-Cray build (already supported,
   `build.sh:35`). State clearly which features fail-soft when the `.so` is absent.
2. **Better:** Make `build.sh` cluster-agnostic — accept `LORRAX_CONTAINER_RUNTIME` (B4),
   and don't hard-require the Cray-MPICH env vars when `LORRAX_PHDF5_MPI_STACK=openmpi`
   (the guard at `build.sh:35-48` already has the `ALLOW_DEFAULT_MPI` escape; document it).
3. **Best (structural):** Wire the `build` dependency group into scikit-build-core so
   `pip install lorrax[ffi]` (or an opt-in extra) compiles `liblorrax_ffi.so` against
   deps discovered via the existing CMake autodetect ladder, with the native libs declared
   as PyPI wheels where they exist (cuSolverMp wheel, conda/spack for HDF5/SLATE documented
   as system prereqs). This is the JAX-grade outcome: `pip install` produces a working
   distributed build on a supported config.

### B6. Separate user-facing install docs from developer/agent notes

**Effort: S–M.**

**Problem:** The "For AI agents" framing (`README.md:39`, `ENVIRONMENT §header:3`,
`src/ffi/AGENTS.md`) and the multi-checkout A/B/C agent-session machinery
(`install.sh:21-27`, `config/README.md:157-171`, modulefile `family()` rationale) are
internal-development concerns interleaved with the install instructions an external user
needs. The KNOWN_SANDBOX_ERRORS / date-stamped rationale comments throughout the scripts
(`build.sh:30`, `run_shifter.sh:64-71`) are developer lore, not user docs.

**Proposal:** Split into `docs/installation.md` + `docs/user-guide/` (external, generic)
vs `docs/developer/` or keep `AGENTS.md`/`PORTING.md` as the internal track. The A/B/C
multi-checkout content moves to the developer track. User docs should contain **no**
`jackm`/`m2651`/`$SCRATCH` literals and no "For AI agents" headers.

---

## Highest-leverage change

**Write the single provider-agnostic "Installation / Build from source" page (B1) whose
core is the real native-stack acquisition recipes (B2).**

Everything else in this lens is downstream of one missing document: the docs never tell a
foreign-cluster researcher how to *obtain* cuSolverMp, parallel HDF5, and SLATE without
copying NERSC's `/opt/cray` tree — even though all three are obtainable independently (a
PyPI wheel for cuSolverMp that `site_config.sh:74` itself endorses; conda-forge/spack for
parallel HDF5; a source build for SLATE), and the CMake build system **already supports**
pointing at arbitrary install locations via `-D` overrides
(`CMakeLists.txt:37-40,255-260,343-350`). The capability exists; only the documentation
that connects it to a non-NERSC user is missing. A support matrix plus three acquisition
recipes plus the non-Shifter `cmake -D...` build invocation converts LORRAX from
"runs on Jack's Perlmutter account" to "buildable on a generic GPU+SLURM cluster" —
which is the stated goal.
