# Agent 1 — Build system & FFI dependency contract

Slice owner: **Build system & FFI dependency contract.** Read-only audit of
`src/ffi/PORTING.md`, `src/ffi/common/cpp/CMakeLists.txt`,
`src/ffi/common/cpp/{build,run_shifter,in_container,select_gpu}.sh`,
`src/ffi/*/scripts/stage_*.sh`, the cuSolverMp/SLATE/phdf5 cpp ABI guards,
and `pyproject.toml`.  All paths in this report are relative to
`/global/homes/j/jackm/software/lorrax_C/` unless prefixed; line refs are
from that checkout as of 2026-05-13.

---

## 1. Scope

Question: **could a competent second user get `liblorrax_ffi.so` built and
importable on Frontier / Polaris / Leonardo / a vanilla Slurm+OpenMPI+Docker
cluster, working only from this repo and PORTING.md?**

In scope: CMake autodetection ladders, the four stage scripts, build.sh
preconditions, the cuSolverMp 0.6→0.7→0.8 ABI dispatch, the
MPI-stack / phdf5-stack / SLATE-stack handshake between `run_shifter.sh`
and `CMakeLists.txt`, the `liblorrax_ffi.so` RPATH, and the Python
packaging side of the FFI build.

Out of scope (other agents): the Lua modulefile shell-function bodies,
`runtime/__init__.py`, env vars on the *runtime* side of cohsex.in, the
ISDF / GW physics, the sandbox `lorrax_agent` overlay.

---

## 2. Current state

### 2.1 The build is decoupled from Python packaging

`pyproject.toml` declares:

- A `[dependency-groups]` table with `build = ["scikit-build-core>=0.11.0",
  "cmake>=4.0.0", "ninja>=1.10.0", "pybind11>=3.0.0", "nanobind>=2.0.0"]`
  (`pyproject.toml:63-69`).
- **No `[build-system]` table.**
- `[tool.uv] package = true` (`pyproject.toml:28-29`) and
  `[tool.setuptools.package-data]` (`pyproject.toml:31-37`).

Consequences:

- Without `[build-system]`, `pip install -e .` defaults to setuptools
  PEP 517, ignores the `build` group, and **does not invoke CMake**.
- The scikit-build-core + nanobind line items in `dependency-groups.build`
  are orphan infrastructure: nothing reads them.  scikit-build-core is the
  canonical "pyproject.toml → CMake → C++ shared lib" backend for exactly
  this case ([docs](https://scikit-build-core.readthedocs.io/en/latest/guide/getting_started.html))
  and `pyproject.toml` looks 50% migrated to it.
- The FFI is built by a human running `bash src/ffi/common/cpp/build.sh`
  inside Shifter, and `ffi/common/ffi_loader.py:69-71` *globs*
  `src/ffi/common/cpp/build/liblorrax_ffi*.so` to find it.  PORTING.md
  never says "you must run build.sh"; it shows the command in §"Build
  system" but doesn't flag that `pip install` is a no-op for the .so.

### 2.2 Three coupled CMake autodetection ladders

`src/ffi/common/cpp/CMakeLists.txt` has three independent probe chains:

| Dep | Probe order |
|------|-------------|
| cuSolverMp | (a) `-DCUSOLVERMP_{INCLUDE,LIB}_DIR`, (b) `-DNVHPC_ROOT`, (c) env `$NVHPC_ROOT`/`$HPCSDK_ROOT`/`$NVHPC_SDK_PATH`, (d) glob `/opt/nvidia/hpc_sdk/Linux_x86_64/*` then `/lorrax_nvhpc/*` then `/usr/local/nvhpc/Linux_x86_64/*` *sorted descending* (`CMakeLists.txt:66-79`), (e) `cuda-cusolvermp-12-X` standalone package under `/usr/local/cuda{,−12.9,−12.8}` (`:106-117`). |
| Parallel HDF5 | `-DHDF5_ROOT`, env `$HDF5_ROOT`/`$HDF5_DIR`, then bind-mount `/lorrax_phdf5/lib` (`CMakeLists.txt:257-263`), then `find_package(HDF5 COMPONENTS C REQUIRED)`. Hard error if `HDF5_IS_PARALLEL` is false (`:267-273`). |
| MPI | env `$LORRAX_MPI_INCLUDE_DIR`/`$LORRAX_MPICH_LIB_DIR`, then `find_path(mpi.h HINTS ${HDF5_INCLUDE_DIRS} /opt/hpcx/ompi/include)` (`:283-296`), then `find_library(mpi HINTS ${LORRAX_MPICH_LIB_DIR})` (`:327-334`).  Falls back to `/opt/hpcx/ompi/lib` with a `WARNING` if `LORRAX_MPICH_LIB_DIR` is unset (`:302-316`). |
| SLATE | env `$LORRAX_SLATE_INSTALL_DIR`, else hardcoded `/global/homes/j/jackm/software/slate/install` (`:347-350`). Then `slate_DIR = ${prefix}/lib64/cmake/slate` etc. |
| NCCL header | `/usr/include/nccl.h`, else `${NVHPC_ROOT}/comm_libs/${NVHPC_CUDA_SUBDIR}/nccl/include` (`:215-223`). |
| CUDA runtime | `CUDA_TOOLKIT_ROOT` defaults to `/usr/local/cuda` (`:161-166`). |

Cross-checks present:
- CUDA-major-version match between `${CUDA_TOOLKIT_ROOT}/version.json` and
  the NVHPC `math_libs/<x.y>/` subdir, hard error on mismatch (`:174-213`).
- `cusolverMp.h` and `libcusolverMp.so` existence (`:130-139`).

### 2.3 The build.sh / run_shifter.sh / CMakeLists handshake

The MPI choice flows through three files in a tight cycle:

1. `run_shifter.sh` reads `LORRAX_PHDF5_MPI_STACK` (`mpich`|`openmpi`,
   default `mpich` since commit `4e2d8f1`), picks `SHIFTER_MODULES`,
   `MPI_LIB_DIR_CT`, `MPI_INCLUDE_DIR_CT`, `MPI_TYPE_DEFAULT`
   (`run_shifter.sh:52-78`), and exports
   `LORRAX_MPI_INCLUDE_DIR`+`LORRAX_MPICH_LIB_DIR`+`LORRAX_PHDF5_MPI_STACK`
   into the Shifter env (`:152-155`).
2. `build.sh` *refuses to run* if either of the first two env vars is
   unset (`build.sh:35-48`, commit `929be9a`), citing
   `KNOWN_SANDBOX_ERRORS.md 2026-05-10`.  Override: `LORRAX_FFI_ALLOW_DEFAULT_MPI=1`.
3. CMake reads those env vars; if absent and the override is on, falls
   back to `/opt/hpcx/ompi/lib` with a `WARNING` — which produces a .so
   with `DT_NEEDED libmpi.so.40`, then segfaults at runtime under
   `--module=mpich` (this is the bug 929be9a was added to prevent).

That triple-source agreement is what makes the .so loadable.  It is
nowhere documented as a single contract; PORTING.md mentions
`LORRAX_PHDF5_MPI_STACK` only in the phdf5 stack-choice section.

### 2.4 Vendor matrix (what links into `liblorrax_ffi.so`)

| Dep | Where it comes from | Version pinned by | Build-time check | Runtime check |
|------|---------------------|-------------------|------------------|---------------|
| cuSolverMp | bind-mounted `/lorrax_nvhpc/<subpath>` | `LORRAX_NVHPC_SUBPATH` in `site_config.sh` (current: `0.7.2_cuda12.9/...`) | path + header existence | `cusolverMpGetVersion` runtime banner + ABI dispatch (`context.cc:189-298`) |
| libcal | NVHPC `lib64/libcal*` (staged together by `stage_nvhpc.sh` / `stage_pypi.sh`) | same as cuSolverMp | linked unconditionally (`CMakeLists.txt` doesn't list it; comes in via `libcusolverMp.so`'s NEEDED) | only on the ≤0.6.x CAL path; 0.7+ creates a CAL wrapper over NCCL only as a shim |
| cuBLASMp | NVHPC `math_libs/<cuda>/lib64/libcublasmp.so` | same as cuSolverMp | linked unconditionally (`CMakeLists.txt:436-437`) | shares dispatch with cuSolverMp |
| NCCL | container `/lib/x86_64-linux-gnu/libnccl.so` then NVHPC fallback (`CMakeLists.txt:240-243`) | container choice (`nvcr.io/nvidia/jax:25.04-py3` ⇒ NCCL 2.26.3 per `context.cc:198-202` comment) | header existence | `ncclGetVersion`; warn if `mp_version ≥ 800 && nccl < 22700` |
| cuSolver (single-GPU) | `/usr/local/cuda` | container | `find_library` required | — |
| cuSolverMg | `/usr/local/cuda` | container | required | — |
| cudart | `/usr/local/cuda/lib64` | container | required | — |
| Parallel HDF5 | bind-mounted `/lorrax_phdf5/lib` (`stage_cray.sh` or `stage_openmpi.sh`) | shell script | `find_package(HDF5)` + `HDF5_IS_PARALLEL` | header / link |
| MPI | `LORRAX_MPICH_LIB_DIR` (in container: `/opt/udiImage/modules/mpich` for Cray; `/opt/hpcx/ompi/lib` for OpenMPI) | `LORRAX_PHDF5_MPI_STACK` | `find_library(mpi HINTS …)` | `MPI_Init_thread` in phdf5/SLATE FFIs |
| SLATE | host build at `~/software/slate/install` accessed inside container via siteFs | hardcoded path | `find_package(slate CONFIG)` optional, sets `LORRAX_HAVE_SLATE` | — |
| blaspp / lapackpp | bundled alongside SLATE | implicit via slateConfig | `find_dependency` inside slateConfig | — |
| Cray libsci, libxpmem, liblustreapi, libmpi_gtl_cuda | staged at `/lorrax_slate/lib` by `slate/scripts/stage_cray.sh` | shell script | not checked by CMake; pulled in at runtime via SLATE's NEEDED tags | LD_LIBRARY_PATH + LD_PRELOAD |
| jax.ffi headers | `python3 -c "import jax.ffi; print(jax.ffi.include_dir())"` (`CMakeLists.txt:146-156`) | container's `/opt/jaxlibs` | must succeed | XLA FFI version baked in header must match runtime |

### 2.5 ABI dispatch implemented in the .so

`src/ffi/cusolvermp/cpp/context.cc:189-298` reads
`cusolverMpGetVersion` and `ncclGetVersion` at context-create time and
dispatches:

- `mp_version < 700`: build a real `cal_comm_t` over NCCL callbacks
  (3 shims `cal_nccl_allgather` / `_req_test` / `_req_free`,
  `context.cc:37-…`) and pass it to `cusolverMpCreateDeviceGrid`.
- `mp_version ≥ 700`: pass the `ncclComm_t` directly (cast through
  `reinterpret_cast<cal_comm_t>`), per
  [NVIDIA's CAL→NCCL migration note](https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html).
- Rank-0 stderr banner; rank-0 warning on `(mp<700 && p>1 && q>1)` and
  on `(mp≥800 && nccl<22700)` (missing `ncclCommWindowRegister`,
  cf. [nccl#1784](https://github.com/NVIDIA/nccl/issues/1784)).

This is the right call.  The corresponding cuBLASMp dispatch is at
`context.cc:349-…` and reuses the same `cusolvermp_version` field — good
single source of truth.

---

## 3. NERSC-isms

| Item | Where | Travels to other clusters? |
|------|-------|---------------------------|
| Shifter as the container runtime | `run_shifter.sh:135`, `0.1.0.lua:194` | **No.** Shifter is NERSC/Cray-specific. Apptainer is closest analogue but uses `--bind` not `--volume`, and `--module=mpich` has no equivalent. Frontier uses [Singularity/Apptainer](https://docs.olcf.ornl.gov/), Polaris uses Apptainer, Leonardo uses Singularity. |
| `--volume` restricted to `/pscratch` + a few siteFs paths | `stage_cray.sh:7-11`, `stage_openmpi.sh:18-22` | **No.** This restriction is `udiRoot.conf`-driven, NERSC-only.  Other runtimes happily bind `/opt`. Half the staging script complexity exists to copy libs *out of* `/opt/cray` into `/pscratch`. |
| `/opt/udiImage/modules/mpich` | `0.1.0.lua:75`, `run_shifter.sh:63`, `site_config.sh:91` | **No.** This path is Shifter's `--module=mpich` mount target. Any non-Shifter runtime needs a different bind layout (e.g. Apptainer `--bind /opt/cray/pe/mpich/…:/opt/mpich` or `pmix`-based hostmpi). |
| `libmpi_gtl_cuda.so.0` LD_PRELOAD trick | `0.1.0.lua:181`, `run_shifter.sh:130-133`, `stage_cray.sh:52` | **Cray-only.** Frontier (Cray Slingshot) needs the same. Polaris (HPE Slingshot but GPU-aware MPI is per-MPI, not via libgtl), Leonardo (HDR IB), generic OpenMPI: no analogue. |
| `--mpi=cray_shasta` | `0.1.0.lua:273`, `run_shifter.sh:72`, `site_config.sh:68` | **No.** Frontier supports cray_shasta; everyone else uses `pmi2`/`pmix`. |
| `--constraint=gpu` | `site_config.sh:54`, `0.1.0.lua:240` | **No.** Constraint names are site-defined. |
| `module load cray-hdf5-parallel cray-mpich` | `stage_cray.sh:21-23` | Cray sites only. |
| `LORRAX_FFI_NVHPC_DIR` defaulting to `/pscratch/sd/${USER:0:1}/${USER}/...` | `stage_pypi.sh:35`, `stage_nvhpc.sh:25` | NERSC `$SCRATCH` layout. Other clusters: `/lustre/…`, `/work/…`, etc. |
| Hardcoded `/global/homes/j/jackm/software/slate/install` as the SLATE default | `CMakeLists.txt:348` | **No.** Username + filesystem layout both NERSC-specific. |
| siteFs path `/global/u2`, `/global/common/software/nersc9/darshan` | `run_shifter.sh:117`, `stage_cray.sh:72-74` | NERSC-only. |
| `salloc … bash -c "sleep 100000"` allocation pattern | `0.1.0.lua:241` | Works on any Slurm cluster, but the qos/account names are site-specific. |

Some of these *would degrade gracefully* under a porting attempt
(e.g. `--constraint` name is one sed away). The Shifter bind layout and
the `--module=mpich` GTL/PMI stack are the load-bearing NERSC-isms.

---

## 4. Defect catalog

### Tagged FRAGILE — works today, would silently break

1. **`pyproject.toml` declares `jax[cuda13]>=0.9.0` (`:9, :60-62`).**
   The pinned container is `nvcr.io/nvidia/jax:25.04-py3`, which ships
   JAX 0.5.x with CUDA 12.9 (PORTING.md §"Hard requirements" L13-14 and
   `stage_pypi.sh:10` confirm).  PORTING.md says the minimum is JAX 0.5,
   but pyproject demands ≥0.9.0 with CUDA-13 wheel extras.  Anyone who
   actually runs `pip install -e .` inside the container will trigger a
   JAX upgrade attempt against the wrong CUDA, then either silently bypass
   the bundled libcusolverMp (FFI ABI breaks) or fail outright. **[FRAGILE]** **[COMPAT]**
2. **No `[build-system]` table.** `pip install -e .` does NOT trigger
   `src/ffi/common/cpp/build.sh`.  The `dependency-groups.build` table
   listing scikit-build-core + nanobind is unused. A new user following
   the obvious flow (`git clone; pip install -e .`) will get a working
   Python package import that crashes the first time `get_lib()` is
   called because no `.so` exists. **[FRAGILE]** **[LOC-COST]**
3. **NVHPC autodetect glob `list(SORT … ORDER DESCENDING)`
   (`CMakeLists.txt:71, 85`)** picks the *lexically* newest NVHPC subdir,
   not the *numerically* newest. `26.10` sorts *before* `26.9`. On a
   cluster with both 26.9 and 26.10 installed this picks 26.9; the build
   succeeds but the runtime banner shows the wrong library. **[FRAGILE]**
4. **NCCL fallback path `${NVHPC_ROOT}/comm_libs/${NVHPC_CUDA_SUBDIR}/nccl/include`
   (`CMakeLists.txt:218-219`)** but `stage_pypi.sh` does not stage any
   NCCL header. The path-exists check at `:221` would fail if container
   `/usr/include/nccl.h` is missing AND `NVHPC_ROOT` is a stage_pypi tree
   (which only contains `math_libs`, not `comm_libs`). Currently masked
   by the container shipping `nccl.h` at `/usr/include`. **[FRAGILE]**
5. **`stage_pypi.sh:88-95` cusolverMp header fallback** reuses the 0.6.0
   header from a hardcoded directory
   `${LORRAX_FFI_NVHPC_DIR}/25.5_cuda12.9/.../cusolverMp.h`. Header for
   0.7.2 is not shipped in the wheel; if 0.6.0 isn't already staged, the
   script aborts. So `stage_pypi.sh` for 0.7.x silently depends on having
   run `stage_nvhpc.sh` for 25.5 first. **[FRAGILE]**
6. **Manual `MPI::MPI_CXX` IMPORTED target (`CMakeLists.txt:362-367`)**
   stands in for `find_package(MPI)` so SLATE's `slateConfig.cmake` finds
   "an MPI." If SLATE's CMake ever requires additional properties on
   `MPI::MPI_CXX` (e.g. `INTERFACE_COMPILE_DEFINITIONS` from a real
   `find_package(MPI)` call) the stand-in silently drops them. Stays
   silent until SLATE bumps its config. **[FRAGILE]**
7. **HDF5 path-chain assumes `/lorrax_phdf5/lib`
   (`CMakeLists.txt:258`, `:418`, `0.1.0.lua:153, 196`).**  Hardcoded
   bind-mount target. If a user customises the bind-mount, both build and
   runtime break — the `INSTALL_RPATH` literal at `CMakeLists.txt:418`
   bakes the path into the .so. **[FRAGILE]** **[LOC-COST]**
8. **The cuSolverMp 0.6 → 0.7 → 0.8 history is a moving ABI target.**
   Three commits in three months (`c52fbd2` add 0.7 dispatch, `65e2f59`
   default to 0.7.2, `929be9a` MPI-stack guard) all came from the same
   ABI break. NVIDIA's [release notes](https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html)
   indicate CAL→NCCL was a one-time event, but 0.8 introduced an NCCL
   ≥2.27 requirement (`ncclCommWindowRegister`, cf. `context.cc:227-251`
   warning + [nccl#1784](https://github.com/NVIDIA/nccl/issues/1784)).
   Bumping to NVHPC 25.11 (CUDA 13, cuSolverMp 0.9+) is the most likely
   near-term break. **[FRAGILE]**
9. **`cusolvermp_version >= 700` is the only branch condition
   (`context.cc:196, 355`).** If NVIDIA introduces a third ABI tier at
   1.0 (or another CUDA major bump), this dispatch needs another arm —
   no tests exist to detect it. **[FRAGILE]**
10. **CUDA-major check
    (`CMakeLists.txt:174-213`)** parses `${CUDA_TOOLKIT_ROOT}/version.json`
    with a fragile inline regex
    `"\"cuda\"[^{]*{[^\"]*\"version\"[ :]*\"([0-9]+\\.[0-9]+)"`.  If
    NVIDIA changes the JSON layout the regex silently returns nothing,
    skipping the check.  Falls back to `nvcc --version` (good), but if
    both probes fail the entire ABI check is skipped silently — no
    warning. **[FRAGILE]**
11. **`-DNVHPC_ROOT` and `-DCUSOLVERMP_INCLUDE_DIR` paths are honoured
    only on the first cmake invocation** (cached). Re-running build.sh
    after editing `site_config.sh` does *not* update the linked NVHPC —
    you need `--fresh` (`build.sh:58-61`). PORTING.md does not say so.
    **[FRAGILE]**
12. **`stage_pypi.sh:72-75` produces three symlinks
    (`libcusolverMp.so.${SHORT}.0` ← `libcusolverMp.so.0` ←
    `libcusolverMp.so`)**, but the `find_library` chain in CMakeLists is
    only `find_library(cudart …)` and friends — for cuSolverMp itself the
    build uses literal path `${CUSOLVERMP_LIBDIR}/libcusolverMp.so`
    (`:436`).  If a future stage script forgets the `.so` ⇒ `.so.0`
    symlink, the build silently picks up `.so.0` (or fails).  This is
    inverted: the convention is fragile because it's not enforced. **[FRAGILE]**
13. **`liblorrax_ffi.so` RPATH (`CMakeLists.txt:417-418`)** literally
    embeds `/lorrax_phdf5/lib;/lorrax_slate/lib` in the binary.  Move
    the .so to a different cluster's container with different bind
    targets and the RPATH search fails — silent fallback to
    LD_LIBRARY_PATH only. **[FRAGILE]** **[COMPAT]**
14. **`stage_cray.sh` (slate) shim `libreadline.so.7 → libreadline.so.8`
    (`slate/scripts/stage_cray.sh:64-66`)** assumes minor-version compat
    of libreadline. A future container with libreadline.so.9 would
    silently dlopen but break unpredictably (the script comment
    acknowledges this is a "tiny surface" hack). **[FRAGILE]** **[LOC-COST]**
15. **`stage_cray.sh` (phdf5) SONAME shim
    `libmpi_gnu_{91,110,123}.so.12 → /opt/udiImage/modules/mpich/libmpi.so.12`
    (`phdf5/scripts/stage_cray.sh:84-86`)** — three SONAMEs are
    enumerated.  If Cray ships a new compiler triple (`libmpi_gnu_130.so.12`)
    or the GCC-12.X.X minor changes, the symlink list goes out of date
    silently — phdf5 fails to load at runtime. **[FRAGILE]**

### Tagged COMPAT — will not work on a different cluster as written

16. **Hardcoded SLATE install path `/global/homes/j/jackm/software/slate/install`
    (`CMakeLists.txt:348`).** A new user gets `slate_FOUND = FALSE`
    silently, `LORRAX_HAVE_SLATE = 0`, the FFI builds *without* SLATE
    targets, and any code path that calls `slate_*` ops crashes at
    Python-level "FFI target not registered."  PORTING.md mentions the
    override but the default is jackm's $HOME. **[COMPAT]** **[LOC-COST]**
17. **`run_shifter.sh:44`** defaults `NVHPC_HOST` to
    `/pscratch/sd/j/jackm/lorrax_nvhpc` — another user's `$SCRATCH` won't
    match. Equivalent defaults at `run_shifter.sh:54, 61, 107`. **[COMPAT]**
18. **`LORRAX_SRC` and `LORRAX_SITE` defaults (`run_shifter.sh:95-96`)**
    point at hardcoded user homes (`/global/u2/j/jackm/...`). Other users
    must set them explicitly. PORTING.md doesn't mention this. **[COMPAT]**
19. **The phdf5 OpenMPI alternative (`stage_openmpi.sh`)** pins
    `hdf5-1.14.6-mpi_openmpi_h8367ee7_8.conda` linked against OpenMPI 4.x
    (`libmpi.so.40`). Sites with OpenMPI 5 (libmpi.so.80) need a
    different wheel; the script gives no hint. **[COMPAT]**
20. **The whole `--module=mpich` Shifter idiom** (`run_shifter.sh:135`,
    `0.1.0.lua:194`) has no Apptainer/Singularity analogue. PORTING.md
    §"For non-Shifter runtimes" hand-waves "swap the shifter invocation
    for `apptainer exec --nv --bind …`" but doesn't address how
    Apptainer's `--bind` interacts with the host MPI without an
    equivalent of `--module=mpich`'s SONAME injection. On Polaris /
    Leonardo / a generic Slurm+Docker cluster, the user has to invent
    that mechanism themselves. **[COMPAT]**
21. **Cray-PE-specific build assumption for SLATE** (`stage_cray.sh:48-52`):
    libsci + libmpi_gtl_cuda + xpmem only exist on Cray sites. On
    Polaris / Leonardo, SLATE has to be built against OpenBLAS or
    similar, and the staging script has no equivalent. **[COMPAT]**
22. **`select_gpu.sh` reads `SLURM_LOCALID`** (`select_gpu.sh:12`).
    PMIx implementations sometimes don't set `SLURM_LOCALID`; PMI2
    does. The script defaults to 0, which silently coalesces all ranks
    onto GPU 0 if the env var is missing — looks like it works but
    serializes everything. **[COMPAT]** **[FRAGILE]**
23. **`cuda-cusolvermp-12-X` standalone-package fallback
    (`CMakeLists.txt:106-117`)** assumes a Debian-style `/usr/local/cuda`
    layout. On a RHEL/Rocky host where `cusolverMp` ships under
    `/usr/local/cuda-12.X/targets/x86_64-linux/lib` *as a symlink chain*,
    `find_library` with `NO_DEFAULT_PATH` may not resolve transitive
    deps. **[COMPAT]**

### Tagged LOC-COST — exists only because of NERSC peculiarities

24. **`select_gpu.sh` + `in_container.sh`** (`select_gpu.sh:1-14`,
    `in_container.sh:1-15`) — both files exist exclusively to work
    around Shifter quirks (`SLURM_LOCALID` per rank,
    `MPICH_GPU_SUPPORT_ENABLED` unset by `--module=mpich`). On
    non-Shifter, both files are dead weight. ~30 LoC total. **[LOC-COST]**
25. **All four stage scripts** (`stage_nvhpc.sh`, `stage_pypi.sh`,
    `stage_cray.sh` ×2, `stage_openmpi.sh`) are NERSC bind-mount-restriction
    workarounds.  On a cluster with sane bind-mount policy (Apptainer
    with default `--bind /opt`), 3 of 5 stage scripts collapse into
    "point CMake at the installed lib." ~250 LoC + a maintenance burden
    (every NVHPC bump, every Cray PE bump). **[LOC-COST]**
26. **`stage_cray.sh` (phdf5) MPICH SONAME shim layer** — three symlinks
    to paper over the cray-pe-vs-shifter SONAME mismatch
    (`phdf5/scripts/stage_cray.sh:84-86`). Pure Cray-ism. **[LOC-COST]**
27. **Manual `MPI::MPI_CXX` IMPORTED target** (`CMakeLists.txt:362-367`)
    only exists because the build avoids `find_package(MPI)` to keep MPI
    selection under user control. A normal `find_package(MPI)` workflow
    is shorter and gives more downstream-CMake compatibility. The cost
    of the current approach is the 5-line workaround. **[LOC-COST]**
28. **`run_shifter.sh:82-93` "if staged dir missing, print 7 lines of mkdir
    + cp instructions"** — recapitulates `stage_nvhpc.sh` inline. Either
    the stage script should be auto-run from here, or this block is dead
    documentation. **[LOC-COST]**
29. **`run_shifter.sh` SLATE preload conditional (`:130-133`)** + the
    in-container re-assert at `in_container.sh:13` exist solely to work
    around Shifter's `module_mpich_siteEnvUnset`. **[LOC-COST]**

### PORTING.md gaps

30. **PORTING.md §"Build system" doesn't mention `run_shifter.sh`** as
    the required wrapper around `build.sh`. The reader sees "Build:
    `bash src/ffi/common/cpp/build.sh`" (L36) and tries to run it on the
    host. It fails immediately on `LORRAX_MPI_INCLUDE_DIR` unset. The
    fix is documented only in `build.sh:35-48` (the error message
    itself). **[FRAGILE]** **[LOC-COST]**
31. **PORTING.md never says how to get `.so` from the build/ directory
    into the Python import path.** `ffi_loader.py` globs the build dir,
    so it Just Works *if* you build inside the repo, but a user copying
    `liblorrax_ffi.so` to a different location must set
    `LORRAX_FFI_SO` — which is mentioned only in the docstring of
    `ffi_loader.py:18-20`. **[LOC-COST]**
32. **PORTING.md says (L17) "SLATE: any version" but `find_package(slate CONFIG)`
    requires a specific `slate_DIR` layout that ICL's CMake config only
    started shipping in 2023.** No lower version bound is stated;
    `git log` of the SLATE submodule (n/a — it's external) would matter.
    [icl-utk-edu/slate](https://github.com/icl-utk-edu/slate) ships its
    own CMake config but doesn't publish ABI policy
    ([blaspp/lapackpp release pages](https://github.com/icl-utk-edu/blaspp)
    confirm cadence is ~annual but make no ABI promise). **[FRAGILE]**
33. **PORTING.md says (L12) "JAX with `jax.ffi`: 0.5"** but does not say
    that `jax.ffi.register_ffi_target` defaulted to api_version=1 at some
    point ([JAX FFI docs](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.register_ffi_target.html),
    [JAX FFI tutorial discussion](https://github.com/jax-ml/jax/discussions/26602))
    and that older calls assuming api_version=0 break.
    `ffi_loader.py:199-203` uses default api_version, so a JAX downgrade
    on a new cluster might silently route to the wrong dispatch path. **[FRAGILE]**
34. **PORTING.md mentions `jax.ffi.include_dir()`** (only via
    `CMakeLists.txt:146-156`'s `execute_process` line, never in the
    porting checklist). XLA FFI ABI is keyed to the JAX version *the
    .so was built against*; mixing build-time and run-time JAX silently
    breaks ([jax#34047](https://github.com/jax-ml/jax/issues/34047)).
    **[FRAGILE]**

---

## 5. Blitz proposals (ranked by leverage)

For each: scope ≈1 day. "Locks in" = the test/CI/check that prevents the
defect from regressing.

### P1 — Wire `pip install -e .` to the FFI build (defects #1, #2, #16)

- **Change.** Add `[build-system] requires = [...]` and
  `build-backend = "scikit_build_core.build"` to `pyproject.toml`; promote
  `src/ffi/common/cpp/CMakeLists.txt` to the top-level
  `CMakeLists.txt` (or a thin wrapper at repo root that `add_subdirectory`s
  it); install the .so into the `lorrax/ffi/common/cpp/` site-packages
  dir at `pip install` time.  Drop the orphan `dependency-groups.build`.
  Loosen `jax[cuda13]>=0.9.0` to `jax[cuda12]>=0.5,<0.9` or split into a
  `[project.optional-dependencies] cuda12`/`cuda13` matrix.
- **Addresses.** #1 [FRAGILE/COMPAT], #2 [FRAGILE/LOC-COST], partially
  #16 [COMPAT] (if `SLATE` becomes a CMake-detected optional component).
- **Leverage axis.** Installability — the *first thing* a new user does
  on a new cluster goes from "fail silently" to "fail with a CMake error
  message that names the missing dep."  Maintainability secondary
  (the duplicate dep-group declaration goes away).
- **Risk.** Editable installs need
  `pip install --no-build-isolation -Ceditable.rebuild=true -ve .` for
  fast iteration; might surprise users accustomed to setuptools'
  in-place build. Mitigate with a one-line note in the README.
- **Test.** A `tests/build_smoke/test_pip_install.py` that runs
  `pip install -e .` in a tmpdir against a CMake stub (no real
  cuSolverMp), and asserts the .so lands in the expected path.

### P2 — Emit a `VENDOR_VERSIONS.txt` from the build (defects #3, #4, #5, #8, #10, #11, #32, #33)

- **Change.** At configure time, CMake writes a
  `build/VENDOR_VERSIONS.txt` containing: resolved `NVHPC_ROOT`,
  `NVHPC_CUDA_SUBDIR`, `cusolverMp` library SONAME (from
  `readelf -d $CUSOLVERMP_LIBDIR/libcusolverMp.so | grep SONAME`),
  resolved cudart version, NCCL version (probe `nccl.h NCCL_MAJOR`),
  HDF5 version + `IS_PARALLEL`, MPI lib SONAME, SLATE
  `slate_VERSION`, JAX version (`jax.__version__` + `jax.ffi`
  api_version).  Print at end of `cmake` configure step; also write to
  disk.  Add a `cmake -DDRY_RUN=ON` mode that *only* writes this file
  and exits, no compile.
- **Addresses.** #3, #4, #5 (the file would expose 0.6.0 fallback),
  #8, #10 (replace inline JSON regex with a hard fail if not parseable),
  #11 (forces user to re-cmake, surfacing stale cache), and
  PORTING.md gaps #32, #33.
- **Leverage.** Maintainability — author + porter both have one file to
  diff between two builds.  "What changed in the dependency graph since
  last week?" becomes a one-line answer.
- **Risk.** Low.
- **Test.** A `tests/build_smoke/test_vendor_versions.py` that asserts
  the file exists after `bash build.sh` and contains all expected keys.

### P3 — Strict mode in `build.sh` + CMake (defects #6, #7, #9, #11, #12, #14, #15)

- **Change.** Make every CMake `WARNING` become a `FATAL_ERROR` when
  `LORRAX_BUILD_STRICT=1` (or `build.sh --strict`).  Specifically:
  the HPC-X OpenMPI fallback (`CMakeLists.txt:310-315`); the
  `LORRAX_HAVE_SLATE=0` silent skip; the `slate_FOUND = FALSE` case;
  the missing-readline-shim case; the missing-SONAME-shim case (check
  each enumerated `libmpi_gnu_*.so.12` exists post-stage).  Default
  off so existing workflows are unchanged; CI runs with `--strict`.
- **Addresses.** #6, #7, #9, #11, #12, #14, #15, #16, #22.
- **Leverage.** Installability — a porter who runs `build.sh --strict`
  on day 1 hits every silent fallback as a hard error.
- **Risk.** False positives the first time a porter runs it; the
  upside is they then *configure*, not silently miss.
- **Test.** A CI job (or a documented `tests/build_smoke/`) builds with
  `--strict` against a non-NERSC `lorrax_phdf5_openmpi` stage.

### P4 — Out-of-container build smoke test (defects #20, #21, #23, #24)

- **Change.** Add `tests/build_smoke/test_out_of_container.sh` that runs
  the CMake configure step on a host *without* Shifter (e.g. a Polaris
  or Frontier login node, or a generic Linux box with CUDA 12 + OpenMPI
  installed), with all four bind-mount targets unmounted, exercising
  the override paths (`-DNVHPC_ROOT=`, `-DHDF5_ROOT=`,
  `-DLORRAX_SLATE_INSTALL_DIR=`, `-DLORRAX_MPI_INCLUDE_DIR=`,
  `-DLORRAX_MPICH_LIB_DIR=`).  Doesn't need to *build* — `cmake` config
  + a `--dry-run ninja` is enough to exercise the autodetection ladder.
  Wire it into the CI checkpoint workflow.
- **Addresses.** #20, #21, #23, #24, #25 (validates that the bind-mount
  defaults are overridable), portions of #27.
- **Leverage.** Installability — protects PORTING.md's "Checklist for a
  new cluster" from rotting between author-only Perlmutter runs.
- **Risk.** Needs a non-NERSC machine in CI; could be a GitHub-Actions
  Ubuntu runner with a CUDA-stub layer that satisfies `find_library`.
- **Test.** Self-locking — the CI job *is* the test.

### P5 — Single MPI-stack contract file (defects #18, #19, #20, #28, #30)

- **Change.** Move the
  `(LORRAX_PHDF5_MPI_STACK, SHIFTER_MODULES, MPI_LIB_DIR_CT, MPI_INCLUDE_DIR_CT, MPI_TYPE_DEFAULT)`
  quintuple out of the per-shell case-statement in `run_shifter.sh`
  (`:52-78`) into a CMake-readable + shell-readable config (e.g.
  `config/mpi_stacks/cray_mpich.cmake` + matching `.sh`). Make
  `build.sh` and `run_shifter.sh` source the same file.  Document the
  contract in `PORTING.md`.  Bonus: delete the recapitulated
  mkdir+cp instructions in `run_shifter.sh:82-93`.
- **Addresses.** #18 [COMPAT] (hardcoded user paths get one defaulting
  layer), #19 [COMPAT] (extending stack list is a single new file),
  #28 [LOC-COST], #30 [PORTING gap].
- **Leverage.** Maintainability — adding a new MPI stack (e.g. OpenMPI 5
  for Polaris, Cray Slingshot-12 for a future system) becomes
  copy-and-edit one file, not three.
- **Risk.** Slight: needs CMake to source a `.cmake` from `config/`,
  which is normal-but-not-obvious to a porter.
- **Test.** `test_mpi_stack_round_trip.sh`: for each stack file,
  source it as shell and parse it as CMake, assert the five values
  match.

### P6 — Numerically-sorted NVHPC autodetect + explicit version pin (defect #3)

- **Change.** Replace `list(SORT … ORDER DESCENDING)` at
  `CMakeLists.txt:71, 85` with a SemVer-aware compare. Or — preferred,
  smaller change — add `CMAKE_MATCH_*` regex parsing and use
  `VERSION_GREATER` for the comparison. Tighten the
  `NVHPC_CUDA_SUBDIR` detection at `:84-93` similarly.
- **Addresses.** #3.
- **Leverage.** Low absolute, very high relative-to-cost; tiny patch
  prevents an annoying class of "right SDK installed, build picked the
  wrong one" surprises after NVHPC 26.10 ships.
- **Risk.** None.
- **Test.** Unit-test the regex+compare in a `CMakeLists.txt` toy
  project (run as part of `tests/build_smoke/`).

---

## 6. Open questions (most important section)

These are things I could not resolve from desk research, even with the
web. They are concrete blockers for declaring "PORTING.md is enough."

1. **Frontier with Apptainer (Slingshot-11 + Cray MPICH + ROCm/CUDA)** —
   Frontier has Cray MPICH but uses Apptainer, not Shifter. Apptainer's
   `--bind` doesn't have a `--module=mpich` analogue (i.e. an automatic
   SONAME injection layer).  Does the staging scheme work if `/opt/cray`
   is bind-mounted directly into the image, or does the libmpi_gnu_*.so.12
   SONAME mismatch still bite?  I'd need to actually try it.  The
   suspect path is `phdf5/scripts/stage_cray.sh:84-86`'s SONAME shim
   layer — if Apptainer's bind happens *before* dlopen, the symlink
   shim might not even be on `LD_LIBRARY_PATH` early enough.

2. **Polaris with OpenMPI 5** — `stage_openmpi.sh` pins HDF5 against
   OpenMPI 4 (libmpi.so.40). Polaris has been migrating to OpenMPI 5
   (libmpi.so.80). conda-forge's hdf5 builds for OpenMPI 5 exist
   ([conda-forge label `cf202301`](https://conda-forge.org/)) but the
   exact wheel name is not in the script. Does the cusolverMp NCCL
   bootstrap path care about MPI presence at all on a cluster that has
   *no* MPI at process-init time?  (PORTING.md L19-20 says no, but I
   couldn't verify because `ffi.cusolvermp` shares `liblorrax_ffi.so`
   with `ffi.phdf5`, which forces MPI to be linkable even when the
   user calls only cusolverMp paths.)

3. **NVIDIA HPC SDK 26.x** ships cuSolverMp 0.9+ (per
   [HPC SDK 26.1 release notes](https://docs.nvidia.com/hpc-sdk/hpc-sdk-release-notes/index.html))
   and CUDA 13. If the JAX container also moves to CUDA 13
   (`nvcr.io/nvidia/jax:26.04-py3` and later), does the existing
   `mp_version >= 700` branch still produce a working build? The
   CUDA-major-version check at `CMakeLists.txt:174-213` would *prevent*
   the build if CUDA 12 toolkit is mixed with CUDA 13 NVHPC — but the
   `jax[cuda13]>=0.9.0` line in `pyproject.toml` implies the author has
   already mapped out a CUDA-13 migration. The current code has no
   tested CUDA-13 path that I can find.

4. **JAX `jax.ffi.include_dir()` ABI stability** — JAX 0.5 → 0.9 is at
   least three ABI bumps in the XLA FFI header
   ([jax#34047](https://github.com/jax-ml/jax/issues/34047) shows a
   regression in 0.8.2; nothing in PORTING.md says which JAX-versions
   produce mutually-compatible .so files). Is `liblorrax_ffi.so` built
   against JAX 0.5 callable from JAX 0.7? From 0.9? Without a CI matrix
   I cannot say; the only safe answer is "build the .so on every JAX
   upgrade," and PORTING.md should say so.

5. **`SLATE` API stability across releases.** [icl-utk-edu/slate](https://github.com/icl-utk-edu/slate)
   doesn't publish a SemVer policy and the bundled `blaspp`+`lapackpp`
   are tagged independently. The current `CMakeLists.txt:355-357` pins
   SLATE/blaspp/lapackpp config dirs to whatever the host install has.
   If the user upgrades SLATE between sessions, do the
   `lorrax_ffi/slate/cpp/*.cc` translation units still compile? No
   build-time version check exists in CMakeLists; no test exercises
   the SLATE API surface.

6. **NCCL ≥ 2.27 staging path.** `context.cc:242-251` warns if NCCL
   <2.27 is loaded with cuSolverMp 0.8+, but neither
   `src/ffi/cusolvermp/scripts/` nor PORTING.md describe how to *stage*
   NCCL 2.27+ ahead of the container's bundled `libnccl.so.2`. The
   straightforward path is `LD_PRELOAD` or a stage_pypi-style NCCL
   wheel extraction, but I haven't seen this validated.
   ([nccl#1784](https://github.com/NVIDIA/nccl/issues/1784) suggests
   it's a real bug class.)

7. **The `LORRAX_FFI_NVHPC_DIR` default `/pscratch/sd/${USER:0:1}/${USER}/...`
   convention** (in stage scripts) — does `${USER:0:1}` work on every
   cluster's `$SCRATCH` layout?  Frontier uses
   `/lustre/orion/scratch/$USER/proj-shared`. Polaris uses `/lus/eagle/projects/...`.
   This convention is NERSC-only; a porter has to either re-edit every
   `stage_*.sh` or set `LORRAX_FFI_NVHPC_DIR` and friends explicitly.
   The cleaner fix is making the env-var path the *only* path (no NERSC
   default in the scripts).

8. **What test exists that would catch a botched FFI link?** I see one
   end-to-end test trio in PORTING.md (the `lxrun python3 -u -m
   common.cusolvermp_eigh_test` block), but no unit-level FFI symbol
   check in `tests/`. A `ldd liblorrax_ffi.so | grep "not found"` smoke
   check would catch most #13 [FRAGILE] and #15 [FRAGILE] regressions —
   but I can't tell from reading the code whether anyone runs it.

---

## Sources

- [cuSOLVERMp Release Notes](https://docs.nvidia.com/cuda/cusolvermp/release_notes/index.html)
- [NCCL `ncclCommWindowRegister` issue #1784](https://github.com/NVIDIA/nccl/issues/1784)
- [NVIDIA HPC SDK 26.1 release notes](https://docs.nvidia.com/hpc-sdk/hpc-sdk-release-notes/index.html)
- [scikit-build-core getting-started guide](https://scikit-build-core.readthedocs.io/en/latest/guide/getting_started.html)
- [nanobind packaging docs](https://nanobind.readthedocs.io/en/latest/packaging.html)
- [JAX FFI docs](https://docs.jax.dev/en/latest/ffi.html) and [`register_ffi_target`](https://docs.jax.dev/en/latest/_autosummary/jax.ffi.register_ffi_target.html)
- [JAX FFI tutorial discussion #26602](https://github.com/jax-ml/jax/discussions/26602)
- [JAX FFI regression issue #34047](https://github.com/jax-ml/jax/issues/34047)
- [icl-utk-edu/slate](https://github.com/icl-utk-edu/slate), [blaspp](https://github.com/icl-utk-edu/blaspp), [lapackpp](https://github.com/icl-utk-edu/lapackpp)

`Agent 1 done — see agent_1.md`
