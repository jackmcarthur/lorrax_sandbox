# LORRAX dependency declaration & wiring — antipatterns and a cleaner model

**Lens:** how LORRAX's dependencies — especially the three native FFI runtime libs
(NVHPC cuSolverMp, parallel HDF5, Cray SLATE) — are *declared, obtained, versioned, and
wired*. Not doc prose. **READ + PROPOSE ONLY.**

Repo: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D` (branch `agent/dep-home-migration`).

---

## 0. Executive summary

LORRAX has two disjoint dependency systems that do not know about each other:

1. **A Python-packaging system** (`pyproject.toml`) that declares pure-Python deps
   (`jax`, `h5py`, `numpy`, …) and *thinks* it produces an installable package.
2. **An out-of-band native system** — a hand-rolled mesh of `stage_*.sh` copy scripts,
   a standalone CMake build (`build.sh`), Shifter bind-mounts at fixed container paths
   (`/lorrax_nvhpc`, `/lorrax_phdf5`, `/lorrax_slate`), and env-var path resolution
   scattered across `run_shifter.sh`, `site_config.sh`, the Lmod modulefile, and
   `CMakeLists.txt`.

The native system is **invisible to packaging**. `pip install lorrax` / `uv sync`
produces a tree that cannot run any FFI codepath, because (a) `liblorrax_ffi.so` is a
gitignored build artifact that no install step ever produces, and (b) the three system
libs it links against are neither declared, versioned, nor fetched by anything Python
can see. The `build` dependency-group (scikit-build-core/cmake/ninja/pybind11/nanobind)
that *would* bridge the two systems is declared but **never wired in** — there is no
`[build-system]` table, so it is dead weight.

This report enumerates the antipatterns with `file:line` citations, then proposes a
target architecture modeled on how JAX and other native-extension Python packages handle
CUDA/native deps, plus a migration path.

---

## 1. Antipatterns (current state → problem → fix)

### A1. No `[build-system]` table; the native extension is never built on install
**Current state.** `pyproject.toml` has `[project]`, `[project.scripts]`,
`[tool.uv] package = true` (line 28-29), and a `build` dependency-group
(`pyproject.toml:63-69`: scikit-build-core, cmake, ninja, pybind11, nanobind). But there
is **no `[build-system]` table at all** (confirmed: `grep -n build-system pyproject.toml`
→ no match). With `[tool.uv] package = true` and no `build-system`, uv/pip fall back to
the legacy **setuptools** backend, which only copies `.py` files. It never invokes CMake,
never compiles `src/ffi/common/cpp/`, and never produces `liblorrax_ffi.so`.

**Problem.** The single most important build artifact in the package — the FFI shared
object that every cuSolverMp/phdf5/SLATE codepath dlopen's via
`src/ffi/common/ffi_loader.py:84` — is produced **only** by a human running
`bash src/ffi/common/cpp/build.sh` by hand, inside a Shifter shell, on a node where three
system libs happen to be bind-mounted. A fresh `git clone` + `uv sync` yields an
"installed" package that raises `FileNotFoundError: Could not locate liblorrax_ffi*.so.
Build with: bash src/ffi/common/cpp/build.sh` (`ffi_loader.py:89-93`) the first time any
FFI is touched. This is the **single biggest from-scratch onboarding cliff** (CONTEXT.md
validated fact).

**Fix.** Add a real `[build-system]` table that compiles the extension on install. The
`build` group already names exactly the right tool — **scikit-build-core** — it is simply
not connected:
```toml
[build-system]
requires = ["scikit-build-core>=0.11", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
cmake.source-dir = "src/ffi/common/cpp"
wheel.py-api = "cp312"
```
This makes `pip install .` / `uv sync` run CMake → ninja → produce `liblorrax_ffi.so`
into the wheel automatically (this is exactly what JAX's own `jaxlib` and packages like
`nanobind`-based extensions do). See §2 for the find_package work this requires.

### A2. The `build` dependency-group is vestigial / misleading
**Current state.** `pyproject.toml:63-69` declares a `build` group with
scikit-build-core, cmake, ninja, **pybind11**, and **nanobind**. Yet
`src/ffi/common/ffi_loader.py:3` states plainly: *"The library is a plain C shared object
(**no pybind/nanobind**)."* The CMake build (`CMakeLists.txt`) never `find_package`s
pybind11 or nanobind, and `grep` finds no `#include <pybind11...>`/`<nanobind...>`
anywhere in `src/ffi/`.

**Problem.** Two failure modes. (1) A reader looking at `pyproject.toml` reasonably
concludes the FFI is a pybind/nanobind extension and that some build backend consumes
this group — neither is true; it is pure documentation rot encoded as a dependency list.
(2) Because the group is not referenced by `[build-system].requires`, installing it
(`uv sync --group build`) gives you cmake/ninja but still does **not** trigger a build —
it is a footgun that looks like the build path but isn't.

**Fix.** Either wire it into `[build-system].requires` (per A1) and **drop pybind11**
(genuinely unused), or delete the group. Keeping a dependency list that contradicts the
code is worse than having none.

### A3. Native runtime deps are not captured in packaging — manually staged out-of-band
**Current state.** The three FFI runtime libs are obtained by hand-run copy scripts:
- `src/ffi/cusolvermp/scripts/stage_nvhpc.sh` (NVHPC subset) /
  `src/ffi/cusolvermp/scripts/stage_pypi.sh` (PyPI wheel).
- `src/ffi/phdf5/scripts/stage_cray.sh` (Cray HDF5 + MPICH-ABI shim) /
  `stage_openmpi.sh`.
- `src/ffi/slate/scripts/stage_cray.sh` (Cray libsci + `libmpi_gtl_cuda` + xpmem +
  lustreapi + a `libreadline.so.7→.so.8` shim, `slate/scripts/stage_cray.sh:65`).

None of these are invoked by any install step. They `cp -aL` files into a `/pscratch` (now
`$HOME/software`) tree that nothing in `pyproject.toml` references.

**Problem.** "Install LORRAX" is really a 6-step manual runbook (`PORTING.md:78-106`):
`module spider nvhpc` → run 3 stage scripts → build SLATE from source → edit
`site_config.sh` → `install.sh` → `build.sh`. There is no single command, no manifest of
what versions were staged, and no way to reproduce or verify a stage. A teammate who runs
`uv sync` and nothing else gets a package that cannot do distributed linear algebra or
parallel I/O and won't tell them why until runtime.

**Fix.** See §2. The right model is: cuSolverMp comes from a **versioned PyPI wheel**
(already half-done — `stage_pypi.sh` proves the wheel exists:
`nvidia-cusolvermp-cu12==0.7.2.888`), declared as an optional extra; HDF5/MPI/SLATE are
declared as a **system-library contract** (one config file naming prefixes + a
`find_package`-driven build), not a pile of copy scripts.

### A4. `liblorrax_ffi.so` is a gitignored artifact a fresh clone lacks
**Current state.** `.gitignore:72` (`src/ffi/**/*.so`) and `.gitignore:71`
(`src/ffi/**/cpp/build/`) exclude both the artifact and its build tree. Confirmed:
`git ls-files src/ffi | grep '\.so$'` → none tracked. Runtime discovery
(`ffi_loader.py:63-81`) searches `cpp/build/`, `$LORRAX_FFI_SO`, and `sys.path` for
`liblorrax_ffi*.so` — i.e. it expects a file that only `build.sh` produces.

**Problem.** Correct to gitignore a build artifact — the antipattern is gitignoring it
**with no automated build to regenerate it on install** (A1). The two together mean the
package's most important file simply does not exist after a clone, and the only recovery
is the manual `build.sh` cliff.

**Fix.** A1 (build on install) makes the `.so` a wheel artifact, so the gitignore is then
correct and harmless. Keep the gitignore; fix the build.

### A5. Hardcoded user/site absolute paths baked into shipped files
**Current state.** Person-specific `jackm` / `/global/homes/j/jackm` / `scratchperl`
paths appear in **shipped, tracked** files (not just the gitignored build cache):
- `config/perlmutter/site_config.sh:25` `LORRAX_SITE_PACKAGES="$HOME/scratchperl/.isdf/isdf_venvs/isdf_site"`
- `config/perlmutter/site_config.sh:29` `LORRAX_DEPS="/pscratch/sd/j/jackm/lorrax_sandbox/sources"`
- `src/ffi/common/cpp/run_shifter.sh:95` `: "${LORRAX_SRC:=/global/u2/j/jackm/software/lorrax/src}"`
- `src/ffi/common/cpp/run_shifter.sh:96` `LORRAX_SITE:=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site`
- `src/ffi/common/cpp/run_shifter.sh:108` `SLATE_INSTALL_HOST=...:-/global/homes/j/jackm/software/slate/install`
- `src/ffi/common/cpp/CMakeLists.txt:348` SLATE default `/global/homes/j/jackm/software/slate/install`
- `PORTING.md:32` documents the SLATE auto-probe as `/global/homes/<u>/software/slate/install`.

(The gitignored `build/` cache additionally bakes `/pscratch/sd/j/jackm/...` into
`CMakeCache.txt` and every `*.cc.o.d` — confirmed present in this checkout. That's
expected for a build dir but reinforces why it must be regenerated, not shipped.)

**Problem.** Every one of these is a default that only works for one person on one
machine. `run_shifter.sh` is the canonical FFI launcher and its defaults point at
`jackm`'s home. A different user on Perlmutter who relies on the script defaults gets
silent wrong `PYTHONPATH`/`LD_LIBRARY_PATH` (their code runs against Jack's checkout, or
fails). `site_config.sh` is *meant* to be edited per-site, but `run_shifter.sh` and
`CMakeLists.txt` bypass it with their own hardcoded fallbacks.

**Fix.** No tracked file should contain a username. Defaults must be `$HOME`-relative or
unset-with-loud-error. `run_shifter.sh` should derive `LORRAX_SRC` from its own location
(`$SCRIPT_DIR/../../../..`), not a hardcoded `jackm` path. The SLATE default in
`CMakeLists.txt:348` should be `$ENV{HOME}/software/slate/install` or empty. See small
fixes S1–S4.

### A6. No version pinning of the system libs
**Current state.** The host-side native libs are unpinned. The *only* place a version is
asserted is `LORRAX_NVHPC_SUBPATH="0.7.2_cuda12.9/..."` in `site_config.sh:87` — a
**string baked into a path**, not a checked constraint. `stage_cray.sh` (phdf5) hardcodes
a fallback `1.12.2.9` HDF5 path (`phdf5/scripts/stage_cray.sh:36`) and MPICH `9.0.1`
(`:41`) "observed on Perlmutter as of April 2026" — i.e. whatever the module system
happened to expose that day. SLATE is "built from source against the target MPI"
(`PORTING.md:17`) with no recorded commit/version. `PORTING.md:9-17` gives *minimums* in
prose but nothing enforces them.

**Problem.** The dependency set is whatever was on the login node when someone last ran
the stage scripts. There is no lockfile, no recorded (cuSolverMp, HDF5, MPICH, libsci,
SLATE-commit, CUDA, NCCL) tuple, and no failure if the host bumps cray-mpich to 9.1 with
a different SONAME. The detailed correctness notes in `site_config.sh:74-86` (0.6.0
silently returns wrong answers on Px>1,Py>1 meshes; 0.7.2 is the validated baseline)
prove version *matters enormously* — yet nothing pins it.

**Fix.** Pin cuSolverMp via PyPI version in an extra (`nvidia-cusolvermp-cu12==0.7.2.888`).
For HDF5/MPI/libsci/SLATE — which come from the host module system — record the exact
versions in a per-site **lockfile/manifest** that the stage step writes and the build step
verifies (e.g. emit a `staged.lock` JSON with resolved SONAMEs + module versions; fail the
build if the live host disagrees). Add a runtime `lrx_version_info`-style check
(`ffi_loader.py:270` already reports CUDA/NCCL versions) that asserts the cuSolverMp ABI.

### A7. Bind-mount-at-fixed-container-path coupling
**Current state.** Three host trees are bind-mounted at hardcoded container paths
`/lorrax_nvhpc`, `/lorrax_phdf5`, `/lorrax_slate` (`run_shifter.sh:101-111`). These fixed
paths are then **hardcoded a second time** in unrelated places:
- `CMakeLists.txt:69-70` NVHPC autodetect globs `/lorrax_nvhpc/*`.
- `CMakeLists.txt:258-262` HDF5 root defaults to `/lorrax_phdf5`.
- `CMakeLists.txt:64` (run_shifter) MPICH include `=/lorrax_phdf5/include`.
- `CMakeLists.txt:418` `INSTALL_RPATH` bakes `/lorrax_phdf5/lib;/lorrax_slate/lib` into
  the `.so`.
- `run_shifter.sh:117,132` LD_LIBRARY_PATH / LD_PRELOAD reference the same.
- The Lmod modulefile (`config/modulefiles/lorrax/0.1.0.lua:181`) `LD_PRELOAD`s
  `/lorrax_slate/lib/libmpi_gtl_cuda.so.0`.

**Problem.** The container-internal mount point is a magic constant duplicated across the
launcher, CMake, the modulefile, and the RPATH baked into the binary. Changing it requires
edits in ≥5 files, and the RPATH means the `.so` is **only loadable when those exact mount
points exist** — it is welded to Shifter-on-Perlmutter. On Apptainer/Singularity (a stated
goal, CONTEXT.md) or bare-metal, the RPATH points at non-existent dirs and the loader falls
back to LD_LIBRARY_PATH, which then must replicate the same fiction. `PORTING.md:108-112`
hand-waves "swap the shifter invocation for apptainer --bind ..." but the
`/lorrax_*`-baked RPATH and the 5-way duplication make that non-trivial.

**Fix.** Define the mount points **once** (e.g. in `site_config.sh` as
`LORRAX_FFI_NVHPC_MOUNT=/lorrax_nvhpc`) and reference that variable everywhere. Better:
when the build-on-install model (§2) puts cuSolverMp inside the Python env, the `.so` can
RPATH `$ORIGIN`-relative into the wheel's bundled libs (JAX's model) and the
bind-mount-at-fixed-path coupling disappears for the wheel-shippable deps. HDF5/MPI stay
host-provided but should be located by `find_package(HDF5)`/`find_package(MPI)` with
`HDF5_ROOT`/`MPI_HOME` env contracts, not a hardcoded `/lorrax_phdf5`.

### A8. JAX version skew: declared `>=0.9.0`, runtime is ~0.5.3
**Current state.** `pyproject.toml:9-10` and the `jax` group (`:59-62`) pin
`jax[cuda13]>=0.9.0` + `jaxlib>=0.9.0`. The production Shifter image is
`nvcr.io/nvidia/jax:25.04-py3` (`site_config.sh:32`, `run_shifter.sh:45`), which ships
**JAX ~0.5.3** (CONTEXT.md validated fact). `PORTING.md:14` even lists the JAX minimum as
**0.5** and says it "must match CUDA major" — directly contradicting the `>=0.9.0` pin.
Note also `cuda13` extra vs the container's CUDA 12.9 (`stage_pypi.sh:22`,
`CMakeLists.txt` links 12.9): the extra requests a CUDA-13 jaxlib build.

**Problem.** Three contradictions in one dependency line: (1) `>=0.9.0` excludes the
0.5.3 the code actually runs against; a clean `uv sync` would pull a JAX the FFI was never
built/tested against (and `jax.ffi.include_dir()` in `CMakeLists.txt:147` would emit
headers for a different XLA FFI API version than the runtime — `build.sh:62-67` explicitly
warns the FFI API version is baked into the header and must match the runtime). (2)
`cuda13` vs CUDA-12.9 container is a major-version mismatch. (3) The pin disagrees with the
package's own porting doc. This is the classic "declared deps describe an aspirational
future, the container freezes the real past" skew.

**Fix.** Pin to what runs: `jax>=0.5.3,<0.6` with the CUDA extra matching the container
(`cuda12`). If JAX is **provided by the container** (it is — the NVIDIA image bundles it),
then JAX should be an *expected-present* dep, not a `>=` pull: declare it under an extra
(`pip install lorrax[cuda]`) for non-container installs, but mark it provided in the
container path so `uv sync` inside the container does not clobber the image's jaxlib. The
FFI ABI dependency on the exact jaxlib is real (`build.sh:62-67`) — that's an argument for
pinning narrowly, not loosely.

### A9. Env-var path resolution scattered across shell, CMake, Lua, and config
**Current state.** The same physical paths are resolved by **four** independent
mechanisms with overlapping-but-not-identical variable names and defaults:
- `config/perlmutter/site_config.sh:102-108` defines `LORRAX_FFI_{NVHPC,PHDF5,SLATE}_DIR_DEFAULT`.
- `run_shifter.sh:44,80,107` reads `LORRAX_FFI_{NVHPC,PHDF5,SLATE}_DIR` with **its own**
  hardcoded defaults (`$HOME/software/lorrax_nvhpc`, etc.) that *duplicate* but could
  drift from site_config's.
- `CMakeLists.txt:33-128` has a 5-rung NVHPC autodetect ladder + separate
  `HDF5_ROOT`/`HDF5_DIR`/`LORRAX_MPI_INCLUDE_DIR`/`LORRAX_MPICH_LIB_DIR`/
  `LORRAX_SLATE_INSTALL_DIR` resolution (`:257-350`).
- The modulefile (`0.1.0.lua:108-110`) reads the same `LORRAX_FFI_*_DIR` env vars again
  with `env_or(...)` and `@..._DEFAULT@` template substitution.
- `build.sh:23` introduces **yet another** name, `LORRAX_NVHPC_ROOT` (default
  `/lorrax_nvhpc/25.5_cuda12.9`), distinct from `LORRAX_FFI_NVHPC_DIR`.

**Problem.** There is no single source of truth for "where is cuSolverMp." The launcher,
the build, the module, and the config each re-derive it, with subtly different env-var
names (`LORRAX_FFI_NVHPC_DIR` vs `LORRAX_NVHPC_ROOT` vs `NVHPC_ROOT` vs `HPCSDK_ROOT`) and
different hardcoded fallbacks. Changing a path means auditing four files; a stale default
in any one silently wins depending on which entrypoint the user happened to invoke. The
NVHPC autodetect ladder alone (`CMakeLists.txt:42-128`, ~85 lines) is a smell: it exists
to paper over the absence of a single contract.

**Fix.** One config file (`site_config.sh`, already the intended owner) defines the
canonical variables; `run_shifter.sh`, the modulefile, and `build.sh` **source it** rather
than re-defaulting. CMake reads the same variables via `-D`/env passed by `build.sh`
(no independent ladder). Collapse `LORRAX_NVHPC_ROOT`/`LORRAX_FFI_NVHPC_DIR`/`NVHPC_ROOT`
to one name. This is the "single config/env contract" JAX-class packages use (one
`find_package`, env honored, no per-script reinvention).

### A10. Two MPI stacks selected by a runtime env var that must agree at build time
**Current state.** `LORRAX_PHDF5_MPI_STACK` (mpich|openmpi) is read at **runtime** by
`run_shifter.sh:42` to pick mount paths, `--mpi=` flavor, and LD paths — but the choice is
also baked into the **build** (`CMakeLists.txt:276-335`: MPI lib/include resolved from
`LORRAX_MPI_*` env, producing a `.so` with a specific `DT_NEEDED libmpi_*.so.12` vs
`libmpi.so.40`). `build.sh:35-48` fails loudly if the MPI env vars are unset to prevent a
silently-wrong `.so` — good — but nothing guarantees the *runtime* `LORRAX_PHDF5_MPI_STACK`
matches the stack the `.so` was *built* against.

**Problem.** A user can build the `.so` against MPICH and then launch with
`LORRAX_PHDF5_MPI_STACK=openmpi` (or vice versa); the mismatch surfaces as a runtime
SONAME/segfault (the exact failure `build.sh:40-43` and KNOWN_SANDBOX_ERRORS 2026-05-10
describe). The build-time/runtime stack agreement is a manual invariant, unenforced.

**Fix.** Record the stack the `.so` was built against (embed in the binary or a sidecar
`build_info.json`), and have `ffi_loader.py` / `run_shifter.sh` assert the runtime stack
matches. Better long-term: with `find_package(MPI)` and the wheel built per-stack, ship
the stack tag in the wheel metadata.

---

## 2. Target dependency architecture

The guiding model: **make the native extension a first-class part of the Python package**
(JAX/jaxlib, `h5py`, `mpi4py`, scikit-build-core exemplars), and reduce "system libs" to a
small, explicit, locatable contract instead of a copy-script mesh.

### Layer 1 — Build the extension on install (scikit-build-core + CMake)
Add `[build-system]` (A1) so `pip install .` compiles `liblorrax_ffi.so`. The existing
`CMakeLists.txt` is already 90% of a proper config-driven build — it just needs to be
**driven by the standard backend** instead of `build.sh`. scikit-build-core hands CMake
the right `Python3_EXECUTABLE`/install paths; the `.so` lands in the wheel under
`lorrax/ffi/`. `ffi_loader.py:63-81` already searches `sys.path` for the `.so`, so an
installed-into-site-packages artifact is found with no loader change.

This is precisely the pattern JAX uses for `jaxlib` (Bazel there, but the contract is
identical: the C++/CUDA extension is a wheel artifact, not a hand-built side file), and
that `nanobind`/`pybind11` extension packages use via scikit-build-core's documented
"CMake `find_package`" flow.

### Layer 2 — cuSolverMp from a versioned wheel, as an extra
`stage_pypi.sh` already proves the clean path exists: `nvidia-cusolvermp-cu12` is on PyPI
with versioned, ABI-meaningful releases (`stage_pypi.sh:10-27`). Promote it to a real
declared dependency:
```toml
[project.optional-dependencies]
cuda12 = ["jax[cuda12]>=0.5.3,<0.6", "nvidia-cusolvermp-cu12==0.7.2.888",
          "nvidia-nccl-cu12>=2.26"]
```
`pip install lorrax[cuda12]` then *fetches the validated 0.7.2 build* (fixing A6 for
cuSolverMp) and the wheel's `libcusolverMp.so.0` lands in site-packages, which CMake's
`find_package` (or a `nvidia.cu12` import-path probe) locates at build time — no
`stage_nvhpc.sh`, no `/lorrax_nvhpc` bind-mount, no RPATH-to-mount-point (A7). This is the
NVIDIA-recommended way to consume CUDA math libs in Python (the `nvidia-*-cu12` wheel
family). The `.so` then RPATHs `$ORIGIN/../nvidia/cu12/lib`, the way JAX/CUDA wheels do.

### Layer 3 — Host MPI/HDF5/SLATE as an explicit system-library contract
HDF5/MPI/libsci/SLATE legitimately come from the host (Cray PE). Keep that, but make it a
**contract, not a copy mesh**:
- **One config file** (`config/<cluster>/site_config.sh`, already the intended owner)
  declares: `HDF5_ROOT`, `MPI_HOME`/`MPICH_DIR`, `SLATE_DIR`, and the bind-mount points as
  **single variables** (A9, A7). Every consumer (`build.sh`, `run_shifter.sh`,
  modulefile) **sources this file** instead of re-defaulting.
- **CMake uses standard `find_package`** — `find_package(HDF5 COMPONENTS C)` (already
  present, `CMakeLists.txt:266`) and `find_package(MPI)` (currently *avoided*,
  `:276-296`, in favor of a hand-rolled `LORRAX_MPI_*` path scheme to pin the ABI). The
  ABI-pinning concern is real, but it can be expressed as `MPI_HOME=<stage>` +
  `MPIEXEC`/`MPI_C_COMPILER` hints to a standard `find_package(MPI)`, which is more
  portable than the bespoke `LORRAX_MPICH_LIB_DIR` + manual `MPI::MPI_CXX` stand-in
  (`:362-367`).
- **Stage scripts become version-recording, not just copying** — emit `staged.lock`
  (resolved module versions + SONAMEs) that `build.sh` verifies against the live host
  (A6). Even better, replace the `stage_*.sh` mesh with a declarative manifest the build
  consumes.

### Layer 4 — Optional reproducible recipe (spack / pixi / conda)
For the "different cluster, no Shifter" goal (CONTEXT.md), provide **one** reproducible
environment recipe so users don't hand-build SLATE + stage Cray libs:
- A **spack** environment (`spack.yaml`) pinning `slate`, `hdf5+mpi`, `nccl`,
  `cusolvermp`, and `jax` — spack already packages SLATE and parallel HDF5, and is the
  HPC-native answer. This replaces "clone icl-utk-edu/slate, build from source"
  (`PORTING.md:87-90`) with `spack install`.
- Or a **pixi**/conda-forge env for non-Cray sites (`hdf5=*=mpi_*`, `openmpi`, JAX), which
  is what `stage_openmpi.sh` is informally reaching toward.

### Layer 5 — Proper extras and a support matrix
Replace the single muddled dependency list with extras that map to install paths
(mirrors JAX's `jax[cuda12]` / `jax[cuda13]` UX and PySCF/ASE optional-deps):
```
lorrax                 # pure-python core (docs build, parsers) — no GPU
lorrax[cuda12]         # JAX-CUDA12 + cusolvermp wheel + nccl
lorrax[ffi]            # build the native extension (implies a system-lib contract)
lorrax[docs]           # mkdocs-material etc. (currently jammed into core deps, A11)
lorrax[dev]            # pytest, flake8 (already a group)
```

### A11 (bonus, found while mapping extras) — docs tooling is in *runtime* deps
`pyproject.toml:12-15` puts `mkdocs`, `mkdocs-material`, `mkdocstrings`,
`mkdocstrings-python` in `[project.dependencies]` (core runtime). These are doc-build-only.
Every `pip install lorrax` drags in the entire mkdocs stack. **Fix:** move to a `docs`
extra/group.

---

## 3. Migration path (incremental, each step independently shippable)

1. **Stop the bleeding (S-fixes, ~1 hr):** remove `jackm` paths from tracked files
   (S1–S4); fix the JAX pin to `>=0.5.3,<0.6` + `cuda12` (A8); move mkdocs to a `docs`
   extra (A11); delete or fix the contradictory `build` group's pybind11 (A2).
2. **Wire the build backend (A1):** add `[build-system]` = scikit-build-core; point
   `tool.scikit-build.cmake.source-dir` at `src/ffi/common/cpp`. Keep `build.sh` as a thin
   wrapper that calls `pip install -e .` so existing muscle memory still works. Gate the
   FFI build behind the `[ffi]` extra so the pure-python `pip install lorrax` still works
   GPU-less.
3. **cuSolverMp via wheel (Layer 2):** add `[cuda12]` extra with
   `nvidia-cusolvermp-cu12==0.7.2.888`; teach CMake to find it in site-packages
   (`importlib`-resolved path) ahead of the `/lorrax_nvhpc` ladder; retire `stage_nvhpc.sh`
   for the common case (keep for sites that want the full SDK).
4. **Single config contract (A9, A7):** make `run_shifter.sh`, the modulefile, and
   `build.sh` all `source site_config.sh`; collapse `LORRAX_NVHPC_ROOT`/`LORRAX_FFI_NVHPC_DIR`
   to one name; define mount points once.
5. **Version lock for host libs (A6):** stage scripts emit `staged.lock`; build verifies.
6. **Reproducible recipe (Layer 4):** add `spack.yaml` (Cray) and a conda/pixi env
   (non-Cray); update PORTING.md to point at them instead of the manual runbook.

---

## 4. Single highest-leverage change

**Add a real `[build-system]` table (scikit-build-core + CMake `find_package`) so the
native extension is compiled on `pip install` / `uv sync`, and ship cuSolverMp as the
already-existing `nvidia-cusolvermp-cu12==0.7.2.888` PyPI wheel via a `[cuda12]` extra.**

This single move converts the worst antipattern (A1/A4 — the gitignored `.so` that no
install produces, the biggest onboarding cliff per CONTEXT.md) into a normal wheel build,
activates the otherwise-dead `build` group (A2), removes the `/lorrax_nvhpc` bind-mount +
RPATH coupling for the one dep that can be wheel-shipped (A7), pins the
correctness-critical cuSolverMp version (A6), and gives `pip install lorrax[cuda12]` the
same one-command UX as `pip install jax[cuda12]` — the professionalism bar set by
CONTEXT.md. It is the linchpin that makes "another researcher can build and run LORRAX"
true instead of aspirational.
