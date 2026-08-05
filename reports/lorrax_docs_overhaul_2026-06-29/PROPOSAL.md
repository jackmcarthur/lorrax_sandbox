# LORRAX docs / dependency overhaul — consolidated proposal

**Synthesis of 4 lenses:** portability, dependency-architecture, docs-IA, onboarding.
**Repo under review:** `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D` (branch `agent/dep-home-migration`).
**Mode:** PROPOSE ONLY. No repo file edited by any analyst or by this synthesis.
**Source reports:** `agent_portability.md`, `agent_dependency-architecture.md`, `agent_docs-ia.md`, `agent_onboarding.md` (same directory).

---

## 1. Executive summary

LORRAX's documentation is a flat folder of 34 SHOUTING_CAPS Markdown files framed *"For AI agents,"* in which durable references sit beside transient dev scratchpads, and every install path silently assumes Jack's NERSC Perlmutter account. Measured against the JAX/PySCF/ASE bar (landing → Install → Quickstart → User Guide → API reference → Architecture → Contributing) there is no Installation page, no Quickstart, no rendered API reference, and no Contributing page — even though the project already hard-depends on the mkdocs-material + mkdocstrings stack (`pyproject.toml:12-15`) and never configured it (no `mkdocs.yml` outside `.venv`). The build story is worse than the prose: `pyproject.toml` has **no `[build-system]` table**, so `pip install`/`uv sync` never compiles the native FFI extension; `liblorrax_ffi.so` is a gitignored artifact that no install step produces, so a fresh clone dead-ends at `FileNotFoundError … Build with: bash src/ffi/common/cpp/build.sh` the first time any FFI path is touched. The three native libs (cuSolverMp, parallel HDF5, SLATE) are obtained only by `cp`-ing NERSC's `/opt/cray` and `/opt/nvidia` trees — yet all three are obtainable independently (a PyPI wheel `nvidia-cusolvermp-cu12==0.7.2.888` that `site_config.sh:74` itself endorses; conda-forge/spack HDF5; a SLATE source build) and the CMake build already accepts `-D` overrides to point anywhere. Declared deps also contradict reality (`jax[cuda13]>=0.9.0` vs the container's JAX ~0.5.3 / CUDA 12.9). The encouraging finding across all four lenses: **the content and the capability already exist and are high quality; what is missing is the documentation and packaging wiring that connects them to a non-NERSC user.**

---

## 2. Tier 1 — small immediate doc fixes (no restructuring)

Prioritized, deduplicated across all four lenses. "Mech" = ready-to-apply mechanical edit (string/path swap, no judgement call). Priority: P0 = blocker/credibility, P1 = correctness, P2 = polish.

| # | Pri | Mech | Location | Problem | Fix |
|---|-----|------|----------|---------|-----|
| T1 | P0 | ✅ | `README.md:14` | `gw_isdf/gw_jax.py` — package is `gw/`, `gw_isdf/` does not exist | `s|gw_isdf/|gw/|g` (also `w_isdf.py`→`gw/w_isdf.py`) |
| T2 | P0 | ✅ | `README.md:21-24` | quick-start runs `python -m gw.gw_jax -i cohsex.in`; no `cohsex.in` in repo root → fails on fresh clone | Point at bundled fixture `uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in`; add note "runs end-to-end on CPU, no GPU/native build" |
| T3 | P0 | ✅ | `README.md:39`, `ENVIRONMENT_COMPREHENSIVE.md:2-3`, `CODEBASE_COMPREHENSIVE.md:3`, `advanced/README.md:15` | "For AI agents" framing tells humans the docs aren't for them | Rewrite intros human-first; move agent-routing to `AGENTS.md` only |
| T4 | P0 | ✅ | `ENVIRONMENT_COMPREHENSIVE.md:238-240,361`, `config/README.md:109-111` | FFI default paths still show `$SCRATCH/...`; relocated to `$HOME/software` on 2026-06-24 (`site_config.sh:99-104`) — docs point at purged scratch | Replace all with `$HOME/software/lorrax_{nvhpc,phdf5_cray/stage,slate_cray/stage}`; §7 "under `$SCRATCH`"→"under `$HOME/software`" |
| T5 | P0 | ✅ | `ENVIRONMENT_COMPREHENSIVE.md:67` | `uv sync --no-install-project --locked` skips editable install → `ModuleNotFoundError: gw`; contradicts `README.md:21` bare `uv sync` | Drop `--no-install-project`; align with README |
| T6 | P1 | — | `pyproject.toml:9`, `ENVIRONMENT_COMPREHENSIVE.md:29` | `jax[cuda13]>=0.9.0` pinned; container ships JAX ~0.5.3 / CUDA 12.9; `PORTING.md:14` says JAX ≥0.5 — mutually contradictory | Reconcile to one truth: pin `jax>=0.5.3,<0.6` + `cuda12`, or state both as distinct supported configs in a matrix (overlaps Tier 3 A8) |
| T7 | P1 | — | `docs/index.md:18,35` | Links to nonexistent `formalism.md` + `examples/`; "Key modules" cites `src/isdf/common/wfnreader.py` (absent; real is `src/common/load_wfns.py`) | Repoint to `archive/formalism.md`/real theory page + `src/common/load_wfns.py`; drop/replace `examples/` with `tests/regression/cohsex_debug/` |
| T8 | P1 | — | `config/README.md:129` | Labels `LORRAX_MPI_TYPE=pmix` "legacy OpenMPI (not wired up)" while `run_shifter.sh:31,58` + `PORTING.md:144` treat it as the active OpenMPI runtime — the one knob non-Cray clusters need | State consistently: `pmix` is the OpenMPI launch protocol, not default (Cray faster on PM), correct for OpenMPI clusters |
| T9 | P1 | — | `ENVIRONMENT_COMPREHENSIVE.md:252-261`, `src/ffi/AGENTS.md:17-37`, `stage_openmpi.sh:6` | Three docs name three different "default" phdf5 stacks (`run_shifter.sh:42`/`PORTING.md:134` say mpich since 2026-04-20) | Pick one canonical default; correct `stage_openmpi.sh:6-9` header (now fallback, not default); note OpenMPI is the portable one |
| T10 | P1 | — | `ENVIRONMENT_COMPREHENSIVE.md:348-378` (§7) | "Generic SLURM" bare-venv example only runs the pure-JAX no-FFI path; silently omits that FFI features need native libs + a built `.so` | Add admonition: example is pure-JAX only; distributed FFI needs the native stack (link Tier 2/3); stage knobs assume Cray PE |
| T11 | P2 | — | `ENVIRONMENT_COMPREHENSIVE.md:281-290` (§5.6), `:242-250` | `LD_PRELOAD=libmpi_gtl_cuda.so.0` + `MPICH_GPU_SUPPORT_ENABLED=1` presented as universal; they are Cray-MPICH-only | Label "(Cray MPICH stack)"; add OpenMPI/UCX note (`MPICH_*`/`gtl` vars do not apply) |
| T12 | P2 | — | `config/README.md:29-33` | "`lxpre cohsex.in 640`" opaque; 3 preprocessing modules + outputs invisible | Add 3-line expansion: `centroid.kmeans_cli`→centroids, `psp.get_dipole_mtxels`→`dipole.h5`, `gw.kin_ion_io_chunked`→`kin_ion.h5` (`0.1.0.lua:325-334`) |
| T13 | P2 | ✅ | `NEW_WINDOW_MINIMAX_GUIDELINES.md:1` | Begins with raw LLM artifact `"Absolutely — here is a cleaned-up…"` | Strip conversational preamble (or archive whole file) |
| T14 | P2 | — | `ENVIRONMENT_COMPREHENSIVE.md:263-269` (§5.4) | FFI build shown as one command; omits the two hard prereqs (GPU alloc + staged NVHPC) `build.sh:6,35-47` require | Prepend "Prereqs: (a) `stage_nvhpc.sh` has run; (b) hold a GPU allocation (`lxalloc`)" |
| T15 | P2 | — | `README.md:22` | "~15s" pytest claim; the lone collected test is a `@pytest.mark.regression` end-to-end subprocess, slower on CPU | Re-label "regression smoke test (CPU, ~1–2 min)" or add a `-m "not regression"` fast lane |

Note: the "remove `jackm`/personal absolute paths from tracked files" fixes (`run_shifter.sh:95-96,108`, `CMakeLists.txt:348`, `site_config.sh:25,29`) are listed as Tier 3 step S1 because they touch shipped scripts, not docs — but the *doc* placeholders (`$LORRAX_ROOT`, `<module-name>` instead of `lorrax_A|B|C`, `/path/to/lorrax_X`) are mechanical Tier-1 edits.

---

## 3. Tier 2 — docs restructuring (information architecture)

**Action:** create `mkdocs.yml` (the stack is already paid for, never configured), wire mkdocs-material + mkdocstrings, and define a `nav` that separates **user docs** from **developer/agent notes**. Target tree (durable topic name; source file + disposition in parens):

```
mkdocs.yml                       NEW — wire the already-declared stack
docs/
  index.md                       REWRITE landing (from index.md + README ¶1-2; drop broken links)
  installation/
    index.md                     REWRITE — support matrix + from-source (generic parts of ENV §7)
    perlmutter.md                MOVE — site-specific (config/README + ENV Perlmutter §§; lorrax_A|B|C, lxrun live here)
    ffi-native-libs.md           NEW — cuSolverMp/phdf5/SLATE acquisition + liblorrax_ffi.so build (#1 cliff)
  quickstart.md                  MERGE — the 3 snippets (README:18-24 / index:53-58 / AGENTS:59-86); anchor on the fixture
  user-guide/
    inputs.md, running-gw.md, centroids.md, outputs.md   NEW (from sandbox COHSEX docs)
  theory/
    overview.md (archive/formalism.md, promoted), physics.md (PHYSICS_COMPREHENSIVE),
    isdf-zeta-vq.md (ZETA_V_Q_ALGORITHMS), minimax-quadrature.md (MINIMAX_QUADRATURE),
    symmetry.md (SYMMETRY_COMPREHENSIVE)                  KEEP, rename, de-emoji
  architecture/
    codebase.md (CODEBASE_COMPREHENSIVE), memory-model.md (MEMORY_MODEL),
    multihost.md (advanced/jax_multihost)                KEEP, de-"agent"
  api/                           GENERATED by mkdocstrings — not hand-written
  contributing.md                NEW — humanize AGENTS.md:88-104 coding standards
  changelog.md                   NEW
docs/dev/                        NEW — explicitly OUT of site nav
  plans/     ← FREQ_INTEGRATION_REWRITE_PLAN, PLAN_zeta_g_flat_migration, plans/*
  progress/  ← FREQ_INTEGRATION_PROGRESS, SIGMA_FREQ_AUDIT_STATUS
  notes/     ← PROFILING_SUGGESTIONS, NEW_WINDOW_MINIMAX_GUIDELINES, GN_PPM_..._REVISED, AGENT_TODO
  archive/   ← existing docs/archive/** (frozen)
```

**Explicit dispositions on named files:**
- **KEEP (promote into nav, light edit):** `PHYSICS_COMPREHENSIVE.md`, `CODEBASE_COMPREHENSIVE.md`, `MEMORY_MODEL.md`, `SYMMETRY_COMPREHENSIVE.md`, `MINIMAX_QUADRATURE.md`, `ZETA_V_Q_ALGORITHMS.md`, `advanced/jax_multihost.md`, `advanced/HL_GPP_derivation.md`. This is the best material in the repo; it only needs a home.
- **REWRITE:** `index.md` (landing), `README.md` (de-agent, fix paths, lead with Path-A smoke test), `ENVIRONMENT_COMPREHENSIVE.md` → split into `installation/index.md` (generic) + `installation/perlmutter.md` (site).
- **MERGE:** the three "quickstart" snippets → one `quickstart.md` anchored on the shipped `tests/regression/cohsex_debug/` fixture.
- **ARCHIVE into `docs/dev/` (out of nav):** all `*_PLAN*.md`, `*_PROGRESS.md`, `*_AUDIT_STATUS.md`, `AGENT_TODO.md`, `PROFILING_SUGGESTIONS.md`, `NEW_WINDOW_MINIMAX_GUIDELINES.md`, `GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`, `plans/*`. (~10 of ~19 top-level docs are dev ephemera — this is the root IA defect in one number.)
- **CUT (delete):** `pydoc-markdown.yml` (points at nonexistent `gw_isdf`/`isdf` packages); the buggy `docs/gen_api_docs.sh` (line 18 tries to execute the output dir as a command); the pdoc/pydoc-markdown story in README/index; `pdoc` + `pydoc-markdown` from `[dependency-groups].dev` (`pyproject.toml:55-56`); `archive/nufft/` if the NUFFT backend is truly dead.

**Pick one API mechanism — mkdocstrings — and delete the other two.** Today three conflicting, all-broken API mechanisms are referenced (README says `pdoc`; `gen_api_docs.sh` runs `pydoc-markdown` and has a bug; `pydoc-markdown.yml` targets dead packages). mkdocstrings is already a dep, is the JAX path, and renders the existing NumPy-style docstrings `AGENTS.md:90` already mandates.

**Tone conventions to adopt:** mkdocs-material admonitions (`!!! warning` for the FFI build cliff); lowercase topical filenames (no `_COMPREHENSIVE`/`_REVISED`/`_STATUS` in the user tree); no site/person paths in nav docs (`$LORRAX_ROOT`, `<module-name>` placeholders; literal NERSC paths confined to `installation/perlmutter.md`); one source of truth per fact (install info currently duplicated and drifted across README/AGENTS/index/ENV/config-README).

---

## 4. Tier 3 — dependency-architecture refactor

The root defect: **two disjoint dependency systems that don't know about each other** — a Python-packaging system (`pyproject.toml`) that thinks it produces an installable package, and an out-of-band native system (`stage_*.sh` copy scripts, standalone `build.sh`, Shifter bind-mounts at `/lorrax_{nvhpc,phdf5,slate}`, env-var resolution scattered across shell/CMake/Lua/config). The native system is invisible to packaging. Target model (mirrors JAX `jaxlib`, `h5py`, `mpi4py`, scikit-build-core exemplars):

**Layer 1 — build the extension on install.** Add a real `[build-system]` table (scikit-build-core + CMake) pointing `cmake.source-dir` at `src/ffi/common/cpp`. `pip install`/`uv sync` then compiles `liblorrax_ffi.so` into the wheel; `ffi_loader.py:63-81` already searches `sys.path`, so no loader change. The `build` group already names exactly scikit-build-core — it is simply never connected (no `[build-system]` → setuptools fallback that only copies `.py`).

**Layer 2 — cuSolverMp from a versioned wheel, as an extra.** `stage_pypi.sh` already proves the wheel exists. Declare `[cuda12] = ["jax[cuda12]>=0.5.3,<0.6", "nvidia-cusolvermp-cu12==0.7.2.888", "nvidia-nccl-cu12>=2.26"]`. `pip install lorrax[cuda12]` fetches the validated 0.7.2 build (the version that *matters* — 0.6.0 silently returns wrong answers on Px,Py>1 meshes per `site_config.sh:74-86`), lands `libcusolverMp.so.0` in site-packages, and lets the `.so` RPATH `$ORIGIN/../nvidia/cu12/lib` — eliminating the `/lorrax_nvhpc` bind-mount + baked RPATH for this dep.

**Layer 3 — host MPI/HDF5/SLATE as an explicit contract, not a copy mesh.** One config file (`site_config.sh`, the intended owner) declares `HDF5_ROOT`, `MPI_HOME`, `SLATE_DIR`, and the mount points as single variables; `build.sh`, `run_shifter.sh`, and the modulefile **source it** instead of each re-defaulting (today the same path is resolved by 4 mechanisms with 4 different env-var names — `LORRAX_FFI_NVHPC_DIR` vs `LORRAX_NVHPC_ROOT` vs `NVHPC_ROOT` vs `HPCSDK_ROOT`). CMake uses standard `find_package(HDF5/MPI)` driven by those vars instead of the 85-line bespoke autodetect ladder. Stage scripts emit a `staged.lock` (resolved versions + SONAMEs) the build verifies.

**Layer 4 — optional reproducible recipe.** A `spack.yaml` (Cray: `slate`, `hdf5+mpi`, `nccl`, `cusolvermp`) and a conda/pixi env (non-Cray: `hdf5=*=mpi_openmpi_*`, `openmpi`, JAX) to replace the manual "clone icl-utk-edu/slate, build from source" runbook.

**Layer 5 — proper extras + support matrix.** `lorrax` (pure-python core, no GPU), `lorrax[cuda12]`, `lorrax[ffi]`, `lorrax[docs]` (mkdocs stack — **currently wrongly in `[project.dependencies]` so every install drags it in**, `pyproject.toml:12-15`), `lorrax[dev]`.

**Migration path + effort (each step independently shippable):**

| Step | Action | Effort |
|------|--------|--------|
| S1 | Stop the bleeding: strip `jackm` paths from tracked files (`run_shifter.sh:95-96,108`, `CMakeLists.txt:348`, `site_config.sh:25,29` → `$HOME`-relative or unset-with-loud-error); fix JAX pin to `>=0.5.3,<0.6`+`cuda12` (T6); move mkdocs to `docs` extra; drop unused `pybind11` from `build` group (FFI is plain C, `ffi_loader.py:3`) | **S** |
| S2 | Add `[build-system]` = scikit-build-core; gate FFI behind `[ffi]` extra so pure-python `pip install lorrax` still works GPU-less; keep `build.sh` as a thin `pip install -e .` wrapper | **M** |
| S3 | cuSolverMp via wheel: add `[cuda12]` extra; teach CMake to find it in site-packages ahead of the `/lorrax_nvhpc` ladder; retire `stage_nvhpc.sh` for the common case | **M** |
| S4 | Single config contract: `run_shifter.sh`/modulefile/`build.sh` source `site_config.sh`; collapse the 4 NVHPC env-var names to one; define mount points once | **M** |
| S5 | Version lock for host libs: stage scripts emit `staged.lock`; build verifies; record built-against MPI stack so runtime `LORRAX_PHDF5_MPI_STACK` mismatch is caught (not a segfault) | **M** |
| S6 | Reproducible recipe: add `spack.yaml` + conda/pixi env; point PORTING.md at them | **L** |
| S7 | Decouple `$SCRATCH`/`$HOME` filesystem vocab via one `LORRAX_STAGE_ROOT` (default `${SCRATCH:-$HOME}/lorrax-stage`); derive all stage defaults from it | **S** |
| S8 | Promote Apptainer/Singularity to first-class: factor container invocation into a `run_container.sh` wrapper keyed on `LORRAX_CONTAINER_RUNTIME={shifter,apptainer,singularity,none}`; replace the 3 "swap the invocation" hand-waves with one worked Apptainer example | **M–L** |

---

## 5. Cross-lens agreements & tensions

**Convergences (≥2 lenses) — these are the load-bearing findings:**

1. **The `liblorrax_ffi.so` build cliff is the #1 problem** — named by all four lenses (CONTEXT validated fact; dependency-arch A1/A4; portability B5; onboarding C6). Dependency-arch and portability agree on the *same* fix (`[build-system]` + scikit-build-core wiring the already-declared `build` group).
2. **cuSolverMp PyPI wheel is the unlock for the worst native dep** — portability B2 and dependency-arch Layer 2 independently land on `nvidia-cusolvermp-cu12==0.7.2.888`, both citing `site_config.sh:74` / `stage_pypi.sh` as proof it already exists.
3. **`gw_isdf/`→`gw/` and "For AI agents" framing** — all four lenses (T1, T3). Cheap, mechanical, high credibility-signal.
4. **Stale `$SCRATCH`→`$HOME/software` default paths** — portability A6, onboarding C5/fix-4, docs-IA D3 (T4).
5. **A support matrix + split install paths (pure-JAX vs FFI/container) is the missing front door** — portability B1, docs-IA §E `installation/`, onboarding S1/S2. All three propose essentially the same matrix and the same user-vs-dev doc split.
6. **Personal absolute paths in shipped files** — portability A11 and dependency-arch A5 cite the identical lines (`run_shifter.sh:95-96,108`, `CMakeLists.txt:348`).
7. **JAX version skew** — portability A4, dependency-arch A8, onboarding C4, docs-IA F3 (T6).
8. **mkdocs stack is paid-for but unwired** — docs-IA C/G primary; dependency-arch A11 independently flags mkdocs is wrongly in *runtime* deps.

**Tensions / adjudication:**

- **Pin JAX narrowly (`<0.6`) vs keep `>=0.9.0` aspirational.** Dependency-arch argues pin to what runs (the FFI ABI is baked into headers at `build.sh:62-67`, so loose pins are dangerous); the `cuda13`/`>=0.9.0` line is aspirational future. **Adjudication: pin narrowly to `jax>=0.5.3,<0.6`+`cuda12` now** (the FFI ABI argument is decisive), and document CUDA-13/JAX-0.9 as a *separate, untested* matrix row rather than the default. T6/S1.
- **Wheel-build-on-install (Layer 1, "best") vs document-the-build (portability B5 "minimum").** Not a real conflict — they are the endpoints of one ramp. **Adjudication: ship B5-minimum (document the non-Shifter `cmake -D…` invocation) in Tier 1/2 immediately, since it is doc-only and unblocks foreign builders today; pursue the `[build-system]` wiring (S2) as the durable Tier-3 fix.** Document now, package next.
- **Keep vs delete the `build` dependency-group.** Dependency-arch A2 offers both. **Adjudication: wire it (it names the right tool) and drop only the genuinely-unused `pybind11`** — deleting would discard the one correct signal in the file.
- **`stage_*.sh` mesh: retire vs keep.** Portability treats stage scripts as the porting surface to document; dependency-arch wants them replaced by wheels + a manifest. **Adjudication: retire `stage_nvhpc.sh` for the common case once the wheel path lands (S3); keep `stage_cray.sh` for genuine Cray-PE sites but make it version-recording (S5).** Both lenses are satisfied.

No lens contradicts another on direction; the tensions are all about *sequencing and degree*, resolved in §6.

---

## 6. Recommended sequence

**Do first — one afternoon, doc-and-metadata only, no behavior risk (Tier 1 P0 + S1):**
1. T1, T2, T3, T4, T5 — the five P0 fixes. T2 in particular surfaces the *only* command a fresh-clone newcomer can run to success (Path A, `use_ffi_io=false` fixture, no GPU/native build) and is the single highest-leverage onboarding edit.
2. S1 — strip `jackm` paths from tracked files; fix the JAX pin (T6); move mkdocs to a `docs` extra; drop unused `pybind11`. Pure metadata, unblocks everything downstream.

**Do next — the front door (Tier 2 + portability B1/B2 content):**
3. `mkdocs.yml` + the `docs/dev/` split (move all `*_PLAN/_PROGRESS/_AUDIT/TODO` out of nav). This is the docs-IA highest-leverage action and creates the home into which every other fix lands.
4. Write `installation/index.md` (support matrix + 3 tracks) and `installation/ffi-native-libs.md` (the real cuSolverMp-wheel / conda-HDF5 / SLATE-source acquisition recipes + the non-Shifter `cmake -D…` build). This is the portability highest-leverage change and the single biggest content gap. Delete the dead pdoc/pydoc-markdown API mechanisms; turn on mkdocstrings.

**Then — the durable packaging refactor (Tier 3, in order):** S2 → S3 → S4 → S5, each independently shippable. S2 (`[build-system]`) + S3 (cuSolverMp wheel) together are the unanimous #1 fix and convert `pip install lorrax[cuda12]` into JAX-grade one-command UX.

**Defer:** S6 (spack/pixi recipes — valuable but large, and the documented manual recipe from step 4 unblocks foreign builders in the meantime); S7/S8 (`LORRAX_STAGE_ROOT` indirection, Apptainer runtime abstraction — real code refactors worth doing once the doc/packaging spine exists, but not blockers). The user-guide pages (`inputs.md`/`running-gw.md`/etc.) and `contributing.md`/`changelog.md` can be filled incrementally after the nav skeleton is live.
