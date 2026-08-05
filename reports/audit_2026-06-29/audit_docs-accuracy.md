# Audit — Documentation accuracy & consistency lens

Scope: doc edits + moves in `main..agent/docs-tier1` (LORRAX repo
`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`). Read-only. Every link was
resolved on disk and every module/path claim was checked against the actual tree before
asserting a defect. Pre-existing issues surfaced (not introduced) are marked as such.

## Summary

The Tier-1/Tier-2 doc work is, on the whole, careful and *reduces* drift: JAX pin,
`$SCRATCH`→`$HOME/software`, `lorrax_X`→`lorrax`, container image tag, `uv sync` install
command, MPI default, and the dependency-group table are all consistent across
README / AGENTS / index / quickstart / installation/* / ENVIRONMENT_COMPREHENSIVE /
config/README. mkdocs nav is complete (every nav entry resolves; no rendered page omitted;
all four stub pages clearly flagged `!!! note "TODO"`). The installation/ffi-native-libs
recipes are explicitly flagged untested and their referenced scripts / CMake vars / loader
all exist.

The defects are: (1) a **broken preprocessing-module name** (`gw.kin_ion_io_chunked`)
repeated across three newly-authored doc surfaces; (2) one **promoted archive page**
(`theory/overview.md`) that now renders stale package/path references in the live site;
(3) residual **stale-prose path references** in promoted theory + ENVIRONMENT docs that the
editor fixed as *links* but missed as plain code-spans; (4) **move-induced broken relative
links** confined to the build-excluded `dev/` tree. No broken links escape into the
rendered site except the stale code-span prose in (2)/(3).

---

## BLOCKER

### B1. `gw.kin_ion_io_chunked` does not exist — documented preprocessing step #3 is broken (3 doc surfaces, all newly authored/edited this session)

The module `gw.kin_ion_io_chunked` was removed in commit `62b0e94` ("remove dead modules,
relocate CLIs"); the live module is **`gw.kin_ion_io`** (its own docstring reads
`python -m gw.kin_ion_io -i cohsex.in`, described as the "Chunked kin+ion computation").
`src/gw/kin_ion_io_chunked.py` is gone (only a stale `.pyc` remains in `__pycache__`).

The wrong name appears in the changed set at:
- `docs/quickstart.md:44` — `3. **Kinetic + ionic** — python -m gw.kin_ion_io_chunked -i cohsex.in → kin_ion.h5`  (file NEW in `fa9f98b`)
- `docs/installation/perlmutter.md:38` — `3. python3 -m gw.kin_ion_io_chunked -i <in> → kin_ion.h5`  (file NEW in `fa9f98b`)
- `config/README.md:42` — `#   [3/3] python3 -m gw.kin_ion_io_chunked -i cohsex.in -> kin_ion.h5`  (added `+` line in `5e8d970`)

A user copy-pasting any of these gets `No module named gw.kin_ion_io_chunked`.

(The pre-existing `config/modulefiles/lorrax/0.1.0.lua:332` — the actual `lxpre` shell
function — ALSO uses the broken name and was NOT touched this session; so the real `lxpre`
preprocessing step 3 is itself broken upstream. The docs faithfully mirror a broken
modulefile. Flagging the modulefile too since the docs derive from it.)

**Fix:** replace `gw.kin_ion_io_chunked` → `gw.kin_ion_io` in quickstart.md:44,
perlmutter.md:38, config/README.md:42, and `config/modulefiles/lorrax/0.1.0.lua:332`.

---

## SHOULD-FIX

### S1. `docs/theory/overview.md` promoted into the live nav with stale package/path references

`fa9f98b` `git mv`'d the frozen `docs/archive/formalism.md` → `docs/theory/overview.md`
(100% rename, zero content change) and wired it into the nav as **Theory → Overview**. The
content is a pre-existing archive doc, but promotion now *renders* its stale references in
the site:
- `overview.md:54` — "see `src/gw_isdf/gw_jax.py` and `src/gw_isdf/w_isdf.py`" — the
  `gw_isdf` package was renamed to `gw`; `src/gw_isdf/` does not exist (it is `src/gw/`).
- `overview.md:62` — "We use `isdf.psp.kin_ion_io`…" — no `isdf.` top-level package; the
  module is `gw.kin_ion_io`.
- `overview.md:64` — "dipole matrix elements from `isdf.psp.get_dipole_mtxels`" — now
  `psp.get_dipole_mtxels`; also "done on startup of `cohsex_jax`" — `cohsex_jax` is the old
  driver name (now `gw.gw_jax`).
- `overview.md:3` — "condensed… version of the notes in `docs/misc/isdf_context.md`" —
  that file moved to `docs/dev/archive/isdf_context.md`.

This is the **only** rendered-site page carrying `gw_isdf` / `isdf.psp` / `cohsex_jax`
references; the rename was applied correctly everywhere else. Because the page is now
presented as current "Theory: Overview", the stale specifics read as live API.

**Fix:** either update overview.md's paths (`src/gw_isdf/`→`src/gw/`,
`isdf.psp.`→`psp.`/`gw.`, `cohsex_jax`→`gw.gw_jax`, `docs/misc/isdf_context.md`→
`docs/dev/archive/isdf_context.md`), or add a "historical / formalism notes" banner if the
content is deliberately frozen.

### S2. Stale-prose path references in promoted theory + ENVIRONMENT pages (links fixed, code-spans missed)

The editor diligently updated every clickable markdown **link** target for the move (e.g.
`[MEMORY_MODEL.md](MEMORY_MODEL.md)` → `[MEMORY_MODEL.md](../architecture/memory-model.md)`
in physics.md and isdf-zeta-vq.md). But **bare code-span prose** mentions of old paths were
not updated and now point nowhere:
- `docs/theory/physics.md:399` — "**See**: `docs/MEMORY_MODEL.md` for detailed formulas" →
  now `docs/architecture/memory-model.md`.
- `docs/theory/physics.md:496` — "Full derivation in `docs/MINIMAX_QUADRATURE.md`.
  Windowing strategy in `docs/NEW_WINDOW_MINIMAX_GUIDELINES.md`." → now
  `docs/theory/minimax-quadrature.md` and `docs/dev/notes/NEW_WINDOW_MINIMAX_GUIDELINES.md`.
- `docs/ENVIRONMENT_COMPREHENSIVE.md:409` — "Check `MEMORY_MODEL.md` for per-stage
  formulas." → now `architecture/memory-model.md`.
- `docs/theory/isdf-zeta-vq.md:10, 1102` — bare `PHYSICS_COMPREHENSIVE.md`; `:1103`
  bare `MEMORY_MODEL.md`.

These are not broken hyperlinks (they are inline `code` spans), but they assert paths that
no longer exist on a now-rendered page. **Fix:** update the bare path strings to the new
locations (or drop the `docs/` prefix and link them).

Note (NOT a defect): the *link display text* across physics.md / codebase.md /
ENVIRONMENT_COMPREHENSIVE.md still reads the old uppercase filenames
(`[CODEBASE_COMPREHENSIVE.md](architecture/codebase.md)`), but every such link **target
resolves correctly**. Cosmetic only.

### S3. Move-induced broken relative links inside the (build-excluded) `dev/` tree

`fa9f98b` moved dev READMEs deeper without updating their `../` relative links. Confined to
`exclude_docs: dev/` so they never render in the site, but they are dead for anyone browsing
raw files:
- `docs/dev/archive/README.md:21-27` — `../PHYSICS_COMPREHENSIVE.md`,
  `../CODEBASE_COMPREHENSIVE.md`, `../ENVIRONMENT_COMPREHENSIVE.md`, `../MEMORY_MODEL.md`,
  `../MINIMAX_QUADRATURE.md` (was `docs/archive/README.md`; `../` used to hit `docs/`, now
  hits the empty `docs/dev/`). Lines 30-32: `../advanced/`, `../references/`,
  `../AGENT_TODO.md` likewise broken.
- `docs/dev/notes/advanced_README_legacy.md:7,10,19` (was `docs/advanced/README.md`) —
  `jax_multihost.md`, `HL_GPP_derivation.md` (old siblings, now under
  `architecture/`/`theory/`), `../ENVIRONMENT_COMPREHENSIVE.md`.
- `docs/dev/archive/misc/references/README.md:23-24` — `../PHYSICS_COMPREHENSIVE.md`,
  `../MINIMAX_QUADRATURE.md`.

**Fix:** repoint these to the promoted locations, or accept as frozen-archive dead links.
Lower priority than S1/S2 because nothing renders.

---

## NIT

### N1. `docs/installation/ffi-native-libs.md:99` attributes `LORRAX_PHDF5_MPI_STACK` to `build.sh`

The page says "`build.sh` accepts `LORRAX_FFI_ALLOW_DEFAULT_MPI=1`… set
`LORRAX_PHDF5_MPI_STACK=openmpi` accordingly." `LORRAX_FFI_ALLOW_DEFAULT_MPI` IS read by
`build.sh` (lines 33-45), but `LORRAX_PHDF5_MPI_STACK` is consumed by `run_shifter.sh`, the
stage scripts, and `CMakeLists.txt` (lines 279-306) — not `build.sh` directly. The variable
is real and part of the build pipeline, and the page is flagged untested, so impact is low.
**Fix:** reword to "the build pipeline (run_shifter.sh / CMake) selects the stack via
`LORRAX_PHDF5_MPI_STACK`".

### N2. "no Dockerfiles in-tree" is literally false (pre-existing)

`docs/ENVIRONMENT_COMPREHENSIVE.md:55` claims "Docker / docker-compose — no Dockerfiles
in-tree." Four Dockerfiles exist at `docs/dev/archive/docker/`. This claim was ALSO false in
`main` (Dockerfiles were at `docs/archive/docker/`); the move just relocated them. The
intent is clearly "no Dockerfiles in the active build", but the literal statement is wrong.
Pre-existing. **Fix (optional):** "…archived under `docs/dev/archive/docker/`, not used in
the active build."

### N3. Pre-existing broken links surfaced by the moves

- `docs/architecture/multihost.md:49,67` — `slurms/slurms/01-single-host-8-GPUs.slurm` and
  `slurms/02-multihost-2nodes.slurm` resolve to nothing. **Pre-existing** carryover from an
  externally-authored multihost tutorial; `slurms/` never existed in the repo. The move
  (`docs/advanced/jax_multihost.md`→`docs/architecture/multihost.md`, same depth, 100%
  rename) did not introduce these — but the page is now in the live nav, so the dead links
  render. Not introduced; flagging because promotion surfaces them.
- `docs/dev/progress/SIGMA_FREQ_AUDIT_STATUS.md` and `docs/dev/plans/phdf5_cray_mpich_migration.md`
  contain absolute personal-machine paths (`/home/jackm/...`, `/global/u2/j/jackm/...`) as
  markdown links — pre-existing dev scratch, never repo-resolvable, and in the excluded
  `dev/` tree. Not a real defect.

### N4. `lorrax-bse` console script targets a nonexistent module (pre-existing, packaging not docs)

`pyproject.toml:41` `lorrax-bse = "bse.bse_isdf:main"` — `src/bse/bse_isdf.py` does not
exist (the BSE driver is `bse/bse_jax.py`). README.md:16 advertises `lorrax-bse` as an
available console command. **Pre-existing in `main`** (identical line); the S1 packaging
commit did not introduce it but also did not fix it. Out of strict docs-accuracy scope but
makes a README "available command" claim false. **Fix:** point at the real entry point or
drop `lorrax-bse`.

### N5. Stray non-markdown file in `docs/` root (pre-existing)

`docs/sigma_direct_check.py` sits in the docs root (outside nav). Harmless (mkdocs treats it
as a static asset), pre-existing, not in the changed set. Cosmetic.

---

## Items explicitly verified as CORRECT (not defects)

- **Quickstart fixture runs CPU-only as claimed.** `tests/regression/cohsex_debug/cohsex_test.in`
  exists with `use_ffi_io = false`; ships `WFNsmall.h5`, `centroids_frac_60.txt`,
  `dipole.h5`, `kin_ion.h5`; writes `eqp_test.dat` with reference `eqp_ref.dat`
  (matches quickstart.md:24-27). `tests/test_gw_jax_regression.py` defaults to
  `ISDF_COHSEX_TEST_PLATFORM=auto` (lets JAX pick CPU on a CPU-only box) and supports an
  explicit `cpu` path — so `pytest -q` "CPU, ~1-2 min" holds on a fresh CPU-only clone.
- **`gw_isdf`→`gw` rename correct everywhere except `theory/overview.md`** (see S1). README
  module paths (`centroid/kmeans_isdf.py`, `common/load_wfns.py`, `gw/gw_jax.py`,
  `gw/w_isdf.py`), AGENTS.md doc-table targets (all 6 moved paths), and console scripts
  `gw_jax`/`lorrax-gw`/`lorrax-centroids` all resolve.
- **JAX pin consistent**: `jax[cuda12]>=0.5.3,<0.6` in pyproject.toml,
  installation/index.md, ENVIRONMENT_COMPREHENSIVE.md (table + troubleshooting), with
  `cuda13`/`0.9.0` consistently labeled the untested row. cu13→cu12 drift fully reconciled.
- **Container image consistent**: `nvcr.io/nvidia/jax:25.04-py3` in pyproject comment,
  installation/index.md:59, config/perlmutter/site_config.sh, install.sh, AND the sandbox
  SKILL recipe.
- **`uv sync` install command + "editable install / puts src on sys.path"** consistent
  across README/index/quickstart/installation/ENVIRONMENT; `[tool.uv] package = true`
  backs the editable claim. The old `uv sync --no-install-project --locked` / `uv venv`
  drift was removed.
- **MPI default consistent**: `gpu,mpich` (Perlmutter default) / OpenMPI alternative; no
  new drift.
- **mkdocs nav complete**: all 19 nav targets exist; no rendered (non-dev) page omitted;
  `exclude_docs: dev/` excludes the ~40-file dev tree; all 4 stub pages (api, user-guide,
  contributing, changelog) clearly `!!! note "TODO"` and do not present incomplete content
  as complete.
- **installation/index.md + ffi-native-libs.md untested-flagging is honest**: both carry
  `!!! note "TODO"` / *untested* banners on the non-NERSC rows. Every script
  (`build.sh`, `stage_nvhpc.sh`, `stage_cray.sh`, `stage_openmpi.sh`, slate `stage_cray.sh`,
  `run_shifter.sh`), every CMake var (`CUSOLVERMP_INCLUDE_DIR`, `CUSOLVERMP_LIB_DIR`,
  `HDF5_ROOT`, `LORRAX_SLATE_INSTALL_DIR`, `LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR`,
  `HDF5_IS_PARALLEL`), the loader search order (`ffi_loader.py`: `$LORRAX_FFI_SO` →
  `cpp/build/` → `sys.path`), and the `FileNotFoundError … bash src/ffi/common/cpp/build.sh`
  message all exist / match reality. `src/ffi/PORTING.md` and `src/ffi/AGENTS.md` exist.
- **Cut doc-tooling fully de-referenced**: `pydoc-markdown.yml`, `tools/gen_api_docs.sh`,
  `docs/gen_api_docs.sh`, `pdoc`, `pybind11` are gone from disk AND from all prose; the
  mkdocs purpose line was retargeted to `mkdocs build`. `.gitignore` has `site/`.

---

## Verdict

Docs are materially more consistent than before — JAX/container/install/MPI drift is
genuinely reconciled and nav is complete — but one BLOCKER (`gw.kin_ion_io_chunked` is a
dead module name copied into three new doc surfaces and the live `lxpre` modulefile) plus a
promoted-archive page (`theory/overview.md`) and residual stale-prose paths leak false
specifics into the rendered site and a broken copy-paste command.
