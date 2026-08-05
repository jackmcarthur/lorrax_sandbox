# Audit Lead Verdict — dep-migration + docs-overhaul (2026-06-29)

Consolidated adjudication of the correctness, docs-accuracy, and cleanliness lenses over
the 6 commits `main..agent/docs-tier1` (LORRAX repo) plus the uncommitted sandbox
`SKILL.md`/`runs/**` edits. Findings deduped across lenses; convergence noted. Every
load-bearing claim was re-verified against the live tree (see "Lead verification" notes).

---

## 1. Verdict

**KEEP AFTER FIXES — 1 BLOCKER.** The body of work is fundamentally sound: the FFI dep
migration is byte-clean (container mount targets `/lorrax_*`, every `LD_LIBRARY_PATH`
entry, `LD_PRELOAD`, and the `LORRAX_MPI_*` vars are provably unchanged; only host-side
`--volume` SOURCEs and `*_DEFAULT` read-paths moved), all four relocated dep trees exist
at `$HOME/software`, `pyproject.toml`/`uv.lock` are coherent (jax/jaxlib `0.5.3`/cu12, no
dangling dropped-dep refs), the `$SEL`/`$INC` SKILL wiring matches the validated 4-GPU
recipe, and the docs overhaul genuinely *reduces* drift (JAX pin, container tag, install
command, MPI default, nav all reconciled). Nothing here breaks an install or corrupts a
run. The single blocker is a dead module name (`gw.kin_ion_io_chunked`) copy-pasted into
the documented preprocessing step across three new doc surfaces (plus the pre-existing
`lxpre` modulefile) — a user who follows the quickstart verbatim hits `No module named`
on step 3. That is a one-token fix but it makes the headline "how to run it" instructions
non-functional, so it must land before this is trustworthy. Everything else is
SHOULD-FIX polish (stale paths on a promoted theory page, write-side stager defaults that
lag the read-side migration, in-nav links into the build-excluded `dev/` tree) or NITs.

---

## 2. BLOCKERS (must fix)

### BLOCKER-1 — Documented preprocessing step #3 invokes a removed module `gw.kin_ion_io_chunked`
**Convergence:** docs-accuracy B1 (primary) + cleanliness implicitly via the same surfaces.
**Lead-verified on disk:** `src/gw/kin_ion_io_chunked.py` does NOT exist; only
`src/gw/kin_ion_io.py` is present. Grep confirms the dead name at all four locations below.
**Locations / fix — replace `gw.kin_ion_io_chunked` → `gw.kin_ion_io`:**
- `docs/quickstart.md:44` (file new in `fa9f98b`)
- `docs/installation/perlmutter.md:38` (file new in `fa9f98b`)
- `config/README.md:42` (line added in `5e8d970`)
- `config/modulefiles/lorrax/0.1.0.lua:332` — the live `lxpre` shell function; **pre-existing
  upstream**, NOT touched this session, but it is the real preprocessing path and is broken
  the same way. Fix it here so the docs and the tool agree.

**Why BLOCKER and not SHOULD-FIX:** this is the central, copy-pasteable "run the pipeline"
instruction in the two most user-facing new pages; a fresh user is guaranteed to hit it.
The fix is trivial (rename), so there is no cost to requiring it.

---

## 3. SHOULD-FIX (real, non-blocking)

### SF-1 — `docs/theory/overview.md` promoted into live nav with stale package/path/driver refs
**Convergence (3/3 lenses):** docs-accuracy S1, cleanliness SHOULD-FIX 2 + 3a/3b — all the
same root cause. `fa9f98b` git-mv'd the frozen `docs/archive/formalism.md` → `theory/overview.md`
(R100, no content change) and wired it as **Theory → Overview**, so its stale specifics now
render as live API. **Lead-verified on disk** (grep): all of the following are present in
`overview.md`:
- `:54` `src/gw_isdf/gw_jax.py` / `src/gw_isdf/w_isdf.py` — `src/gw_isdf/` does not exist;
  real dir is `src/gw/`. (The same overhaul fixed this exact error in `README.md`, so this
  is a missed proofread spot, not a deliberate keep.)
- `:62` `isdf.psp.kin_ion_io`, `:64` `isdf.psp.get_dipole_mtxels` — no `isdf.` top-level
  package; now `gw.`/`psp.`.
- `:64` `cohsex_jax` — old driver name; now `gw.gw_jax`.
- `:3` `docs/misc/isdf_context.md` — `docs/misc/` never existed; file is now
  `docs/dev/archive/isdf_context.md`.
**Fix:** update the paths/driver name in `overview.md`, OR add a "frozen formalism notes"
banner. AND resolve the duplication noted by cleanliness SHOULD-FIX 3a: `physics.md:3`
claims it "Consolidates: formalism.md, …" yet formalism.md still ships unchanged as
overview.md — a "merge that wasn't merged." Either drop `formalism.md` from the physics.md
"Consolidates" list (keep overview.md as a standalone intro) or drop the Theory→Overview
nav entry.

### SF-2 — In-nav rendered pages link into the build-excluded `docs/dev/` tree (dead links in the built site)
**Source:** cleanliness SHOULD-FIX 1. **Introduced by this diff** — the link targets were
deliberately rewritten this branch to `../dev/notes/…` / `../dev/progress/…`, but the new
`mkdocs.yml:26-27 exclude_docs: dev/` removes those pages from the build, so the rendered
HTML carries dead links (non-strict build exits 0, so it ships them silently).
**Locations:** `docs/architecture/codebase.md:639-641`, `docs/theory/physics.md:680,824-826`,
`docs/ENVIRONMENT_COMPREHENSIVE.md:460`. **Fix:** de-link these (plain text "see developer
notes under `docs/dev/`"), or add the specific pages to nav. The new `exclude_docs` and the
new cross-ref targets are mutually inconsistent — pick one.

### SF-3 — Stale bare code-span path prose on promoted pages (links were fixed, prose was not)
**Source:** docs-accuracy S2 (cleanliness NIT 8 flags the same prose mentions as its two
bare-prose cases). The editor correctly updated every clickable link *target* but left
inline `code`-span path strings asserting old locations on now-rendered pages:
- `docs/theory/physics.md:399` `docs/MEMORY_MODEL.md` → `docs/architecture/memory-model.md`
- `docs/theory/physics.md:496` `docs/MINIMAX_QUADRATURE.md` → `docs/theory/minimax-quadrature.md`;
  `docs/NEW_WINDOW_MINIMAX_GUIDELINES.md` → `docs/dev/notes/NEW_WINDOW_MINIMAX_GUIDELINES.md`
- `docs/ENVIRONMENT_COMPREHENSIVE.md:409` `MEMORY_MODEL.md` → `architecture/memory-model.md`
- `docs/theory/isdf-zeta-vq.md:10,1102` `PHYSICS_COMPREHENSIVE.md` → `physics.md`; `:1103`
  `MEMORY_MODEL.md` → `architecture/memory-model.md`
**Fix:** update the bare strings (or convert to proper links).

### SF-4 — Write-side FFI stagers still default to `$SCRATCH` while consumers now read `$HOME/software`
**Source:** correctness SHOULD-FIX 1. Migration `dc5f7f7` moved the READ-side defaults
(`run_shifter.sh`, `site_config.sh`) to `$HOME/software/lorrax_*/stage`, but the WRITE-side
stagers still default `LORRAX_FFI_{PHDF5,SLATE}_DIR` to `/pscratch/sd/${USER:0:1}/${USER}/…`.
**Locations:** `src/ffi/phdf5/scripts/stage_cray.sh:31`, `src/ffi/slate/scripts/stage_cray.sh:23`,
prose `src/ffi/slate/README.md:146`. A fresh user who runs the stagers then the launcher
(no override) stages into `$SCRATCH`, the launcher's `[[ -d ]]` check fails, and the
phdf5/slate bind-mount is silently skipped — losing parallel HDF5 / SLATE with no error.
**Not a blocker** (VALIDATED FACT: live deps already exist at `$HOME`; both stagers honor the
`LORRAX_FFI_*_DIR` override, so the validated session is unaffected). **Fix:** repoint the
two defaults + README line to `$HOME/software/lorrax_*_cray/stage`.

### SF-5 — `run_shifter.sh` header comment now lies about the default it documents
**Source:** cleanliness SHOULD-FIX 4. `dc5f7f7` changed the code defaults (lines 54/61) to
`$HOME/software/…` but left the documenting header comment at `run_shifter.sh:18` and `:29`
stating the old `/pscratch/sd/$USER/lorrax_phdf5_{cray,openmpi}/stage`. Comment now
contradicts code in the same file. **Fix:** update lines 18/29 to the `$HOME/software` paths.
(Same migration root cause as SF-4; fix together.)

### SF-6 — Move-induced broken relative links inside the build-excluded `dev/` tree
**Source:** docs-accuracy S3. `fa9f98b` moved dev READMEs deeper without fixing their `../`
links: `docs/dev/archive/README.md:21-32`, `docs/dev/notes/advanced_README_legacy.md:7,10,19`,
`docs/dev/archive/misc/references/README.md:23-24`. **Confined to `exclude_docs: dev/`** so
nothing renders; dead only for raw-file browsers. Lowest of the SHOULD-FIXes — repoint or
accept as frozen-archive.

---

## 4. NITS (polish)

- **N-1 (correctness NIT 2)** — `pyproject.toml:80`: `nanobind` left in the `build` group is
  as unused as the dropped `pybind11` (FFI is ctypes plain-C; no `NB_MODULE`/`nanobind_add_*`
  anywhere; two comments explicitly say it is NOT used). Pure dead build dep, no runtime
  impact, unused before too. Drop it or add a "reserved for future bindings" note.
- **N-2 (correctness NIT 3)** — `run_shifter.sh:103` `LORRAX_SITE` now defaults empty (was
  the isdf_site path). Intended de-personalization; current FFI-test consumers import none
  of h5py/scipy/matplotlib so nothing breaks. On record only: a future Python step via
  run_shifter needing h5py would ImportError unless the caller exports `LORRAX_SITE`.
- **N-3 (docs-accuracy N1)** — `docs/installation/ffi-native-libs.md:99` attributes
  `LORRAX_PHDF5_MPI_STACK` to `build.sh`; it is actually consumed by run_shifter.sh / stage
  scripts / CMakeLists.txt. Var is real, page flagged untested. Reword to "the build pipeline
  (run_shifter.sh / CMake)".
- **N-4 (cleanliness NIT 5)** — In-nav TODO stubs point at unrendered/absent paths:
  `docs/changelog.md:5` → `docs/dev/progress/` (excluded); `docs/user-guide/index.md:8-9` →
  `docs/docs_gwjax/COHSEX_INPUT.md` (sandbox-only, absent in lorrax_D). Repoint to git
  history / drop the bullet.
- **N-5 (cleanliness NIT 8 / docs-accuracy S2 note)** — ~34 link *labels* still read old
  UPPERCASE names (e.g. `` [`MEMORY_MODEL.md`](memory-model.md) ``); **targets resolve
  correctly** so these are NOT broken — cosmetic label lag only. (The two genuinely-stale
  bare-prose cases are already covered under SF-3.)
- **N-6 (cleanliness NIT 9 / docs-accuracy N5)** — `site_config.sh:99` comment dated 06-24 vs
  commit 06-29 (consistent with SKILL.md: relocated 06-24, committed 06-29 — informational);
  `pyproject.toml:39-42 [tool.uv.workspace] members = [ ]` empty (pre-existing on main);
  `docs/sigma_direct_check.py` stray in docs root (pre-existing, harmless static asset).

---

## 5. False alarms / disagreements / pre-existing (adjudicated)

These were raised by an auditor but are NOT defects of this work, or are pre-existing issues
merely surfaced. Cross-checked against CONTEXT VALIDATED FACTS.

- **nvhpc 0.7.0/0.8.0 not copied to `$HOME`** — NOT a bug. CONTEXT explicitly validates this
  as an intentional, accepted tradeoff (LU-bug regression; production uses 0.7.2 + 25.5). Do
  not flag.
- **Missing `libcal` under the 0.7.2 subpath** — correctness lens checked and cleared it:
  `libcusolverMp.so` links `libnccl.so.2` directly (the documented CAL→NCCL ABI fix), libnccl
  is in the container. Consistent with the validated 4-GPU run. Not a defect.
- **`run_shifter.sh` empty-`LORRAX_SITE` / LORRAX_SRC auto-derivation** — verified correct: no
  trailing-colon PYTHONPATH bug, FFI-test consumers unaffected. The behavior change is the
  intended de-personalization (NIT N-2 only).
- **`lorrax-bse = bse.bse_isdf:main` targets nonexistent module** (docs-accuracy N4) —
  **PRE-EXISTING, identical line on `main`**; the S1 packaging commit neither introduced nor
  fixed it. Real packaging/README defect but out of scope for this effort; track separately,
  do not attribute to this work.
- **"no Dockerfiles in-tree" literally false** (docs-accuracy N2) — **PRE-EXISTING** (false on
  main too; the move just relocated the four Dockerfiles to `docs/dev/archive/docker/`). Intent
  is "none in the active build." Not introduced. Optional reword.
- **`architecture/multihost.md:49,67` broken `slurms/slurms/…` links** (docs-accuracy N3 /
  cleanliness NIT 7) — **PRE-EXISTING**, byte-identical to `main:docs/advanced/jax_multihost.md`
  (R100 rename, same depth). The diff did not introduce or worsen them; promotion into nav
  merely surfaces them. Externally-authored tutorial carryover; `slurms/` never existed. Clean
  up opportunistically, but this is not a defect of this effort.
- **`SKILL.md:136` "JAX 0.7.2" for the jax:25.04 image** (correctness pre-existing note) —
  committed HEAD, NOT in this session's diff; likely a confusion with cuSolverMp 0.7.2. Out of
  scope; pre-existing.
- **nvhpc subpath split (`25.5` in run_shifter vs `0.7.2` in SKILL/VI3)** — PRE-EXISTING in
  both diffs, both dirs exist, production uses 0.7.2 per VALIDATED FACTS. Not a migration
  regression.
- **Dev-scratch absolute personal paths in `docs/dev/progress|plans/*`** (docs-accuracy N3) —
  pre-existing dev scratch in the excluded `dev/` tree, never repo-resolvable. Not a defect.
- **TODO-stub nav pages rendered empty** (cleanliness NIT 6) — intentional Tier-2 scaffolding;
  admonitions are honest (`!!! note "TODO"`). Per CONTEXT this is deliberate. Not a defect;
  track so stubs don't rot.

**Net:** no auditor produced a false positive that contradicts a VALIDATED FACT. The two
"BLOCKER candidates" reduce to a single real blocker (BLOCKER-1); the cleanliness lens
correctly declined to raise any blocker. All three verdicts converge: keep-after-fixes.
