# Audit — Cleanliness, leftover cruft, and git hygiene

Lens: stale references to moved/cut files, leftover artifacts/half-done scaffolding,
CUT items truly gone, commit hygiene, sandbox run-script uniformity. READ-ONLY.

Scope: `lorrax_D` 6 commits `main..agent/docs-tier1`; sandbox uncommitted
`skills/execute_workflow/SKILL.md` + `runs/**` scripts.

---

## Summary

The migration is **cleanly separable and surgical** (only host-side SOURCE paths +
`*_DEFAULT` defaults change; container targets `/lorrax_*` and all `LD_*` paths
untouched). The CUT items (`pydoc-markdown.yml`, `docs/gen_api_docs.sh`,
`pdoc`/`pydoc-markdown`/`pybind11`) are **fully removed and de-referenced** from
pyproject/README/AGENTS. Commits map to their messages with one cosmetic caveat. The
sandbox run-script edits are **uniform** (no half-migrated script, no mixed
`$SCRATCH`/`$HOME` within a script). The real defects are in the *docs-overhaul*
surface: the new mkdocs nav excludes `docs/dev/`, yet several promoted in-nav pages
were given fresh links *into* `docs/dev/` → dead links in the rendered site; and a
couple of promoted theory pages carry stale code/doc paths that the same overhaul
fixed elsewhere.

---

## Findings

### SHOULD-FIX 1 — In-nav pages link into `docs/dev/`, which mkdocs excludes from the build
- **Location:** `docs/architecture/codebase.md:639-641`, `docs/theory/physics.md:680,824,825,826`, `docs/ENVIRONMENT_COMPREHENSIVE.md:460` (link targets `../dev/notes/…`, `../dev/progress/…`, `dev/notes/AGENT_TODO.md`); `mkdocs.yml:26-27` `exclude_docs: dev/`.
- **Problem:** Introduced by this diff. The `git diff` of `codebase.md` shows the targets were *deliberately rewritten* this branch from `GN_PPM_…md` (sibling) to `../dev/notes/GN_PPM_…md`, etc. — so they resolve on disk, but `mkdocs.yml` now adds `exclude_docs: dev/`, so those pages are NOT in the rendered site. In the built HTML these are dead links. (mkdocs non-strict build still exits 0 — a VALIDATED FACT — so it doesn't fail the build, it just ships broken links.) The new mkdocs `exclude_docs` and the new cross-reference targets are mutually inconsistent.
- **Fix:** Either drop the dev-note links from the rendered pages (they are developer-only references), or convert them to plain inline text ("see the developer notes under `docs/dev/`") with no link, or add the specific pages back into nav. Cheapest: del"link"-ify them.

### SHOULD-FIX 2 — Promoted `theory/overview.md` references a non-existent module dir `src/gw_isdf/`
- **Location:** `docs/theory/overview.md:54` — "see `src/gw_isdf/gw_jax.py` and `src/gw_isdf/w_isdf.py`".
- **Problem:** `src/gw_isdf/` does not exist; the real dir is `src/gw/`. This is *pre-existing text* (R100 rename of `docs/archive/formalism.md`, content unchanged) but the overhaul **promoted this file into the rendered nav** (Theory > Overview), surfacing the stale path into the published docs. The same overhaul fixed the identical `gw_isdf/` → `gw/` error in `README.md` (see the README diff: `gw_isdf/gw_jax.py` → `gw/gw_jax.py`), so this is a missed spot in the proofread, not a deliberate keep.
- **Fix:** `src/gw_isdf/` → `src/gw/` in `overview.md:54` (and proofread the rest of the newly-promoted `overview.md`, which was R100-imported wholesale).

### SHOULD-FIX 3 — `theory/overview.md` "Consolidates" overlap + stale `docs/misc/isdf_context.md` ref
- **Location:** `docs/theory/overview.md:3` and `docs/theory/physics.md:3`.
- **Problem (a) duplication:** `physics.md:3` claims it **"Consolidates: `formalism.md`, …"** — but `formalism.md` was promoted unchanged to `theory/overview.md` and is STILL in the nav (Theory > Overview). So the formalism content now lives in BOTH the consolidated `physics.md` AND the standalone `overview.md` — overlapping/duplicated theory in the rendered site, with a "Consolidates" claim that implies it was folded in and retired. This is the "MERGE that wasn't actually merged" pattern: physics.md says it absorbed formalism.md, but formalism.md (as overview.md) is still shipped alongside it.
- **Problem (b) stale path:** `overview.md:3` (pre-existing R100 text, now in nav) says it is "a condensed … version of the notes in `docs/misc/isdf_context.md`" — `docs/misc/` never existed and `isdf_context.md` now lives at `docs/dev/archive/isdf_context.md`. Dangling path in a published page.
- **Fix:** Decide whether `overview.md` is a short standalone intro (then remove `formalism.md` from the physics.md "Consolidates" list to stop claiming it was merged) or fully fold it in (then drop the Theory > Overview nav entry). Either way fix the `docs/misc/isdf_context.md` reference (point to `dev/archive/isdf_context.md` or drop it).

### SHOULD-FIX 4 — `run_shifter.sh` header comment still documents the old `$SCRATCH` default for `LORRAX_FFI_PHDF5_DIR`
- **Location:** `src/ffi/common/cpp/run_shifter.sh:18` and `:29` ("default `LORRAX_FFI_PHDF5_DIR`: `/pscratch/sd/$USER/lorrax_phdf5_{cray,openmpi}/stage`").
- **Problem:** Collateral staleness from the migration. The migration commit `dc5f7f7` changed the *code* defaults (lines 54/61) `/pscratch/sd/j/jackm/…` → `$HOME/software/…`, but left the documenting header comment stating the old `$SCRATCH`-style path. Before the migration the comment was consistent with the code default; the migration made it lie. The comment now contradicts the code in the same file.
- **Fix:** Update lines 18 and 29 to `$HOME/software/lorrax_phdf5_{cray,openmpi}/stage` to match the new defaults.

### NIT 5 — In-nav stub pages point users at unrendered / non-existent paths
- **Location:** `docs/changelog.md:5` → `docs/dev/progress/` (excluded from site); `docs/user-guide/index.md:8-9` → `docs/docs_gwjax/COHSEX_INPUT.md` (sandbox-only path, does not exist in `lorrax_D`).
- **Problem:** Both are TODO-stub nav pages that direct the reader to a path that is not in the rendered site (dev/ is excluded) or not in the repo at all (`docs/docs_gwjax/` is a sandbox dir). Low impact (admonition-stub pages), but they're "follow this link" pointers that go nowhere for a non-NERSC reader.
- **Fix:** In `changelog.md`, point at the git history / GitHub releases instead of `docs/dev/progress/`. In `user-guide/index.md`, drop the `docs/docs_gwjax/COHSEX_INPUT.md` bullet (or note it lives in the separate sandbox repo).

### NIT 6 — TODO-stub nav pages are wired into nav while empty
- **Location:** `docs/api/index.md`, `docs/user-guide/index.md`, `docs/changelog.md` (all bodies are a single `!!! note "TODO"`); `docs/contributing.md` and `docs/installation/index.md` are substantive.
- **Problem:** Three nav entries render essentially empty TODO pages. This is intentional Tier-2 scaffolding (CONTEXT) and the admonitions are honest, so not a defect — flagged only so it's a tracked, deliberate stub, not forgotten cruft. (api/index.md correctly documents that mkdocstrings is wired and only per-module pages are pending — accurate.)
- **Fix:** None required now; consider a tracking issue so the stubs don't rot.

### NIT 7 — `architecture/multihost.md` carries pre-existing broken `slurms/…` links (PRE-EXISTING)
- **Location:** `docs/architecture/multihost.md:49` (`slurms/slurms/01-single-host-8-GPUs.slurm` — note the doubled `slurms/slurms/`) and `:67` (`slurms/02-multihost-2nodes.slurm`).
- **Problem:** No `docs/architecture/slurms/` directory exists; these links are dead. **Pre-existing**: this file is a R100 rename of `docs/advanced/jax_multihost.md` and the link text is byte-identical to `main` (`git show main:docs/advanced/jax_multihost.md` has the same `slurms/slurms/…`). The move promoted a page that already had broken links into the nav, but the diff did not introduce or worsen them.
- **Fix (optional):** Out of scope for this effort, but since the page is now in the published nav, the doubled-`slurms/` typo and the missing slurm assets should be cleaned up or the links removed.

### NIT 8 — Cosmetic: old UPPERCASE doc names retained as link *labels* in promoted pages (PRE-EXISTING content, not broken)
- **Location:** ~34 mentions across `docs/theory/*`, `docs/architecture/*`, `docs/ENVIRONMENT_COMPREHENSIVE.md` (e.g. `` [`MEMORY_MODEL.md`](memory-model.md) ``, `` [`PHYSICS_COMPREHENSIVE.md`](../theory/physics.md) ``).
- **Problem:** The link *targets* were correctly updated to the new paths (verified: my link-checker found no broken in-nav target except math false-positives), but the human-readable labels still say the old `UPPERCASE_NAME.md`. A reader sees "MEMORY_MODEL.md" but it resolves to `memory-model.md`. Purely cosmetic — not a broken link. Two are bare prose with no link to the right place: `physics.md:496` "Full derivation in `docs/MINIMAX_QUADRATURE.md`" and `physics.md:399` "See `docs/MEMORY_MODEL.md`".
- **Fix:** Optional polish — relabel to the new names or convert the two bare-prose mentions into proper links.

### NIT 9 — Cosmetic date/`[tool.uv.workspace]` items (mostly PRE-EXISTING)
- `config/perlmutter/site_config.sh:99` migration comment dated `2026-06-24` while commit `dc5f7f7` is `2026-06-29`; SKILL.md uses the same `2026-06-24`, so the two are mutually consistent (relocation done 06-24, committed 06-29). Informational, not a defect.
- `pyproject.toml:39-42` `[tool.uv.workspace] members = [ ]` is whitespace-only/empty — **pre-existing on `main`**, not introduced here.

---

## Things checked and found CLEAN (do not re-flag)

- **CUT items truly gone & de-referenced:** `pydoc-markdown.yml` and `docs/gen_api_docs.sh` are deleted from the working tree; zero references to `pydoc-markdown`, `gen_api_docs`, or `pybind11` remain anywhere (only one `pdoc` mention, in the out-of-nav `docs/dev/notes/AGENT_TODO.md`). `pdoc`/`pydoc-markdown` removed from `[dependency-groups].dev`; `pybind11` removed from the `build` group; mkdocs moved into a `docs` extra. `mkdocs.yml`'s `mkdocstrings` plugin is backed by `mkdocstrings` + `mkdocstrings-python` in the `docs` extra — coherent.
- **No leftover old doc dirs:** `docs/advanced/`, `docs/archive/`, `docs/plans/`, `docs/progress/`, `docs/misc/`, `docs/notes/` are all fully gone (git mv left no empty husks). No empty dirs under `docs/`.
- **Migration is surgical & separable:** `dc5f7f7` touches only the `*_DEFAULT` / `NVHPC_HOST` / `PHDF5_DEFAULT` / `SLATE_DIR` lines; `ca840ec`'s edits to the same two files are a disjoint line region (personal-path stripping). No overlap. Container mount targets `/lorrax_nvhpc|phdf5|slate` and all `LD_LIBRARY_PATH`/`LD_PRELOAD` container paths are UNCHANGED. No residual `jackm`/`/pscratch` hardcoded paths remain in `run_shifter.sh`, `site_config.sh`, `CMakeLists.txt`.
- **Commit hygiene:** Each commit's file set matches its message. `pyproject.toml` is touched by two commits but each split is internally coherent and documented (S1 `ca840ec`: JAX pin + docs extra + pybind11; Tier-2 `fa9f98b`: pdoc/pydoc-markdown removal alongside the CUT of the API-doc tooling). `ea1ea3c` is uv.lock-only. `290de0a` is exactly `+site/` in `.gitignore`. No build output / cache committed; `build/`, `src/ffi/**/cpp/build/`, `.jax_cache`, `*.log` are all gitignored, and the local `profile/` clutter is untracked (pre-existing, not in the diff).
- **README/AGENTS repointing is complete and correct:** all old `docs/UPPERCASE.md` links repointed to new paths (and the README `gw_isdf/` → `gw/` and "For AI agents" removal applied). README/AGENTS are repo-root docs (not in the rendered `docs_dir`), so their `docs/dev/…` links are valid filesystem links.
- **README/quickstart fixture is real & consistent:** `tests/regression/cohsex_debug/cohsex_test.in` exists; README and `quickstart.md` reference it identically.
- **Sandbox run scripts are uniform:** every changed `runs/**` script repaths exactly the three `--volume` SOURCE paths `/pscratch/sd/j/jackm/lorrax_*` → `/global/homes/j/jackm/software/lorrax_*` (= `$HOME/software`) and only those; container targets, `LD_LIBRARY_PATH`, `LD_PRELOAD`, `PYTHONPATH`, and `JAX_COMPILATION_CACHE_DIR` (still `/pscratch`) are correctly untouched. No script half-migrated; no mixed `$SCRATCH`/`$HOME` within a script.
- **SKILL.md is self-consistent:** deps mounted from literal `/global/homes/j/jackm/software/…`; `$SEL`/`$INC` wrappers added uniformly to all Step 5/6 srun commands; the "never set CUDA_VISIBLE_DEVICES" pitfall rewritten coherently. `select_gpu.sh`, `in_container.sh`, and the `centroid.kmeans_cli` entrypoint all exist and are tracked in `lorrax_D`.
- **Placeholder tokens are legitimate:** `<module-name>` (ENVIRONMENT 150 / config/README 168) is a genuine user-substitution placeholder in an install example; `$LORRAX_ROOT`/`$LORRAX_*` are documented modulefile env vars. Not leftover cruft.

---

## Verdict

Migration and CUT-item removal are clean and surgical; commit hygiene is sound. The
only real defects are in the docs-overhaul surface: promoted in-nav pages link into the
build-excluded `docs/dev/` tree (dead links in the rendered site) and two promoted
theory pages carry stale `src/gw_isdf/` and `docs/misc/isdf_context.md` paths plus a
"Consolidates formalism.md" claim that contradicts formalism.md still shipping as
Theory > Overview — all SHOULD-FIX polish, none blocking.
