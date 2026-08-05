# LORRAX docs — Information-Architecture & Professional-Quality review

Lens: **docs IA & professional quality vs the JAX/PySCF/ASE/GPAW bar.**
Scope: `README.md`, `AGENTS.md`, `docs/index.md`, and the full `docs/` tree
(34 `.md` files, ~13k lines) under
`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.
Mode: READ + PROPOSE ONLY. Citations are `file:line` / `file §section`.

---

## 0. The central finding (one sentence)

LORRAX has *no docs site* — it has a **flat folder of 34 shouting-caps Markdown
files** in which durable user/reference docs (`PHYSICS_COMPREHENSIVE.md`,
`ENVIRONMENT_COMPREHENSIVE.md`) sit side-by-side with transient developer
scratchpads (`*_PLAN*.md`, `*_PROGRESS.md`, `*_AUDIT_STATUS.md`,
`AGENT_TODO.md`), every file is framed *"For AI agents"* rather than for users,
and the **API-doc pipeline that the deps are paid for (mkdocs-material +
mkdocstrings) is not wired at all** — three different, mutually contradictory,
partly-broken API mechanisms are referenced instead.

Against the JAX/PySCF/ASE bar (landing → Install → Quickstart → User Guide → API
reference → Architecture → Contributing → Changelog), LORRAX currently has: a
landing-ish file with broken links, no Install page (only a 455-line
`*_COMPREHENSIVE.md`), no Quickstart page, no rendered API reference, no
Contributing, no changelog. The raw material for excellent physics/architecture
docs *exists* and is high quality — it is just unstructured and mis-addressed.

---

## A. What exists today (inventory)

`find docs -name '*.md'` → 34 files. Grouped by what they actually are:

**Durable user/reference (KEEP, promote):**
- `docs/index.md` (58 L) — landing page
- `docs/PHYSICS_COMPREHENSIVE.md` (977 L)
- `docs/CODEBASE_COMPREHENSIVE.md` (641 L)
- `docs/ENVIRONMENT_COMPREHENSIVE.md` (455 L)
- `docs/MEMORY_MODEL.md` (1004 L)
- `docs/SYMMETRY_COMPREHENSIVE.md` (665 L)
- `docs/MINIMAX_QUADRATURE.md` (340 L)
- `docs/ZETA_V_Q_ALGORITHMS.md` (1146 L)
- `docs/advanced/jax_multihost.md` (870 L), `docs/advanced/HL_GPP_derivation.md` (186 L)

**Transient dev notes mixed into the top level (ARCHIVE/CUT — should not ship):**
- `docs/AGENT_TODO.md` (292 L) — self-labelled "parking lot… NOT the user's
  current priorities" (`AGENT_TODO.md:3`)
- `docs/FREQ_INTEGRATION_PROGRESS.md` (71 L) — `Status: Stages 1-3 completed…
  Owner: Codex` (`FREQ_INTEGRATION_PROGRESS.md:3-4`)
- `docs/FREQ_INTEGRATION_REWRITE_PLAN.md` (651 L) — `Status: Implementation
  blueprint` (`:3`)
- `docs/SIGMA_FREQ_AUDIT_STATUS.md` (121 L) — `(Handoff)`, `Date: 2026-03-31`
  (`:1,3`)
- `docs/PLAN_zeta_g_flat_migration.md` (373 L) — `Working doc… so the agent can
  resume` (`:3-5`), embeds `SLURM_JOBID=52841861` and run paths (`:17`)
- `docs/plans/unified_slab_io.md` (291 L), `docs/plans/phdf5_cray_mpich_migration.md` (197 L)
- `docs/PROFILING_SUGGESTIONS.md` (342 L) — audit/opinion note
- `docs/NEW_WINDOW_MINIMAX_GUIDELINES.md` (478 L) — opens with the raw LLM
  artifact `"Absolutely — here is a cleaned-up, self-contained note…"`
  (`NEW_WINDOW_MINIMAX_GUIDELINES.md:1`)
- `docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` (459 L) — `_REVISED` in the
  filename is a version marker; README calls it "most recent dev push"
  (`README.md:37`)

**Already-archived (fine, but see §D):** `docs/archive/**` (15 files incl. a
`misc/references/` paper dump and an `nufft/` dead-backend folder).

So **~10 of the ~19 top-level docs are developer ephemera** living in the same
namespace a new user would browse. That is the IA problem in one number.

---

## B. Benchmark gap table (vs JAX/PySCF/ASE)

| Expected section (the bar) | LORRAX today | Gap |
|---|---|---|
| Clear landing page | `docs/index.md` exists | Links to **nonexistent** `formalism.md` and `examples/` (`index.md:18`); duplicates README; "For AI agents" framing in README (`README.md:39`) |
| **Installation** (support matrix: OS/CUDA/MPI/cluster; pip / container / source) | `ENVIRONMENT_COMPREHENSIVE.md` (455 L) | Not a page, it's a wall; Perlmutter-only; site paths `/path/to/lorrax_X`, `lorrax_A` (`ENVIRONMENT_COMPREHENSIVE.md:65,151,161`); no matrix |
| **Quickstart / tutorial** | scattered: `README.md:18-24`, `index.md:53-58`, `AGENTS.md:59-86` | Three different "quick" snippets, no end-to-end worked example, `examples/` dir does not exist |
| **User Guide** | — | none; physics/memory docs are reference, not task-oriented guides |
| **API reference** (mkdocs-material + mkdocstrings are deps) | **not wired** | see §C — 3 conflicting, broken mechanisms; `docs/api/` does not exist |
| **Architecture / internals** | `CODEBASE_COMPREHENSIVE.md`, `MEMORY_MODEL.md`, `SYMMETRY_COMPREHENSIVE.md` | Good content, but addressed to agents (`CODEBASE_COMPREHENSIVE.md:3`) and not in a nav |
| **Contributing** | — | none (no coding-standards page for *humans*; the only coding standards live in `AGENTS.md:88-104`, addressed to agents) |
| **Changelog** | — | none in repo |
| **Theory / math** | strong (`PHYSICS_COMPREHENSIVE.md`, `MINIMAX_QUADRATURE.md`, `ZETA_V_Q_ALGORITHMS.md`) | best part of the docs; just needs a home in nav |

---

## C. The API-doc pipeline is triple-broken (highest-value structural fix)

The repo *pays for* the JAX-grade stack — `mkdocs>=1.6.1`, `mkdocs-material>=9.6.20`,
`mkdocstrings>=0.30.1`, `mkdocstrings-python>=1.18.2` are **hard runtime
dependencies** (`pyproject.toml:12-15`) — yet there is **no `mkdocs.yml`
anywhere in the repo** (only the copies inside `.venv/`). Three competing,
non-working API stories coexist:

1. **README** says use `pdoc`: `uv add pdoc` then `uv run -- bash
   docs/gen_api_docs.sh` (`README.md` / `index.md:24-31`).
2. **`docs/gen_api_docs.sh`** actually invokes **`pydoc-markdown`**, not pdoc
   (`gen_api_docs.sh:13-21`), and contains a bug: line 18 is
   `"${OUT_DIR}" >/dev/null 2>&1 || true` — i.e. it tries to **execute the output
   directory as a command** (almost certainly a corrupted `rm -rf "${OUT_DIR}"`).
3. **`pydoc-markdown.yml`** points at packages that **no longer exist**:
   `packages: [gw_isdf, isdf.common, isdf.isdf_init]` (`pydoc-markdown.yml:4-7`),
   but the real `src/` layout is `gw/ common/ centroid/ …` (no `gw_isdf`, no
   `isdf` package). So even after fixing the script bug, it generates nothing.

Net effect: `docs/api/` is referenced by `index.md:22` ("Generated Markdown lives
under `docs/api/`") and `archive/README.md`, but the directory does not exist and
cannot be produced by any documented command. **Three tools (`pdoc`,
`pydoc-markdown`, `mkdocstrings`) are declared; zero work.**

Proposal: **pick one — mkdocstrings — and delete the other two.** It's already a
dep, it's the JAX/`mkdocs-material` path, and it renders docstrings live from the
existing NumPy-style docstrings that `AGENTS.md:90` already mandates. Remove
`pdoc` and `pydoc-markdown` from `[dependency-groups].dev`
(`pyproject.toml:55-56`), delete `pydoc-markdown.yml`, delete or rewrite
`docs/gen_api_docs.sh`, and stop the README/index from advertising the pdoc flow.

---

## D. Antipatterns to fix (small immediate fixes — current → problem → fix)

1. **Broken landing-page links.**
   `index.md:18` → "See formalism details in formalism.md. For runnable
   examples, see examples/." Neither `docs/formalism.md` nor `examples/` exists
   (`formalism.md` was archived to `docs/archive/formalism.md`; no `examples/`
   tree). **Fix:** point to `archive/formalism.md` or, better, to a real Theory
   page; either create a minimal `examples/` or delete the sentence.

2. **"For AI agents" framing in user-facing docs.**
   `README.md:39`, `CODEBASE_COMPREHENSIVE.md:3`, `ENVIRONMENT_COMPREHENSIVE.md:3`,
   `advanced/README.md:15`, `archive/misc/references/README.md:18`. **Fix:**
   rewrite intros to address *users/contributors*; move the agent-routing
   guidance into `AGENTS.md` only (where it belongs).

3. **Site-/person-specific absolute paths in user docs.**
   `ENVIRONMENT_COMPREHENSIVE.md` uses `/path/to/lorrax_X` and concrete module
   variants `lorrax_A|B|C` (`:65,151,156,161,195,283,321`);
   `PLAN_zeta_g_flat_migration.md:17` hard-codes `SLURM_JOBID=52841861` and a
   `runs/MoS2/...` path; `plans/*.md` and `SIGMA_FREQ_AUDIT_STATUS.md` embed
   `/pscratch`, `/global`, `jackm`, `$SCRATCH`, `HOME/software` (grep hit list in
   §A). **Fix:** in user docs use placeholders (`$LORRAX_ROOT`,
   `<your-module-name>`); confine the literal `lorrax_A|B|C` multi-checkout
   scheme to a developer note, not the install page.

4. **Module-name inconsistency `lorrax` vs `lorrax_X`.**
   `README.md:26` says `module load lorrax`; `AGENTS.md:79` and
   `ENVIRONMENT_COMPREHENSIVE.md:161` say `module load lorrax_X (X = A|B|C)`.
   **Fix:** one canonical name in user docs.

5. **Stale `gw_isdf/` paths.**
   `README.md:14` references `gw_isdf/gw_jax.py`; `[project.scripts]` uses
   `gw.gw_jax:main` (`pyproject.toml:23-24`); `index.md:35` correctly says
   `src/gw/gw_jax.py`. The archive is riddled with `src/gw_isdf/...`
   (`archive/formalism.md:54`, `archive/cohsex_jax_physics.md` passim). **Fix:**
   correct README to `src/gw/`; leave archive frozen but note it's stale.

6. **Raw LLM artifact left in a doc.**
   `NEW_WINDOW_MINIMAX_GUIDELINES.md:1` literally begins `"Absolutely — here is a
   cleaned-up, self-contained note…"`. Unprofessional; **Fix:** strip the
   conversational preamble (or archive the whole file, §E).

7. **Emoji / chat decoration in reference docs.**
   `AGENT_TODO.md` (🏗️ headers, `:7`), `advanced/jax_multihost.md`,
   `archive/nufft/NUFFT_BACKEND_STATUS.md`. Acceptable in dev notes, not in
   shipped reference docs. **Fix:** de-emoji anything promoted to the site.

8. **Filename version markers.**
   `GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`, `ctsp_revised.md`. Versioning belongs
   in git, not filenames. **Fix:** drop `_REVISED` on promotion.

9. **SHOUTING_SNAKE_CASE filenames everywhere.** Every top-level doc is
   `ALL_CAPS_LIKE_THIS.md`. The bar (JAX/ASE) uses lowercase topic names
   (`installation.md`, `quickstart.md`). **Fix:** rename on the move into the new
   tree (§E).

---

## E. Target docs IA (structural proposal)

Adopt mkdocs-material (already a dep) with this `nav`. Names are the *durable
topic*, with the current source file in parentheses (KEEP / MERGE / REWRITE).

```
mkdocs.yml  (NEW — wire the stack that's already paid for)
docs/
  index.md                     Landing: what LORRAX is, 5-line pitch, links
                               (REWRITE from current index.md + README ¶1-2;
                                drop broken formalism.md/examples links)
  installation/
    index.md                   Support matrix (OS/CUDA/MPI/cluster) + from-source
                               (REWRITE: extract the *generic* parts of
                                ENVIRONMENT_COMPREHENSIVE §7 "Generic SLURM")
    perlmutter.md              Site-specific (MOVE from config/README.md +
                                ENVIRONMENT_COMPREHENSIVE Perlmutter sections;
                                this is where lorrax_A|B|C, lxrun belong)
    ffi-native-libs.md         The cuSolverMp/phdf5/SLATE build step
                               (NEW — the #1 onboarding cliff per CONTEXT facts)
  quickstart.md                One end-to-end COHSEX run (MERGE the 3 snippets in
                                README:18-24 / index:53-58 / AGENTS:59-86)
  user-guide/
    inputs.md                  cohsex.in reference (NEW/from sandbox docs)
    running-gw.md, centroids.md, outputs.md
  theory/
    overview.md                (= archive/formalism.md, promoted)
    physics.md                 (KEEP PHYSICS_COMPREHENSIVE.md, rename)
    isdf-zeta-vq.md            (KEEP ZETA_V_Q_ALGORITHMS.md)
    minimax-quadrature.md      (KEEP MINIMAX_QUADRATURE.md)
    symmetry.md                (KEEP SYMMETRY_COMPREHENSIVE.md)
  architecture/
    codebase.md                (KEEP CODEBASE_COMPREHENSIVE.md, de-"agent")
    memory-model.md            (KEEP MEMORY_MODEL.md)
    multihost.md               (KEEP advanced/jax_multihost.md)
  api/                         (GENERATED by mkdocstrings — NOT hand-written)
  contributing.md              (NEW — humanize AGENTS.md:88-104 coding standards)
  changelog.md                 (NEW)
```

Everything not in that tree moves out of the user namespace:

```
docs/dev/   (NEW, or a top-level DEVNOTES/ — explicitly NOT part of the site nav)
   ├ plans/        ← FREQ_INTEGRATION_REWRITE_PLAN, PLAN_zeta_g_flat_migration,
   │                 plans/unified_slab_io, plans/phdf5_cray_mpich_migration
   ├ progress/     ← FREQ_INTEGRATION_PROGRESS, SIGMA_FREQ_AUDIT_STATUS
   ├ notes/        ← PROFILING_SUGGESTIONS, NEW_WINDOW_MINIMAX_GUIDELINES,
   │                 GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED, AGENT_TODO
   └ archive/      ← existing docs/archive/** (frozen; keep as-is)
```

KEEP / MERGE / ARCHIVE / CUT / REWRITE summary:

- **KEEP (promote into nav, lightly edit):** `PHYSICS_COMPREHENSIVE.md`,
  `CODEBASE_COMPREHENSIVE.md`, `MEMORY_MODEL.md`, `SYMMETRY_COMPREHENSIVE.md`,
  `MINIMAX_QUADRATURE.md`, `ZETA_V_Q_ALGORITHMS.md`, `advanced/jax_multihost.md`,
  `advanced/HL_GPP_derivation.md`.
- **REWRITE:** `index.md` (landing), `README.md` (de-agent, fix paths),
  `ENVIRONMENT_COMPREHENSIVE.md` → split into `installation/` (generic) +
  `installation/perlmutter.md` (site).
- **MERGE:** the three "quick start" snippets → one `quickstart.md`.
- **ARCHIVE (into `docs/dev/`, out of nav):** all `*_PLAN*.md`, `*_PROGRESS.md`,
  `*_AUDIT_STATUS.md`, `AGENT_TODO.md`, `PROFILING_SUGGESTIONS.md`,
  `NEW_WINDOW_MINIMAX_GUIDELINES.md`, `GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`,
  `plans/*`.
- **CUT (delete):** `pydoc-markdown.yml`, the pdoc/pydoc-markdown story in
  README/index, the buggy `docs/gen_api_docs.sh` (replace with `mkdocs build`),
  `archive/nufft/` if the NUFFT backend is truly dead (`NUFFT_BACKEND_STATUS.md`).

---

## F. Tone & conventions to adopt (the JAX bar)

1. **Audience = users/contributors, not agents.** Strip "For AI agents" from all
   nav docs; keep agent routing solely in `AGENTS.md` + `docs/dev/`.
2. **mkdocs-material admonitions** (`!!! note`, `!!! warning`) instead of bold
   ad-hoc callouts; especially a `!!! warning` for the FFI build cliff.
3. **Versioned, generic commands.** Pin the *actual* JAX version (CONTEXT flags
   `pyproject.toml` says `jax[cuda13]>=0.9.0` but the image runs ~0.5.3) and use
   `uv run` consistently; no bare `python -m`.
4. **No site/person paths in nav docs** — `$LORRAX_ROOT`, `<module-name>`
   placeholders; literal NERSC paths only in `installation/perlmutter.md`.
5. **Lowercase topical filenames; no `_COMPREHENSIVE`/`_REVISED`/`_STATUS`
   suffixes** in the user tree.
6. **One source of truth per fact.** Today install info is split across
   `README.md`, `AGENTS.md`, `index.md`, `ENVIRONMENT_COMPREHENSIVE.md`,
   `config/README.md` — collapse to the `installation/` pages and cross-link.

---

## G. Single highest-leverage change

**Wire the mkdocs-material + mkdocstrings site that the project already depends
on, and define its `nav` (§E) — moving every `*_PLAN/_PROGRESS/_AUDIT/TODO` file
out of the user namespace into `docs/dev/`.** This one action: (a) turns the dep
investment in `pyproject.toml:12-15` into an actual rendered site; (b) forces the
user-vs-developer separation that is the root IA defect; (c) auto-generates the
missing API reference from existing docstrings; and (d) gives a concrete home
into which the §D small fixes land. It converts "a folder of 34 caps-lock files
for agents" into "a JAX-grade docs site for researchers" with the content that
already exists.
