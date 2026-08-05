# Agent 2 — Documentation / skills / instructional-text lens

Scope: every `.md`, `SKILL.md`, modulefile comment, template, and narrative in the
sandbox that describes where the four runtime-dep dirs live or how the shifter env is
assembled, and that would read stale/wrong after they move to `$HOME/software/`.
Read-only on everything except this file. Run scripts are Agent 1's; I only quote them
to characterize the doc-vs-reality divergence.

## TL;DR

The prose/doc surface is **remarkably clean** of scratch dep paths. AGENTS.md,
agents_xprof.md, PARSE_OUTPUTS.md, KNOWN_SANDBOX_ERRORS.md, all templates, and every
SKILL *except* `execute_workflow` contain **zero** references to the four scratch dep
dirs. The migration's doc footprint is essentially three things:

1. **`skills/execute_workflow/SKILL.md` — the one active instruction that matters**, and
   it is already wrong in a *different* way than the migration (no bind mounts at all).
2. **One historical CHANGELOG entry** (dated `2026-04-16`) that names the scratch nvhpc
   path — leave alone (point-in-time record, per the contract).
3. **The `reports/lorrax_install_maintain_blitz_2026-05-13/` campaign** — a dated audit
   that uses *container-internal* mount paths (`/lorrax_phdf5`, stable under migration)
   and `$SCRATCH/.jax_cache` (a cache, not a dep). Leave alone.

Crucially: **`$HOME/software/` already exists and is already the install convention.**
`ls /global/homes/j/jackm/software/` shows `slate`, `lorrax_A/B/C`, `lorrax_phdf5_openmpi`,
etc. The run scripts already point `PYTHONPATH` and `LD_LIBRARY_PATH` at
`/global/homes/j/jackm/software/{lorrax_C,slate}` today. So migrating the *deps* (nvhpc,
phdf5 stages, slate stage) into the same tree is consistent with established practice,
not a new convention.

---

## 1. Every doc file:line that mentions a scratch dep path or the staging/build process

| File:line | Classification | Current text | Proposed text | Why |
|---|---|---|---|---|
| `skills/execute_workflow/SKILL.md:142` | **MUST CHANGE** (active instruction) | `SITE=$HOME/scratchperl/.isdf/isdf_venvs/isdf_site` | `SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site` (keep literal; Shifter may not expand `$HOME` — note already states this two lines down) | Path component is fine post-migration but is the spot to also add the missing bind mounts (see §2). `isdf_site` is unaffected by the dep move; included only because it's part of the same prefix block that must change. |
| `skills/execute_workflow/SKILL.md:143-146` | **MUST CHANGE** (active instruction) | `SHIFTER="shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 \` `--env=PYTHONPATH=/global/u2/j/jackm/software/lorrax/src:$SITE \` `--env=JAX_ENABLE_X64=1 \` `--env=HDF5_USE_FILE_LOCKING=FALSE"` | Add `--module=gpu,mpich`, the three `--volume` bind mounts (sourced from the **new** `$HOME/software` paths), the `LD_LIBRARY_PATH`/`LD_PRELOAD`/`MPICH_GPU_SUPPORT_ENABLED` envs, and fix the dead `PYTHONPATH` (see §2 for the exact recommended block). | This prefix is the **single source of truth** an agent copies to run GWJAX on Perlmutter, and it (a) has no bind mounts, (b) points `PYTHONPATH` at `/global/u2/j/jackm/software/lorrax/src` which **does not exist** (verified: only `lorrax_A/B/C` exist there, not `lorrax`). Both problems compound at migration time. |
| `CHANGELOG.md:2398-2401` | **HISTORICAL — leave alone** | `- NVHPC (for cuSOLVERMp ONLY): /opt/.../25.5/ staged to /pscratch/sd/j/jackm/lorrax_nvhpc and bind-mounted into Shifter at /lorrax_nvhpc — ...` | (no change) | Under dated header `## 2026-04-16: JAX FFI scaffolding`. CONTEXT explicitly excludes historical CHANGELOG entries. It is an accurate point-in-time record, not active guidance; not actively misleading because it is clearly dated. |
| `reports/lorrax_install_maintain_blitz_2026-05-13/*.md` (agent_1/2/3/4, round2_*, consensus, bispinor_smoke_runbook) | **HISTORICAL — leave alone** | Many refs to `/lorrax_nvhpc`, `/lorrax_phdf5`, `/lorrax_slate` (container-internal mount points) and `$SCRATCH/.jax_cache`; one to `/pscratch/sd/j/jackm/lorrax_nvhpc` (`agent_1.md:267`, as a *cited problem*: "another user's `$SCRATCH` won't…"). | (no change) | Dated one-shot audit campaign. The container-internal paths (`/lorrax_*`) are the mount **targets** and are invariant under the migration (only the `--volume` LHS changes). `$SCRATCH/.jax_cache` is a JIT cache, not one of the four deps. `agent_1.md:267` is *arguing for* exactly this migration — leaving it is fine and even supportive. |
| `modulefiles/lorrax_agent/1.0.lua:92` (comment) | **No change needed** | `-- live at $HOME/software/lorrax_<LETTER>, so the trailing component is` | (no change) | Already documents the `$HOME/software` convention. Confirms the migration target is the existing convention, not a new one. |
| `modulefiles/lorrax_agent/1.0.lua:133` | **No change needed** | `local shifter_cmd = "$LORRAX_SHIFTER"` | (no change) | The sandbox modulefile splices the shifter string **verbatim from the base `lorrax_X` module** (`config/modulefiles/lorrax/*.lua` inside each source checkout). The bind-mount literals live *there*, not in this sandbox file. That base modulefile is source-tree / Agent-1 territory; flag only — see Risks. |
| `modulefiles/lorrax_agent/1.0.lua:50`, `:246` (comments) | **Pre-existing doc rot, not migration** | `# (one-time build, see docs/ENVIRONMENT_COMPREHENSIVE.md §3.5).` and `# ... (see docs/ENVIRONMENT_COMPREHENSIVE.md §3.5).` | (out of scope for this migration; flag) | `docs/ENVIRONMENT_COMPREHENSIVE.md` **does not exist** in the sandbox `docs/`; the file only exists inside `sources/lorrax_*/docs/`. This is an already-broken doc link unrelated to the dep move. Noted for completeness; do not fix as part of migration. |

**Files audited and confirmed clean (zero dep-path / staging references):**
`AGENTS.md`, `agents_xprof.md`, `PARSE_OUTPUTS.md`, `KNOWN_SANDBOX_ERRORS.md`,
`skills/build_inputs/SKILL.md`, `skills/compare/SKILL.md`,
`skills/checkpoint/SKILL.md`, `skills/profiling_stack/SKILL.md`, all of `docs/**`
(BGW/QE/gwjax specs are physics/format docs), all of `templates/**`.
`execute_workflow/SKILL.md:143` is the **only** `SHIFTER=` prefix defined anywhere in
`skills/`, and the only `--volume`/`shifter --` invocation in `skills/`.

---

## 2. Focused: `execute_workflow/SKILL.md` vs the run scripts — what is the divergence?

### What the SKILL says (lines 134-152)
> "GWJAX runs in the Shifter container (NVIDIA JAX image, JAX 0.7.2 / Python 3.12) for
> multi-GPU execution."
```bash
SITE=$HOME/scratchperl/.isdf/isdf_venvs/isdf_site
SHIFTER="shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 \
    --env=PYTHONPATH=/global/u2/j/jackm/software/lorrax/src:$SITE \
    --env=JAX_ENABLE_X64=1 \
    --env=HDF5_USE_FILE_LOCKING=FALSE"
```
No `--volume`, no `mpich` module, no `LD_LIBRARY_PATH`, no `LD_PRELOAD`,
no `MPICH_GPU_SUPPORT_ENABLED`.

### What the run scripts actually do
`runs/CrI3/04_gw_6x6_600b_2026-06-17/run_cri3_now.sh:13-15` (Cray path, current CrI3/VI3):
```bash
VOL="--volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc \
     --volume=/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage:/lorrax_phdf5 \
     --volume=/pscratch/sd/j/jackm/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:\
/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:... \
     --env=MPICH_GPU_SUPPORT_ENABLED=1"
SHIFTER_OM="shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich $VOL ... $LDP \
     --env=PYTHONPATH=$OMWT/src:$SITE:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
```
`runs/Si/C_vqsph_si10_ref/run.sh:12-19` (OpenMPI path) is the same shape with the openmpi
phdf5 stage and `lorrax_nvhpc/25.5_cuda12.9` and `PYTHONPATH=.../software/lorrax_C/src`.

### Characterization — is the SKILL a different code path, or just stale?
It is **stale, not an alternate single-GPU path.** Evidence:
- The SKILL's prefix is used in the SKILL for *both* the single-GPU preprocessing steps
  (5a/b/c) **and** the multi-GPU `gw.gw_jax` step (step 6). So it is not a "lite,
  single-GPU only" variant — it claims to cover the full multi-GPU run, which is exactly
  what the heavy run scripts do.
- Its `PYTHONPATH` target `/global/u2/j/jackm/software/lorrax/src` **does not exist on
  disk** (verified: that tree holds `lorrax_A`, `lorrax_B`, `lorrax_C`, not `lorrax`). A
  prefix with a dead PYTHONPATH cannot have been the working invocation for the current
  runs. The run scripts use the real `…/software/lorrax_{B_orbmag_wt,C}/src`.
- It says "JAX 0.7.2 / Python 3.12" in prose but pins `--image=...jax:25.04-py3`; the
  CHANGELOG/source env doc tie the FFI deps (cuSolverMp, phdf5, slate) to that very
  image. A run that loads `gw.gw_jax` with cuSolverMp/SLATE FFI **requires** the three
  bind mounts; without them the FFI `.so`s fail to resolve. So the SKILL prefix as
  written would only work for pure-JAX paths that never touch the FFI — i.e. it predates
  the FFI/distributed-GPU stack becoming the default.

Conclusion: the SKILL froze at an earlier era (pre-FFI, single `lorrax/src` checkout)
and was never updated when (a) the build split into `lorrax_A/B/C/D` checkouts under
`$HOME/software`, and (b) the nvhpc/phdf5/slate bind mounts became mandatory. The
migration is the right moment to fix it, because whatever paths we write now should be
the post-migration `$HOME/software` ones.

### Recommendation (concrete) — replace SKILL.md:139-151 with
```bash
# $HOME on Perlmutter = /global/homes/j/jackm ; Shifter does NOT expand $HOME,
# so all paths below are literal. Deps live in $HOME/software (40 GB quota, not purged).
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
LORRAX_SRC=/global/homes/j/jackm/software/lorrax_B/src   # the active checkout

# bind-mount the three host dep dirs (post-migration locations)
VOL="--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc \
     --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 \
     --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate"

SHIFTER="shifter --module=gpu,mpich --image=nvcr.io/nvidia/jax:25.04-py3 $VOL \
    --env=PYTHONPATH=$LORRAX_SRC:$SITE \
    --env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    --env=JAX_ENABLE_X64=1 \
    --env=HDF5_USE_FILE_LOCKING=FALSE"
```
Caveats to leave as a SKILL note: (i) the OpenMPI runs use a different phdf5 stage
(`lorrax_phdf5_openmpi/stage`) and `lorrax_nvhpc/25.5_cuda12.9` and no `LD_PRELOAD`/`mpich`
module — so the SKILL should say "Cray-MPICH variant shown; for the OpenMPI/hpcx variant
swap the phdf5 stage and drop the gtl preload". (ii) Because the run scripts already pull
these via the base `LORRAX_SHIFTER` modulefile, the cleanest long-term fix is for the
SKILL to tell agents to **`module load lorrax_X` and use `$LORRAX_SHIFTER`** rather than
hand-roll the prefix — but that is a larger refactor; for this migration, getting the
literal paths right is sufficient. **Whoever updates the base modulefile and the run
scripts (Agent 1) and whoever updates this SKILL must use the same post-migration paths.**

---

## 3. Docs that describe the dep dirs as BUILD ARTIFACTS (need a "how to rebuild/relocate" note)

| Doc | Treats deps as | Needs rebuild/relocate note? |
|---|---|---|
| `CHANGELOG.md:2398-2401` (`2026-04-16` entry) | **Build artifact** — explicitly: NVHPC `/opt/.../25.5/` "**staged to** `/pscratch/.../lorrax_nvhpc`". Also names the staging driver: "Build: `src/ffi/common/cpp/build.sh` via `run_shifter.sh`". | It's historical, so don't edit it — but it confirms these are *staged copies* of a host NVHPC tree, **not** built-in-place. → A plain `cp -a` of the scratch dirs to `$HOME/software` reproduces them faithfully (they are extracted subsets, no absolute-path baking in the data itself). The only path-baking is in the **consumer** RPATH/env, which the run scripts/modulefile control. |
| `reports/lorrax_install_maintain_blitz_2026-05-13/agent_1.md:103-114` | **Build artifact, with provenance** — names the staging scripts: phdf5 stage built by `stage_cray.sh`/`stage_openmpi.sh`; slate libs "staged at `/lorrax_slate/lib` by `slate/scripts/stage_cray.sh`". | Historical; don't edit. But it's the authoritative record that the four dirs are **outputs of `stage_*.sh` scripts living in the LORRAX source tree**, so "how to rebuild" already exists (re-run those scripts pointing the output at `$HOME/software`). The migration note in the orchestrator's plan should cite these scripts as the canonical rebuild path, and note that a copy is sufficient for *relocation* (rebuild only needed if the host toolchain changes). |
| `reports/lorrax_install_maintain_blitz_2026-05-13/agent_1.md:242` | **Build artifact with a baked path** — "embeds `/lorrax_phdf5/lib;/lorrax_slate/lib` in the binary" (FFI `INSTALL_RPATH`). | The RPATH bakes the **container-internal** mount target (`/lorrax_phdf5`), which is invariant under migration. So the FFI `.so` does **not** need rebuilding for the move — only the `--volume` LHS changes. This is the key reassurance: relocation is copy-only on the consumer side. |

Net: the deps **are** build artifacts, but they are *staged copies / extracted subsets*,
so for relocation a `cp -a` (or `rsync`) is sufficient and no recompile of either the
deps or the LORRAX FFI is required. A rebuild is only needed if the underlying host
NVHPC / Cray-MPICH / SLATE versions change, in which case `build.sh`/`stage_cray.sh`/
`stage_openmpi.sh` (in the LORRAX source tree, not the sandbox) are the procedure.

---

## 4. Docs that, if left unchanged, would actively mislead a future agent into recreating the deps on scratch

**Only one, and indirectly:** `skills/execute_workflow/SKILL.md` is the doc an agent is
told (by AGENTS.md "Read order") to read before running anything. After the migration:
- Its prefix has **no bind mounts at all**, so an agent following it verbatim would get
  FFI load failures and — with no pointer to where the deps live — might "fix" it by
  re-staging nvhpc/phdf5/slate onto `$SCRATCH` (the path the old run scripts and the
  `2026-04-16` CHANGELOG entry show), recreating exactly what we moved off scratch.
- Its dead `PYTHONPATH=/global/u2/j/jackm/software/lorrax/src` would also push an agent
  to "recreate" a `lorrax/` checkout. → fixing this to a real `…/software/lorrax_B/src`
  is part of the same edit.

No *other* doc actively misleads: AGENTS.md/templates don't mention deps; the CHANGELOG
and blitz reports are dated and clearly historical, so an agent reading them for "current
state" has the date as a guard. The risk is concentrated entirely in the one SKILL.

A defensive addition worth making regardless of migration: a one-line note in
`execute_workflow/SKILL.md` (and/or `KNOWN_SANDBOX_ERRORS.md`) stating **"the nvhpc/phdf5/
slate dep dirs live in `/global/homes/j/jackm/software/` (not `$SCRATCH`); never re-stage
them onto scratch — scratch is purged."** That single sentence is the cheapest insurance
against re-creation.

---

## 5. Open questions / risks

1. **Base `LORRAX_SHIFTER` modulefile is the real source of the bind mounts, and it is
   inside the source checkouts** (`sources/lorrax_X/config/modulefiles/lorrax/*.lua`),
   which `modulefiles/lorrax_agent/1.0.lua:133` splices verbatim. I treated it as
   source-tree / Agent-1 territory and did not edit. **Risk:** if Agent 1 scopes only
   `runs/**` scripts, the modulefile-driven `lxrun` path (the *other* way agents launch
   GWJAX) keeps the scratch `--volume` strings and silently defeats the migration.
   Someone must own `sources/lorrax_*/config/modulefiles/lorrax/*.lua`. Flag to
   orchestrator.
2. **`sources/lorrax_*/docs/ENVIRONMENT_COMPREHENSIVE.md` documents the deps via env-var
   indirection** with scratch defaults: `$LORRAX_FFI_NVHPC_DIR` (default
   `$SCRATCH/lorrax_nvhpc`), `$LORRAX_FFI_PHDF5_DIR` (default `$SCRATCH/lorrax_phdf5_cray/
   stage`), `$LORRAX_FFI_SLATE_DIR` (default `$SCRATCH/lorrax_slate_cray/stage`) — but
   `LORRAX_SLATE_INSTALL_DIR_DEFAULT` already = `$HOME/software/slate/install`. This is
   the *cleanest* migration lever (just change three `_DIR` defaults to `$HOME/software/
   …`), but it lives in the **source repo**, not the sandbox. Out of my prose lens; flag
   that the real fix may be one-line-per-default in the source CMake/site_config rather
   than editing run scripts at all. The modulefile reference to this doc
   (`docs/ENVIRONMENT_COMPREHENSIVE.md`) is also **broken in the sandbox** (file only
   exists under `sources/`), a pre-existing rot worth fixing opportunistically.
3. **Two MPI variants, two stages, two nvhpc subdirs.** Cray runs use
   `lorrax_phdf5_cray/stage` + `lorrax_nvhpc/0.7.2_cuda12.9`; OpenMPI runs use
   `lorrax_phdf5_openmpi/stage` + `lorrax_nvhpc/25.5_cuda12.9`. Any SKILL update must
   show (or at least name) both, or it will mislead whichever variant it omits. The
   nvhpc dir is shared (one dir, versioned subdirs) — good, only one copy.
4. **`$HOME/software/lorrax_phdf5_openmpi` already exists but is incomplete.** It contains
   only `cache/` (conda build cache: `hdf5.conda`, `libaec.conda`), **not** the `stage/`
   dir the run scripts mount. So the openmpi phdf5 *stage* still lives only on scratch —
   the migration genuinely needs to copy `stage/` (or rebuild from the cache). Don't
   assume the $HOME tree is already complete just because the dir name is present.
5. **`u2` vs `homes`.** `/global/u2/j/jackm` and `/global/homes/j/jackm` resolve to the
   same tree (verified via `readlink -f`). SKILL.md uses `u2` for PYTHONPATH; run scripts
   use `homes`. Either works, but **pick one literal and use it consistently** in the
   updated SKILL to avoid a future agent thinking they're two different stores.
6. **CHANGELOG forward-entry, not a back-edit.** Per AGENTS.md rule #4, the migration
   should be recorded as a *new dated* CHANGELOG entry ("deps relocated to $HOME/software")
   rather than editing the `2026-04-16` block. That keeps history honest and gives future
   agents a dated "current state" signal that supersedes the old scratch references.
