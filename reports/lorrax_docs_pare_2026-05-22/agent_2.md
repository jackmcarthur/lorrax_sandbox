# Agent 2 — Install / environment / porting / overview docs

**Auditing**: `README.md`, `AGENTS.md`, `docs/CODEBASE_COMPREHENSIVE.md`,
`docs/ENVIRONMENT_COMPREHENSIVE.md`, `config/README.md`, `src/ffi/PORTING.md`
on branch `agent/install-blitz-integration` (HEAD `3079a1f`).

**Out of scope**: physics/theory docs, memory model, frequency-integration docs
(Agents 1, 3, 4 cover those).

---

## 1. Per-doc verdict

| Doc | Lines | Action | Justification |
|---|---:|---|---|
| `README.md` | 39 | **KEEP + spot-fix** | Accurate overview; doc-index is stale (missing SYMMETRY_COMPREHENSIVE, ZETA_V_Q_ALGORITHMS; GN_PPM guide is listed but is "REVISED" noise) |
| `AGENTS.md` | 112 | **KEEP + spot-fix** | Code map and run commands are current; Key documentation table is stale; `SIGMA_FREQ_AUDIT_STATUS.md` is the only doc not in README |
| `docs/CODEBASE_COMPREHENSIVE.md` | 641 | **KEEP + spot-fix** | Module map accurate; 2 stale issues (see §3); no renamed functions found at the May refit |
| `docs/ENVIRONMENT_COMPREHENSIVE.md` | 455 | **MERGE + trim** | 60% of content duplicates `config/README.md`; remaining 40% is Blitz-drifted (see §3); candidate for becoming a one-page pointer |
| `config/README.md` | 269 | **KEEP** | Most accurate and up-to-date of the three Perlmutter docs; received Blitz #5 additions; should be the single authority for cluster usage |
| `src/ffi/PORTING.md` | 225 | **KEEP + spot-fix** | Structure is correct; 3 stale items from Blitz #0 and #3 (see §3) |

---

## 2. Per-doc breakdown

### 2.1 `README.md` (39 lines)

**Status**: largely current, two issues.

**Doc-index staleness**: the listed docs are:
- PHYSICS_COMPREHENSIVE, CODEBASE_COMPREHENSIVE, ENVIRONMENT_COMPREHENSIVE, MEMORY_MODEL, MINIMAX_QUADRATURE, GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED

What's missing from README but exists in `docs/`: SYMMETRY_COMPREHENSIVE,
ZETA_V_Q_ALGORITHMS, SIGMA_FREQ_AUDIT_STATUS (this one is in AGENTS.md).

`GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` is listed as a top-level reference doc.
Its "REVISED" name signals it superseded something — if it's now the canonical
PPM guide, rename and drop "REVISED" from the README entry (and from the file).

**Quick-start is correct**: `uv sync`, `uv run python -m pytest -q`, `uv run python -m gw.gw_jax`. No issues.

**QUICKSTART gap**: README's Quick Start section is 3 lines that dump the user
into either `uv run` (local) or `config/README.md` (Perlmutter). That's fine —
see §5 for quickstart recommendation.

### 2.2 `AGENTS.md` (112 lines)

**Status**: code map is accurate. Two issues.

**Key documentation table** (§"Key documentation"): lists 6 docs.
Missing: ENVIRONMENT_COMPREHENSIVE (in README but not AGENTS.md),
SYMMETRY_COMPREHENSIVE, ZETA_V_Q_ALGORITHMS.
Includes SIGMA_FREQ_AUDIT_STATUS (not in README — valid, agents need it).

**Recommendation**: AGENTS.md table is the agent-facing index — it should be
the more complete one. README's list is the human first-reader index — keep it
shorter. Canonicalize: add ENVIRONMENT_COMPREHENSIVE to AGENTS.md table (it
exists and is the install surface; agents working on cluster setup need it).
Add one-line mentions of SYMMETRY_COMPREHENSIVE and ZETA_V_Q_ALGORITHMS.

**"How to run → Perlmutter" section**: still shows
`LORRAX_NGPU=1 lxrun ...` examples but doesn't show `LORRAX_NNODES`. Now that
multi-node lxrun is wired upstream (Blitz #5), add `LORRAX_NNODES=2 lxrun`
example here (same as the one in `config/README.md`).

**Coding standards**: current and accurate.

### 2.3 `docs/CODEBASE_COMPREHENSIVE.md` (641 lines)

**Status**: module map is accurate; function hierarchy matches current code.
No `src/isdf/` references in CODEBASE_COMPREHENSIVE (the rename was already
reflected here). No references to `gflat_to_rchunk` or old sphere-idx accessors
— those are not mentioned at all, so no drift from May refit.

**Two stale issues found**:

1. **§6.1 cohsex.in reference points to sandbox, not repo**:
   > Canonical reference: `docs/docs_gwjax/COHSEX_INPUT.md` in the sandbox
   > (not in this repo)

   This is an honest admission but the right fix is Blitz #2 (move the doc
   into the repo). Until then, the note is accurate — leave it.

2. **§8 "Agent todos" link at the bottom**:
   > `AGENT_TODO.md`

   The file exists (292 lines) and its own header says "these are NOT the
   user's current priorities." The link in CODEBASE_COMPREHENSIVE's "Next
   Steps" section sends agents to a parking-lot document. Recommend: drop
   the `AGENT_TODO.md` link from the "Next Steps" footer here and in
   ENVIRONMENT_COMPREHENSIVE.

**Overall drift estimate**: < 10%. KEEP with the two spot-fixes above.

### 2.4 `docs/ENVIRONMENT_COMPREHENSIVE.md` (455 lines)

This is the most drifted doc in the slice. **Four confirmed staleness issues**:

**Issue 1 (Blitz #0 drift) — allocator table §3.1**:
```
| `XLA_PYTHON_CLIENT_ALLOCATOR` | `platform` | Use CUDA async mempool ... |
| `TF_GPU_ALLOCATOR`            | `cuda_malloc_async` | ... |
```
The modulefile now sets `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (confirmed in
`0.1.0.lua:168`). The doc table still shows `platform` + `TF_GPU_ALLOCATOR`.
This is two rows wrong.

**Issue 2 (Blitz #0 drift) — §3.2 heading and §8.3**:
Section 3.2 is titled "Why the platform allocator, not MEM_FRACTION=0.95" —
but the allocator is now `cuda_async`, not `platform`. The explanation
(`platform` = `cudaMallocAsync`) is approximately correct but the name is
wrong. §8.3 fix recommendation says "confirm `XLA_PYTHON_CLIENT_ALLOCATOR=platform` are set" — wrong value.

**Issue 3 (Blitz #1 drift) — §1.1 JAX pin**:
```
| **jax[cuda13]** | ≥0.9.0 | ...
```
`pyproject.toml` now declares `jax>=0.5` with no `cuda13` extra (with an
explicit comment explaining why). Both the version and the extra specifier
are stale. Same issue in §1.2 (`build` group description) and §8.1 fix
step 4 ("jaxlib must be the CUDA build (`jax[cuda13]`)").

**Issue 4 (Blitz #3 / not yet fixed) — MPI variable names in PORTING.md
table also appear in the MPI override discussion**:
§5.5 uses `$LORRAX_MPI_INCLUDE_DIR` / `$LORRAX_MPICH_LIB_DIR` — these
are the old CMake override names. `config/mpi_stacks/cray_mpich.cmake` now
uses `LORRAX_MPI_INCLUDE_DIR_CT` and `LORRAX_MPI_LIB_DIR_CT` as the
canonical names. (See also PORTING.md below.)

**Issue 5 (doc lie, not fixed by Blitz #5) — multi-node example**:
Line 322:
```bash
LORRAX_NNODES=2 LORRAX_NGPU=8 lxrun python3 -u -m gw.gw_jax -i cohsex.in
```
`LORRAX_NGPU` is GPUs **per node** (confirmed in `0.1.0.lua:302,326`).
With 2 nodes and 4 GPUs/node on Perlmutter, the correct command is
`LORRAX_NGPU=4`, giving `total_ranks = 2×4 = 8`. With `LORRAX_NGPU=8` the
srun invocation becomes `--gres=gpu:8 -N 2 -n 16` — requests 8 GPUs per
node, which Perlmutter nodes don't have. The expected topology comment
says `jax.process_count() == 8` (not 16), confirming `NGPU=8` is wrong.
**This lie also appears in `0.1.0.lua:30` in the same form.**

**Issue 6 (Blitz #5 — not documented) — LORRAX_CONTAINER_*_PATH**:
§5.2 bind-mount table uses the old host-side variables:
- `$LORRAX_FFI_NVHPC_DIR`, `$LORRAX_FFI_PHDF5_DIR`, `$LORRAX_FFI_SLATE_DIR`

The modulefile now also exposes `LORRAX_CONTAINER_NVHPC_PATH`,
`LORRAX_CONTAINER_PHDF5_PATH`, `LORRAX_CONTAINER_SLATE_PATH` (the in-container
mount targets) as overridable env vars. PORTING.md has a section on these;
ENVIRONMENT_COMPREHENSIVE doesn't mention them.

**Issue 7 (Blitz #4 — dist-init docs stale)**:
§6.1 describes dist-init via the old `_maybe_init_jax_distributed()` function
and its `_LORRAX_JAX_DISTRIBUTED_DONE` sentinel guard. Code now uses
`runtime.init_jax_distributed()` (confirmed in `gw_jax.py:13,18`).

**Issue 8 (Blitz #5 — bare-venv fallback stale)**:
§7 "Generic SLURM clusters" bare-venv example still sets
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` + `TF_GPU_ALLOCATOR=cuda_malloc_async`.

**Structural redundancy with `config/README.md`**:
The following sections of ENVIRONMENT_COMPREHENSIVE duplicate `config/README.md`
with the same or less accuracy:

| ENVIRONMENT_COMPREHENSIVE section | Equivalent config/README.md section | Winner |
|---|---|---|
| §4 "Perlmutter via Lmod module" (§4.1–4.4) | §Quick Start + §Usage + §Per-invocation cost | `config/README.md` (more current) |
| §5.2 "Bind-mounts" table | `config/README.md` bind-mount table | `config/README.md` (more current) |
| §5.5–5.6 MPI stack | `config/README.md` Unified Cray MPICH section | `config/README.md` |

The non-redundant material in ENVIRONMENT_COMPREHENSIVE is:
- §1 Dependencies table (accurate but JAX pin stale)
- §2.1 Local dev install (accurate)
- §3 JAX configuration (has stale allocator; worth keeping the env-var table concept)
- §6.2–6.4 Multi-host / non-SLURM details (accurate)
- §8 Troubleshooting (has 2 stale items; otherwise good)

**Recommendation**: Replace ENVIRONMENT_COMPREHENSIVE with a ~100-line document
that covers only the non-redundant material (Dependencies, Local dev install,
JAX env vars, Non-SLURM distributed init, Troubleshooting) and points to
`config/README.md` for all Perlmutter-specific content. The file would then
justify its own existence: it's for non-Perlmutter / non-cluster usage.

### 2.5 `config/README.md` (269 lines)

**Status**: the most current doc in the slice. Blitz #5 landed correctly
here (LORRAX_NNODES, LORRAX_CONTAINER_*_PATH via the overlay section, multi-user
concurrency section).

**One remaining lie in the `lxrun` example in the module description** (same as
ENVIRONMENT_COMPREHENSIVE §6.3): the comment block in `0.1.0.lua` at line 30
shows `LORRAX_NNODES=2 LORRAX_NGPU=8 lxrun ...`. The config/README.md does
NOT repeat this error — `config/README.md` correctly shows `LORRAX_NNODES=2`
without specifying `NGPU=8`. **Only `0.1.0.lua` and `ENVIRONMENT_COMPREHENSIVE`
have the `NGPU=8` error.**

**Allocator in "What `module load lorrax` provides" table**: still shows
`XLA_PYTHON_CLIENT_ALLOCATOR | platform` (line ~63). The modulefile now sets
`cuda_async`. One-line fix.

**`TF_GPU_ALLOCATOR | cuda_malloc_async` row**: should be deleted (removed from
the modulefile per Blitz #0).

### 2.6 `src/ffi/PORTING.md` (225 lines)

**Status**: good structure, 3 stale items.

**Issue 1 (Blitz #0) — Runtime section env var block**:
```
XLA_PYTHON_CLIENT_ALLOCATOR=platform  # = cudaMallocAsync (via TF_GPU_ALLOCATOR)
TF_GPU_ALLOCATOR=cuda_malloc_async
```
Should be a single line: `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`. The comment
"via TF_GPU_ALLOCATOR" is now wrong.

**Issue 2 (Blitz #3) — CMake autodetection table**:
```
| MPI | `$LORRAX_MPI_INCLUDE_DIR` / `$LORRAX_MPICH_LIB_DIR` | ... |
```
After Blitz #3, the single-source `config/mpi_stacks/cray_mpich.cmake` defines
`LORRAX_MPI_INCLUDE_DIR_CT` and `LORRAX_MPI_LIB_DIR_CT`. The PORTING.md
table uses the old names. The `config/mpi_stacks/` directory isn't mentioned
anywhere in PORTING.md — either add a pointer to it or update the variable
names to `_CT` suffixed form.

**Issue 3 (Blitz #3, partial) — "two-source-of-truth" MPI problem**:
PORTING.md does NOT still describe MPI as a two-source-of-truth problem
(the concern in the slice prompt). The document already has a unified
Cray-first framing. However, the old variable names in the CMake autodetect
table are the residual evidence of the old problem.

**Well-done sections** (accurate, keep): Hard requirements table (JAX≥0.5
correctly stated), build system, staging section, Perlmutter-specific section,
phdf5 stack choice and tuning knobs, gotchas.

**select_gpu.sh PBS/PMIx**: PORTING.md says `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`
set by `select_gpu.sh` — does not mention the PBS/PMIx fallback chain that was
added. This is a minor omission (it's implementation-level detail, not a porting
step), but the Gotchas section would benefit from a one-liner noting the script
also covers `PMIX_LOCAL_RANK` / `PMI_LOCAL_RANK` / `OMPI_COMM_WORLD_LOCAL_RANK`.

---

## 3. Per-section verdicts (finer granularity)

### `docs/ENVIRONMENT_COMPREHENSIVE.md` internal sections

| Section | Action | Note |
|---|---|---|
| §1 Dependencies | KEEP + fix | Change `jax[cuda13]≥0.9.0` → `jax≥0.5` (3 places) |
| §2.1 Local dev install | KEEP | Accurate |
| §2.2 Perlmutter | MERGE INTO config/README | Trivial 2-liner pointing to config/README |
| §3.1 Env var table | KEEP + fix | Replace `platform`+`TF_GPU_ALLOCATOR` with `cuda_async` (Blitz #0) |
| §3.2 "Why platform" explanation | KEEP + fix header | Change heading to "Why cuda_async" |
| §3.3–3.5 Device selection, mock, inspect | KEEP | Accurate |
| §4 Perlmutter via Lmod | DELETE | 100% duplicated in config/README.md with higher accuracy |
| §5.1–5.3 FFI targets, bind-mounts, staging | MERGE INTO PORTING.md | Already in PORTING.md more accurately |
| §5.4 Building .so | MERGE INTO PORTING.md | Already in PORTING.md §Build system |
| §5.5–5.6 MPI stack | MERGE INTO PORTING.md/config/README | Duplicated; `_CT` variable name fix needed |
| §6 Multi-host | KEEP | Non-redundant; §6.1 needs dist-init function name fix |
| §7 Generic SLURM / bare-venv | KEEP + fix | Good; fix allocator vars in bare-venv example |
| §8 Troubleshooting | KEEP + fix | §8.3 fix allocator name; otherwise accurate |

After deletion/merge, ENVIRONMENT_COMPREHENSIVE becomes ~100 lines covering
Dependencies, Local dev, JAX config, Multi-host non-Perlmutter, Troubleshooting.

### `docs/CODEBASE_COMPREHENSIVE.md` internal sections

All sections KEEP. Two spot-fixes:
- §9 "Next Steps" footer: drop `AGENT_TODO.md` link.
- §6.1: add a note that Blitz #2 (deferred) will move `COHSEX_INPUT.md` into the repo.

---

## 4. Cross-cutting recommendations

### 4.1 Allocator is wrong in four places

Blitz #0 changed the modulefile to `cuda_async` but did not update the three
prose documents that describe the allocator setting:
1. `docs/ENVIRONMENT_COMPREHENSIVE.md` §3.1 table + §3.2 heading + §8.3 + §7 bare-venv
2. `config/README.md` "What `module load lorrax` provides" table (2 rows)
3. `src/ffi/PORTING.md` Runtime section env block

One pass, four files, ~10 lines of text. This is a fragile pattern: the
modulefile is the authoritative source but the docs are independently
maintained copies. Recommendation: make `0.1.0.lua` the single copy of the env
var table; docs link to it with "`module show lorrax` for the current list."

### 4.2 JAX pin wrong in one doc, correct in code

`pyproject.toml` says `jax>=0.5` (correct). `ENVIRONMENT_COMPREHENSIVE.md` says
`jax[cuda13]>=0.9.0` in three places. PORTING.md correctly says `jax>=0.5` in
its Hard Requirements table. One doc needs 3 fixes.

### 4.3 Duplicate Perlmutter cluster docs (ENVIRONMENT §4 vs config/README)

Both describe lxalloc/lxrun/lxshell with slightly different completeness and
accuracy. `config/README.md` is more accurate post-Blitz. The duplication is
the primary reason ENVIRONMENT_COMPREHENSIVE is hard to maintain: when the
module changes, two docs need updating. After ENVIRONMENT §4 is deleted,
future cluster changes need only touch `config/README.md` and `PORTING.md`.

### 4.4 Two doc-indices that disagree

`README.md` and `AGENTS.md` both list "key docs" but don't agree:
- `ENVIRONMENT_COMPREHENSIVE` is in README but not AGENTS.md.
- `SIGMA_FREQ_AUDIT_STATUS` is in AGENTS.md but not README.
- `SYMMETRY_COMPREHENSIVE`, `ZETA_V_Q_ALGORITHMS`, `MEMORY_MODEL` are not
  in either README or AGENTS.md's Key documentation table.

**Recommendation**: Do NOT create a new `docs/_index.md` (would be a third
index). Instead:
- `AGENTS.md` Key documentation table → the agent-facing index (complete).
  Add ENVIRONMENT_COMPREHENSIVE, SYMMETRY_COMPREHENSIVE, ZETA_V_Q_ALGORITHMS.
- `README.md` doc-list → the first-reader "what exists" pointer (shorter, 5-6 lines).
  Keep physics, codebase, environment, memory model; drop the REVISED guide from
  the top-level list (it's PPM-specific, not general orientation).

This is cheaper than a new `_index.md` and doesn't add a third truth.

### 4.5 `LORRAX_NGPU=8` doc lie: not fixed by Blitz #5

Two locations still carry `LORRAX_NNODES=2 LORRAX_NGPU=8`:
- `docs/ENVIRONMENT_COMPREHENSIVE.md:322`
- `config/modulefiles/lorrax/0.1.0.lua:30` (the comment block at top)

Both should read `LORRAX_NGPU=4` for a 2-node Perlmutter job (4 GPUs/node).
The `lxrun` implementation at `0.1.0.lua:336` correctly uses `--gres=gpu:${ngpu}
-N ${nnodes} -n $((nnodes*ngpu))`, so `NGPU=8` on a 2-node job would request
8 GPUs per node — impossible on Perlmutter. **This is a latent user-facing bug.**

### 4.6 `docs/index.md` is orphaned and stale

`docs/index.md` (58 lines) has two confirmed stale references:
- `src/isdf/common/wfnreader.py` — path doesn't exist; renamed to `src/file_io/wfnreader.py`
- `examples/` — doesn't exist
- Links to `formalism.md` and `docs/api/` — neither exists

This file appears to be from an earlier MkDocs scaffold that was never wired
into a live site. It is not linked from README.md or AGENTS.md. **Recommend
DELETE** (or ARCHIVE if there's any chance a rendered docs site gets built).

### 4.7 `docs/AGENT_TODO.md` — should not be linked from reference docs

The file header says its contents are not current priorities. `CODEBASE_COMPREHENSIVE`
and `ENVIRONMENT_COMPREHENSIVE` both link to it in "Next Steps" footers. Remove
those links. The file can stay as a parking lot, but it shouldn't be
advertised as a reference.

---

## 5. What to write fresh / gaps

### 5.1 No quickstart path for "first calculation on Perlmutter in 10 minutes"

The install-blitz consensus noted a QUICKSTART.md was deferred pending
Blitz #1-#3 landing. Those are now landed. The right artifact is a
**section at the bottom of `config/README.md`**, not a new file:

**Proposed outline (40-50 lines)**:

```
## Quickstart: first GW calculation on Perlmutter

### Prerequisites (one-time)
- [ ] Clone LORRAX to $SCRATCH
- [ ] Edit config/perlmutter/site_config.sh (account, scratch path)
- [ ] bash config/perlmutter/install.sh
- [ ] Run the 3 FFI staging scripts (cusolvermp/phdf5/slate)
- [ ] bash src/ffi/common/cpp/build.sh (inside run_shifter.sh)

### First run (Si test case)
1. lxalloc
2. Copy templates/cohsex.in → your run dir
3. Set WFN.h5 path in cohsex.in
4. lxpre cohsex.in 300
5. lxrun python3 -u -m gw.gw_jax -i cohsex.in

### Expected output
- eqp.dat with quasiparticle energies
- gw.out timing summary
```

This goes inside `config/README.md`, which is already the Perlmutter reference.
No new file needed. Target: ~50 lines.

**Why not README.md**: README's Quick Start is intentionally minimal (3 lines).
The Perlmutter-specific steps belong in `config/README.md`.

**Why not a new `docs/QUICKSTART.md`**: would be a fourth Perlmutter doc surface.

### 5.2 CI existence not documented

`.github/workflows/ci.yml` exists (Blitz #4). Neither AGENTS.md nor ENVIRONMENT_COMPREHENSIVE mentions that CI exists. One sentence in AGENTS.md "Before committing" section: "CI (GitHub Actions) runs pytest on every PR." Trivial addition.

---

## 6. Open questions

**Q1**: Is `config/README.md` the canonical source for cluster docs? If so,
`docs/ENVIRONMENT_COMPREHENSIVE.md` §4 should be deleted, not merged. The
author needs to confirm this is the intended direction before an agent
trims 100 lines of ENVIRONMENT_COMPREHENSIVE.

**Q2**: The `LORRAX_NGPU=8` lie in `0.1.0.lua:30` (the Lua comment block, not
the function implementation) — is this actually causing problems for users, or
does it only matter for docs? The implementation at line 336 is correct; only
the example comment is wrong.

**Q3**: `docs/index.md` — is this wired to a live MkDocs/mkdocstrings build
pipeline anywhere? If not, DELETE is clean. If there's a future plans for a
rendered docs site, ARCHIVE instead.

**Q4**: Should the `config/README.md` allocator table (`platform` + `TF_GPU_ALLOCATOR`)
be fixed in a quick follow-up commit? It's a one-minute change but it's a
correctness issue (anyone reading the doc and manually setting env vars before
`module load lorrax` would set the wrong allocator).

**Q5**: Blitz #2 (cohsex.in schema + `COHSEX_INPUT.md` moves into repo) is
deferred. `CODEBASE_COMPREHENSIVE.md §6.1` notes that the canonical cohsex.in
reference is in the sandbox, not the repo. Should that note become more
prominent (e.g., a warning block) until Blitz #2 lands?

---

## Summary of concrete spot-fixes needed (ranked by user impact)

| Priority | File | Fix | Lines |
|---|---|---|---|
| 1 (bug) | `config/modulefiles/lorrax/0.1.0.lua` | Change `LORRAX_NGPU=8` → `LORRAX_NGPU=4` in comment at line 30 | 1 |
| 1 (bug) | `docs/ENVIRONMENT_COMPREHENSIVE.md` | Change `LORRAX_NGPU=8` → `LORRAX_NGPU=4` at line 322 | 1 |
| 2 (stale) | `config/README.md` | Replace `platform` + `TF_GPU_ALLOCATOR` rows with `cuda_async` in the env-var table | 2 |
| 2 (stale) | `docs/ENVIRONMENT_COMPREHENSIVE.md` | Replace allocator references: §3.1 table (2 rows), §3.2 title, §7 bare-venv, §8.3 fix | ~8 |
| 2 (stale) | `src/ffi/PORTING.md` | Replace allocator 2-line block with `cuda_async` | 2 |
| 3 (stale) | `docs/ENVIRONMENT_COMPREHENSIVE.md` | Fix JAX pin: 3 occurrences of `jax[cuda13]>=0.9.0` → `jax>=0.5` | 3 |
| 3 (stale) | `docs/ENVIRONMENT_COMPREHENSIVE.md` | §6.1: replace `_maybe_init_jax_distributed` with `runtime.init_jax_distributed` | 2 |
| 3 (stale) | `src/ffi/PORTING.md` | Update CMake autodetect table: `_MPI_INCLUDE_DIR` → `_MPI_INCLUDE_DIR_CT`, add pointer to `config/mpi_stacks/` | 3 |
| 4 (cleanup) | `docs/index.md` | DELETE (orphaned, two stale paths, no incoming links) | -58 |
| 4 (cleanup) | `docs/CODEBASE_COMPREHENSIVE.md` | Drop `AGENT_TODO.md` from Next Steps footer | 1 |
| 4 (cleanup) | `docs/ENVIRONMENT_COMPREHENSIVE.md` | Drop `AGENT_TODO.md` from Next Steps footer | 1 |
| 5 (gap) | `AGENTS.md` | Add ENVIRONMENT_COMPREHENSIVE, SYMMETRY_COMPREHENSIVE, ZETA_V_Q_ALGORITHMS to Key documentation table | 3 |
| 5 (gap) | `README.md` | Add SYMMETRY_COMPREHENSIVE, ZETA_V_Q_ALGORITHMS to doc-list | 2 |
| 5 (gap) | `AGENTS.md` | Add one-line note that CI exists (GitHub Actions) | 1 |
