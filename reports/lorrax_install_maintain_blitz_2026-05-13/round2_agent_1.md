# Round-2 reconciliation — Agent 1 (build / FFI slice)

I read `agent_2.md` (container/MPI), `agent_3.md` (runtime/env), and
`agent_4.md` (docs/cohsex.in/CI). Below is the cross-slice convergence,
the genuine disagreements, my own slice-update marks, the top-5 ranked
blitzes, and the open questions only the author can resolve.

---

## A. Cross-slice convergence

Defects flagged by me **and** at least one other agent. Strongest tag
shown; tagging disagreements noted. "Fix-agree?" = whether the proposed
remediations point in the same direction.

### A1. `pyproject.toml` has no `[build-system]` → `pip install -e .` does **not** build the FFI

- Flagged by **Agent 1 D#2** [FRAGILE / LOC-COST] and **Agent 4 D-36** [FRAGILE].
- Strongest tag: **FRAGILE** (both agents agree). Add LOC-COST because the
  `dependency-groups.build` table is dead infrastructure.
- Fix-agree? **Yes.** Agent 1 P1 and Agent 4 B-5 both prescribe wiring
  `scikit-build-core` as the actual backend. Identical landing zone.

### A2. `jax[cuda13]>=0.9.0` requirement vs container's CUDA 12 / JAX 0.5.x

- **Agent 1 D#1** [FRAGILE / COMPAT], **Agent 4 D-13 + D-14** [FRAGILE], echoed
  by **Agent 3 §6 Q (no direct number)** as "what JAX actually runs?".
- Strongest tag: **FRAGILE** (Agent 4) + **COMPAT** (Agent 1, on the assumption
  a porter actually triggers `pip install`). Use both.
- Fix-agree? **Direction yes**, mechanism diverges. Agent 1 wants the dep
  loosened/split into `cuda12`/`cuda13` extras; Agent 4 wants the
  fact-of-the-matter ("which JAX runs in production?") nailed down before
  changing anything (Q-4). The right order is Agent 4's: answer the question,
  then act.

### A3. Hardcoded container mount paths `/lorrax_{nvhpc,phdf5,slate}`

- **Agent 1 D#7 + D#13** [FRAGILE / LOC-COST / COMPAT] (RPATH bake-in is mine),
  **Agent 2 D3** [LOC-COST].
- Strongest tag: **FRAGILE** (because RPATH ships into the .so → silently
  fails on a different bind layout). Agent 2 only carries LOC-COST; bump it.
- Fix-agree? **Yes.** Agent 1 P5 and Agent 2 B5 both want a single set of
  `LORRAX_CONTAINER_*_PATH` knobs read by both CMake and the modulefile.

### A4. MPI-type knob split across `run_shifter.sh` and `0.1.0.lua`

- **Agent 1 §2.3 + D#30** [FRAGILE / LOC-COST gap in PORTING.md], **Agent 2 D2 +
  D20** [COMPAT + LOC-COST/FRAGILE], **Agent 3 [COMPAT-1]**.
- Strongest tag: **COMPAT** (wrong value silently gives each rank its own
  `MPI_COMM_WORLD`, per Agent 3's reading of the SLURM MPI guide).
- Fix-agree? **Yes.** Agent 1 P5 and Agent 2 B1 both prescribe a single
  `LORRAX_MPI_TYPE` source-of-truth read by both the build wrapper and the
  modulefile. Agent 3 adds a *runtime* assert
  (`jax.process_count() == proc_count`) that catches the same wrong-value
  bug from the other end. Combine.

### A5. `select_gpu.sh` reads `SLURM_LOCALID` and silently coalesces ranks on GPU 0

- **Agent 1 D#22** [COMPAT / FRAGILE], **Agent 2 D7** [COMPAT / FRAGILE].
- Tagging disagreement: **Agent 3 [LOC-COST-5]** explicitly says
  "`select_gpu.sh` is portable" — i.e. *no* fragility tag. Agent 3's read is
  too generous: on a PBS cluster (Polaris), `SLURM_LOCALID` doesn't exist,
  the `:-0` fallback fires, all ranks pile onto GPU 0, no error. Agents 1 and
  2 are right; tag stays **COMPAT + FRAGILE**.
- Fix-agree? **Yes** (Agent 1 P3 strict mode would surface it; Agent 2 B4
  proposes alternative MPI/GPU bind strategies; Agent 3 doesn't propose a
  fix because it doesn't see the defect).

### A6. `lxrun` hardcodes `-N 1` — base modulefile is single-node only

- **Agent 2 D1** [COMPAT / FRAGILE], **Agent 4 D-8** [FRAGILE].
- Agent 4 sharpens: `ENVIRONMENT_COMPREHENSIVE.md:322` actually *documents*
  `LORRAX_NNODES=2 LORRAX_NGPU=8 lxrun ...` as the multi-node entry point —
  but the base lxrun never reads `LORRAX_NNODES`. So this is also a
  **doc-vs-code drift** [FRAGILE]. Agent 2 has the code defect; Agent 4 has
  the doc defect; same bug.
- Fix-agree? **Yes**, two paths: (a) wire `LORRAX_NNODES` into base lxrun
  (Agent 2 B10 option a); (b) document the gap (Agent 2 B10 option b /
  Agent 4 B-7). Agent 4 prefers documentation, Agent 2 prefers wiring.
  My read: option (a) is the right move because the doc *already* claims
  this works.

### A7. `in_container.sh` + `select_gpu.sh` exist only to undo Shifter quirks

- **Agent 1 D#24** [LOC-COST], **Agent 2 D6** [LOC-COST / COMPAT],
  **Agent 3 [LOC-COST-5]**.
- All three agree this is NERSC-coping. Agent 3 partially defends
  `select_gpu.sh` (see A5 — disagreement noted there). For
  `in_container.sh`, all three concur it's pure dead weight off Shifter.
- Strongest tag: **LOC-COST**, plus **COMPAT** because copying it verbatim
  to Apptainer adds a confusing extra fork.
- Fix-agree? **Direction yes**, Agent 2 B4 is the most concrete (drop the
  scripts in favour of `srun --gpus-per-task=1` + `--env=`); Agent 1 was
  silent on a fix. Risk: Agent 2 acknowledges
  `--gpus-per-task=1` historically broke JAX topology — see Q in §E.

### A8. `/opt/hpcx/ompi/lib` silent OpenMPI fallback in CMakeLists

- **Agent 1 §2.3 + D#10** [FRAGILE], **Agent 2 D10** [FRAGILE].
- Strongest tag: **FRAGILE**. Both note that `build.sh:35-48`'s guard is
  bypassable (`LORRAX_FFI_ALLOW_DEFAULT_MPI=1`) and only protects the
  `build.sh` entry point — anyone running `cmake … && ninja` by hand
  walks straight into a `DT_NEEDED libmpi.so.40` .so.
- Fix-agree? **Yes.** Agent 1 P3 (strict mode promotes the WARNING to
  FATAL_ERROR); Agent 2 doesn't propose a fix but supports the same
  diagnosis. Should be merged into P3.

### A9. Stage scripts (`stage_*.sh`) exist almost entirely because Shifter forbids `--volume` from `/opt/cray`

- **Agent 1 D#25** [LOC-COST], **Agent 2 D8** [LOC-COST].
- Strongest tag: **LOC-COST**. Agreement is clean: ~250 LoC of NERSC-coping;
  Apptainer/Enroot make 3 of 5 scripts collapse into "point CMake at
  the installed lib".
- Fix-agree? **Yes** in spirit, but neither agent proposes deleting the
  scripts — both want PORTING.md to *flag* them as NERSC-only so a porter
  knows to skip steps.

### A10. JAX coordinator port `12355` hard-coded, no collision handling

- **Agent 2 D15** [FRAGILE], **Agent 3 [FRAGILE-3]**.
- Strongest tag: **FRAGILE**. Agent 2 worries about two LORRAX shells on
  the same login node; Agent 3 about two `lxshell`-then-manually-launched
  processes. Same bug.
- Fix-agree? **Yes** — both want collision handling (random port from
  jobid hash, or jitter). Agent 3 also folds this into a doctor-command
  check (B7).

### A11. `MPICH_GPU_SUPPORT_ENABLED=1` + `libmpi_gtl_cuda.so.0` LD_PRELOAD set unconditionally

- **Agent 2 D4 + D5** [COMPAT / FRAGILE], **Agent 3 §3 NERSC-isms** flags
  the same. Agent 1 noted in §3 NERSC-isms but didn't catalog as a defect.
- Strongest tag: **COMPAT** (Cray-only; OpenMPI/UCX needs different env
  vars; on non-Cray the preload file may not exist, leading to dl
  warnings).
- Fix-agree? **Yes.** Agent 2 B8 (guard preload + PMI knobs by
  `LORRAX_MPI_TYPE`) is the cleanest version.

### A12. Image tag `nvcr.io/nvidia/jax:25.04-py3` not digest-pinned

- **Agent 2 D11** [FRAGILE], echoed indirectly by **Agent 4 D-13/D-14** as
  "the (image, NVHPC, NCCL) tuple is an implicit invariant".
- Strongest tag: **FRAGILE**.
- Fix-agree? **Yes** in principle (Agent 2 B6: digest-pin in
  `site_config.sh`); Agent 2 flags Q1 as the gating uncertainty (does
  Shifter support digest refs?).

### A13. Env-var → cohsex.in migration is incomplete; doc never tracked the migration

- **Agent 3 [LOC-COST-1, -2, -3]** (5 SC + 3 ISDF planner + 5 V_q vars
  still in env), **Agent 4 D-3** (the 11 *migrated* keys are not
  documented).
- Strongest tag: **LOC-COST** (both); Agent 4 adds **FRAGILE** for the
  doc-drift dimension. Use both.
- Fix-agree? **Complementary, not conflicting.** Agent 3 B2 finishes the
  migration; Agent 4 B-1 + B-2 generate the doc from the parser. The
  combined fix is "migrate, then auto-generate" — both halves needed.

### A14. No FFI / Shifter / dist-init smoke test in CI; no CI at all

- **Agent 1 P4** (out-of-container build smoke test) + **Agent 4 D-26 +
  D-27 + D-28 + D-29 + B-4**.
- Strongest tag: **FRAGILE** (vendor-bump silent breakage class). Agent 4
  is more comprehensive (FFI + Shifter + dist init); Agent 1 specifically
  wants to exercise the autodetection ladder on a non-NERSC host.
- Fix-agree? **Yes.** They are different targets of the same CI investment.
  Pair Agent 1's "configure-only smoke on a vanilla Linux box" with
  Agent 4's "weekly Perlmutter Shifter smoke" — together they catch the
  full failure surface.

### A15. Test/bench files inline ~15 copies of distributed-init boilerplate

- **Agent 3 [FRAGILE-9]**, partially **Agent 4 D-29** (the test directory
  layout / collection question).
- Strongest tag: **FRAGILE** (drift risk — `runtime/__init__.py` evolves,
  the 15 copies don't).
- Fix-agree? **Yes** (Agent 3 B5 has the concrete migration plan).

### A16. `lorrax_agent` sandbox overlay vs upstream multi-user concurrency

- **Agent 3 [LOC-COST-4]** (645 lines of pool coordination — necessary?),
  **Agent 4 D-33 + D-35** (sandbox-vs-upstream split, no canonical-source
  rule).
- Strongest tag: **LOC-COST**. Agent 3 questions whether the multi-agent
  pattern should exist at all; Agent 4 questions whether the overlay
  should graduate upstream. Different framings, same artifact.
- Fix-agree? **Diverge**: Agent 3 leans toward "each agent gets its own
  salloc, kill the pool"; Agent 4 leans toward either upstreaming
  `lxstatus` or documenting the gap. See §E for the question.

---

## B. Disagreements

### B1. `select_gpu.sh` portability (resolved by §A5)

Agent 3 [LOC-COST-5] says the script is portable; Agents 1 (D#22) and 2
(D7) say it silently breaks on PBS / non-SLURM. **My judgment: Agents 1
and 2 are right.** Evidence: `select_gpu.sh:12` is literally
`export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}` — the `:-0` fallback is
the bug. Resolution: a one-line PBS fallback (`PMI_LOCAL_RANK` or
`PMIX_LOCAL_RANK`) takes care of it without removing the script. What
*would* resolve it definitively: a PBS test (Polaris) showing all ranks
land on GPU 0.

### B2. Drop `select_gpu.sh` + `in_container.sh` vs keep them

Agent 2 B4 wants both gone (replace with `srun --gpus-per-task=1` +
`--env=`). Agent 3 [LOC-COST-5] would keep `select_gpu.sh` and concedes
`in_container.sh` is dead off Shifter. Agent 1 (this slice) is silent.
**My judgment: split the difference.** Delete `in_container.sh` if/when
the modulefile gains a non-Shifter branch (no defenders); keep
`select_gpu.sh` and patch it for PBS. Agent 2's `--gpus-per-task=1`
proposal hangs on the JAX-tolerates-it question (Agent 2 Q2 in §E) which
no one verified — don't gamble on it.

### B3. Tagging disparity: `jax[cuda13]>=0.9.0` mismatch

Agent 1 D#1 tags **FRAGILE + COMPAT**; Agent 4 D-13/D-14 tags **FRAGILE**
only. Both flag the same code. **Tag should be FRAGILE + COMPAT** —
Agent 1's COMPAT axis is right because a porter who runs
`pip install -e .` on any cluster *will* trip it; that's the
"non-NERSC-as-written" axis.

### B4. Should `lorrax_agent` graduate upstream?

Agent 3 Q6 leans "the overlay shouldn't exist; each agent gets its own
salloc". Agent 4 Q-7 lays out the trade and leans "wait for a second
user". Agent 4 B-7 hedges with options (a) and (b). **No agent has
definitive evidence.** This is a sandbox-design call I'm not equipped to
adjudicate — it requires data on whether pool bugs have actually cost the
author hours. Flag for the user (§E).

### B5. PORTING.md quality

Agent 4 §2 calls PORTING.md "Tight, well-written. Out of date in places."
Agent 1 (this slice) §4.4 catalogs five distinct gaps (D#30–34) and
treats it as primary critique target. Not a true disagreement —
Agent 4's "tight" judgment is at the prose level; Agent 1's defects are
content-level. The prose IS tight; the content has gaps both about (a)
the `run_shifter.sh` build wrapper and (b) the JAX/SLATE version
contract. Agent 4 doesn't dispute these; just didn't enumerate them.

### B6. `XLA_PYTHON_CLIENT_ALLOCATOR=platform` vs `TF_GPU_ALLOCATOR=cuda_malloc_async` conflict

Agent 3 [FRAGILE-2] flags this and proposes B8 to resolve. Nobody else
caught it. **No disagreement, just non-overlap.** Agent 3 is right; this
is a real bug. The fix waits on a `lxrun python -c "import jax; jax.devices()"`
log scrape (Agent 3 Q1). Should be folded into the doctor command (B1
in Agent 3's blitzes).

---

## C. Slice-specific updates to my own draft

For each Agent-1 §4 defect / §5 blitz where round 2 changed my view or
added evidence. (Items not listed: stand as in `agent_1.md`.)

### Defects

- **D#1 (jax[cuda13]>=0.9.0 vs container)** — **CONFIRMED + SHARPENED.**
  Agent 4 D-13/D-14 backs it. Sharpened: Agent 4's Q-4 reframes this as
  "we don't actually know which JAX runs in the container". The defect
  is more pointed than I wrote — the dep declaration may be aspirational
  or stale, but the *runtime* JAX is determined by an
  `LORRAX_SITE_PACKAGES` bind-mount nobody documented. Until that's
  resolved, even loosening the pin (my P1) is premature.
- **D#2 (no `[build-system]`)** — **CONFIRMED.** Agent 4 D-36 echoes
  precisely.
- **D#7 + D#13 (hardcoded `/lorrax_*` and RPATH)** — **CONFIRMED +
  SHARPENED.** Agent 2 D3 lists every cross-file occurrence I missed
  (`phdf5/cpp/ctx.h` defaults; every `stage_*.sh`; the modulefile L154-
  156 + L195-197). Sharper version: the load-bearing strings appear in
  ≥6 files, not 4. P5 needs to widen its scope.
- **D#10 (`/opt/hpcx/ompi/lib` silent fallback)** — **CONFIRMED** by
  Agent 2 D10. No sharpening needed; the diagnosis matches.
- **D#16 (hardcoded SLATE install at `~jackm/software/slate/install`)**
  — **CONFIRMED** by Agent 2's NERSC-isms table row 9 indirectly
  (subpath baked in `site_config.sh:87`). No new sharpening.
- **D#22 (select_gpu.sh + SLURM_LOCALID)** — **CONFIRMED + SHARPENED.**
  Agent 2 D7 names Polaris as the concrete failing cluster (PBS, no
  `SLURM_LOCALID`). Sharper than my "PMIx implementations sometimes don't
  set it". Agent 3's contrary reading is wrong (see §B1).
- **D#24 (in_container.sh + select_gpu.sh exist for Shifter quirks)** —
  **CONFIRMED + REVISED on scope.** Agent 3 [LOC-COST-5] is right that
  `select_gpu.sh` is *partly* generic-SLURM and would be retained; only
  `in_container.sh` is pure Shifter dead weight. Revise the LoC-cost
  estimate from "~30 LoC dead weight" to "~5 LoC dead weight + 5 LoC
  PBS-fallback patch needed for the rest".
- **D#25 (stage scripts as bind-mount workarounds)** — **CONFIRMED** by
  Agent 2 D8. Quantitatively: ~250 LoC matches.
- **D#30 (PORTING.md doesn't mention `run_shifter.sh`)** — **CONFIRMED**
  in spirit by Agent 4 D-7/D-8 (general doc-vs-code drift class). No
  direct quote, but the pattern is identical.

### Blitzes

- **P1 (wire pip install → CMake)** — **CONFIRMED** (Agent 4 B-5 is the
  same blitz). **REVISED order**: Agent 4's Q-4 (which JAX actually
  runs?) must be answered first, otherwise this blitz changes the JAX
  install path under the user without us knowing what was running before.
  New ordering: answer Q-4 → P1.
- **P3 (strict mode)** — **CONFIRMED + SHARPENED** by Agent 2 D10
  (HPC-X fallback is exactly what strict mode would surface). Add to P3:
  the strict mode should also reject the `LORRAX_FFI_ALLOW_DEFAULT_MPI=1`
  override (or at least require an explicit second flag).
- **P4 (out-of-container smoke test)** — **CONFIRMED** by Agent 4 B-4
  (the FFI/Shifter/dist-init smoke). The two are complementary, not
  redundant: P4 is "configure ladder on a non-NERSC box"; B-4 is "FFI
  symbol smoke on a real Perlmutter Shifter". Both should land.
- **P5 (single MPI-stack contract file)** — **CONFIRMED + SHARPENED**
  by Agent 2 B1 + Agent 3 B7(d). Sharpened version: in addition to my
  build-time + runtime config unification, add Agent 3's runtime assert
  (`jax.process_count() == proc_count`). That assert *catches* exactly
  the silent-singleton-init failure mode my proposal *prevents*. They
  belong together.
- **P6 (numerically-sorted NVHPC autodetect)** — unchanged, no other
  agent flagged or contested.

---

## D. Top-5 cross-cutting blitz proposals

Across ~24 ranked items in the four drafts, these are what I would
execute first, in this order. No "depends on which axis" — committed.

### #1 — Wire `[build-system] = scikit-build-core` into `pyproject.toml`

- **Proposed by:** Agent 1 P1, Agent 4 B-5.
- **Acceptance criteria:** `pip install -e .` (run inside the existing
  Shifter container with NVHPC bind-mounts present) builds
  `liblorrax_ffi.so` end-to-end, lands it in
  `lorrax/ffi/common/cpp/` site-packages, and a fresh
  `python -c "import lorrax.ffi.common.ffi_loader as L; L.get_lib()"`
  returns the library handle without falling through to the legacy glob.
  CI hook: `tests/build_smoke/test_pip_install.py` asserts the .so
  exists and `dlopen`-loads.
- **Closes:** Agent 1 D#2 most directly (and exposes D#1 / Agent 4 D-13
  at the right moment).
- **Dependencies on other top-5 items:** None; this is the foundation.
  But see Open Question E1: it must answered first or the JAX dep gets
  re-resolved silently.
- **Real Perlmutter run required?** Yes for the integration test (needs
  the Shifter container with NVHPC bind-mount). The CMake-stub variant
  (no real cuSolverMp) can run desk-side.

### #2 — Schema-validate `cohsex.in` AND auto-generate `COHSEX_INPUT.md` from the parser

- **Proposed by:** Agent 4 B-1 + B-2 (treat as one work unit).
- **Acceptance criteria:**
  (a) `LorraxConfig.from_input_file` raises (or warns under a knob) on
  any unknown key not in an explicit deprecated-keys list;
  (b) a new `tools/gen_cohsex_input_md.py` walks the parser schema and
  writes `lorrax_C/docs/COHSEX_INPUT.md` (the file currently lives in
  the sandbox);
  (c) a CI/pytest job that asserts
  `set(parser._DEFAULTS) == set(documented_keys) ∪ deprecated_keys`,
  fails on drift.
- **Closes:** Agent 4 D-1 / D-3 / D-4 / D-16 in one shot; lays the rails
  for Agent 3 [LOC-COST-1..3] (the remaining env vars to migrate, since
  each new key now auto-documents itself).
- **Dependencies on other top-5 items:** None (parallelizable with #1).
- **Real Perlmutter run required?** No. Pure desk-side work + pytest.

### #3 — Unify the MPI-type knob into one source-of-truth + add a runtime-side mismatch assert

- **Proposed by:** Agent 1 P5, Agent 2 B1, Agent 3 B7(d).
- **Acceptance criteria:**
  (a) `config/mpi_stacks/{cray_mpich,openmpi}.{cmake,sh}` files (one per
  stack) hold the `(LORRAX_PHDF5_MPI_STACK, SHIFTER_MODULES,
  MPI_LIB_DIR_CT, MPI_INCLUDE_DIR_CT, MPI_TYPE_DEFAULT)` quintuple;
  (b) `run_shifter.sh`, `build.sh`, and `0.1.0.lua` all source/read
  this single file (no in-place case statement);
  (c) `runtime/__init__.py` post-init asserts
  `jax.process_count() == int(os.environ.get('SLURM_NTASKS', '1'))` and
  raises a clear error if not (catches the wrong-`--mpi=` singleton-init
  case);
  (d) round-trip pytest: for each stack file, source as shell + parse as
  CMake, assert the five values match.
- **Closes:** Agent 2 D2 / D20, Agent 3 [COMPAT-1] (the two halves of
  the same cross-file invariant), partially Agent 1 D#30.
- **Dependencies on other top-5 items:** Light — parallelizable with #1.
  Some overlap with #5 (the test infrastructure).
- **Real Perlmutter run required?** Partial — the round-trip test is
  desk-side, but the runtime assert needs a real srun launch to verify
  with `--mpi=` purposely set wrong (or skip-marked in CI).

### #4 — `lxrun --dry-run` (a.k.a. "`lorrax doctor`")

- **Proposed by:** Agent 2 B2 (`lxrun --dry-run`), Agent 3 B1
  (`lorrax doctor`). Same artifact, two framings.
- **Acceptance criteria:** A new shell function (or `bin/lorrax-doctor`)
  prints (i) the materialized `srun … shifter … --env=… --bind=…
  in_container.sh "$@"` argv that lxrun would execute, and (ii) a
  green/yellow/red audit of every env-var contract from Agent 3's §2c
  table, plus the LD_LIBRARY_PATH 6-segment chain check, plus the
  `XLA_PYTHON_CLIENT_ALLOCATOR` vs `TF_GPU_ALLOCATOR` conflict check
  (Agent 3 [FRAGILE-2]). Exits 0 only if all green.
- **Closes:** Agent 2 D3 / D4 / D16 (debuggability), Agent 3
  [FRAGILE-1 / -2 / -5 / -7 / -8] + [COMPAT-1] (audit surface). Single
  largest "porter wakes up confused, runs this, gets answers" win.
- **Dependencies on other top-5 items:** Mild — benefits from #3 having
  landed (so the audit can read the unified MPI knob).
- **Real Perlmutter run required?** Construction + dry-run mode is
  desk-side; the green-light test requires a real `module load` + Shifter.

### #5 — FFI / Shifter / dist-init smoke test, run on every (or weekly) Perlmutter CI

- **Proposed by:** Agent 1 P4 (out-of-container configure smoke),
  Agent 4 B-4 (Perlmutter FFI symbol smoke + dist-init smoke).
- **Acceptance criteria:** A `tests/integration/test_smoke_ffi.py` (or
  bash equivalent) that, when run inside the Shifter container with the
  built FFI: (a) `import _lorrax_ffi`; (b) runs a 64×64 SLATE Cholesky;
  (c) calls `jax.distributed.initialize()` and asserts
  `jax.process_count() == 1` for the single-process case; (d) on
  failure, prints a `ldd liblorrax_ffi.so | grep "not found"`. PLUS a
  desk-side configure-only smoke (Agent 1 P4) that runs the
  `CMakeLists.txt` autodetect ladder on a non-NERSC Linux box with
  CUDA-stub libs to validate the override paths
  (`-DNVHPC_ROOT=`, `-DHDF5_ROOT=`, `-DLORRAX_SLATE_INSTALL_DIR=`,
  `-DLORRAX_MPI_INCLUDE_DIR=`, `-DLORRAX_MPICH_LIB_DIR=`).
- **Closes:** Agent 4 D-26 / D-27 / D-28 / D-29 (the no-CI cluster);
  Agent 1 D#1 (would have caught the JAX wrong-CUDA pin); Agent 1 D#15
  (would have caught a broken libmpi SONAME shim).
- **Dependencies on other top-5 items:** Soft dependency on #1 (after
  scikit-build-core lands, the smoke can rely on `pip install -e .` to
  produce the .so consistently).
- **Real Perlmutter run required?** Yes for the FFI / Shifter / dist-init
  test (scheduled weekly is enough). The configure-only desk-side smoke
  can run on any GitHub-hosted Linux runner.

---

## E. Open questions for the user

These are gates. Without an answer, the corresponding blitz can't
land cleanly.

### E1. Which JAX *actually* executes inside the container at production time?

`pyproject.toml:14` declares `jax[cuda13]>=0.9.0`. PORTING.md says "JAX
0.5+, container `nvcr.io/nvidia/jax:25.04-py3`" (which ships JAX ≈0.5
with CUDA 12). `ENVIRONMENT_COMPREHENSIVE.md:86` hints that
`LORRAX_SITE_PACKAGES` is bind-mounted from the host into the container —
**does the host site-packages dir override the container's JAX, and if
so what version actually runs?** A 30-second
`lxrun python -c "import jax; print(jax.__version__, jax.__file__)"`
resolves it. **Gates blitz #1 (and Agent 1 D#1, Agent 4 D-13/D-14, Agent 4 Q-4).**

### E2. Has LORRAX been tested on any non-NERSC cluster, ever?

PORTING.md reads as if it's been validated; no agent found evidence
that it has. If yes (a private workstation? a previous group cluster?),
a one-line callout per known-tested cluster turns most §A COMPAT items
into "documented". If no, PORTING.md is *aspirational* and a second
user is a research project. **Gates blitz #5 acceptance criteria
(do we believe the desk-side configure smoke is sufficient or do we
need a real second-cluster integration?).** Echoes Agent 4 Q-6.

### E3. Which GPU allocator is actually active in production —
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` or `TF_GPU_ALLOCATOR=cuda_malloc_async`?

`0.1.0.lua:130-131` sets both; per JAX docs they're mutually exclusive.
The intent comment matches `cuda_malloc_async`, but JAX-prefixed names
have precedence in current XLA — so the *platform* allocator may be
silently in charge, with cuSOLVERMp/NCCL VRAM-sharing working by
accident. A startup-log scrape (look for the JAX line that records the
chosen allocator) resolves it. **Gates the resolution of Agent 3
[FRAGILE-2] / B8.**

### E4. Should `lorrax_agent` graduate upstream now or wait?

Agent 3 questions whether the 645-line pool layer should exist at all
(Q6); Agent 4 Q-7 frames it as "wait for a second user vs. preempt".
Neither has data on whether pool bugs have actually cost hours. **Gates
blitz #6-style work** (Agent 4 B-7) on `lxstatus` upstreaming and
informs the longer-term sandbox-vs-upstream split (Agent 4 D-32 / D-33 /
D-34 / D-35).

### E5. Does Shifter on Perlmutter actually accept digest-pinned image refs (`...:25.04-py3@sha256:...`)?

Image-pinning blitz (Agent 2 B6) hangs on this. `shifterimg lookup` and
`shifter --image=` historically take a tag, not a digest. **Gates B6.**
Echoes Agent 2 Q1.

### E6. Does `srun --gpus-per-task=1` work with current JAX `local_device_ids` semantics?

Agent 2's B4 (drop `select_gpu.sh` and `in_container.sh` in favour of
`--gpus-per-task=1`) hangs on this. The `run_shifter.sh:163-167` comment
says it broke JAX topology sync in the past, but the comment is
undated. Without verification, B4 stays out of the top 5. **Gates B4.**
Echoes Agent 2 Q2.

### E7. Are the cohsex.in keys in §4.1 of Agent 4's draft (`use_chunked_isdf`,
`sigma_debug_split_contrib`, `write_no_head_vw`) actually deprecated, or were
they meant to land in `_DEFAULTS` and never did?

Agent 4 Q-8 raises this. The schema-generator blitz (#2) needs the
answer to build the right deprecated-keys allow-list. If they were
intended → fix the parser, not the doc. **Gates blitz #2.**

### E8. Is the `tests/test_gw_jax_regression.py` `ISDF_COHSEX_TEST_PLATFORM=cpu` mode a meaningful safety net?

Agent 4 Q-2 raises this. If the CPU branch covers <30% of the
production code paths, the smoke test (#5) needs to be more ambitious
than "weekly Perlmutter run". A `pytest --co` + coverage report on the
existing fixture answers it. **Affects blitz #5 scoping.**

---

`Agent 1 round 2 done — see round2_agent_1.md`
