# Consensus — LORRAX install/maintain blitz (2026-05-13)

Synthesis of 4 round-1 audits (`agent_{1..4}.md`) + 4 round-2 reconciliations
(`round2_agent_{1..4}.md`). Each agent owned one slice in round 1 (build/FFI,
MPI/Shifter, runtime/env, docs/synthesis) and produced a committed top-5
ranking in round 2. This document is the cross-slice consensus.

---

## TL;DR

Five blitz items rank-ordered by all four agents. Two open questions block
or shape three of the five. The blitz is **mostly desk-doable** — only two
items need a real Perlmutter session, and one (Q1: which JAX runs in
production?) is a 30-second one-liner that unblocks the rest.

---

## Author updates (post round-2)

Two decisions from the author that simplify and re-prioritize the plan:

### Decision 1 — Allocator question (was Q2) is resolved: switch to `cuda_async`

The current dual setting in `config/modulefiles/lorrax/0.1.0.lua:130-131`
(`XLA_PYTHON_CLIENT_ALLOCATOR=platform` AND
`TF_GPU_ALLOCATOR=cuda_malloc_async`) is wrong on two counts:

- Per JAX docs, the JAX-prefixed name wins; `platform` is "very slow,
  not recommended for general use" (eager `cudaMalloc`/`cudaFree` per
  allocation). LORRAX has been running on the debug-grade allocator.
- The `TF_GPU_ALLOCATOR` legacy name only matters in pre-XLA TF; with
  current JAX it has no effect.

**Fix**: replace lines 130-131 with a single
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`. This enables the CUDA
stream-ordered memory pool (`cudaMallocAsync`), which:

- grows on demand (no 75%-upfront grab);
- **releases memory back to the device pool on free**, unlike BFC;
- shares the device pool with any other library that allocates via
  `cudaMallocAsync` — including cuSolverMp workspaces, if allocated
  that way.

This is a sub-1-hour Day-0 change. **Promote to blitz item #0.**

The deeper fix for FFI ↔ JAX memory coexistence (relevant because
cuSolverMp's per-q workspace is ≤ ¼ of HBM and JAX's peak is elsewhere
in the pipeline): cuSOLVER does not internally `cudaMalloc` — *the
caller allocates the workspace*. The clean pattern is:

```python
nbytes = cusolvermp_xeigh_bufferSize(handle, ...)
workspace = jnp.empty(nbytes // dtype.itemsize, dtype)  # JAX-pool
result = jax.ffi.ffi_call(...)(matrix, workspace, ...)
del workspace  # back to the pool, eligible for the next q
```

If `src/ffi/cusolvermp/eigh.py` / `batched.py` already query
`bufferSize` and allocate the workspace JAX-side, this is already
optimal — only Decision 1 (the allocator setting) needs to change. If
the workspace is allocated C++-side via `cudaMalloc` (outside the
pool), migrating it to a JAX-allocated buffer becomes **blitz item
#6** (small, but high leverage: it removes the dual-pool issue
entirely). **Action**: read those two files to confirm which path
they're on. (Not blocking the main 5; ~2-hour blitz once verified.)

NCCL buffer / startup recycling for the per-q cuSolverMp loop: per
commit `24bfa5f`, LORRAX already caches the cuSolverMp NCCL comm by
`(p, q, layout)` mesh signature, so reuse across the q-loop is
already happening. `ncclCommInitRank` (100ms–1s) only fires once per
mesh shape. The only marginal squeeze available: add the cuSolverMp
comm to `runtime/nccl_warmup` so the first per-q call doesn't pay
NCCL channel-init at the cost of the user-visible timer.

### Decision 2 — Cray HPC stack is the committed common denominator; Shifter is the canonical container *today* but Apptainer (Frontier, Polaris, etc.) is the *aspirational* second target; non-NVIDIA (Frontier MI250X) is acknowledged as a future refactor

Refined position from the author:

- **Cray MPICH stays the canonical MPI**: it's common to most current and
  near-future scientific clusters (Perlmutter, Frontier, LUMI, several
  university Slingshot installs). The `--mpi=cray_shasta`,
  `MPICH_GPU_SUPPORT_ENABLED=1`, `libmpi_gtl_cuda` machinery is
  load-bearing infrastructure, not NERSC-coping. Keep it.
- **Shifter is the canonical container *today* on Perlmutter**, but
  Frontier/Polaris/most-other-Cray-sites use Apptainer. An Apptainer
  companion image + modulefile path *is* a maintained goal (long-term).
  It's still not a 1-day blitz — that's a 1-2 week project once the
  groundwork is in place. **Configurable container-mount paths
  (`LORRAX_CONTAINER_{NVHPC,PHDF5,SLATE}_PATH`) are back in the top
  tier of work as future-proofing**, since they're the prerequisite for
  the Apptainer port to be mechanical when scheduled.
- **AMD MI250X / ROCm (Frontier GPUs) is an acknowledged future
  refactor**, out of scope for this blitz. cuSOLVERMp → hipsolverMp or
  SLATE-only path, NCCL → RCCL, cuBLASMp → rocBLASmp, etc. The
  scikit-build-core / `[build-system]` work (#1) should anticipate
  this by keeping the FFI build conditional on a backend toggle
  (`LORRAX_GPU_BACKEND=cuda|rocm`) even if only `cuda` is wired today.
- The ~250 LoC of `stage_*.sh` scripts are **infrastructure**, not
  LOC-cost defects. Re-tag from `[LOC-COST]` to "Cray-canonical, well
  documented, parametrize for future Apptainer port."
- `in_container.sh`, `select_gpu.sh` — **stay**, with `select_gpu.sh`
  still getting the 5-line PBS/PMIx fallback chain.
- PORTING.md re-frames as **"Perlmutter deployment guide + Cray-HPC
  porting reference (Apptainer + Frontier are work-in-progress
  targets)"**.
- Blitz item #3 (unify MPI-stack contract) still shrinks vs. the
  original framing: the `(LORRAX_MPI_TYPE, SHIFTER_MODULES, MPI_LIB_DIR,
  MPI_INCLUDE_DIR)` quintuple unifies into a single file, but only one
  MPI stack (`cray_mpich`) needs to be live; alternate stack files are
  stubs for now. Still ships the post-init
  `assert jax.process_count() == SLURM_NTASKS` runtime check.
- Configurable container-mount paths **back in the top tier**: not a
  blocker, but worth doing in the same week — promote to **Blitz #5b**
  (alongside the overlay split). Both are ≤0.5 day each.
- Open Question Q6 (has LORRAX ever run elsewhere?) clarified:
  not yet; goal *is* multi-cluster reach over time.

---

## The five blitz items, committed

Each item lists its component round-1 proposals, scope, acceptance criteria,
dependencies, and whether it's desk-doable or needs a live cluster.

### #1 — Wire `[build-system] = scikit-build-core` in `pyproject.toml`; reconcile the JAX pin

Ranked #1 by Agents 1, 2, 3; #2 by Agent 4 (universal top-2).
Source: A1 P1 + A4 B-5 + A1 #1 + A4 D-13/D-14.

- **What changes.** `pyproject.toml` gets a real `[build-system]` block
  pointing at scikit-build-core; `src/ffi/common/cpp/CMakeLists.txt`
  becomes the (or sub-included from) top-level CMake target.
  `liblorrax_ffi.so` installs to `lorrax/ffi/common/cpp/` site-packages
  on `pip install`. The orphan `dependency-groups.build` table goes
  away. The `jax[cuda13]>=0.9.0` pin loosens to match the JAX *actually*
  bind-mounted into the container.
- **Acceptance.**
  - `pip install -e .` inside the existing Shifter container produces a
    working `liblorrax_ffi.so` end-to-end.
  - `python -c "from ffi.common.ffi_loader import get_lib; get_lib()"`
    succeeds without `LORRAX_FFI_SO` set.
  - New `tests/build_smoke/test_pip_install.py` asserts the .so exists
    and `dlopen`-loads (CMake-stub variant for CI; real-NVHPC variant
    for Perlmutter).
- **Closes.** A1 #2 (silent no-op pip install), A1 #1 / A4 D-13/D-14
  (JAX pin vs container mismatch), partial A4 D-36.
- **Dependencies.** **Q1** must be answered before this lands (or the JAX
  pin gets re-resolved silently). No code dependencies on items #2-#5.
- **Doable from desk?** Mostly, yes — pyproject/CMake refactor is
  desk-side. **Needs one Shifter session** to validate the install
  actually links and imports.
- **Why first.** Foundational: changes the first thing a porter does
  (`git clone; pip install`) from "silently fails to build the FFI" to
  "fails with a CMake error message that names the missing dep, or
  succeeds." Every other blitz benefits from this being in place.

### #2 — `cohsex.in` schema validation + auto-generate `COHSEX_INPUT.md` from parser + finish the env-var → cohsex.in migration + move the doc into `lorrax_C/`

Ranked #1 by Agent 4; #2 by Agents 1, 2, 3 (universal top-2).
Source: A4 B-1 + A4 B-2 + A3 B2.

- **What changes.** Introduce `src/gw/_cohsex_schema.py` with a single
  `Field(name, type, default, doc)` table covering all 77 parser keys
  plus the 13 env-controlled knobs still in code (A3's LOC-COST-1/2/3:
  5 SC + 3 ISDF planner + 5 V_q). Add `_UNKNOWN_KEY_POLICY` plus an
  alias allow-list for deprecated keys (`output_file`,
  `use_chunked_isdf`, `sigma_debug_split_contrib`, `write_no_head_vw`).
  Add `tools/gen_cohsex_input_md.py` that walks the schema and emits
  `lorrax_C/docs/COHSEX_INPUT.md` (the doc currently lives in the
  sandbox, not the repo). Update `templates/cohsex.in` and the
  regression-test fixture.
- **Acceptance.**
  - `pytest tests/test_cohsex_schema.py` asserts
    `set(parser._DEFAULTS) == set(documented_keys) ∪ deprecated_keys`.
  - `pytest tests/test_cohsex_doc_freshness.py` regenerates the .md and
    fails CI on diff.
  - `read_lorrax_input("templates/cohsex.in")` emits no warnings.
  - No `LORRAX_SC_*` / `LORRAX_V_Q_*` / `ISDF_CHUNK_*` reads in
    `src/gw/` or `src/common/isdf_fitting.py` outside debug-only paths.
- **Closes.** A4 D-1 through D-6, D-12, D-16, D-18, D-30, D-32; A3
  LOC-COST-1/2/3; A3 B2; A3 B4 (ENV_VARS doc surface partly subsumed).
- **Dependencies.** **Q4** answer shapes the deprecated-keys allow-list
  (are those keys typos or intentional auto-derived flags?). **Q5**
  determines whether strict mode is default-error or default-warn.
- **Doable from desk?** **Yes, fully.** No Perlmutter session needed.
- **Why second.** Highest agreement across slices (3 agents ranked it #2,
  the docs-specialist ranked it #1), only fully-desk-doable item in the
  top 3, locks in a maintenance pattern the author already validated
  three times (commits 488e870 / 9fe5fde / 40a4cca).

### #3 — Unify the MPI-stack contract + add post-init `assert size == SLURM_NTASKS` + make `select_gpu.sh` portable across SLURM/PBS/PMIx

Ranked #3 by Agents 1, 2, 4; absorbed by #4 in Agent 3.
Source: A1 P5 + A2 B1 + A2 B8 + A3 B7(d) + the A1/A2/A3 convergence on
`select_gpu.sh` portability (resolved per Section B-1 of round-2 drafts).

- **What changes.** Two parts:
  - **Config side:** create `config/mpi_stacks/cray_mpich.{cmake,sh}` and
    `…/openmpi.{cmake,sh}` files that define the
    `(LORRAX_MPI_TYPE, SHIFTER_MODULES, MPI_LIB_DIR_CT, MPI_INCLUDE_DIR_CT,
    MPI_TYPE_DEFAULT, GTL_PRELOAD)` quintuple. `run_shifter.sh`,
    `build.sh`, and `config/modulefiles/lorrax/0.1.0.lua` all read this
    one source. `LD_PRELOAD libmpi_gtl_cuda.so.0` and
    `MPICH_GPU_SUPPORT_ENABLED=1` (`0.1.0.lua:181-186`) become
    conditional on `LORRAX_MPI_TYPE` starting with `cray`/`mpich`; emit
    `UCX_TLS=...,cuda_copy,gdr_copy,cuda_ipc` + `OMPI_MCA_pml=ucx`
    under `pmix`.
  - **Runtime side:** `init_jax_distributed()` post-init asserts
    `jax.process_count() == int(os.environ.get("SLURM_NTASKS", "1"))`
    (warn first, promote to fatal after one release). Patch
    `select_gpu.sh` from `${SLURM_LOCALID:-0}` to
    `${SLURM_LOCALID:-${PMI_LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}`
    and fail loud if all are unset while `SLURM_NTASKS>1`.
- **Acceptance.**
  - `module load lorrax && env | grep -E '^(MPICH_|LD_PRELOAD|UCX_)'`
    shows Cray-specific vars iff `LORRAX_MPI_TYPE=cray_shasta`.
  - Wrong `LORRAX_MPI_TYPE` produces a clear failure within 1s of
    `init_jax_distributed`, not a silent singleton run.
  - `tests/build_smoke/test_mpi_stack_round_trip.sh`: for each stack
    file, source as shell + parse as CMake, assert the five values match.
- **Closes.** A.4 / A.5 / A.7 from Agent 4's table; A1 #18, A2
  D2/D4/D5/D7/D13/D20, A3 COMPAT-1.
- **Dependencies.** None on items #1, #2.
- **Doable from desk?** Mostly desk; **needs one real `lxrun`** to verify
  the post-init assert fires correctly with `LORRAX_MPI_TYPE` set wrong.
- **Why third.** Every defect this fixes is a *silent* failure
  (wrong-`--mpi=` → singleton init; missing `LORRAX_MPI_TYPE` guard →
  spurious warnings on Apptainer; missing `SLURM_LOCALID` → all ranks
  pile onto GPU 0). Silent failures are the highest-leverage class to
  convert to loud ones.

### #4 — CPU-only smoke-test trio in CI: `import _lorrax_ffi` + mocked `init_jax_distributed()` + cohsex.in parser round-trip; weekly Perlmutter follow-up

Ranked #4-#5 by all 4 agents (universal top-5).
Source: A3 B3 + A4 B-4 + A1 P4 + A1 open Q8.

- **What changes.** New `.github/workflows/ci.yml` (or self-hosted
  equivalent) running on a free GitHub Linux runner. Three tests:
  1. `ldd build/liblorrax_ffi.so | grep "not found"` should be empty
     (catches the RPATH / SONAME-shim regression class from A1 #13, #14,
     #15).
  2. `tests/runtime/test_init.py` with `unittest.mock` on
     `jax.distributed.initialize` — four cases: single-rank, multi-rank
     fast path, multi-rank fallback, sentinel re-entry.
  3. `tests/test_cohsex_schema.py` (rides on #2's schema-validate work).

  Plus collapse the ~15 inline `_LORRAX_JAX_DISTRIBUTED_DONE` copies in
  test/bench files to single imports from `runtime.init_jax_distributed`
  (A3 B5 + FRAGILE-9).

  A separate weekly Perlmutter cron (self-hosted runner or NERSC-action
  equivalent — see Q3) runs the same trio inside Shifter against the real
  FFI build. This is a follow-up, not in-scope for the same blitz.
- **Acceptance.**
  - CI green on fresh PR; all three tests run < 60s on a free Linux
    runner.
  - `grep -L 'from runtime import init_jax_distributed' src/common/*_test.py`
    fails CI when a new inlined dist-init copy lands.
- **Closes.** A.9 (most-cited "no CI" gap); A4 D-26/D-27/D-29/D-30, A3
  FRAGILE-1/3/4/5 + B3, A1 #8 + open Q8.
- **Dependencies.** Soft on #2 (test 3 rides on the schema). Soft on #1
  (test 1 is much cleaner after `pip install` produces the .so).
- **Doable from desk?** **Yes for the CPU runner.** Weekly Perlmutter
  follow-up needs Q3 answered.
- **Why fourth.** Cheapest possible CI is hours away; locks in every
  fix in #1-#3 against regression. Without CI, every fix is one
  non-malicious commit away from being rolled back silently.

### #5 — Split the `lorrax_agent` overlay: promote `LORRAX_NNODES`-aware multi-node `lxrun` + `lxstatus` upstream; leave pool/heartbeat in the sandbox; document the boundary

Ranked #4-#5 by Agents 2/3/4; absorbed under "doctor" theme by Agent 1.
Source: A3 B6 + A4 B-7 + A2 B10 + A4 D-33.

- **What changes.** Copy the multi-node srun construction from
  `modulefiles/lorrax_agent/1.0.lua` into base
  `config/modulefiles/lorrax/0.1.0.lua` *without* the `lx_pool.py
  prelaunch` call and *without* the heartbeat. Add a 30-line `lxstatus`
  to the base (parses `squeue -j $SLURM_JOBID -s`). Add a one-paragraph
  "multi-user-on-shared-allocation needs the sandbox overlay" callout
  to `config/README.md` and `PORTING.md`.
- **Acceptance.**
  - `LORRAX_NNODES=2 lxrun python3 -c "import jax;
    print(jax.process_count())"` returns 2 (or `2 × LORRAX_NGPU`) on a
    2-node allocation **with no overlay loaded**.
  - `module show lorrax` lists `lxrun`, `lxshell`, `lxpre`, `lxalloc`,
    `lxstatus`; **not** `lxattach`/`lxreap`.
  - Sandbox `lorrax_agent` overlay continues to work unchanged.
  - `ENVIRONMENT_COMPREHENSIVE.md:322` no longer contains a doc lie.
- **Closes.** A.3 (lxrun -N 1 hardcoded), A.11 (sandbox-vs-upstream
  split); A2 D1/D17 + B10, A3 LOC-COST-4 + B6, A4 D-8 + D-33.
- **Dependencies.** None on items #1-#4. Becomes load-bearing if Q2
  ("second user planned?") is yes.
- **Doable from desk?** Mostly; **needs one 2-node allocation** (~30
  min) to verify multi-node launch.
- **Why fifth.** Lowest absolute leverage in the top 5, but cheapest.
  Also a prerequisite for any honest "PORTING.md is single source of
  truth" rewrite — the upstream module can't currently launch multi-node,
  so the porting story is wrong before a second user even starts.

---

## Open questions that gate the blitz

Ordered by how much they shift the plan. Each is concrete and
author-answerable (most in under a minute).

### Q1 (gates #1, #2 framing) — Which JAX *actually* runs inside `lxrun` today?

Flagged by all four agents (A1 #1, A4 D-13/D-14, A3 §6 Q1, A2 D11).
`pyproject.toml` declares `jax[cuda13]>=0.9.0`. The container
`nvcr.io/nvidia/jax:25.04-py3` ships JAX 0.5.x + CUDA 12.x. The
modulefile bind-mounts `LORRAX_SITE` site-packages into the container,
possibly overriding the in-image JAX. **No agent could resolve this
without running:**

```bash
lxrun python3 -c "import jax; print(jax.__version__, jax.__file__, jax.devices())"
```

One line of output decides whether #1's "loosen the pin" is 1 day or 3
days.

### ~~Q2~~ — **RESOLVED** by Decision 1 above (switch to `cuda_async`). No longer an open question. Promoted to **Blitz item #0** (sub-1-hour modulefile edit).

### Q3 (gates #4 acceptance) — Is the GitHub-Actions or NERSC-action runner path realistic, or self-hosted?

A4 Q-9. The cost of #4 is dominated by runner setup, not test code. If
NERSC's `@nersc/setup-perlmutter`-style action exists and covers
Shifter+GPU+multi-rank, weekly Perlmutter CI is straightforward; if
not, fallback is a self-hosted runner on a workstation with one GPU.

### Q4 (gates #2 implementation) — Are `sigma_debug_split_contrib`, `write_no_head_vw`, and `use_chunked_isdf` (documented but not in `_DEFAULTS`) deprecated/auto-derived, or are they typos in the doc?

A4 D-1, Q-8. Determines the deprecated-keys allow-list for #2's strict
schema. If intentional → parser should grow them; if typos → doc removes
them. Author's call.

### Q5 (gates #2 default) — When schema-strict mode lands, should unknown cohsex.in keys be a warning or a hard error?

A4 Q-1. Strict + mass-rewrite of templates is cleaner long-term but
breaks every historical `runs/**/cohsex.in` until rewritten. Consensus
recommendation: `LORRAX_COHSEX_STRICT=1` opt-in (default warning), so
new code can adopt it and old `runs/` re-execute as-is.

### ~~Q6~~ — **RESOLVED** by Decision 2 above. Shifter + Cray MPICH are committed; non-NERSC porting is best-effort, not maintained. PORTING.md re-frames as Perlmutter deployment guide. Apptainer companion stays out of scope.

---

## Notable disagreements that were resolved in round 2

### `select_gpu.sh` is NOT pure NERSC-coping

Agent 3 was right that the script is generic SLURM, not Cray-specific
(it would work as-is on Frontier, Leonardo, any university SLURM).
Agents 1+2 were right that the `${SLURM_LOCALID:-0}` fallback silently
breaks on PBS/PMIx clusters. Resolution: keep the script, fix the
fallback chain (folded into #3). The COMPAT tag inflation from round 1
is corrected.

### Apptainer port is NOT a 1-day blitz

Agent 2's round-1 B3 (Apptainer companion image) was demoted in round 2
by Agents 1, 2, and 4. Real Apptainer support needs: the libmpi SONAME
shim ordering verified under `--bind`, GPU-aware MPI passthrough
configured, stage-script deletion validated. ~1-2 weeks, not 1 day.
**Replaced by A2 B5 — make container-mount paths configurable** —
which IS a 1-day blitz and unblocks a future Apptainer port. (A2 B5
is the honorable-mention #6; rolled into #3's "single contract" spirit.)

### `lorrax_agent` 645-LoC overlay is not "dead weight"

Agent 3's round-1 framing ("the overlay could be 50 lines") was
walked back in round 2. The pool coordination solves a real
multi-process / shared-allocation problem and is sandbox-appropriate.
But the *universal* slice (multi-node lxrun, lxstatus) belongs
upstream — hence #5.

### Two-allocator question is real, not a tagging artifact

Agent 3 alone caught `XLA_PYTHON_CLIENT_ALLOCATOR` ↔ `TF_GPU_ALLOCATOR`
in round 1. Round 2 confirmed across agents: this is a genuine
single-agent convergent finding (with the JAX docs). Promoted to Q2.

---

## Items deliberately NOT in the top 5

- **QUICKSTART.md / in-tree example cohsex.in** (A4 B-3) — premature
  until #1-#3 fix what the quickstart would describe. Schedule for
  week 2.
- **`docs/_index.md` router + link-check CI** (A4 B-9) — depends on
  #4's CI scaffold; week 2.
- **Apptainer companion** (A2 B3) — out of scope. Replaced by
  configurable container-mount paths via #3.
- **Memory-model consolidation** (A4 B-6) — owned by the parallel
  `zeta_rchunk_memory_model` team; not install-blitz scope.
- **Image digest pinning** (A2 B6) — low leverage, contingent on
  Shifter feature support (Q8 in A4's draft); defer.
- **`MAINTENANCE_TODO.md`** (A4 B-8) — bookkeeping; fold into #2's
  housekeeping pass.

---

## Suggested execution order

If the author has one week of blitz time:

1. **Day 0 (~1 hour):**
   - **Blitz #0**: replace `XLA_PYTHON_CLIENT_ALLOCATOR=platform` +
     `TF_GPU_ALLOCATOR=cuda_malloc_async` with a single
     `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` in `0.1.0.lua:130-131`.
     Smoke a representative run, eyeball memory + iteration time.
   - Answer **Q1** (which JAX runs) from a live `lxshell` — one-line
     `python -c "import jax; print(jax.__version__, jax.__file__)"`.
   - Read `src/ffi/cusolvermp/eigh.py` + `batched.py` to see if the
     workspace is JAX-allocated or C++-side `cudaMalloc`. If the
     latter: queue Blitz #6 (move workspace allocation to JAX-side
     `jnp.empty`).
2. **Day 1-2:** **Blitz #2** (cohsex schema + auto-doc + finish env
   migration) — fully desk-doable, biggest maintainability win.
3. **Day 2-3:** **Blitz #1** (scikit-build-core wiring + JAX pin
   fixup) — desk-side refactor + one Shifter session to verify.
4. **Day 3-4:** **Blitz #3** (single source of truth for `--mpi=`,
   portable `select_gpu.sh`, post-init `assert jax.process_count() ==
   SLURM_NTASKS`) — narrower scope than originally because no
   non-Cray MPI path needs to be modeled.
5. **Day 4-5:** **Blitz #4** (CPU smoke CI). Locks in #1-#3.
6. **Day 5:** **Blitz #5** (multi-node `lxrun` + `lxstatus` upstream;
   document overlay boundary). Lower urgency since no second user
   planned, but cheap.

If only **two days**: Day-0 (#0 allocator fix + Q1 + workspace check)
and #2 (cohsex schema). Single biggest maintainability ROI.

If only **one day**: #0 alone in the morning (the perf regression
risk is real if `platform` is winning), #2 in the afternoon.

---

## Source artifacts

All eight drafts live in
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/`:

- Round 1 (slice audits, ~2700 lines total):
  `agent_1.md` (build/FFI, 588 lines),
  `agent_2.md` (MPI/Shifter, 562 lines),
  `agent_3.md` (runtime/env, 764 lines),
  `agent_4.md` (docs/synthesis, 754 lines).
- Round 2 (reconciliation, ~2000 lines total):
  `round2_agent_1.md`, `round2_agent_2.md`,
  `round2_agent_3.md`, `round2_agent_4.md`.
- Shared briefing: `CONTEXT.md`.
- Tmux launchers: `launch_team.sh`, `launch_round2.sh`.
