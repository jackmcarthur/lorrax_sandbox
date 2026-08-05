# Agent 4 — Round 2 (synthesis): docs / config / CI ↔ build / MPI / runtime

Synthesis pass. I re-read agents 1/2/3, then re-graded my round-1 catalogue,
then ranked across slices. Where I'm tie-breaker is "doc vs. code"
(Section B) and "what gets done in week 1 of an install-maintain blitz"
(Section D).

Abbreviations: A1/A2/A3/A4 = the four agents. Citations like `A2 D5` =
defect 5 in agent_2.md.

---

## Section A — Cross-slice convergence

Defects independently flagged by at least two agents. I weight 3- and
4-agent agreement above 2-agent agreement. Tags use the round-1 lens
(FRAGILE / COMPAT / LOC-COST).

| # | Defect (one-liner) | Agents | Strongest tag | Fix one-liner |
|---|---|---|---|---|
| A.1 | `jax[cuda13]>=0.9.0` pin in `pyproject.toml` conflicts with the `nvcr.io/nvidia/jax:25.04-py3` container (JAX 0.5.x, CUDA 12.x) — nobody runs `pip install` because the FFI build is decoupled, so the mismatch is invisible. | A1 #1, A4 D-13/D-14, A3 Q2 | **FRAGILE+COMPAT** | Pick the version the bind-mounted site-packages actually ships, document the bind-mount→runtime-JAX precedence, then loosen the pin to `>=0.5` until a real CUDA-13 migration is planned. |
| A.2 | No `[build-system]` table; `pip install -e .` is a silent no-op for `liblorrax_ffi.so`. The orphan `dependency-groups.build` (scikit-build-core + nanobind) is half-migrated. | A1 #2, A4 D-36 | **FRAGILE+LOC-COST** | Wire scikit-build-core as the real backend, drop the orphan group, install the .so into site-packages on `pip install`. (A1 P1 = A4 B-5.) |
| A.3 | `lxrun` hardcodes `-N 1`; multi-node only works via the sandbox overlay, but `PORTING.md` doesn't say so and `ENVIRONMENT_COMPREHENSIVE.md` *implies* multi-node works with `LORRAX_NNODES`. | A2 D1/D17, A3 LOC-COST-4 surface, A4 D-8 | **FRAGILE+COMPAT** | Either port the overlay's `LORRAX_NNODES` into base lxrun (without the pool-coordination layer) or add a one-paragraph "base is single-node" callout in `config/README.md` + `PORTING.md`. |
| A.4 | `select_gpu.sh` reads only `SLURM_LOCALID`; silently coalesces all ranks onto GPU 0 on PBS/PMIx. | A1 #22, A2 D7, A3 §3 row "select_gpu.sh" | **COMPAT+FRAGILE** | 5-line change: also read `PMI_LOCAL_RANK`/`PMIX_LOCAL_RANK`; fail loud if none are set when `SLURM_NTASKS > 1`. |
| A.5 | MPI-type selector is split: `LORRAX_MPI_TYPE` (runtime, Lua L73/L270) + `LORRAX_PHDF5_MPI_STACK` (build, `run_shifter.sh`); these are independent strings tied together by code review, and a wrong value silently gives singleton MPI_COMM_WORLDs. | A1 #18, A2 D2/D20, A3 COMPAT-1 | **COMPAT+FRAGILE** | One source of truth (a `config/mpi_stacks/{cray_mpich,openmpi}.{cmake,sh}` file or just one env var), plus a post-init `assert size == SLURM_NTASKS` (A3 B7d). |
| A.6 | `/lorrax_nvhpc` / `/lorrax_phdf5` / `/lorrax_slate` bind-mount strings appear as load-bearing literals in 4+ files (modulefile, `run_shifter.sh`, `CMakeLists.txt`, every `stage_*.sh`), and are baked into the FFI `INSTALL_RPATH`. | A1 #7/#13, A2 D3, A3 §2b table | **LOC-COST+FRAGILE** | Make them `LORRAX_CONTAINER_{NVHPC,PHDF5,SLATE}_PATH` env vars with the current values as defaults; the RPATH gets baked at build time from the build-side var. (A2 B5.) |
| A.7 | `MPICH_GPU_SUPPORT_ENABLED=1` + `LD_PRELOAD=libmpi_gtl_cuda.so.0` are unconditional in the Lua, even on lxshell (one rank, no GPU comm) and on a notional non-Cray port. The symmetric UCX/OpenMPI knobs are nowhere. | A2 D4/D5/D13, A3 COMPAT-1 | **COMPAT+FRAGILE** | Guard both behind `LORRAX_MPI_TYPE` starting with `cray`/`mpich`; emit `UCX_TLS=…,cuda_copy,gdr_copy,cuda_ipc` + `OMPI_MCA_pml=ucx` for `pmix`. (A2 B8.) |
| A.8 | `pyproject.toml` lacks a `[build-system]`, and `PORTING.md` §"Build system" doesn't say the FFI build needs `run_shifter.sh` (not bare `bash build.sh`) — so the obvious workflow `git clone; pip install -e .` silently produces a no-FFI install. | A1 #30 + #2, A4 D-36 | **FRAGILE+LOC-COST** | Covered jointly by A.2 (wire scikit-build-core) and a PORTING.md edit naming `run_shifter.sh` as the required wrapper. |
| A.9 | No CI, no smoke test for FFI symbol resolution, JAX-distributed bootstrap, or Shifter modulefile load. The c52fbd2 ABI break ("CAL→NCCL at 0.7") is exactly what CI would catch. | A1 #8 + open Q8, A3 B3, A4 D-26/D-27/D-28/D-29 | **FRAGILE** | A minimal smoke trio: `import _lorrax_ffi`; a 64×64 distributed Cholesky; a CPU-only `jax.distributed.initialize()` test. Even a weekly self-hosted runner is a step up from zero. (A4 B-4 + A3 B3.) |
| A.10 | The env-var-to-`cohsex.in` migration trend (488e870 / 9fe5fde / 40a4cca) is incomplete: 5 SC knobs, 3 ISDF planner knobs, 5 V_q tunables, 2 Lustre-stripe knobs still in env vars. None are in `COHSEX_INPUT.md` regardless. | A3 LOC-COST-1/2/3 + FRAGILE-7, A4 D-3/D-4 | **LOC-COST+FRAGILE** | Finish the migration. The author's pattern is established; ~half-day of mechanical work + doc regen. (A3 B2.) |
| A.11 | The sandbox `lorrax_agent` overlay carries logic upstream lacks (lxstatus, multi-node lxrun, allocation tag for `lxattach`). Doc gap: PORTING.md and `config/README.md` don't warn that the upstream module alone won't survive shared-allocation multi-user use, and one of the things the overlay does (lxrun multi-node) is universally useful, not sandbox-specific. | A2 D17, A3 LOC-COST-4 + B6, A4 D-33/D-34/D-35 | **LOC-COST** | Promote the universal slice (multi-node lxrun + `lxstatus`) upstream; leave the pool/heartbeat layer sandbox-only with a clearly-marked dependency. (A3 B6 + A4 B-7.) |
| A.12 | Hardcoded user-home paths (`/global/homes/j/jackm/software/slate/install` in CMake, `LORRAX_FFI_NVHPC_DIR` defaulting to `/pscratch/sd/${USER:0:1}/${USER}/…`, `LORRAX_SRC`/`LORRAX_SITE` defaulting to `/global/u2/j/jackm/…`). | A1 #16/#17/#18, A2 site_config NERSC-isms, A4 §3 | **COMPAT** | One templating pass: every host path overridable, no jackm-string surviving the install.sh patching step. |
| A.13 | The ~250 LoC of `stage_*.sh` exists exclusively because Shifter restricts `--volume` source paths to `/pscratch`. Apptainer/Enroot have no such restriction. The PORTING.md "for non-Shifter, swap `apptainer exec`" line vastly understates this. | A1 #25, A2 D8 | **LOC-COST+COMPAT** | Document. The scripts are correct for Perlmutter; the porting story needs an explicit "delete stage scripts; bind-mount directly" branch. |
| A.14 | JAX coordinator port `12355` is hardcoded with no override and no collision handling — relevant any time two ranks-0 land on the same first-node of an allocation (the agent overlay's bread-and-butter). | A2 D15, A3 FRAGILE-3 | **FRAGILE** | Trivial: derive port from `int(os.environ.get("SLURM_JOBID","0")) % 10000 + 20000` or similar; fall back to env override. |
| A.15 | Sentinel `_LORRAX_JAX_DISTRIBUTED_DONE` leaks across `srun` steps in the same shell — `module unload && module load` mid-session followed by a new `lxrun` skips distributed init silently. Fragile/undocumented invariant. | A3 FRAGILE-5, A4 D-15 (informal-rules problem) | **FRAGILE** | Name the sentinel with `SLURM_STEP_ID` suffix, document in a `MAINTENANCE_TODO.md`. (A3 B7c.) |
| A.16 | PORTING.md gaps: no mention of `jax.ffi.include_dir()` ABI keying to the build-time JAX version (`A1 #34`), no statement of the SLATE version contract (`A1 #32`), no explanation of the bind-mount triad's runtime requirements (Agent 2's whole audit), no mention of the multi-node ceiling (A.3 above), `cohsex.in` schema lives in the sandbox (A4 D-32). | A1 #30-#34, A2 D17, A4 D-13/D-32 | **LOC-COST+FRAGILE** | Treat PORTING.md as the single porting source-of-truth and rewrite it against this round's defect list. The current doc is partly aspirational (A1 open Q1, A4 Q-6). |

Convergence on `[FRAGILE]`-vs-`[COMPAT]`: A.1, A.5, A.9, A.15 are
mostly fragility-on-Perlmutter; A.4, A.7, A.13 are mostly portability;
A.2, A.6, A.11, A.12, A.16 are both at once.

---

## Section B — Disagreements

Real disagreements between the four drafts, with a tie-break.

### B.1 — Should `select_gpu.sh` survive?

- **A2 (B4)**: drop `select_gpu.sh` AND `in_container.sh`; use `srun
  --gpus-per-task=1` + `shifter --env=`.
- **A3 (LOC-COST-5)**: `select_gpu.sh` is "a fine pattern" and "portable";
  `in_container.sh` is dead weight outside Shifter, can go.
- **A1 (#22)**: `select_gpu.sh` is COMPAT+FRAGILE on PBS/PMIx.
- **A4**: didn't take a position.

**Resolution.** A3 wins on `select_gpu.sh` but A1 wins on the contents:
keep the script, fix it to also read `PMI_LOCAL_RANK`/`PMIX_LOCAL_RANK`
and fail loud when SLURM_NTASKS>1 with no localid. A2's `--gpus-per-task`
swap is blocked on A3's open question Q2 ("does it still break JAX
topology sync in 2026?") — `run_shifter.sh:163-167` has a comment saying
this broke historically, undated. Verifying the swap costs a real run;
fixing `select_gpu.sh` costs five lines. Do the cheap thing first.

For `in_container.sh`: A2 + A3 agree it's Shifter-specific. A4
synthesis: keep it on Perlmutter, but tag it explicitly as
NERSC-only in `PORTING.md` so a porter knows to delete it under
Apptainer rather than puzzle over what it does.

### B.2 — Apptainer port: 1-day blitz or open research project?

- **A2 (B3)**: ranks a `config/apptainer/` companion image + def file at
  position 7/10 in the blitz list — "highest risk of these, needs cluster
  access to validate."
- **A1 (Open Q1)**: explicitly cannot resolve whether Apptainer's `--bind`
  handles the libmpi_gnu_*.so.12 SONAME shim correctly; would have to
  actually try it.
- **A4**: out of scope (not in my slice).

**Resolution.** A1 wins. A2's "1-day blitz" is unrealistic — A2 actually
admits this in the risk line. A real Apptainer port is a 1-2 week effort
once the SONAME shim, GPU-aware MPI passthrough, and stage-script
deletion are all verified on a non-NERSC site. The author's blitz list
should *not* contain Apptainer support; it should contain "make the
modulefile and CMake honor configurable container-mount paths so a
future Apptainer port is mechanical" (A2 B5 — that one IS a day).

### B.3 — Is `jax[cuda13]>=0.9.0` a docs problem or a code problem?

I'm the natural arbiter here (Section instructions).

- **A1 (#1)**: code problem — the pin will actively break a clean
  `pip install -e .` on the wrong CUDA.
- **A4 (D-13/D-14)**: docs problem — `pyproject.toml`'s pin is a lie that
  describes nothing currently running, and the actual JAX comes from the
  bind-mounted `LORRAX_SITE` site-packages.
- **A3 (Q2)**: open question — "what JAX *actually* runs at production
  time?".

**Resolution.** Both. The pin is wrong for two distinct reasons:

1. It would actively break a fresh install attempt
   (A1's framing) — code problem.
2. It documents an aspirational future state, not the current production
   state — docs problem.

The first symptom only fires once A.2 lands (pip install triggers the
build). Today, no one notices because no one runs `pip install -e .`.
After A.2 lands, this becomes a real install-time failure. So: fix the
pin (code) AND replace the PORTING.md "JAX 0.5+" line with the actual
bind-mounted runtime version (docs). The order is forced — code change
has to land first, doc change is mechanical after.

### B.4 — `XLA_PYTHON_CLIENT_ALLOCATOR=platform` vs `TF_GPU_ALLOCATOR=cuda_malloc_async`

- **A3 (FRAGILE-2 + B8)**: these are mutually exclusive; one is the
  *intended* choice (`cuda_malloc_async` per the in-Lua comment); the
  other might be silently active.
- **A1, A2, A4**: didn't catch this.

**Resolution.** A3 is right; this is a genuine FRAGILE the other three
of us missed. The fix is one line in `0.1.0.lua:130-131` and depends on
empirically verifying which is currently winning. This is a Section D
candidate — high leverage, sub-day, but blocked on one real run.

### B.5 — Should the `lorrax_agent` overlay graduate upstream?

- **A2 (D17)**: yes-ish; multi-node lxrun, at least.
- **A3 (LOC-COST-4 + B6)**: upstream `lxstatus` only; leave pool/heartbeat
  in the sandbox; questions whether the multi-agent A/B/C/D pattern
  should exist at all.
- **A4 (D-33/D-35 + B-7)**: split between two options, leaned reactive
  ("wait for a second user").

**Resolution.** Synthesis answer is A3's, sharpened: promote
`LORRAX_NNODES`-aware multi-node lxrun AND `lxstatus` upstream
(both are universally useful, sandbox-independent). Leave
`lxattach`/`lxreap`/`lx_pool.py` in the sandbox. A.3 + A.11 collapse to
one blitz under this resolution.

### B.6 — Image digest pinning

- **A2 (D11, B6)**: pin `nvcr.io/nvidia/jax:25.04-py3@sha256:…`.
- **A2 (own open question Q1)**: but it's unclear whether Shifter
  actually honors digest refs end-to-end.
- **Others**: didn't address.

**Not really a disagreement** — A2's blitz is internally hedged. Carry
it through as "low-priority, pending feature-support spike." Not in the
top 5.

---

## Section C — My slice-specific updates

Restating each round-1 defect/blitz with **CONFIRMED** / **SHARPENED** /
**REVISED**, in light of the other three drafts.

### Defects (round-1 numbering)

| # | Status | Note |
|---|---|---|
| D-1 (`use_chunked_isdf` documented, not in `_DEFAULTS`) | **CONFIRMED** | Nobody else touched cohsex.in keys. |
| D-2 (`compute_mode` enum not in doc) | **CONFIRMED** | Same. |
| D-3 (11 new keys from 488e870/9fe5fde) | **SHARPENED** | A3's table 2f confirms the exact key list and adds 5 SC + 3 ISDF + 5 V_q still-not-migrated keys. The "migration is in flight, doc is stale" framing is right, but the scope is bigger than I thought: ~13 keys to add now + ~10-15 keys to migrate-then-add. Bumps B-1+B-2 leverage. |
| D-4 (~50% of parser keys undocumented) | **CONFIRMED** | |
| D-5 (`output_file` deprecated, doc & template still use it) | **CONFIRMED** | |
| D-6 (`bare_coulomb_cutoff` 4·ecutwfc vs BGW ecutwfc) | **CONFIRMED** | |
| D-7 (`lxkill` documented, not defined) | **CONFIRMED** | A3 §2b table confirms the four `set_shell_function` calls; no `lxkill` among them. |
| D-8 (`lxrun -N 1` hardcoded; `LORRAX_NNODES` unread) | **SHARPENED** | A2 D1/D17 and A3 §2d both confirm and add color: the overlay silently fixes this, so anyone using the sandbox doesn't notice. Now defect A.3. |
| D-9 (`docs/index.md` stale `src/isdf/...` paths) | **CONFIRMED** | |
| D-10 (`AGENT_TODO.md` stale paths) | **CONFIRMED** | |
| D-11 (no in-tree cohsex.in example) | **CONFIRMED** | |
| D-12 (sandbox `templates/cohsex.in` is rotten) | **CONFIRMED** | |
| D-13 / D-14 (jax-cuda13 vs container) | **SHARPENED** | A1 #1 promotes this from "docs problem" to a real install-time bug-in-waiting. Now A.1; re-categorized as both [FRAGILE] and [COMPAT]. |
| D-15 (no-`np.concatenate` rule under-enforced) | **CONFIRMED** | |
| D-16 (parser silently accepts unknown keys) | **CONFIRMED** | A3 didn't dig into the parser, but its B4 (ENV_VARS.md) is the env-var-side analogue. |
| D-17 (nullable-float keys fragile) | **CONFIRMED** | |
| D-18 (inconsistent enum validation) | **CONFIRMED** | |
| D-19 (two memory model packages) | **CONFIRMED** | |
| D-20 (two `aot_memory*` modules same name) | **CONFIRMED** | |
| D-21 (`read_cohsex_input` alias) | **CONFIRMED** | |
| D-22 (four `*_COMPREHENSIVE.md`) | **CONFIRMED** | |
| D-23 (PHYSICS §11 supersedes §3-5, but §3-5 still present) | **CONFIRMED** | |
| D-24 (~2500 lines of in-progress / `_REVISED` docs) | **CONFIRMED** | |
| D-25 (`AGENT_TODO.md` half-archived) | **CONFIRMED** | |
| D-26 (no CI) | **CONFIRMED** | Now A.9. |
| D-27 (no FFI smoke test) | **SHARPENED** | A1 open Q8 wants the same. A3 wants distributed-init unit tests with mocked SLURM (its B3) — complementary, not overlapping. A4 should reuse A1's "`ldd liblorrax_ffi.so | grep "not found"`" as the dirt-cheap FFI smoke. |
| D-28 (no Shifter smoke test) | **CONFIRMED** | |
| D-29 (`tests/active/` collected?) | **REVISED** | Still don't know without running pytest. A3 didn't resolve it either. Lower priority than I had — move to open Q. |
| D-30 (no cohsex.in parser test) | **CONFIRMED** | |
| D-31 (regression test pinned to CPU) | **CONFIRMED** | |
| D-32 (COHSEX_INPUT.md in wrong repo) | **CONFIRMED** | |
| D-33 (overlay sandbox-only, single-user-on-shared-install bug) | **SHARPENED** | A2 D17 + A3 LOC-COST-4 reframe this: split the overlay. Multi-node lxrun + `lxstatus` belong upstream; pool/heartbeat stay sandbox. A.11 above. |
| D-34 (templates/ in sandbox) | **CONFIRMED** | |
| D-35 (sandbox tooling re-implementing upstream entry points) | **CONFIRMED** | |
| D-36 (no `[build-system]`) | **SHARPENED** | A1 #2 makes this defect A.2 above; the depth of the build-system gap is much worse than I framed it. The orphan `dependency-groups.build` is a 50%-migration tell. |
| D-37 (`docs/index.md` `examples/` link broken) | **CONFIRMED** | |
| D-38 (mkdocs in runtime deps) | **CONFIRMED** | |
| D-39 (two doc indices in README & AGENTS.md) | **CONFIRMED** | |

### Blitz proposals (round-1)

| # | Status | Note |
|---|---|---|
| B-1 (schema-validate cohsex.in) | **CONFIRMED + PROMOTED** | A3 B2 (migrate remaining env vars) feeds the same schema; do them together. Top-1 in Section D. |
| B-2 (regenerate COHSEX_INPUT.md from parser) | **CONFIRMED** | Same. Top-1 in Section D. |
| B-3 (QUICKSTART.md) | **CONFIRMED** | Lower priority than I had — A1/A2/A3 all show the install story is broken upstream of "does the user have an input file"; fixing builds and runtime first is higher leverage. |
| B-4 (smoke-test CI for FFI / dist init) | **SHARPENED** | A1 (#8 + open Q8) and A3 (B3) both want this; CPU-only `init_jax_distributed` test (A3 B3) is the easy first step. Top-2 in Section D. |
| B-5 (wire scikit-build-core in `[build-system]`) | **CONFIRMED** | A1 P1 is essentially the same proposal with build-side detail. Top-3 in Section D. |
| B-6 (consolidate two memory models) | **REVISED** | The zeta-rchunk-memory-model parallel team is converging; not the install-blitz's job. Drop from top-5. |
| B-7 (promote `lorrax_agent` overlay) | **SHARPENED** | A3 B6 wants only the universal slice. A.11 above; replace my "two-option" framing with "split it: multi-node lxrun + lxstatus upstream; rest stays". |
| B-8 (`MAINTENANCE_TODO.md`) | **CONFIRMED** | Cheap. |
| B-9 (`docs/_index.md` router + link-check CI) | **CONFIRMED** | |
| B-10 (mkdocs to docs-group) | **CONFIRMED** | Pure hygiene. |

---

## Section D — Top-5 cross-cutting blitz proposals

Cross-slice ranking. Ordering reflects what would most reduce the "could
a second user install LORRAX" risk per author-day, weighted by how much
the four agents agree.

### **D.1 — Schema-validate `cohsex.in` from a single Field table; generate `COHSEX_INPUT.md` from it; finish the env→cohsex.in migration; move the doc into `lorrax_C/`.**

This is **A4 B-1 + A4 B-2 + A3 B2** as one unit.

- **Scope (~2 days):** introduce `src/gw/_cohsex_schema.py` with a single
  `Field(name, type, default, doc)` table covering all 77 parser keys
  plus the 13 env vars marked LOC-COST-1/2/3 by A3. Migrate those env
  vars to schema-table keys. Add `_UNKNOWN_KEY_POLICY` + alias table for
  deprecated keys (`output_file`, `use_chunked_isdf`,
  `sigma_debug_split_contrib`, `write_no_head_vw`). Add
  `tools/gen_cohsex_input_md.py` that emits `lorrax_C/docs/COHSEX_INPUT.md`
  and a pytest diff. Update `templates/cohsex.in` (sandbox) and the
  regression-test fixture.
- **Acceptance:**
  - `pytest tests/test_cohsex_schema.py` asserts parser ⇄ schema bijection.
  - `pytest tests/test_cohsex_doc_freshness.py` regenerates the .md and
    fails on diff.
  - `read_lorrax_input("templates/cohsex.in")` raises no warnings.
  - No `LORRAX_SC_*` / `LORRAX_V_Q_*` / `ISDF_*` reads in `src/gw/` or
    `src/common/isdf_fitting.py` outside of debug-only paths.
- **Dependencies:** none.
- **Desk vs. real-run:** **desk only**. Verifiable by reading the parser
  and running pytest on a CPU.
- **Addresses:** A.10 (env→cohsex.in not finished); A.16 (PORTING / docs
  drift on COHSEX_INPUT); A4 D-1..D-6, D-12, D-16, D-18, D-30, D-32;
  A3 LOC-COST-1/2/3 + B2 + B4.
- **Why #1:** highest agent-agreement (3 out of 4 in some form), no
  blockers, locks in a maintenance pattern the author has already
  validated three times (488e870/9fe5fde/40a4cca), and produces an
  in-repo reference doc that a second user can read.

### **D.2 — Wire scikit-build-core as the real `[build-system]`; loosen the JAX pin; install `liblorrax_ffi.so` into site-packages on `pip install`.**

This is **A1 P1 + A4 B-5**, with A1's deeper build-system framing.

- **Scope (~1-1.5 days):** add `[build-system] requires = ["scikit-build-core>=0.11",
  "nanobind>=2.0", "cmake>=4.0", "ninja>=1.10"]` and `build-backend =
  "scikit_build_core.build"` to `pyproject.toml`. Promote
  `src/ffi/common/cpp/CMakeLists.txt` (or wrap from repo root). Drop the
  orphan `dependency-groups.build`. Loosen `jax[cuda13]>=0.9.0` to match
  the bind-mounted runtime JAX (likely `>=0.5,<0.9` with `cuda12`).
  Confirm via `python -c "import jax; print(jax.__version__)"` inside
  the container — see Section E Q1.
- **Acceptance:**
  - `pip install -e .` inside the container produces a working
    `import _lorrax_ffi` AND `liblorrax_ffi.so` at the expected path.
  - `LORRAX_FFI_SO` override still works.
  - A new `tests/build_smoke/test_pip_install.py` runs in CI.
- **Dependencies:** depends on Section E Q1 (real JAX version inside the
  container). Half a day to spike Q3 (does scikit-build-core handle the
  CMake autodetect ladder cleanly).
- **Desk vs. real-run:** **needs one real container session** to confirm
  the JAX version + run the smoke test. Build wiring itself is desk-work.
- **Addresses:** A.1 (JAX pin mismatch), A.2 (no `[build-system]`),
  A.8 (PORTING gap); A1 #1/#2, A4 D-13/D-14/D-36.
- **Why #2:** unblocks every other install-side blitz. Without this, the
  build is a manual `bash build.sh` ritual that PORTING.md half-describes
  and nobody else can reproduce.

### **D.3 — Single MPI-stack contract + post-init `assert size == SLURM_NTASKS` + portable `select_gpu.sh`.**

This is **A1 P5 + A2 B1 + A2 B8 + A3 B7d**.

- **Scope (~1 day):** create `config/mpi_stacks/cray_mpich.{cmake,sh}`
  and `…/openmpi.{cmake,sh}` files that define the
  `(mpi_type, shifter_modules, mpi_lib_dir_ct, mpi_include_dir_ct,
   gpu_aware_env)` quintuple. Have `0.1.0.lua` and `run_shifter.sh`
  source the same file. Add the GTL preload + `MPICH_GPU_SUPPORT_ENABLED`
  guard so they only fire under `cray_mpich`. Add
  `assert jax.process_count() == int(os.environ["SLURM_NTASKS"])` to
  `init_jax_distributed`. Patch `select_gpu.sh` to fall back to
  `PMI_LOCAL_RANK`/`PMIX_LOCAL_RANK`.
- **Acceptance:**
  - `module load lorrax && env | grep -E '^(MPICH_|LD_PRELOAD|UCX_)'` shows
    Cray-specific vars iff `LORRAX_MPI_TYPE=cray_shasta`.
  - Wrong `LORRAX_MPI_TYPE` produces a clear failure within the first
    second of `init_jax_distributed`, not a silent singleton run.
  - `select_gpu.sh` on PBS/PMIx pins ranks correctly.
- **Dependencies:** none of the other blitz items.
- **Desk vs. real-run:** mostly desk + one real `lxrun` to verify the
  assertion fires correctly.
- **Addresses:** A.4, A.5, A.7 (three of the strongest 3-agent convergences);
  A1 #18, A2 D2/D4/D5/D7/D13/D20, A3 COMPAT-1.
- **Why #3:** every defect this fixes is a *silent* failure
  (wrong-`--mpi=` → singleton init; missing `LORRAX_MPI_TYPE` guard →
  spurious warnings on Apptainer; wrong localid → all ranks on GPU 0).
  Silent failures are the highest-leverage class to convert to loud ones
  before a second user touches the code.

### **D.4 — CPU-only smoke-test trio in CI: `import _lorrax_ffi`, mocked `init_jax_distributed()`, `cohsex.in` parser round-trip.**

This is **A3 B3 + A4 B-4 (CPU subset) + A1 open Q8 (`ldd` smoke)**.

- **Scope (~1 day):** new `.github/workflows/ci.yml` (or self-hosted
  equivalent) running on a free GitHub Linux runner: install the package
  with a CMake stub that satisfies `find_library` for cuSolverMp/SLATE
  but doesn't compile real CUDA. The three tests are:
  1. `ldd build/liblorrax_ffi.so | grep "not found"` should be empty
     (catches A.6's RPATH/SONAME class).
  2. `tests/runtime/test_init.py` with `unittest.mock` patching
     `jax.distributed.initialize` (A3 B3): four cases — single-rank,
     multi-rank fast path, multi-rank fallback, sentinel re-entry.
  3. `tests/test_cohsex_schema.py` (round-trips D.1's schema-validation).
- **Acceptance:** CI green on a fresh PR; all three tests run < 60s.
- **Dependencies:** D.1 lands first (test #3); D.2 lands first if we want
  real `pip install` in CI (otherwise CI uses a stub).
- **Desk vs. real-run:** **desk only** for the CPU runner. A real-GPU
  follow-up (weekly Perlmutter-self-hosted) is its own future blitz, not
  in this top-5.
- **Addresses:** A.9 (most agent-cited "no CI" gap); A4 D-26/D-27/D-29/
  D-30, A3 FRAGILE-1/3/4/5 + B3, A1 #8 + open Q8.
- **Why #4:** the cheapest possible CI is hours from existing. Even a
  CPU-only run catches A.15 (sentinel drift), A.11 (overlay/upstream
  contract via lxrun --dry-run snapshot), and the parser drift in D.1.
  Real-GPU CI is downstream of getting a CPU pipeline at all.

### **D.5 — Split the `lorrax_agent` overlay: promote `LORRAX_NNODES`-aware multi-node `lxrun` + `lxstatus` upstream; leave `lxattach`/`lxreap`/`lx_pool.py` sandbox-only; document the boundary in `PORTING.md` and `config/README.md`.**

This is **A3 B6 + A4 B-7** unified with **A2 D17 + A2 B10**.

- **Scope (~0.5-1 day):** copy the multi-node srun construction from
  `modulefiles/lorrax_agent/1.0.lua` into base `0.1.0.lua` (without the
  `lx_pool.py prelaunch` call and without the heartbeat). Add a 30-line
  `lxstatus` to base. Add one paragraph each to `PORTING.md` and
  `config/README.md` describing the multi-user-on-shared-install gap and
  pointing at the sandbox overlay for the pool-aware variant.
- **Acceptance:**
  - `LORRAX_NNODES=2 lxrun python3 -c "import jax;
    print(jax.process_count())"` returns the right count on a 2-node
    allocation, without the overlay loaded.
  - `module show lorrax` lists `lxrun`, `lxshell`, `lxpre`, `lxalloc`,
    `lxstatus` as shell functions; **not** `lxattach`/`lxreap`.
  - The sandbox overlay continues to work unchanged.
- **Dependencies:** none.
- **Desk vs. real-run:** **needs one 2-node allocation** to verify
  multi-node launch. ~30 minutes.
- **Addresses:** A.3 (lxrun -N 1), A.11 (sandbox-vs-upstream split);
  A2 D1/D17 + B10, A3 LOC-COST-4 + B6, A4 D-33.
- **Why #5:** lowest absolute leverage of the five, but cheapest. Also
  prerequisite for A4's "PORTING.md is the single porting source of
  truth" rewrite — if the upstream module can't launch multi-node, the
  porting story is wrong before a second user even starts.

### Did NOT make the top 5

- **QUICKSTART.md / examples** (A4 B-3, B-9): low-effort, but premature
  until D.1-D.5 have fixed the things the quickstart would describe.
  Schedule for week 2.
- **`docs/_index.md` router + link-check CI** (A4 B-9): worth doing,
  but the link-check is dependent on D.4's CI scaffold existing first.
- **Apptainer companion** (A2 B3): rejected per B.2. Replace with A2 B5
  ("configurable container-mount paths") which IS a 1-day blitz and
  unblocks a future Apptainer port. **A2 B5 is a strong honorable
  mention** — would be #6.
- **Memory-model consolidation** (A4 B-6): tracked by a separate parallel
  team; not install-blitz scope.
- **Resolving `XLA_PYTHON_CLIENT_ALLOCATOR` vs `TF_GPU_ALLOCATOR`**
  (A3 B8, B.4 above): unblocked by one log inspection on a real run; high
  leverage, sub-day, but blocked on Section E Q2. Schedule for day 2 of
  the blitz, after D.2's container session.
- **`MAINTENANCE_TODO.md`** (A4 B-8, A3 B4 ENV_VARS.md): bookkeeping
  hygiene, do alongside D.1.
- **Image digest pinning** (A2 B6): low leverage, contingent on Shifter
  feature support; defer.

---

## Section E — Open questions for the user

The synthesis-agent instruction says my Section E is the closest thing
to the user's first read. Ordered by what most changes the blitz plan.

**Q1. What JAX version actually runs inside `lxrun` today?** One
`lxrun python3 -c "import jax; print(jax.__version__, jax.devices())"`
resolves three open items at once: my D-13/D-14, A1 open Q4, A3 Q2.
Until this is known, D.2's "loosen the pin" line is guesswork. The
pin says `jax[cuda13]>=0.9.0`; the container ships JAX 0.5.x with
CUDA 12.x; the bind-mounted `LORRAX_SITE` may override either. **One
line of output decides whether D.2 is 1 day or 3 days.**

**Q2. Which of `XLA_PYTHON_CLIENT_ALLOCATOR=platform` or
`TF_GPU_ALLOCATOR=cuda_malloc_async` is actually winning?** A3
FRAGILE-2. The intent comment in `0.1.0.lua:127-128` matches the
async pool; one log line of `XLA_*` startup chatter, captured from
any recent run, decides whether removing `platform` is safe or
introduces a perf regression. Out-of-band: if `platform` is winning,
the author's current runs are slow-per-allocation and OOM-resistant
*by accident*; this is potentially the most consequential single line
in the modulefile.

**Q3. Has LORRAX ever been built or run on a non-NERSC machine —
even your workstation?** My Q-6 plus A2's open Q1/Q4/Q5. PORTING.md
reads as if it has been validated; my read is that it hasn't. If the
answer is "no", the entire PORTING.md should be marked
"aspirational" in its header. If "yes, on machine X", the
breakage-points are knowable and one or two cluster-specific callouts
turn most of Section A's [COMPAT] items into documented gaps rather
than open research.

**Q4. Is there a second user actually planned in the next ~6 months,
or is the blitz preventive?** A4 Q-7. Reactive vs. preventive
ordering changes whether D.5 (overlay split) is week-1 work or
week-N: with no second user, the sandbox overlay's pool layer is
fine where it is; the upstream gap is theoretical. With a second
user soon, D.5 jumps in priority. A1's open Q1-Q5 mostly only
matter if a real port is on the calendar.

**Q5. Are `sigma_debug_split_contrib` and `write_no_head_vw` really
deprecated/auto-derived, or do you want them as user-settable keys?**
My D-1 lists them as "documented but not in `_DEFAULTS`"; A3 didn't
look at `cohsex.in` keys. Before D.1's regen produces a "correct"
doc, I need to know whether these names should be on the
allow-list-deprecated path (silently emitted from inside the
parser) or are honest typos in the doc.

**Q6. Should `--gpus-per-task=1` be re-tested with the current JAX?**
A2 open Q2 + B.1 above. The undated comment in `run_shifter.sh:163-167`
is the only justification for `select_gpu.sh`'s continued existence.
If `--gpus-per-task=1` works in 2026, A2 B4 becomes viable in a future
blitz; if not, the comment should be dated, sharpened, and pinned in
the maintenance TODO.

**Q7. Is the multi-agent A/B/C/D pattern's coordination cost (`lx_pool.py`
at 645 lines) worth the savings vs. four independent salloc's?** A3
Q6. Strictly out-of-scope for the install/maintain blitz, but it's
upstream of how aggressively to split D.5 — if the answer is "I'd be
fine with separate sallocs", a lot of the sandbox overlay's
complexity can be archived.

**Q8. Does Shifter on Perlmutter accept digest-pinned image refs at
`shifter --image=…@sha256:…`?** A2 Q1. Sub-day spike for the author;
either confirms B6 (image digest pinning) as viable or de-prioritizes
it permanently.

**Q9. Are `LORRAX_FFI_NVHPC_HOST`/`_PHDF5_HOST`/`_SLATE_HOST` read by
any tooling I can't see (external `runs/` scripts, `lxstatus` follow-ups)?**
A3 Q5. Trivial to answer from your shell history. If unused, drop
them from the modulefile to shrink the env surface; if used, they
need a doc entry in the ENV_VARS.md table A3 B4 proposes.

---

Agent 4 round 2 done — see round2_agent_4.md
