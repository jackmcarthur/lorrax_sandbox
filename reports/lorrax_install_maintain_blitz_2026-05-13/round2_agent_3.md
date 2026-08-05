# Agent 3 — Round 2 (runtime / env-vars / launcher / distributed-init)

Source drafts read: `agent_1.md` (build/FFI), `agent_2.md` (Cray MPI +
Shifter), `agent_3.md` (mine, runtime), `agent_4.md` (docs/cohsex/CI).
All section line refs below cite the respective draft files.

---

## Section A — Cross-slice convergence

Defects that I and ≥1 other agent independently flagged. **Strongest tag**
= the worst of the per-agent tags. **Fix alignment** = whether our proposed
remedies agree (and if not, who I side with).

### A1. `--mpi=cray_shasta` hardwired with no validate-after-init
- Mine: §3, §4 [COMPAT-1]; agent 2: D2 [COMPAT], D20 [LOC-COST/FRAGILE]
  (the build-time and runtime copies disagree); agent 1: row in §3 vendor
  matrix (cluster portability).
- Strongest tag: **[COMPAT]**.
- Fix alignment: agent 2's B1 (unify into one `LORRAX_MPI_TYPE_DEFAULT`
  source) + my B7(d) (post-init `assert jax.process_count() ==
  proc_count`). These compose — table-driven knob + loud assertion. Adopt
  both.

### A2. `lxrun` hardcodes `-N 1`; `LORRAX_NNODES` is overlay-only
- Mine: §2e (overlay vs base); agent 2: D1 [COMPAT/FRAGILE], D17
  [COMPAT]; agent 4: D-8 [FRAGILE] (docs explicitly promise it works).
- Strongest tag: **[COMPAT]** (silently single-node on a port).
- Fix alignment: agent 2's B10 option (a) — port a stripped-down
  multi-node lxrun upstream — is the cleanest. Agent 4 would settle for
  option (b) (document). I side with agent 2: this is the single largest
  hidden ceiling in the base module and the overlay's mux for free-node
  selection is the *only* sandbox-specific part.

### A3. Bind-mount path strings `/lorrax_{nvhpc,phdf5,slate}` are load-bearing across ≥4 files
- Mine: §2d, [FRAGILE-7]-adjacent; agent 1: defect #7 [FRAGILE/LOC-COST],
  defect #13 [FRAGILE/COMPAT] (baked into `INSTALL_RPATH`); agent 2: D3
  [LOC-COST].
- Strongest tag: **[FRAGILE]+[COMPAT]** (RPATH).
- Fix alignment: agent 2's B5 (env-var-ize the three names) + agent 1's
  P5 (single MPI-stack contract file) cover the build side; runtime side
  needs the same vars threaded through 0.1.0.lua. Agreed.

### A4. `pip install -e .` does not build the FFI; `pyproject.toml` lacks `[build-system]`
- Mine: out of slice in round 1 but noted as a doctor-CLI consumer
  in §B1; agent 1: defect #2 [FRAGILE/LOC-COST] and P1; agent 4: D-36
  [FRAGILE] + B-5.
- Strongest tag: **[FRAGILE]**.
- Fix alignment: agent 1's P1 (wire scikit-build-core) and agent 4's B-5
  are the same proposal. Adopt verbatim. Runtime consequence in my slice:
  once pip-installable, `lorrax doctor` (my B1) becomes a console_script
  entry point with no extra wiring.

### A5. `liblorrax_ffi.so` and JAX-version contract is undocumented
- Mine: open question Q1 (which JAX runs in the container, given
  `jax[cuda13]>=0.9.0` pinned but image ships JAX 0.4/0.5 with CUDA 12);
  agent 1: defect #1 [FRAGILE/COMPAT], open Q4 (ABI stability across
  0.5→0.9); agent 4: D-13, D-14 [FRAGILE].
- Strongest tag: **[FRAGILE]**+**[COMPAT]**.
- Fix alignment: agent 1's P2 (`VENDOR_VERSIONS.txt` from CMake) +
  agent 4's B-2 (parser-generated docs). My env-var inventory feeds the
  same generator (`lorrax doctor` reads it; `VENDOR_VERSIONS.txt` is the
  build-time twin). Bundle as one artifact pipeline.

### A6. Coordinator port `12355` hardcoded
- Mine: [FRAGILE-3]; agent 2: D15 [FRAGILE], Q7 (theoretical collision).
- Strongest tag: **[FRAGILE]**.
- Fix alignment: I propose deriving the port from `$SLURM_JOBID` (mod
  range) so two concurrent jobs on the same node-0 cannot collide; agent
  2 suggests `random.randint(seed=$SLURM_JOBID)`. Same idea, mine is
  deterministic so reattach works. Adopt deterministic.

### A7. `LD_PRELOAD libmpi_gtl_cuda.so.0` set unconditionally, even by `lxshell`
- Mine: §3 row 3, [COMPAT-2] context; agent 2: D4 [COMPAT/FRAGILE], D13
  [FRAGILE].
- Strongest tag: **[COMPAT]**.
- Fix alignment: agent 2's B8 (gate PRELOAD and `MPICH_GPU_SUPPORT_ENABLED`
  on `LORRAX_MPI_TYPE`) is the right shape; my B7 doesn't address it.
  Adopt agent 2's B8 as-is.

### A8. `select_gpu.sh` assumes `SLURM_LOCALID`, silently coalesces ranks on missing var
- Mine: §3 row 6 ("not NERSC-only" — I called it portable); agent 1:
  defect #22 [COMPAT/FRAGILE]; agent 2: D7 [COMPAT/FRAGILE].
- Strongest tag: **[COMPAT]+[FRAGILE]**.
- Fix alignment: I underweighted this; agents 1 and 2 are right that PBS
  + PMIx clusters break silently. The fix is a 3-line fallback chain
  `SLURM_LOCALID || PMI_LOCAL_RANK || PMIX_LOCAL_RANK`, plus a hard error
  if none are set. **Revise my [LOC-COST-5] to reflect this.**

### A9. `HDF5_USE_FILE_LOCKING=FALSE` is a silent corruption hazard off-Lustre
- Mine: §3 row 1, [FRAGILE-mention], open Q7; agent 4: implied in §3
  bare-venv discussion.
- Strongest tag: **[FRAGILE]**.
- Fix alignment: my Q7 proposal — `stat -f` filesystem detect + only set
  on Lustre — is correct but adds Lua. Practical alternative: keep
  unconditional set, document loudly in PORTING.md (agent 4's B-3 area).
  Adopt: document first, detect later.

### A10. cohsex.in parser silently accepts unknown keys + 50% of keys are undocumented
- Mine: §2c table footnote — `~25 production env-var reads still active
  post-migration`, the env→cohsex migration is incomplete ([LOC-COST-1
  through 3]); agent 4: D-1, D-3, D-4, D-16 [FRAGILE], B-1 schema
  validation.
- Strongest tag: **[FRAGILE]**.
- Fix alignment: agent 4's B-1 (schema-validate + alias allow-list) is
  the right shape. My migration blitz (B2) lands keys that B-1 then
  validates. Sequence: my B2 first (or simultaneously), then agent 4's
  B-1 with the new keys already present.

### A11. The 15× duplicated distributed-init boilerplate / two parallel sentinels
- Mine: [FRAGILE-9] + B5; not flagged by others in detail, but agent
  4's D-26..D-29 (no CI for dist init) is the upstream consequence.
- Strongest tag: **[FRAGILE]**.
- Fix alignment: agent 4 B-4 (smoke-test CI for FFI + dist init) and my
  B5 (delete inlined dist-init dances) are complementary — collapse
  duplicates then test the one survivor. Adopt both, sequenced (collapse
  first).

### A12. `XLA_PYTHON_CLIENT_ALLOCATOR=platform` and `TF_GPU_ALLOCATOR=cuda_malloc_async` set simultaneously — **single-agent convergent finding**
- Mine alone: [FRAGILE-2], B8, Q1. None of the other agents flagged this.
  Per the round-2 instruction, the convergence here is between round-1
  me and the JAX docs.
- Web check (this round): the JAX GPU memory allocation page
  ([docs.jax.dev/en/latest/gpu_memory_allocation.html][a]) documents
  both vars but is **silent on precedence when both are set**. JAX
  discussion #6102 and issue #417 both describe them as alternate paths
  to the same XLA allocator switch. Reading the XLA source intent
  ([openxla.org gpu_architecture][b]) and Lambda's JAX-on-NVIDIA guide
  ([lambda.ai][c]), the practical resolution in current XLA:
  `XLA_PYTHON_CLIENT_ALLOCATOR` is the JAX-prefixed selector consumed
  by `xla::pjrt::gpu::CreateGpuAllocator`; `TF_GPU_ALLOCATOR` is a
  legacy TensorFlow-era var that XLA still honors as a fallback to
  pick `cuda_async`. **Order of resolution is not contractually
  guaranteed.** The danger: `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
  almost certainly wins in current XLA, which means our intent
  ("async pool from line 127-128 comment") is **not** the actual
  behavior. "Platform" is a per-cudaMalloc allocator — VERY slow, used
  for OOM debugging only.
- Strongest tag: **[FRAGILE]** with strong suspicion of correctness/perf
  impact.
- Fix alignment: my B8 (drop one, choose `cuda_malloc_async` to match
  intent, verify via JAX-startup log on a real run). Promoted into
  Section D below.

**Convergence count**: 12 (A1–A12). Five would also have been
convergent across two agents if you grant proximate framing (the env-vs-
cohsex story; the documentation-vs-modulefile drift on lxrun;
`KNOWN_SANDBOX_ERRORS.md` as the institutional-memory antipattern).
Twelve is enough; not pushing further.

---

## Section B — Disagreements

### B1. Whether the agent-overlay should graduate upstream

- **My round-1 position** (§4 [LOC-COST-4], open Q6): suggested the
  overlay's 645 LoC is "high LoC-cost for what amounts to picking a
  free node and tagging the step," with `<50 lines of Lua/bash`
  potentially sufficient. Hedged with "question for the author."
- **Agent 4** (D-33, B-7): proposes graduating `lxstatus / lxattach /
  lxreap` upstream (option a) OR documenting the gap (option b), leans
  option b because "no second user yet."
- **Agent 2** (B10): suggests the multi-node `lxrun` capability should
  upstream, agnostic on the pool coordination.
- **Who is right.** Agent 4. My round-1 framing was too dismissive of
  `lx_pool.py`. On re-reading `agent_4.md:580-595`, the relevant point
  is: the overlay solves a real concurrency problem that *will* arise
  the moment a second user touches the install, but graduating the full
  pool coordination now adds upstream maintenance burden for a feature
  with one user. Document the gap (B-7 option b) is correct.
- **Evidence resolving it.** The CONTEXT.md (§6 tagging convention) is
  explicit that LOC-COST is "exists only to cope with sandbox" — and
  `lx_pool.py` is *not* that: it solves the genuine multi-process /
  shared-allocation problem that any production deployment will hit.
  My tagging was incorrect. **Revise [LOC-COST-4]** (Section C).

### B2. Whether `in_container.sh` / `select_gpu.sh` are dead weight off-Shifter

- **My round-1 position** ([LOC-COST-5]): `in_container.sh` is dead
  weight on Apptainer; `select_gpu.sh` is portable.
- **Agent 1** (defect #24, P3): groups both into "Shifter-quirk
  workarounds, ~30 LoC, dead weight elsewhere."
- **Agent 2** (D6, D7, B4): wants to drop both, replace with `--env=`
  + `--gpu-bind=closest`/`--gpus-per-task=1`; explicitly flags Q2 — `--gpus-per-task=1`
  has historically broken JAX topology sync.
- **Who is right.** Agent 1 + my position partially. `select_gpu.sh` is
  the wrong target: agents 1/2 both flag (correctly, per A8 above) that
  `SLURM_LOCALID` is not even portable across Slurm-vs-PBS, so the
  script needs a portability *patch*, not deletion. `in_container.sh`
  is genuine dead-weight outside Shifter. Agent 2's B4 proposes deletion
  contingent on `--gpus-per-task=1` working — agent 2's own Q2 admits
  this hasn't been retested. **Recommendation: keep `select_gpu.sh` and
  fix its rank-resolution per A8; conditionalize `in_container.sh` on
  `LORRAX_CONTAINER_RUNTIME=shifter`.**

### B3. `LORRAX_NGPU=4` default

- **My round-1**: §3 row 7 — flagged as Perlmutter-specific (A100
  4-per-node).
- **Agent 2** (table row 12): same observation; explicitly notes
  Frontier=8 GCDs, Polaris=4 ok.
- **No real disagreement.** Both agents would site-config the default;
  difference is only in scope (mine: needs override on non-A100; agent
  2: agreed). No revision needed.

### B4. CI scope: weekly Perlmutter smoke vs CPU-only GitHub runner

- **My round-1**: B3 proposes unit-testing `init_jax_distributed()`
  with mocked SLURM env on CPU.
- **Agent 4** (B-4, Q-9): proposes weekly Perlmutter run via a NERSC
  GitHub-action-style runner with Shifter + GPU, with a CPU-only
  fallback for parser/import regressions.
- **Who is right.** Both, layered. Agent 4's Q-9 correctly flags that
  the runner infrastructure is the cost driver, not the test code.
  Concrete: my B3 (mocked CPU unit test of `init_jax_distributed`)
  is independent of any runner and should land first. Agent 4's B-4
  weekly cron is the layer above. No conflict.

### B5. Whether to schema-validate cohsex.in strictly or with warnings

- **Agent 4** (B-1, Q-1): explicitly leans strict + mass-rewrite of
  templates.
- **My position**: I didn't propose a policy in round 1, but my B2
  (migrate remaining env vars to cohsex.in) would *add* keys that
  agent 4's strict policy would then enforce. I support strict: this is
  the precedent set by 488e870/9fe5fde (loud cleanups), and the
  alternative ("user sets flag, nothing happens") is the exact
  discoverability hole those commits attacked.

### B6. Ranking divergence

Slight: I ranked B1 (`lorrax doctor`) first; agent 4 ranked B-1 (cohsex
schema-validate) + B-2 (regen docs) as the highest-leverage pair. Both
are right for their respective slices; combined Section D ranking below.

---

## Section C — My slice-specific updates

Each of my round-1 defects/blitz items, marked **CONFIRMED** /
**SHARPENED** / **REVISED**.

| Item | Verdict | Justification |
|---|---|---|
| [FRAGILE-1] `set_default_env` order-dependent | **CONFIRMED** | No other agent contradicted; agent 4 D-26 (no CI) means it could rot silently. |
| [FRAGILE-2] dual-allocator env vars | **SHARPENED** | Web check confirms JAX docs are silent on precedence; intent does not match likely behavior. Promote to Section D as B8 (now ranked higher). |
| [FRAGILE-3] coordinator port 12355 | **CONFIRMED**, fix sharpened | Agent 2 D15 + Q7 confirm. Sharpen fix: derive port deterministically from `$SLURM_JOBID` mod a 1024-port range so reattach works. |
| [FRAGILE-4] swallowed exception in dist-init | **CONFIRMED** | No conflict. |
| [FRAGILE-5] sentinel survives across `module unload/load` | **CONFIRMED** | No conflict. Sharpen: include `$SLURM_STEP_ID` and `$$` in the sentinel name. |
| [FRAGILE-6] `--immediate=10` in overlay lxrun | **CONFIRMED**, downgrade in priority | Overlay-only; not load-bearing for upstream. |
| [FRAGILE-7] dual Lustre-stripe naming schemes | **CONFIRMED** | Agent 4 D-1..D-4 doc-vs-code drift is the same pattern; my fix (standardize prefix) is the right shape. |
| [FRAGILE-8] `ISDF_JAX_CACHE_DIR` vs `JAX_COMPILATION_CACHE_DIR` | **CONFIRMED** | Agent 4 (D-39, two indices of truth) is the same drift pattern. |
| [FRAGILE-9] 15× duplicated dist-init | **CONFIRMED**, sharpened | Agent 4 D-29 (`tests/active/test_reshard_all_to_all.py` quarantined) is downstream evidence. Sharpen: a single grep-based CI check that fails when an inline `_LORRAX_JAX_DISTRIBUTED_DONE` lands outside `src/runtime/`. |
| [FRAGILE-10] `nccl_warmup` couples to 2D-mesh | **CONFIRMED** | Latent; no urgency. |
| [COMPAT-1] `--mpi=cray_shasta` | **CONFIRMED**, escalated | Agent 2 D2 + D20 (build/runtime split) is worse than I caught. Adopt agent 2 B1 alongside my B7(d). |
| [COMPAT-2] Shifter is NERSC-only | **CONFIRMED** | Agent 2 B3 (Apptainer.def companion) is the way through. |
| [COMPAT-3] `--overlap` Slurm-version | **CONFIRMED** | Niche. |
| [COMPAT-4] hardcoded `LORRAX_MPICH_LIB_DIR` | **CONFIRMED** | Subsumed by agent 1 P5. |
| [COMPAT-5] `lxalloc` bakes `m2651/interactive` in overlay | **CONFIRMED** | Sandbox-only. |
| [LOC-COST-1] 5 SC env vars not migrated | **CONFIRMED** | My B2. |
| [LOC-COST-2] 3 ISDF planner env vars | **CONFIRMED** | My B2. |
| [LOC-COST-3] 5 V_q tunables | **CONFIRMED** | My B2. |
| [LOC-COST-4] `lx_pool.py` 645 LoC overhead | **REVISED** | Per disagreement B1: the overlay solves a real concurrency problem; the 645 LoC is not "dead weight" but "feature with one user." Re-tag as a maintainability item, not a LOC-cost defect. Agent 4 B-7 option (b) (document the gap) is correct. |
| [LOC-COST-5] `select_gpu.sh` / `in_container.sh` | **REVISED** | I said `select_gpu.sh` was portable; agents 1/2 are right that it silently breaks on PBS/PMIx (per A8). Add a fallback chain. `in_container.sh` is conditional-on-Shifter, not unconditional dead weight. |
| [LOC-COST-6] unused `LORRAX_FFI_*_HOST` exports | **CONFIRMED** | Trivial. Drop with B4 (ENV_VARS.md). |
| [LOC-COST-7] inconsistent `LORRAX_*` / `ISDF_*` / etc. prefixes | **CONFIRMED** | Subsumed by B2 + B4. |
| B1 `lorrax doctor` CLI | **SHARPENED** | Per A4: becomes a `console_script` entry point once agent 1's P1 lands. Add a `--check-allocator-actual` mode that scrapes the JAX-startup log to resolve [FRAGILE-2] empirically (closes my Q1). |
| B2 migrate remaining env vars | **CONFIRMED** | Sequence: before agent 4's B-1 strict schema validation. |
| B3 unit-test `init_jax_distributed()` | **CONFIRMED** | Mocked, CPU-only. Independent of any runner setup. |
| B4 ENV_VARS.md | **CONFIRMED** | Subsumes my [LOC-COST-6, 7]. |
| B5 kill duplicated dist-init boilerplate | **CONFIRMED** | Sequenced before agent 4's B-4 weekly cron. |
| B6 upstream `lxstatus` only | **REVISED → DROP / FOLD** | Per B1 disagreement: agent 4 B-7 option (b) (document gap) is the right call. My B6 collapses into agent 4 B-7. |
| B7 fail-fast wrappers on runtime | **CONFIRMED**, sharpened | (d) the `assert jax.process_count() == proc_count` post-init catches the wrong-`--mpi=` singleton-init footgun (A1). Land as warning first, then promote to fatal after one release. |
| B8 resolve dual-allocator env vars | **SHARPENED**, promoted | Per A12: must verify empirically. Becomes Section D #2. |

---

## Section D — Cross-cutting blitz proposals (top 5, ranked)

Each item: acceptance criteria, dependencies, **desk vs real-run**.
**I commit to this ranking.**

### D1. Wire `pip install -e .` to the FFI build (= agent 1 P1 + agent 4 B-5)

- **Axis.** Installability.
- **Acceptance.** `[build-system]` block in `pyproject.toml` declares
  `scikit-build-core>=0.11.0`; `pip install -e .` inside a Shifter
  shell produces `liblorrax_ffi.so` in the expected site-packages
  location *or* the build dir; `python -c "from ffi.common import
  ffi_loader; ffi_loader.get_lib()"` succeeds without `LORRAX_FFI_SO`
  set. CI lock: `tests/build_smoke/test_pip_install.py` with a CMake
  stub.
- **Dependencies.** None (foundational). Unblocks D2, D5.
- **Classification.** **Desk** for the scikit-build-core wiring; **real
  run** for the in-container build smoke test. The pyproject edits can
  be drafted and reviewed without an allocation.
- **Why first.** This is the single change that turns "obvious pip flow
  is a silent no-op" into "obvious flow works or fails loudly."

### D2. Schema-validate cohsex.in + generate COHSEX_INPUT.md from parser (= agent 4 B-1 + B-2), with my env→cohsex migration (= my B2) landed first

- **Axis.** Maintainability (primary); installability (via discoverable
  config surface).
- **Acceptance.** Single `_FIELDS` table in `src/gw/_cohsex_schema.py`
  is the source of truth; parser raises on unknown keys (with a small
  `_DEPRECATED_KEYS` allow-list that emits `DeprecationWarning`);
  COHSEX_INPUT.md regenerated by `tools/gen_cohsex_input_md.py`;
  diff-CI test guards drift. All 13 keys identified in my round-1
  [LOC-COST-1..3] (5 SC + 3 ISDF planner + 5 V_q) are in the schema.
- **Dependencies.** Land my B2 (env→cohsex migration) *first or
  simultaneously* so the new keys are present when strict validation
  turns on. Otherwise the first run of strict mode rejects existing
  cohsex.in files for the wrong reason.
- **Classification.** **Desk.** No allocation needed; parser tests run
  on CPU.

### D3. Resolve `XLA_PYTHON_CLIENT_ALLOCATOR` vs `TF_GPU_ALLOCATOR` (= my B8, promoted)

- **Axis.** Maintainability + **correctness** (currently unknown
  whether the running allocator matches author intent).
- **Acceptance.** (i) On a real Perlmutter `lxrun`, scrape XLA's
  startup log to record which allocator is active (printed via
  `XLA_FLAGS=--xla_dump_to=…` or by `JAX_DEBUG_LOG_MODULES=jax`).
  (ii) Drop whichever env var is *not* in effect from `0.1.0.lua`.
  (iii) If the active allocator is `platform`, this is a latent
  performance bug — switch deliberately to `cuda_malloc_async` (or
  the JAX-prefixed `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`,
  preferred since it's the actively supported name) and re-measure
  GW driver iteration time to confirm.
- **Dependencies.** Needs one real `lxrun` allocation (~30 minutes).
  Otherwise standalone.
- **Classification.** **Real-run required.** The JAX docs
  ([docs.jax.dev][a]) do not specify precedence; the JAX issue tracker
  ([jax-ml/jax#417, #19035][d]) discusses both vars as alternates
  without precedence guarantees. Cannot resolve at the desk.
- **Why high.** If the platform allocator is winning, every GW run
  since the modulefile was written has been running with the
  debug/slow allocator. Worst case: real perf left on the table; best
  case: a one-line modulefile cleanup. Either way the *uncertainty*
  is the bug.

### D4. Multi-node `lxrun` upstream + documented overlay split (= agent 2 B10(a) + agent 4 B-7(b))

- **Axis.** Installability (a porter's first multi-node attempt
  works) + maintainability (overlay scope clarified).
- **Acceptance.** Base `0.1.0.lua` `lxrun` reads `LORRAX_NNODES`
  (default 1), passes through `-N $LORRAX_NNODES`; the overlay's
  free-node-pool logic stays in the sandbox but `config/README.md`
  gains a one-paragraph "if you have multiple concurrent users on
  one allocation, you need the sandbox `lorrax_agent` overlay" note;
  `docs/ENVIRONMENT_COMPREHENSIVE.md:322` example actually works.
- **Dependencies.** None. Independent.
- **Classification.** **Desk** for the modulefile patch and doc;
  **real run** (≥2 nodes) to validate end-to-end.

### D5. Fail-fast runtime contract + unit-test it on CPU (= my B7 + B3 + agent 4 B-4 CPU layer)

- **Axis.** Maintainability.
- **Acceptance.** `src/runtime/__init__.py` gains: (a) `assert "jax"
  not in sys.modules` at top of `set_default_env`, (b) `warnings.warn`
  in the first-attempt exception swallow, (c) sentinel name includes
  `$SLURM_STEP_ID` and `$$`, (d) coordinator port derived
  deterministically from `$SLURM_JOBID`, (e) post-init `if
  jax.process_count() != proc_count: warn` (promote to assert after
  one release). New `src/runtime/test_init.py` exercises (a)–(d) with
  `unittest.mock`; collapses the 15× duplicated boilerplate (= my B5).
  CI lock: `grep -L 'from runtime import init_jax_distributed'
  src/common/*_test.py` fails CI on a new inlined copy.
- **Dependencies.** None for the CPU unit tests. (e) needs a real
  multi-rank allocation for promotion to `assert`, but lands as `warn`
  immediately.
- **Classification.** **Desk** for all of (a)-(d) and the test/CI
  hooks; **real run** for promoting (e) to fatal.

### Ranking commitment

1. **D1** — `pip install -e .` builds FFI.
2. **D2** — cohsex schema + generated doc + env→cohsex migration.
3. **D3** — allocator conflict resolution (correctness uncertainty).
4. **D4** — multi-node lxrun upstream.
5. **D5** — runtime contract assertions + dist-init test + collapse
   duplicates.

Rationale: D1 is foundational and unblocks the "second user" persona
that all the other defects implicitly assume away. D2 closes the
discoverability hole the author was already attacking. D3 is the only
**possible correctness issue** in my slice — if I'm right and `platform`
is winning, every benchmark and perf report is mismeasured. D4 closes
the largest hidden ceiling in the base module. D5 retires ~150 LoC of
duplicate dist-init code and adds the CI scaffolding agent 4's B-4
weekly cron will eventually layer on top of.

Items deliberately not in top-5: agent 2's Apptainer.def (B3) — high
leverage but >1 day and not the bottleneck today; agent 1's
VENDOR_VERSIONS.txt (P2) — useful but covered partially by D2's
generated-docs precedent; agent 4's QUICKSTART.md (B-3) — would be
top-3 if a second user were imminent. Defer.

---

## Section E — Open questions only the user can resolve

1. **[from A12 / D3]** What allocator is *actually* active on a current
   Perlmutter `lxrun` — `platform` (slow/debug) or `cuda_malloc_async`?
   30-minute test: `lxrun python3 -c "import jax;
   print(jax.devices()); print(jax.live_arrays())"` with
   `JAX_DEBUG_LOG_MODULES=jax`, scrape startup banner. If you've
   already noticed perf regressions in mid-2026 you can't account for,
   this is a candidate root cause.

2. **[from B1 disagreement / D4]** Is there a second user (real or
   planned in the next 6 months)? Agent 4's B-7 graduation question
   only matters if so. If "no second user this year," document the
   gap (option b) and move on. If yes, the overlay's multi-user
   coordination needs an upstream story.

3. **[from agent 1 Q1-Q3]** Has LORRAX ever run on a non-NERSC
   cluster? If yes, which one and what broke? If no, PORTING.md is
   *aspirational*, which is fine but should be labeled — affects
   whether D1 + D4 are sufficient or whether a real port is needed
   as a forcing function.

4. **[from agent 4 Q-4 / my Q1]** What's the JAX version actually
   running inside the Shifter image? `pyproject.toml` says
   `jax[cuda13]>=0.9.0`; the image
   `nvcr.io/nvidia/jax:25.04-py3` ships an older JAX with CUDA 12. Is
   the host JAX bind-mounted over via `LORRAX_SITE`, and if so does
   that mean the in-container JAX version is whatever you `uv sync`'d
   most recently? A 30-second `python -c "import jax;
   print(jax.__version__, jax.devices())"` inside an `lxshell`
   resolves it.

5. **[from my round-1 Q4]** Is `JAX_COMPILATION_CACHE_DIR` ever
   cleaned, or is it growing unbounded under
   `$SCRATCH/.jax_cache` across A/B/C/D agent variants and code
   states? If it's hit a meaningful fraction of scratch, a cleanup
   blitz belongs in the rotation.

6. **[from my round-1 Q5]** Do `LORRAX_FFI_NVHPC_HOST` /
   `_PHDF5_HOST` / `_SLATE_HOST` exports actually get consumed by
   anything (external scripts, debug habits)? If not, drop them in
   D2's housekeeping pass.

7. **[from agent 4 Q-6]** Has anyone other than you ever run
   `module load lorrax && lxrun python -m gw.gw_jax` on Perlmutter?
   If "no — every install/run has been from your shell," then the
   modulefile may have undocumented assumptions about your shell
   state (PATH, prior `uv sync` artifacts, etc.) that a second user
   on Perlmutter would hit. D1's pip-install smoke would catch
   most of these; worth knowing.

8. **[from agent 4 Q-1 / D2]** For cohsex.in strict-mode rollout,
   how many historical run dirs under `runs/**/cohsex.in` would
   need rewrites? If <50, mass-rewrite is fine; if hundreds, we
   need a separate `lorrax migrate-cohsex` tool before flipping
   strict.

---

`Agent 3 round 2 done — see round2_agent_3.md`

[a]: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
[b]: https://openxla.org/xla/gpu_architecture
[c]: https://lambda.ai/blog/pytorch-to-jax-on-lambda-for-enterprise-ml
[d]: https://github.com/jax-ml/jax/issues/417
