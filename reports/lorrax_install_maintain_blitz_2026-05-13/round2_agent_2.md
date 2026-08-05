# Round 2 — Agent 2 reconciliation (Cray MPI / Shifter slice)

Tie-breaking weight assumed strongest on MPI / container / distributed-init
issues, per the round-2 brief. On build-system internals (Agent 1) and
docs / cohsex.in surface (Agent 4) I record disagreements without
pretending to resolve them.

References below use the shorthand A1 / A2 / A3 / A4 = round-1 agent 1-4
drafts; defect IDs are quoted from their tables.

---

## Section A — Cross-slice convergence

Items flagged by me (A2) AND at least one of A1 / A3 / A4. Tags below are
the strongest assigned by any flagger.

1. **`--mpi=cray_shasta` is the wrong default off-Cray + lives in two
   files** — A2 D2/D20, A1 #5+#20+P5, A3 COMPAT-1.
   Strongest tag: **[COMPAT]**.
   All three agree on the *fix shape* (single source-of-truth file, both
   `run_shifter.sh` and `0.1.0.lua` read it). A1's P5 makes the file
   CMake-readable too; A3 wants a runtime "assert process_count ==
   SLURM_NTASKS" sanity check after init. Combine. No real divergence.

2. **`MPICH_GPU_SUPPORT_ENABLED=1` + `libmpi_gtl_cuda.so.0` LD_PRELOAD
   are Cray-only and set unconditionally** — A2 D4/D5/D6/D13, A1
   #24+#29+#26, A3 COMPAT-2.
   Strongest tag: **[COMPAT]** (also [FRAGILE] under A2 D4).
   Convergent fix: gate both on `LORRAX_MPI_TYPE` content (A2 B8). A1
   wants the gating done at CMake `WARNING→FATAL_ERROR` time
   ("strict mode," P3); A3 wants it surfaced by a `lorrax doctor`
   command (A3 B1). These layer cleanly: gate at the modulefile, verify
   with doctor, fail loud in strict-build.

3. **Hardcoded container mount paths `/lorrax_{nvhpc,phdf5,slate}`** —
   A2 D3, A1 #7+#13+#25.
   Strongest tag: **[FRAGILE][LOC-COST]**.
   A2 wants three env-controlled paths (B5). A1 #13 calls out that the
   bind-target leaks into the .so via `INSTALL_RPATH` — i.e. you can't
   *just* change the modulefile, the .so embeds it. A1 is right and
   sharpens A2 B5: the env-var approach needs a CMake-side echo + a
   regen-RPATH path. Agreement on direction.

4. **`-N 1` hardcoded in base lxrun; multi-node only works via sandbox
   overlay** — A2 D1+D17, A4 D-8.
   Strongest tag: **[COMPAT][FRAGILE]**.
   A4 surfaces the matching doc lie
   (`ENVIRONMENT_COMPREHENSIVE.md:322` shows
   `LORRAX_NNODES=2 LORRAX_NGPU=8 lxrun …` as if it works). My B10
   proposed either wire it or document; A4 escalates because the doc
   *already promises* it works — so "document the gap" is no longer an
   option, you have to ship it. A4 is right, B10 should be "wire it in
   base lxrun".

5. **No `[build-system]` table in `pyproject.toml`; `pip install -e .`
   is a silent no-op for the FFI** — A1 #2+P1, A4 D-36+B-5.
   Strongest tag: **[FRAGILE][LOC-COST]**.
   Both propose scikit-build-core. A1 owns the implementation detail;
   A4 owns the doc consequence. No divergence.

6. **`jax[cuda13]>=0.9.0` pin vs `nvcr.io/nvidia/jax:25.04-py3`
   container (CUDA 12, JAX 0.5.x)** — A1 #1, A4 D-13+D-14+Q-4.
   Strongest tag: **[FRAGILE][COMPAT]**.
   Both say "pyproject and container disagree." A4 Q-4 is the cleanest
   formulation: nobody actually knows what JAX runs at runtime, because
   `LORRAX_SITE` bind-mounts may override the in-image JAX. A 30-second
   `python -c "import jax; print(jax.__version__)"` resolves it; until
   then both agents agree the pin is at best aspirational. Outside my
   slice; record as convergent and defer to A1's P1.

7. **`HDF5_USE_FILE_LOCKING=FALSE` is correct on Lustre, dangerous on
   NFS / non-Lustre** — A3 NERSC-isms table + Q7, A4 §3 (notes
   `gspace_mode=host_cache` related host-RAM assumption, similar
   silent-on-other footgun class). I (A2) did not flag this in my D-list
   but it surfaced in my §2 inventory.
   Strongest tag: **[COMPAT]** (silent-corruption hazard off-Lustre).
   A3 Q7 proposes `stat -f -c %T` detection of Lustre before setting.
   Agree.

8. **JAX coordinator port `12355` hardcoded** — A2 D15, A3 FRAGILE-3.
   Strongest tag: **[FRAGILE]**.
   Both flag it as theoretical-today / real-tomorrow once shared
   allocations land. A3 Q7 (in A3, not A2's Q7) is right that it's
   probably masked by Cray PMI's no-args path. A3's B7(c) "include
   `SLURM_STEP_ID` in the re-entry sentinel" is adjacent but doesn't
   fix the port collision. Both agents want a random port seeded by
   `$SLURM_JOBID`; neither commits a blitz to it.

9. **Stage scripts (`stage_cray.sh`, `stage_openmpi.sh`,
   `stage_pypi.sh`) are pure NERSC-bind-mount-restriction
   workarounds** — A2 D8, A1 #25.
   Strongest tag: **[LOC-COST]**.
   ~250 LoC across three scripts that an Apptainer port wouldn't need.
   Both agents agree they survive the port but as a different shape
   (Apptainer can bind `/opt/cray/pe` directly, so stage_cray's
   purpose disappears). No divergence.

10. **`stage_cray.sh` libreadline.so.7 → .so.8 ABI shim** — A2 D9,
    A1 #14.
    Strongest tag: **[FRAGILE]**.
    A1 phrases it more sharply: assumes minor-version compat. Agreement.

11. **`stage_cray.sh` phdf5 SONAME shim layer
    (`libmpi_gnu_{91,110,123}.so.12`)** — A1 #15+#26. I (A2) folded
    this into the broader Cray-PE SONAME concern in D8/D20; A1's flag
    is sharper.
    Strongest tag: **[FRAGILE][LOC-COST]**.
    Convergent: NERSC compiler bumps silently desync the shim list.
    Acknowledge A1 owns the better formulation.

12. **`/opt/hpcx/ompi/lib` silent CMake fallback** — A2 D10, A1 §2.3
    discussion + #6 [FRAGILE].
    Strongest tag: **[FRAGILE]**.
    A1 is closer to the build internals: the `LORRAX_FFI_ALLOW_DEFAULT_MPI=1`
    bypass on `build.sh:35-48` is the underlying flaw — guard is
    bypassable and only protects build.sh's entry point. Cross-validated.

13. **PORTING.md doesn't say `run_shifter.sh` is the required build
    wrapper, never mentions multi-node ceiling** — A1 #30+#31, A2 D17,
    A4 D-7+D-8 (lxkill, multi-node).
    Strongest tag: **[FRAGILE]** (doc-vs-code drift).
    Three agents converge — PORTING.md is the document, not the code,
    that lies. A4's B-2 (regen COHSEX_INPUT.md from parser) is the same
    pattern applied to a different doc.

14. **5 SC + 3 ISDF + 5 V_q env vars not yet migrated to cohsex.in** —
    A3 LOC-COST-1/2/3 + B2, A4 D-3.
    Strongest tag: **[LOC-COST]** (continues 488e870 / 9fe5fde trend).
    Both agree on the migration. Outside my slice but trivially
    convergent.

15. **Duplicate Lustre stripe env var schemes (`LORRAX_LUSTRE_*` shell,
    `LORRAX_PHDF5_*` Python)** — A3 FRAGILE-7. A2 §2 noted both names
    in the env-var inventory; did not flag as a defect — A3 sharpens.
    Strongest tag: **[FRAGILE]**.
    Acknowledge as convergent, A3 owns.

16. **No FFI / Shifter / distributed-init smoke test in CI** — A4
    D-26/27/28/29, A1 P4 (out-of-container build smoke).
    Strongest tag: **[FRAGILE]**.
    Two different smoke-test proposals: A1's is "CMake configure on a
    host without Shifter" (build-side), A4's is "import .so and run
    a Cholesky" (runtime-side). They're complementary, not competing.
    No divergence — both ship together as one CI workflow.

17. **`select_gpu.sh` + `in_container.sh` exist only to undo Shifter
    quirks** — A2 D6, A1 #24, A3 LOC-COST-5.
    Strongest tag: **[LOC-COST]**.
    The three agents agree on the *cost*. They disagree on the
    *portability*: see Section B item 1.

18. **`LORRAX_SLATE_INSTALL_DIR` defaults to jackm's `$HOME`** — A1
    #16, A2 §3 row 9, A3 (mentions $HOME in coordinator fallback
    chain, different concern). A1 owns the sharpest version.
    Strongest tag: **[COMPAT][LOC-COST]**.
    Defaulted at `CMakeLists.txt:348`; a second user gets
    `slate_FOUND=FALSE` silently. Agreement.

---

## Section B — Disagreements

Real conflicts (not just non-overlap). My slice (MPI/container/runtime
init) is where I commit; build/docs disagreements I flag and defer.

### B-1. `select_gpu.sh` portability tag — A3 says portable, A1/A2 say [COMPAT].

A3 §3 NERSC-isms table: *"select_gpu.sh — Generic SLURM convention, not
NERSC-only. Works on any cluster with SLURM task plugin."*
A1 #22 + A2 D7: tag [COMPAT][FRAGILE], rationale "PBS systems (Polaris)
don't set `SLURM_LOCALID`; falls back to 0 → all ranks pinned to GPU 0
silently."

**Who's right.** A3 is right *for the SLURM-only universe* (Frontier,
Leonardo, NERSC, most university clusters). A1/A2 are right *for the
PBS / non-SLURM universe* (Polaris uses PBS-Pro; LSF on a few legacy
sites). Both correct under different scoping.

**Resolution.** The script is portable to any SLURM cluster — that
covers ≥80% of "second-user clusters." The PBS-fallback case is a real
issue but the cheap fix is one line:
`export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-${PMI_LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}`.
That makes it portable to PBS+pmix and OpenMPI bare-launch too.

**Tag correction.** Downgrade A2 D7 from **[COMPAT][FRAGILE]** to
**[FRAGILE]** (silent fallback to GPU 0 is the real defect; the
PBS-specific framing inflates it). A1's #22 should also be downgraded
on the same grounds. Per the brief: this is the tag inflation A2 was
asked to correct. **A3 is right on the framing; A1/A2 are right on the
substance of the silent-fallback bug.**

**Evidence to resolve fully.** Run `printenv | grep -E
'PMI_|SLURM_|OMPI_'` inside an `lxrun` step on Perlmutter and inside
a PBS-launched srun-equivalent on Polaris. Until somebody does this,
the cheap one-line fallback chain costs nothing and closes both
framings.

### B-2. How urgent is the Apptainer port?

A2 B3 ranks Apptainer companion as **risk-high, leverage-high, position
#7 of 10**. A1 (no equivalent blitz, just notes the gap in #20).
A4 B-7 only addresses overlay-coordination, not Apptainer.
A3 doesn't propose it.

**My read.** I'm overweight the Apptainer rewrite as a "1-day blitz"
proposal because porting to Frontier/Polaris is the canonical
"installable by a second user on a non-NERSC cluster" question. But
*one day is not enough* — A1's Q1 confirms this (the SONAME shim has
to be revisited under Apptainer's bind ordering). The honest version
is: it's a 5-10 day project, not a blitz, and the blitz framing
oversells it.

**Resolution.** Demote A2 B3 to "out of blitz scope; pre-requisite is
A2 B1 + A1 P5 (unified MPI-stack contract) and A1 P4 (out-of-container
CMake smoke test)." Don't pretend it's a 1-day item.

### B-3. Strict-mode build (A1 P3) vs. doctor-tool runtime audit (A3 B1) — overlap?

A1 P3 makes CMake `WARNING → FATAL_ERROR` on the `LORRAX_BUILD_STRICT=1`
toggle. A3 B1 adds a runtime `lorrax doctor` CLI that audits env vars
and config at runtime. Same intent (catch silent-fallback drift),
different layer.

**Who's right.** Both. They check different invariants — the build-side
catches "no SLATE found / wrong NVHPC subdir / HPC-X fallback" at
configure-time; the runtime-side catches "wrong `LD_PRELOAD` /
`MPICH_GPU_SUPPORT_ENABLED` unset / `XLA_PYTHON_CLIENT_ALLOCATOR` vs
`TF_GPU_ALLOCATOR` collision / mesh-count != SLURM_NTASKS." Neither
subsumes the other; ship both, doctor first because runtime is where
the porter spends time.

### B-4. Apptainer / Frontier hybrid SONAME interaction (A1 Q1).

A1's Q1: under Apptainer's `--bind /opt/cray/pe:/opt/cray/pe`,
the libmpi SONAME-shim layer in `phdf5/scripts/stage_cray.sh:84-86` may
not be on `LD_LIBRARY_PATH` early enough to intercept. A2 (me) did not
flag this and I have no Apptainer-on-HPE-Cray data. **Defer to A1.**
This is exactly the kind of "cannot resolve without a live cluster"
question that the round-1 prompts told us to keep in §6.

### B-5. `XLA_PYTHON_CLIENT_ALLOCATOR=platform` vs
`TF_GPU_ALLOCATOR=cuda_malloc_async` collision (A3 FRAGILE-2 / B8 / Q1).

A3 flags that the two vars are mutually exclusive and one wins silently.
A1/A2/A4 didn't catch this — it's purely A3's slice. I confirm A3 is
right after a re-read of `0.1.0.lua:129-131`: both are set, both target
the XLA GPU allocator, and per JAX 0.6 docs
([JAX GPU memory](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)),
`XLA_PYTHON_CLIENT_ALLOCATOR` is selected first. **A3 is right;
nobody else looked**. Verify on a live run before deleting the loser.

### B-6. Should `lorrax_agent` overlay graduate upstream?

A2 (me, §3 NERSC-isms table) said the overlay is out of scope.
A3 LOC-COST-4 questions whether the 645 lines of `lx_pool.py` should
exist at all.
A4 B-7 + D-33 proposes a partial graduation (lxstatus only).

**My read** (and tie-breaker weight goes to A2 on this since it's a
launcher/MPI question): A4 B-7 option (a) — promote `lxstatus` only,
keep multi-agent pool logic in the sandbox — is the right call. The
pool logic is a sandbox-specific A/B/C/D agent-fleet pattern; a
second user with one allocation doesn't need it. `lxstatus` is
universally useful. A3's "delete the whole 645 lines" framing is
overreach; A4 B-7 is right.

---

## Section C — A2 slice updates

For each Agent-2 round-1 defect / blitz, marked
CONFIRMED / SHARPENED / REVISED with one-line justification.

### Defects

| # | Status | Note |
|---|--------|------|
| D1  (lxrun -N 1 hardcoded) | **SHARPENED** | A4 D-8 exposed the doc lie at `ENVIRONMENT_COMPREHENSIVE.md:322`; "wire it" now obligatory, not optional |
| D2  (cray_shasta in two files) | **CONFIRMED** | A1 P5 + A3 COMPAT-1 echo; A1 sharpens it to CMake-side too |
| D3  (hardcoded `/lorrax_*` paths) | **SHARPENED** | A1 #13 reveals `INSTALL_RPATH` embeds the path in the .so → my B5 needs to regen RPATH too |
| D4  (LD_PRELOAD GTL unconditional) | **CONFIRMED** | A1 #29, A3 COMPAT-2 echo |
| D5  (MPICH_GPU_SUPPORT_ENABLED no guard) | **CONFIRMED** | A1 #24, A3 COMPAT-2 echo |
| D6  (in_container.sh exists only for Shifter quirk) | **CONFIRMED** | A1 #24, A3 LOC-COST-5 |
| D7  (select_gpu.sh SLURM_LOCALID) | **REVISED** | Demote [COMPAT]→[FRAGILE]; A3 is right that it's not NERSC-specific. Bug is the silent fallback to 0, not the SLURM-ness. See §B-1 |
| D8  (stage_cray exists only because of Shifter `--volume` restriction) | **CONFIRMED** | A1 #25 |
| D9  (libreadline.so.7→.so.8 shim) | **CONFIRMED** | A1 #14 echoes |
| D10 (HPC-X OpenMPI silent fallback in CMakeLists) | **CONFIRMED** | A1 §2.3 + #6 sharpens it: bypass is `LORRAX_FFI_ALLOW_DEFAULT_MPI=1`, not just absence |
| D11 (image not digest-pinned) | **CONFIRMED** | Nobody else flagged; A2-only |
| D12 (supplemental site-packages built by hand) | **CONFIRMED** | A4 D-37/D-38 has overlap (mkdocs deps); my D12 is about `isdf_site` venv, distinct concern |
| D13 (lxshell LD_PRELOAD stays set) | **CONFIRMED** | A2-only; minor |
| D14 (`family("lorrax")` swap is single-shell) | **CONFIRMED** | A2-only; sandbox-A/B/C/D concern, A3 LOC-COST-4 adjacent |
| D15 (coordinator port 12355 hardcoded) | **SHARPENED** | A3 FRAGILE-3 + Q7 cover the same; convergent. Fix: random port seeded by `$SLURM_JOBID`, not just include step-id in sentinel |
| D16 (no `lxrun --dry-run`) | **CONFIRMED** | A2-only; B2 still the cheapest win |
| D17 (PORTING.md doesn't mention multi-node base path) | **CONFIRMED** | A4 D-7+D-8 echo |
| D18 (`LORRAX_DARSHAN_LIB_DIR` no probe) | **CONFIRMED** | A2-only; minor |
| D19 (`salloc … sleep 100000`) | **CONFIRMED** | A1 §3 table notes it's Slurm-specific. Minor |
| D20 (MPI_TYPE default split) | **CONFIRMED** | Subsumed by A1 P5 + A2 B1 unified-contract proposal |
| D21 (NCCL multi-node env not set) | **CONFIRMED** | Nobody else flagged at this depth; A3 mentions NCCL warmup but not the missing OFI plugin env. A2-strongest |
| D22 (`/dep` mpich subpath blind append) | **CONFIRMED** | A2-only; minor |
| D23 (SLATE Comm_dup assumes WORLD) | **CONFIRMED** | A2-only; future-tense |
| D24 (cuSOLVERMp NCCL UID broadcast key) | **CONFIRMED** | A2-only; future-tense |

### Blitz proposals

| ID | Status | Note |
|----|--------|------|
| B1 (unify MPI-type knob) | **CONFIRMED** | A1 P5 proposes the same with CMake-readable file. Combine into one blitz |
| B2 (lxrun --dry-run) | **CONFIRMED** | Still cheapest unlock; no echo, A2-original |
| B3 (Apptainer companion) | **REVISED** | Cannot be a 1-day blitz. Demote to "10-day project, prerequisite: B1 + A1 P4." See §B-2 |
| B4 (drop in_container.sh + select_gpu.sh) | **REVISED** | A3's stance (select_gpu is portable) corrects me. Drop "drop both" framing; keep "drop in_container.sh under Apptainer" only, add the PMI_LOCAL_RANK fallback to select_gpu.sh per §B-1 |
| B5 (configurable container mount paths) | **SHARPENED** | A1 #13 surfaces that `INSTALL_RPATH` embeds the path in the .so; B5 must also include the RPATH update path, not just modulefile + CMake env-read |
| B6 (digest-pin image) | **CONFIRMED** | Still contingent on Shifter feature support (my Q1). Low priority |
| B7 (requirements.txt for site-packages) | **CONFIRMED** | A4 didn't flag; A2-only. Could fold into A1 P1's pyproject overhaul (declare as `[dependency-groups.runtime-site]`) |
| B8 (guard LD_PRELOAD + PMI by LORRAX_MPI_TYPE) | **CONFIRMED** | A1 P3 strict-mode partly subsumes (build-time); B8 covers the modulefile-time gate. Both ship |
| B9 (print_mounts.sh) | **CONFIRMED** | Subsumed by A3 B1 `lorrax doctor`. Roll into doctor; don't ship as separate blitz |
| B10 (multi-node base or doc) | **SHARPENED** | A4 D-8: doc-only is no longer an option (ENV_COMPREHENSIVE already lies). Must wire it |

---

## Section D — Cross-cutting top 5 blitz proposals

Hard ranking, no hedging. Each item assumes ~1 day unless flagged.

### #1 — Wire `[build-system]` to scikit-build-core; reconcile JAX pin against container

- **Proposers.** A1 P1, A4 B-5, A4 D-36, A1 #1.
- **Acceptance criteria.**
  - `pyproject.toml` declares `[build-system] requires = [...]` and
    `build-backend = "scikit_build_core.build"`.
  - `pip install -e .` inside the container produces a working
    `liblorrax_ffi.so` at the expected import path.
  - `jax[cuda12]` extra (or split cuda12/cuda13) replaces the broken
    `jax[cuda13]>=0.9.0` line; PORTING.md table matches.
  - A new `tests/build_smoke/test_pip_install.py` runs in CI.
- **Closes most directly.** A1 #2 (silent no-op pip install).
- **Dependencies on this list.** None. Independent.
- **Desk-doable or live-cluster?** Desk-doable for the
  pyproject/CMake refactor; **needs a live Perlmutter Shifter shell**
  to validate the install actually links and imports. Half desk, half
  cluster.

### #2 — Cohsex.in schema validation + autogenerate `COHSEX_INPUT.md` + finish env→cohsex migration

- **Proposers.** A4 B-1, A4 B-2, A3 B2.
- **Acceptance criteria.**
  - Single `_SCHEMA` table in `src/gw/gw_config.py` carries
    `(name, type, default, one-line-doc)` for every cohsex.in key.
  - Unknown keys produce a warning (or error under
    `LORRAX_COHSEX_STRICT=1`); deprecated keys (`output_file`,
    `use_chunked_isdf`, etc.) hit an explicit allow-list.
  - `tools/gen_cohsex_input_md.py` regenerates `docs/COHSEX_INPUT.md`
    from the schema; pytest diffs generated-vs-committed.
  - 5 SC + 3 ISDF + 5 V_q env vars from A3 LOC-COST-1/2/3 migrated to
    cohsex.in keys with documented defaults.
  - `COHSEX_INPUT.md` moves from sandbox into `lorrax_C/docs/`.
- **Closes most directly.** A4 D-1..D-6, D-16, D-30, D-32.
- **Dependencies.** None within this list. Independent of #1.
- **Desk-doable.** Fully desk-doable. No cluster needed.

### #3 — `lorrax doctor` runtime audit CLI + `lxrun --dry-run`

- **Proposers.** A3 B1 (doctor), A2 B2 (dry-run), A2 B9 (mounts —
  rolled in).
- **Acceptance criteria.**
  - `lorrax doctor` reads every env var the runtime consumes, prints
    green/yellow/red status with remediation hint per finding.
  - Specific checks (covered by my §A items 2, 8 and §B-5):
    `MPICH_GPU_SUPPORT_ENABLED` present; `LD_PRELOAD` present *and*
    matches `LORRAX_MPI_TYPE`; `XLA_PYTHON_CLIENT_ALLOCATOR` ↔
    `TF_GPU_ALLOCATOR` collision detection; `HDF5_USE_FILE_LOCKING`
    sane for FS type via `stat -f -c %T`; post-init
    `jax.process_count() == SLURM_NTASKS`.
  - `lxrun --dry-run` (or `lxprint`) emits the fully materialised
    srun + shifter argv + env injections to stdout, exits 0.
  - One smoke test: `lorrax doctor --help` works in CPU-only mode.
- **Closes most directly.** A2 D16 (no dry-run); A3 FRAGILE-1/2/3/4/5/7/8.
- **Dependencies.** Independent of #1 and #2 in principle; rides
  better *after* #2's schema lands because doctor can then validate
  cohsex.in against the schema too.
- **Desk-doable.** Fully desk-doable.

### #4 — Unify the MPI-stack contract into one config file

- **Proposers.** A2 B1, A1 P5.
- **Acceptance criteria.**
  - A single `config/mpi_stacks/{cray_mpich,openmpi5,…}.cmake` plus
    matching `.sh` carries
    `(LORRAX_PHDF5_MPI_STACK, SHIFTER_MODULES, MPI_LIB_DIR_CT,
    MPI_INCLUDE_DIR_CT, MPI_TYPE_DEFAULT, GTL_PRELOAD)`.
  - `run_shifter.sh`, `build.sh`, `0.1.0.lua` all read from this one
    source. The `LORRAX_PHDF5_MPI_STACK={mpich,openmpi}` branch in
    `run_shifter.sh:51-78` collapses to a sourced include.
  - `LD_PRELOAD` and `MPICH_GPU_SUPPORT_ENABLED` in `0.1.0.lua:181-186`
    are conditional on the `MPI_TYPE` value (my B8).
  - A 4-line CI test (`tests/build_smoke/test_mpi_stack_round_trip.sh`):
    load module, print `$LORRAX_SHIFTER`, grep for `--mpi=…`, assert
    matches the chosen stack file.
- **Closes most directly.** A2 D2, A2 D20 (and unblocks A2 D4 + D5 +
  B8).
- **Dependencies.** A prerequisite for any Apptainer port (out-of-blitz)
  and for #5's out-of-container CMake test.
- **Desk-doable.** Fully desk-doable for the refactor; **one Perlmutter
  validation run** to confirm the modulefile still produces the same
  `$LORRAX_SHIFTER` string.

### #5 — FFI / build / Shifter / distributed-init smoke test suite + minimal CI

- **Proposers.** A1 P4 (out-of-container CMake configure), A4 B-4 (FFI
  + shifter + dist-init smoke).
- **Acceptance criteria.**
  - `tests/build_smoke/test_out_of_container.sh`: CMake configure on a
    host without Shifter, all four bind-mount targets unmounted,
    override env vars set. Exercises the autodetect ladder.
  - `tests/integration/test_smoke_ffi.py`: (a) `import _lorrax_ffi`,
    (b) one Cholesky via `common.slate_cholesky_trsm_test -n 64`,
    (c) `jax.distributed.initialize()` + `jax.process_count() == 1`.
  - `tests/integration/test_modulefile.sh`: fresh Lmod env,
    `module load lorrax`, diff `env | grep -E '^(LORRAX_|LD_|JAX_|
    XLA_|HDF5_|MPICH_)'` against checked-in expected (A3 Q10).
  - At least a weekly scheduled CI run on Perlmutter (NERSC's
    `@nersc/setup-perlmutter`-style runner per A4 Q-9). CPU-only
    fallback runner if NERSC runner not available.
- **Closes most directly.** A4 D-26, D-27, D-28, D-29; locks A1 P3
  (strict-mode) and prevents regressions of #1-#4.
- **Dependencies.** Most leverage *after* #1 (build-system) is in
  place, because then `pip install` is the test entry. Without #1 the
  smoke test would itself need to invoke `bash build.sh`. Sequentially:
  #1 → #5 unlocks the rest.
- **Desk-doable.** No — **requires live Perlmutter runner setup** at
  minimum. The CPU fallback portion (`test_out_of_container.sh` on a
  GitHub-hosted runner) is desk-doable.

### Ranking rationale

#1 is first because it changes the *first thing a porter does* from
"silently fail" to "fail with a CMake error message." No other blitz
matters if the porter never gets a `.so`.

#2 is second because it's the only fully desk-doable item and it
unlocks `lorrax doctor`'s schema awareness in #3. Maintainability
keystone; closes ~12 doc-vs-code defects in one stroke.

#3 is third because it's the porter's debugging surface — once the
build works, they need a tool that tells them *why* a runtime knob
isn't behaving. `lorrax doctor` is the highest-leverage diagnostic
artifact across all four drafts.

#4 is fourth because it removes the only cross-file invariant that
silently fails: MPI-type selection living in two places. Lower than
#3 because the porter has to hit a runtime issue first to discover
they care.

#5 is fifth because it locks in the previous four. Without CI, every
fix in #1-#4 is rolled-back-able by a non-malicious commit. With CI,
the floor stops moving.

**Not in top 5 but adjacent:** Apptainer companion (#7 in my round-1
ranking, demoted in §B-2 — not a blitz); consolidate aot vs gflat
memory models (A4 B-6, important but parallel team owns it); upstream
`lxstatus` (A4 B-7 option a, minor).

---

## Section E — Open questions for the user

These gate one or more blitz items. Most apply across slices.

1. **(blocks #1, A4 Q-4)** What is the JAX version *actually* running
   in production today inside the Shifter container? `jax[cuda13]>=0.9.0`
   in pyproject vs `nvcr.io/nvidia/jax:25.04-py3` (JAX 0.5.x + CUDA 12)
   in PORTING.md. A 30-second `python -c "import jax; print(jax.__version__,
   jax.devices())"` inside `lxshell` resolves this. Until then we don't
   know whether the pin is wrong, the container tag is wrong, or
   `LORRAX_SITE` bind-mounting overrides the container's JAX.

2. **(blocks #3, A3 Q1, A3 FRAGILE-2)** Which GPU allocator is
   actually active — `XLA_PYTHON_CLIENT_ALLOCATOR=platform` or
   `TF_GPU_ALLOCATOR=cuda_malloc_async`? The two are set simultaneously
   in `0.1.0.lua:130-131`. The JAX startup log reports which one
   XLA picked; one line of stderr capture from a live run resolves it.
   Drop the loser as part of #3.

3. **(blocks #2, A4 Q-1, Q-8)** When the schema-strict mode lands,
   should unknown cohsex.in keys be a warning or a hard error? A4 leans
   strict + one-shot template rewrite; the author runs hundreds of
   cohsex.ins across `runs/`, so strict-mode would break all old
   sandbox runs on re-execution. Recommendation: warning by default,
   `LORRAX_COHSEX_STRICT=1` opt-in.

4. **(blocks #2, A4 Q-8)** Are `sigma_debug_split_contrib` and
   `write_no_head_vw` (doc-only, parser doesn't know) and the converse
   (parser-only, doc doesn't know — `gn_ppm`, `hl_ppm`, `cublasmp`,
   `aot_chunk` per A4 D-2) drift bugs, or are they intentional /
   auto-derived? Author's call.

5. **(blocks #4, A1 Q1)** Under Apptainer + `--bind /opt/cray/pe`, is
   the `phdf5/stage_cray.sh:84-86` SONAME shim layer still on
   `LD_LIBRARY_PATH` early enough to intercept libmpi resolution? I
   can't tell without an Apptainer-on-HPE-Cray host. The author's
   answer determines whether the unified MPI-stack file (#4) can
   represent the Frontier case as just a "different `--bind` syntax"
   or has to model the shim ordering too.

6. **(blocks #5, A4 Q-9)** Is the GitHub-Actions + NERSC-runner CI
   path realistic, or will it have to be a self-hosted runner? The
   cost of #5 is dominated by the runner setup, not the test code.

7. **(blocks #4 / Apptainer scope)** A4 Q-6: has the author *ever*
   actually run LORRAX on a non-NERSC cluster, even a workstation? If
   yes, the breakage-points are known and PORTING.md can be sharpened
   from aspirational to validated. If no, then PORTING.md is currently
   a research roadmap, and the second-user story is genuinely untested.

8. **(blocks A4 B-7 / sandbox graduation)** Is there a planned second
   LORRAX user in the next 6 months? A4 Q-7: graduating the
   `lorrax_agent` overlay upstream pre-emptively (option a) costs
   maintenance surface for one user; reactive graduation (option b)
   means the second user is a research project. The author's hiring /
   collaboration outlook determines which.

9. **(blocks A2 D7 / §B-1 resolution)** Does any current LORRAX user
   actually launch on PBS? If no, A3's "select_gpu.sh is portable" is
   right and the PBS-fallback line is dead weight; if yes-in-future,
   the cheap one-line fallback is essentially free insurance. Answer
   determines tag (FRAGILE only vs FRAGILE+COMPAT).

10. **(blocks #5 / smoke-test scope)** Does
    `tests/active/test_reshard_all_to_all.py` get collected by default
    pytest (A4 Q-5)? `pyproject.toml:43` `testpaths = ["tests"]` should
    include it, but the "active" naming suggests quarantine. If
    collected: D-29 is wrong, dist-init smoke already runs. If not:
    the test is dead code and #5's dist-init smoke test is genuinely
    new. A 5-second `uv run pytest --collect-only tests/ | grep
    test_reshard` answers it.

---

`Agent 2 round 2 done — see round2_agent_2.md`
