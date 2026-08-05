You are **Agent 1 of 4** in an independent multi-agent audit of LORRAX's
install/maintain surface.

Three other agents (Agents 2, 3, 4) are working the same overall task in
parallel tmux panes right now. You cannot see their work and they cannot
see yours. After all four of you finish, the orchestrator will collect
drafts and run a discussion round.

**Your shared briefing is at:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/CONTEXT.md`

**Read it first.** It has the overall task, code/path map, NERSC-isms
checklist, constraints, and the suggested 6-section output structure.

---

## Your slice: **Build system & FFI dependency contract**

You own the audit of how LORRAX gets *compiled* and what it links
against. The deliverable answers: *"Could a competent second user check
out LORRAX on a different cluster (Frontier, Polaris, Leonardo, or a
generic Slurm+OpenMPI+Docker cluster) and get `liblorrax_ffi.so`
built and importable from JAX without one-on-one help from the author?"*

Tag every finding in your defect catalog with **[FRAGILE]**,
**[COMPAT]**, or **[LOC-COST]** per CONTEXT.md §6. The build-system
slice is especially rich in [FRAGILE] (ABI pins, autodetect drift) and
[LOC-COST] (NERSC-specific staging scripts another cluster wouldn't
need).

**Web search is encouraged.** When you suspect ABI fragility or want to
check a vendor version-history claim, use `WebSearch` / `WebFetch`.
Concrete prompts: cuSOLVERMp release notes & ABI between 0.6→0.7→0.8;
NCCL API stability across CUDA 12.x; SLATE / blaspp / lapackpp release
cadence; nanobind ↔ scikit-build-core compatibility; `jax.ffi`
stability across JAX 0.4 → 0.5 → current. Cite URLs in your report.

### Primary read targets

1. `src/ffi/PORTING.md` — the existing porting doc. **This is the
   single most important critique target for you.** Read every line.
2. `src/ffi/common/cpp/CMakeLists.txt` — the unified build. Trace every
   autodetect probe, every `-D` override, every fallback path. Make a
   list of what's hardcoded vs. configurable vs. environment-derived.
3. `src/ffi/common/cpp/build.sh` — driver. What does it assume about
   environment state? About being run from inside a container?
4. `src/ffi/cusolvermp/scripts/stage_pypi.sh`, `src/ffi/slate/scripts/`
   — vendor staging. What does each one fetch? From where? What's
   the version policy?
5. `src/ffi/cusolvermp/cpp/*.cpp`, `src/ffi/phdf5/cpp/*.cpp`,
   `src/ffi/slate/cpp/*.cpp` — skim for ABI assumptions, version
   guards, dlopen patterns.
6. `pyproject.toml` — Python packaging side. How does `pip install -e .`
   trigger the FFI build? What happens if cmake can't find a dep?
7. `git log --oneline --all -- src/ffi/` — recent activity on the FFI
   tree. Pay attention to ABI shifts (`c52fbd2` cuSOLVERMp 0.7+,
   `929be9a` explicit-MPI guard, `2c5e5b5` SLATE auto-detect).

### Specific questions you must answer

- **Vendor matrix.** Make a table of every external dep, the version
  range LORRAX tolerates, how the CMake build discovers it, what
  happens on a miss (hard fail / silent skip / wrong-version link).
- **ABI fragility.** cuSOLVERMp 0.7+ shifted the CAL/NCCL comm dispatch
  (commit `c52fbd2`). What other deps have a similar "version N shifted
  the ABI" history? Are there other deps where LORRAX is one upstream
  release away from breaking?
- **Container-vs-host split.** Which deps are baked into the Shifter
  image (`nvcr.io/nvidia/jax:25.04-py3`) vs. bind-mounted from host
  (`LORRAX_FFI_NVHPC_HOST`, `LORRAX_FFI_PHDF5_HOST`,
  `LORRAX_FFI_SLATE_HOST`)? On a non-NERSC cluster without Shifter,
  what's the equivalent? Does the build assume the bind-mount layout?
- **Hard NERSC paths.** Every absolute path under `/opt/...`,
  `/lorrax_*`, `/global/...`, `~/software/...` in the CMakeLists +
  build.sh + stage_pypi.sh — list them. Which are defaults that can be
  overridden vs. baked in?
- **Silent fallback risks.** commit `929be9a` ("require explicit MPI
  env to avoid silent HPC-X fallback") tells us this has been a real
  bug class. Are there other places where CMake autodetect could pick
  the *wrong* dep without complaining?
- **The PORTING.md gap.** What does PORTING.md *not* tell a new user
  that the code actually requires? List every implicit assumption.

### Blitz proposals — be concrete

For each proposal: file(s) it touches, what changes, what test or CI
check locks it in. ~1-day scope. Examples of the kind of thing you
should propose (not prescriptive; come up with your own):

- A `cmake -DDRY_RUN=ON` mode that prints a vendor-dep matrix with
  resolved paths and version strings, so a user can sanity-check
  before a full build.
- A `tests/build_smoke/` that does an out-of-container build on a
  non-Shifter host as a regression.
- A version-pin table extracted from CMakeLists into a single
  `VENDOR_VERSIONS.txt`, so the porting doc and the build agree.
- A `--strict` mode for build.sh that turns silent autodetect misses
  into errors.

### Your assigned output file

`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/agent_1.md`

Write the complete audit there. Cover the six sections from CONTEXT.md
§6 (scope, current state, NERSC-isms, defect catalog, blitz proposals,
open questions).

### Hard constraints (from CONTEXT.md §5, restated)

- Read-only on `sources/lorrax_C/`. No code edits.
- No compute (no `srun`, `lxrun`, `python` that runs JAX). Desk research
  only.
- Do **not** read `agent_2.md`, `agent_3.md`, or `agent_4.md` in your
  report dir. Those belong to the other agents.
- Stay in this tmux pane.
- Stop when your file is written. Print a one-line summary:
  `Agent 1 done — see agent_1.md`.

Start by reading CONTEXT.md, then PORTING.md, then the CMakeLists.
