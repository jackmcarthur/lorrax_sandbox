# Agent 4 — Docs, onboarding, runtime config surface, CI/test gaps

Slice: the new-user docs trail, the `cohsex.in` runtime-config surface, and
the maintainability ledger (parallel implementations, doc-vs-code drift,
sandbox-vs-upstream split, CI/test gaps). Build/MPI/container fragility is
out of scope and assigned to Agents 1/2/3 — I only mention them where docs
make claims about them.

---

## 1. Scope

**In scope.** Top-level `README.md`, `AGENTS.md`, `docs/*.md`,
`config/README.md`, `src/ffi/PORTING.md`, the
sandbox-side `docs/docs_gwjax/COHSEX_INPUT.md`, the cohsex.in parser
(`src/gw/gw_config.py`), `templates/cohsex.in`, the `tests/` tree,
`pyproject.toml`, the AGENTS-level overlay docs, and recent doc git
history.

**Out of scope.** CMake / FFI internals, MPI flavor selection, container
runtime swap, vendor ABI, Lua mechanics. (Cited only where a doc claim
about them is wrong.)

---

## 2. Current state

Numbers, not vibes:

- **Comprehensive docs**: 4 files in `docs/` named `*_COMPREHENSIVE.md`,
  totalling **3 313 lines** (PHYSICS 1763, CODEBASE 640, ENVIRONMENT 455,
  MEMORY_MODEL 458 — the last is "comprehensive in spirit"). Plus 7 plan /
  audit / progress docs in `docs/` totalling another ~2 700 lines. There
  is no quickstart, no FAQ, no "you are here" router.
- **README.md**: 39 lines. Three lines on "how to run", then the entire
  rest is paragraphs of marketing prose + a doc index. No worked example,
  no "what cohsex.in must contain at minimum", no link to a sample input
  file. Routes a Perlmutter user to `config/README.md` and a local user
  to `uv sync` — both are correct, neither is enough.
- **`config/README.md`**: 244 lines. The most useful onboarding doc by
  far. Goes Quick Start → Usage → modulefile → unified Cray MPICH stack →
  per-invocation cost → multi-checkout → shared install → site-packages
  → file layout → porting. This doc is the de-facto README for
  Perlmutter.
- **`src/ffi/PORTING.md`**: 196 lines. Hard requirements table, build
  table, staging table, runtime checklist, gotchas. Tight, well-written.
  Out of date in places (§3 below).
- **`docs/index.md`**: 58 lines. **Stale.** References
  `src/isdf/centroid/kmeans_isdf.py` and `src/isdf/common/wfnreader.py`;
  the actual paths are `src/centroid/kmeans_isdf.py` and
  `src/file_io/wfnreader.py`. Last touched April; the `src/isdf/` tree
  was renamed and never propagated here.
- **`docs/AGENT_TODO.md`**: 292 lines of "suggestions identified by AI
  agents that are NOT user priorities" (the doc's own header). This is
  the closest thing to a maintenance backlog and it is explicitly
  parked.
- **`tests/`**: 25 unit-test files at `tests/*.py`, plus
  `tests/active/test_eqp_bgw.py` and one regression test
  (`test_gw_jax_regression.py`). No `tests/integration/`, no FFI smoke
  test, no Shifter smoke test, no `jax.distributed.initialize()` test
  in CI (the one that exists is in `tests/active/` and gated on
  subprocess CPU-only execution).
- **CI**: `.github/`, `.gitlab-ci.yml`, `Jenkinsfile` — **none exist**.
  CI is "the author runs `uv run python -m pytest -q` before
  committing" per `AGENTS.md:103`. There is no automated build, test, or
  doc-build trigger.
- **`pyproject.toml`**: 75 lines. scikit-build-core not actually wired
  (`build` is a dependency group, not the actual build backend — it's
  declared `package = true` under `[tool.uv]` with no `[build-system]`
  block at all in what I read). FFI `.so` is built by a separate
  `bash build.sh` invocation, not by `pip install`.
- **Recent activity (last 25 commits)**: dominated by performance /
  shard_map / FFI work in `src/`. **One docs commit** (`b72b591` adding
  PHYSICS §11). Three commits *removing* env-var surface in favour of
  cohsex.in keys (488e870 / 9fe5fde / 40a4cca) — none of which updated
  COHSEX_INPUT.md.

The cohsex.in parser (`src/gw/gw_config.py:145–348`) has a single
`_DEFAULTS` dict with **77 keys**. The reference doc
(`COHSEX_INPUT.md`) explicitly documents **~38** of them across 11
sections. Any unrecognised key in cohsex.in is silently ignored
(`configparser` + `_DEFAULTS.items()` loop at line 437 — no key
intersection check).

---

## 3. NERSC-isms (docs slice)

I'm leaving build/MPI/container NERSC-isms to Agents 1/2/3. From a docs
perspective, the relevant question is: *does the doc tell a non-NERSC
reader where the line is?*

- **`config/README.md`**: explicit. §"Porting to other clusters" lists
  the placeholder set in `site_config.sh` and clearly says non-Shifter
  needs `lxrun`/`lxshell`/`lxpre` rewriting. Good. Would degrade
  gracefully on Frontier (Apptainer + cray_shasta) with documented
  edits; would need real work on Polaris (PBS, not Slurm — `lxalloc`'s
  `salloc` body becomes meaningless) or a university Slurm+Docker
  cluster (no `--module=`, no `select_gpu.sh` semantics if cgroups
  already pin GPUs).
- **`src/ffi/PORTING.md`**: explicit, clean. Has a checklist. Names the
  `stage_openmpi.sh` alternative for non-Cray. Names the
  Apptainer/Singularity rewrite as a one-line swap (correct in spirit;
  in practice the env-passing flag syntax and the `--volume` argument
  format differ enough to be more than a swap).
- **`docs/ENVIRONMENT_COMPREHENSIVE.md` §7**: claims a "bare-venv
  fallback" works on a generic SLURM cluster (lines 366–378). The
  example sets only the JAX env vars and runs `python -m gw.gw_jax`.
  This is **misleading** — without the FFI `.so` built and on
  `LD_LIBRARY_PATH`, large runs hit `ImportError: liblorrax_ffi.so` or
  silently fall back to non-FFI paths whose code paths are not
  separately CI'd. The bare-venv path is a documentation artifact, not
  a tested mode.
- **`COHSEX_INPUT.md`**: NERSC-neutral on the surface. Two latent
  defaults are NERSC-shaped, neither flagged: `memory_per_device_gb=0.0
  → auto-detect via jax.memory_stats() / nvidia-smi` (correct on Cray
  but on a SLURM+cgroup cluster `nvidia-smi` may report the *node* not
  the *cgroup*-allotted slice), and the implicit assumption that ψ(G)
  fits in **host RAM** (the `gspace_mode = host_cache` default is fine
  on Perlmutter's 512 GB nodes; on a 64 GB-per-node cluster it OOMs the
  host).

Pure [COMPAT] items in the docs slice are rare — most NERSC-isms hide
in code Agents 1/2/3 are auditing.

---

## 4. Defect catalog

Numbered, exhaustive within this slice. Tags: **[FRAGILE]** doc-vs-code
drift / silent failure on update; **[COMPAT]** breaks on a non-NERSC
cluster as documented; **[LOC-COST]** dead weight, parallel
implementations, sandbox-vs-upstream redundancy.

### 4.1 Doc-vs-code drift in COHSEX_INPUT.md (the cohsex.in reference)

The single biggest [FRAGILE] cluster in this slice. `COHSEX_INPUT.md`
was last touched **2026-04-04** (commit e8d9351). The parser has been
refactored at least 4 times since.

- **D-1 [FRAGILE]** `COHSEX_INPUT.md` documents **`use_chunked_isdf`
  (default true)** at line 31 as a real key. The parser
  (`gw_config.py:_DEFAULTS`) does **not** contain this key. Setting
  `use_chunked_isdf = false` in cohsex.in is silently a no-op.
  Equivalent for `sigma_debug_split_contrib` (doc line 284,
  not in `_DEFAULTS` — auto-set internally from
  `sigma_freq_debug_output`) and `write_no_head_vw` (doc line 369,
  not in `_DEFAULTS`).
- **D-2 [FRAGILE]** Section 3 ("Calculation Mode") of COHSEX_INPUT.md
  documents only `x_only` / `do_screened` / `bispinor` /
  `self_consistent`. The current canonical axis is
  **`compute_mode = x_only | cohsex | gn_ppm | hl_ppm`** (introduced
  via the `ComputeMode` enum at `gw_config.py:47–80`). The doc has
  **zero references to `compute_mode`, `gn_ppm`, `hl_ppm`,
  `cublasmp`, `aot_chunk`** (`grep -c` confirms 0 hits). The legacy
  flags still parse via the `compute_mode = "auto"` shim
  (`gw_config.py:732–761`); a new user reading the doc has no idea
  that `ppm_model = hl` (line 296–300 of `_DEFAULTS`) is even an
  option.
- **D-3 [FRAGILE]** Recent migrations 488e870 (chunk sizes) and 9fe5fde
  (algorithmic toggles) added **at least 11 new cohsex.in keys**:
  `psig_k_chunk_size`, `gflat_chunk_size`, `vq_g_chunk_size`,
  `cusolvermp_charge`, `cusolvermp_lu`, `gamma_contract_mode`,
  `use_aot_chunk_chooser`, `chunk_chooser_mode`, `gspace_mode`,
  `use_ffi_io`, `isdf_memory_mode`. **None are in COHSEX_INPUT.md.**
  Combined with D-1 (no validation against unknown keys), a user has
  no surfaced way to discover them except by reading the parser
  source.
- **D-4 [FRAGILE]** Other parser keys with no doc entry, not from the
  recent migration: `do_G0`, `no_degen_averaging`, `degen_avg_tol_ry`,
  `centroids_file_current` (bispinor-specific second centroid file),
  `kin_ion_file`, `sigma_diag_file`, `eqp0_file`, `eqp1_file`,
  `bgw_vcoul_file`, `bgw_vcoul_sym_wfn`, `mc_average_vcoul_body`,
  `bare_coulomb_cutoff`, `zeta_cutoff`, `regenerate_minimax_tables`,
  `ppm_head_omega_h_ry`, `sigma_window_edge_factor`,
  `sigma_omega_batch_size`, `ppm_sigma_target_error`,
  `ppm_sigma_max_nodes`, `write_wfn_h5`, `wcoul0_eta`, `debug_omega`,
  the entire BSE block (`get_centroids_fi`, `wfn_fi_min`, `wfn_fi_max`,
  `kgrid_fi`). **Doc covers ~38 of 77 parser keys (≈ 50%).**
- **D-5 [FRAGILE]** COHSEX_INPUT.md §9 documents
  `output_file = "eqp0_noqsym.dat"` as the main output. `gw_config.py`
  L161–170 makes `output_file` a deprecated key — setting it now
  emits a `DeprecationWarning` and is ignored; the actual output keys
  are `sigma_diag_file` / `eqp0_file` / `eqp1_file`. The
  sandbox-template `cohsex.in` (`templates/cohsex.in`) still uses
  `output_file = eqp0.dat` and will trigger that warning on every run.
- **D-6 [FRAGILE]** `bare_coulomb_cutoff` default in the parser is
  `None`; the user-memory `project_bare_coulomb_cutoff_default`
  records that LORRAX's *effective* default is **4·ecutwfc** while
  BGW's is **ecutwfc** — anyone comparing to BGW must set this
  explicitly. This is a known gotcha that lives nowhere in the docs.

### 4.2 Doc-vs-code drift outside COHSEX_INPUT.md

- **D-7 [FRAGILE]** `docs/ENVIRONMENT_COMPREHENSIVE.md:171` documents
  `lxkill` as a shell function. The modulefile
  (`config/modulefiles/lorrax/0.1.0.lua`, `set_shell_function`
  declarations at lines 234, 257, 296, 312) defines **lxalloc, lxrun,
  lxshell, lxpre — no lxkill**. The user is told "cancel allocation,
  unset SLURM_JOBID" and gets `command not found`.
- **D-8 [FRAGILE]** `docs/ENVIRONMENT_COMPREHENSIVE.md:322` shows
  `LORRAX_NNODES=2 LORRAX_NGPU=8 lxrun python3 -u -m gw.gw_jax ...`
  as the multi-node entry point. The modulefile's `lxrun`
  (`0.1.0.lua:279`) hardcodes **`-N 1`** and never reads
  `LORRAX_NNODES`. Multi-node `lxrun` does not work as documented.
  (`run_gw.slurm` batch uses `#SBATCH -N` directly, bypassing this; an
  interactive 2-node run via `lxrun` silently runs on one node.)
- **D-9 [FRAGILE]** `docs/index.md:43–44` references
  `src/isdf/centroid/kmeans_isdf.py` and
  `src/isdf/common/wfnreader.py`. The actual paths are
  `src/centroid/kmeans_isdf.py` and `src/file_io/wfnreader.py` (per
  `AGENTS.md:43,38`). The `src/isdf/` package was renamed and
  `index.md` was never updated. Last touched 2026-03 / 04 era.
  References to a non-existent `formalism.md` (line 19) and
  `examples/` directory (line 19) — neither exists at repo root.
- **D-10 [FRAGILE]** `docs/index.md` and `docs/AGENT_TODO.md` use
  `src/isdf/common/load_wfns.py: 2796 lines, 39 functions` as a
  refactor target — the file actually lives at `src/common/load_wfns.py`
  in the current tree. Anyone trying to act on the TODO will hit a
  missing-file error first.
- **D-11 [FRAGILE]** README.md "Quick start" (line 22) shows
  `uv run python -m gw.gw_jax -i cohsex.in` — but does not say
  `cohsex.in` is not provided in-tree. There is no example
  cohsex.in inside `lorrax_C/`; the only template is in the *sandbox*
  (`/pscratch/sd/j/jackm/lorrax_sandbox/templates/cohsex.in`), not
  shipped with the canonical checkout. A second user does not get this
  file. (The regression test does ship one at
  `tests/regression/cohsex_debug/cohsex_test.in`, but README doesn't
  point at it.)
- **D-12 [FRAGILE]** Sandbox `templates/cohsex.in` is itself stale: it
  uses `output_file = eqp0.dat` (deprecated, D-5), declares
  `sigma_debug_split_contrib = true` and `write_no_head_vw = true`
  (silently dropped, D-1), and has no `compute_mode` key (relies on
  `auto` shim, D-2). This template is what the Build Inputs skill
  copies into every new run dir, so the rot propagates per-run.
- **D-13 [FRAGILE]** `src/ffi/PORTING.md` table (lines 9–17) lists
  `JAX with jax.ffi: 0.5+, container nvcr.io/nvidia/jax:25.04-py3`.
  `pyproject.toml:14` requires `jax[cuda13]>=0.9.0`. The 25.04 image
  (per NVIDIA's tagging convention) ships JAX ≈ 0.4.x with `cuda12`
  — incompatible with the `jax[cuda13]>=0.9.0` requirement. Either
  the dep pin is wrong, the container is mis-named in the doc, or the
  container is bind-mounting `LORRAX_SITE_PACKAGES` over the in-image
  JAX (cf `ENVIRONMENT_COMPREHENSIVE.md:86`). The doc never explains
  which jax actually runs at runtime — the user has to open a Python
  REPL inside the container to find out.
- **D-14 [FRAGILE]** `ENVIRONMENT_COMPREHENSIVE.md` §1.1 lists
  `jax[cuda13]>=0.9.0` as the dep. §3.1 says
  `JAX_PLATFORMS=cuda,cpu` is set automatically. `cuda13` is the
  CUDA-13-targeted JAX wheel; `nvcr.io/nvidia/jax:25.04-py3` ships
  CUDA 12.x. The `cuda13` extra is not what runs in the container.
  Same root cause as D-13.
- **D-15 [FRAGILE]** AGENTS.md "Coding standards" §"JAX sharding
  rules" is the only place "no `np.concatenate`" lives — but
  `src/runtime/__init__.py`, FFI helpers, and several test files do
  concatenate host-side. The rule is informally enforced; new
  contributors will not internalise it from one bullet in AGENTS.md
  with no example.

### 4.3 Parser-side fragility (cohsex.in as a config surface)

- **D-16 [FRAGILE]** Parser silently accepts unknown keys. A typo
  like `sigma_omega_step_eV = 0.5` (capital V) is dropped on the floor
  — the loop at `gw_config.py:437` only iterates `_DEFAULTS.items()`,
  never the section's keys. Combined with D-3 / D-4 (most
  documentation-vs-parser drift is in unknown-key territory), a
  user's "I set the flag but nothing changed" debugging session has
  no diagnostic.
- **D-17 [FRAGILE]** Three nullable-float keys (`vhead`,
  `whead_0freq`, `whead_imfreq`, `bare_coulomb_cutoff`, `zeta_cutoff`,
  `ppm_head_omega_h_ry`, `debug_omega`) are detected by
  `isinstance(default, …) is None`. If a future key has a default of
  `None` but is supposed to be a string (e.g. an optional file path),
  it will be coerced to a float by `section.getfloat(... fallback=None)`
  and crash in subtle ways downstream.
- **D-18 [FRAGILE]** Three of the new `auto | on | off` keys
  (`cusolvermp_charge`, `cusolvermp_lu`) and the four-state
  `gamma_contract_mode = take | einsum | scan` are validated only
  by the consumer downstream; the parser accepts any string. The
  `compute_mode` and `isdf_memory_mode` keys *are* validated (raise
  `ValueError`); the others are not. Inconsistent enforcement.
- **D-19 [LOC-COST]** Two parallel "memory model" trees:
  `src/gw/gflat_memory_model.py` (single-file, 4-peak per-rank model
  per the docstring) and `src/gw/aot_memory_model/` (package with
  `chooser.py`, `cost.py`, `core.py`, `kernels/`, `presets.py`,
  `sweep.py`, `predict_cli.py`, `artifacts/`). Both gated by config
  keys (`use_aot_chunk_chooser`, `chunk_chooser_mode`). The
  zeta-rchunk-memory-model parallel team's report
  (`reports/zeta_rchunk_memory_model_2026-05-13/round4_*`) is
  converging on consolidation. Right now the author maintains two
  memory models that must agree; in practice they will not, and the
  active code path depends on a cohsex.in bool.
- **D-20 [LOC-COST]** Two `aot_memory*` modules: `src/gw/aot_memory_model/`
  (the chooser + artifacts) and `src/runtime/aot_memory.py` (separate
  file under the runtime tree). Same name, different things. A new
  reader following AGENTS.md (§"Where things are" doesn't list either)
  has no way to know which is canonical.
- **D-21 [LOC-COST]** `src/gw/__init__.py:9` re-exports
  `read_lorrax_input` AND its alias `read_cohsex_input`
  (`gw_config.py:494: read_cohsex_input = read_lorrax_input`). The
  second is consumed in 9 places across `src/` (psp, bandstructure,
  gw); the first in 1. The alias is a no-op rename that exists only
  because the input file is *called* cohsex.in but the package no
  longer is "cohsex". Pick one; the dual surface is doc/code drift
  bait.

### 4.4 The `_COMPREHENSIVE` problem

- **D-22 [FRAGILE]** Four files named `*_COMPREHENSIVE.md` totalling
  3 313 lines. A new user has to choose between them with no router.
  The naming pattern says "single source of truth" — `MEMORY_MODEL.md`
  is the longest doc *not* labelled comprehensive but is the same kind
  of thing. There is no convention reader has to internalise; in
  practice the author treats the four CAPS docs as the canonical
  source, but a reader can't know without reading AGENTS.md first.
- **D-23 [LOC-COST]** `docs/PHYSICS_COMPREHENSIVE.md` is 1 763 lines.
  The b72b591 commit added §11 (the current bispinor pipeline) and put
  a "this supersedes §3-5" pointer at the top. §3-5 are still present
  in full. The doc now has explicit forward-pointer rot built in —
  load-bearing prose for the *current* code is at line 1300+, while
  the first 600 lines are flagged "historical". Maintenance debt: any
  subsequent rewrite must do the same dance, until the section count
  approaches double digits.
- **D-24 [LOC-COST]** `docs/AGENT_TODO.md` (292 lines), `docs/plans/`,
  `docs/PLAN_zeta_g_flat_migration.md` (373 lines),
  `docs/FREQ_INTEGRATION_REWRITE_PLAN.md` (651 lines),
  `docs/FREQ_INTEGRATION_PROGRESS.md`,
  `docs/SIGMA_FREQ_AUDIT_STATUS.md`,
  `docs/PROFILING_SUGGESTIONS.md`,
  `docs/NEW_WINDOW_MINIMAX_GUIDELINES.md`,
  `docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` —
  ~2 500 lines of in-progress / suggested / draft / "REVISED" docs
  living next to canonical reference docs. The `_REVISED` suffix
  (note: "revised guide" is a tell — the un-revised version was
  archived) is a code smell. None of these are linked from the
  README or AGENTS.md `## Key documentation` table.
- **D-25 [LOC-COST]** `docs/archive/` exists (with
  `cohsex_jax_physics.md`, `formalism.md`, etc.). Good. But
  `docs/AGENT_TODO.md` is *not* in archive despite the doc itself
  saying its contents are "NOT user priorities". Either it's
  authoritative (then file paths must be kept current — currently
  stale, see D-10) or it's archived. Keeping it half-alive is the
  worst option.

### 4.5 Test coverage gaps

- **D-26 [FRAGILE]** No CI. `AGENTS.md:103` (lorrax-side) says "Run
  `uv run python -m pytest -q` after long running branches (5+
  small commits)". This is a contributor-honesty system. Three
  consequences: (a) regression test doesn't run on every commit, so
  numerical regressions in `gw_jax.py` can land before the next
  manual run; (b) no out-of-author signal that the test still passes
  on a fresh checkout / fresh venv; (c) docs that reference the test
  pass status (e.g. `docs/SIGMA_FREQ_AUDIT_STATUS.md`) are also
  honor-system.
- **D-27 [FRAGILE]** No FFI smoke test. The unit suite has
  `test_slab_io_ffi_contract.py` (pure Python contract on a
  shape/sharding boundary) and that's it. There is no test that
  imports `_lorrax_ffi.so` and exercises a single FFI call (e.g. a
  256×256 Cholesky via `slate_cholesky_trsm_test`). When NVHPC bumps
  to 25.6 and the cuSOLVERMp ABI changes (precedent: c52fbd2,
  "0.7+ ABI shift"), the first signal is a real GW run failing in
  the screening solver.
- **D-28 [FRAGILE]** No Shifter smoke test. There is no test or CI
  hook that loads the module, instantiates the container, and runs a
  one-line `python -c "import jax; print(jax.devices())"` inside.
  Modulefile drift (e.g. someone changes `LD_LIBRARY_PATH` ordering)
  surfaces as a real run breakage at calculation start.
- **D-29 [FRAGILE]** `tests/active/test_reshard_all_to_all.py`
  (the only test that touches `jax.distributed.initialize`) lives in
  `tests/active/`, not in the default test path. `pyproject.toml:43`
  has `testpaths = ["tests"]` and `norecursedirs = ["archive"]`,
  which technically *should* include `tests/active/`. Verify whether
  pytest collects it; if it does, my error — but the directory name
  "active" suggests it's quarantined. Either way, multi-process JAX
  init has no regression test that runs by default.
- **D-30 [FRAGILE]** No regression for the cohsex.in parser itself.
  Given §4.1 (≈40 keys undocumented, 3 documented but unparsed, plus
  the legacy `output_file` deprecation flow), a parser test that
  asserts `set(_DEFAULTS) == set(documented_keys)` would catch every
  defect in §4.1 in CI. Currently nothing locks the doc to the
  parser.
- **D-31 [LOC-COST]** `tests/test_gw_jax_regression.py` is gated on
  `ISDF_COHSEX_TEST_PLATFORM=cpu` for portability — meaning the GPU
  path is never regression-tested. The CPU mode covers physics
  correctness but not the FFI / sharded / shard_map paths that run
  in production.

### 4.6 Sandbox vs. upstream split

- **D-32 [LOC-COST]** `COHSEX_INPUT.md` lives in the **sandbox**
  (`/pscratch/sd/j/jackm/lorrax_sandbox/docs/docs_gwjax/`), not in
  `lorrax_C/docs/`. This is the only authoritative reference for the
  cohsex.in surface, and it ships with the sandbox not the package.
  A second user does not get this file when they `git clone
  lorrax_C`. The whole §4.1 cluster of drift is downstream of the
  fact that the doc is in the wrong repo.
- **D-33 [LOC-COST]** The `lorrax_agent/1.0.lua` overlay (the
  `lxstatus` / `lxattach` / `lxreap` / pool-aware `lxrun` layer) is
  sandbox-only. For a single user on Perlmutter, fine. For two users
  with concurrent allocations on a shared install, there is no
  upstream coordination — two users running `lxrun` from the base
  module on the same allocation will collide. The `lorrax_agent`
  overlay solves this; the upstream module doesn't know about it.
- **D-34 [LOC-COST]** `templates/` lives in the sandbox. The
  canonical `lorrax_C/` checkout has **no** template inputs (no
  `templates/` dir, no example cohsex.in next to the regression
  test besides the test fixture itself). Build-Inputs is a sandbox
  skill, not an upstream tool. A second user has to reverse-engineer
  cohsex.in from the regression fixture + parser source.
- **D-35 [LOC-COST]** `KNOWN_SANDBOX_ERRORS.md` records a recent
  fix (2026-05-12, the `pf.py` `jax.distributed` hang) where the
  workaround was to delegate to `runtime.init_jax_distributed()` —
  the upstream canonical impl. This pattern (sandbox tooling
  re-implementing something subtly different from upstream, and
  silently breaking) will recur. There's no convention for "sandbox
  scripts MUST import upstream entry points".

### 4.7 Misc

- **D-36 [FRAGILE]** `pyproject.toml` declares no `[build-system]`
  (only the `build` dependency-group is named — those are
  scikit-build-core / cmake / nanobind, but they're *not* the build
  backend wired up). `pip install lorrax` will fall back to setuptools
  default, which will not build `liblorrax_ffi.so`. The author works
  around this by running `bash src/ffi/common/cpp/build.sh`
  separately. A pip-installer cannot reproduce this.
- **D-37 [FRAGILE]** `docs/index.md:18` references `examples/` as a
  link target. No `examples/` directory exists in `lorrax_C`. Either
  add it or delete the link.
- **D-38 [FRAGILE]** `docs/index.md:25` documents
  `uv add pdoc; uv run -- bash docs/gen_api_docs.sh`. The script
  exists (`docs/gen_api_docs.sh`); whether the API docs build
  succeeds in CI or against the current src layout is not tested.
  `pyproject.toml` has `mkdocs` / `mkdocs-material` /
  `mkdocstrings-python` as runtime deps (!) — they should be a
  `[dependency-groups.docs]` group, not pulled in for every
  `uv sync`. Cost: anyone installing for runtime drags in mkdocs.
- **D-39 [LOC-COST]** AGENTS.md `## Key documentation` table lists 6
  docs. `README.md` Documentation section lists 6 *different* docs
  (overlapping but not identical: AGENTS.md doesn't mention
  `MINIMAX_QUADRATURE.md`; README doesn't mention
  `SIGMA_FREQ_AUDIT_STATUS.md`). Two indices, no canonical one.

---

## 5. Blitz proposals

Ranked by leverage (installability or maintainability win, ~1-day each).
Top of list = do first.

### B-1. Schema-validate cohsex.in against the parser at parse time
*Touches*: `src/gw/gw_config.py` (50–80 LoC: an `_UNKNOWN_KEY_POLICY`
check + an `_ALIAS_KEYS` set for legacy/deprecated keys). Optional
scaffold: `src/gw/_cohsex_schema.py` to hold a single `Field` table
(name, type, default, doc-one-liner) that both the parser and the doc
generator consume.
*Defects addressed*: D-1, D-3, D-4, D-16, D-18 (and lays the
foundation for D-30).
*Why high-leverage*: maintainability. Locks the parser surface to a
single source of truth. After this, every silent typo or unknown key
becomes a noisy error or warning; a doc-generation step (B-3) can pull
the `Field` table directly. The migration is mechanical: enumerate
keys → assign types → bring `_DEFAULTS` in line.
*Risk*: a strict policy will surface stale keys in user input files.
Mitigate with a configurable warn / error toggle (env var) and an
allow-list of known-deprecated keys (`output_file`,
`use_chunked_isdf`, `sigma_debug_split_contrib`, `write_no_head_vw`)
that emit `DeprecationWarning`s but don't crash.
*CI/test lock-in*: a unit test that round-trips a 0-key `[cohsex]`
section through `LorraxConfig.from_input_file` and asserts no warnings;
a unit test that asserts every key in a `tests/golden_cohsex.in` is
either documented or in the allow-list.

### B-2. Move COHSEX_INPUT.md into the lorrax_C repo and regenerate from the parser
*Touches*: copy `/pscratch/sd/j/jackm/lorrax_sandbox/docs/docs_gwjax/COHSEX_INPUT.md`
to `lorrax_C/docs/COHSEX_INPUT.md`; add `tools/gen_cohsex_input_md.py`
that walks the schema from B-1 and emits the doc; add a pytest that
diffs the generated output against the committed file (fails CI if
they drift).
*Defects addressed*: D-1 through D-6, D-12, D-32 in one stroke.
*Why high-leverage*: installability *and* maintainability. The doc
is currently (a) in the wrong repo, (b) 50% out of date, (c)
guaranteed to drift further as the migration trend continues. Once
generated from the parser, drift is impossible.
*Risk*: lossy if the rich §4–§7 prose (head correction physics, PPM
quadrature, three-window scheme, etc.) doesn't survive the regen.
Mitigation: keep the prose as `## Background` blocks at the top of
each section; only the `### key` paragraphs are auto-generated.
*CI/test lock-in*: the diff test is the lock-in.

### B-3. Add a `QUICKSTART.md` (≤ 100 lines) for the "9-band MoS2 GW on Perlmutter" path
*Touches*: new `QUICKSTART.md` at repo root, plus a
`tests/regression/cohsex_debug/cohsex_test.in` symlink or copy at
`examples/mos2_minimal/cohsex.in`. README links to it as the *first*
doc to read.
*Defects addressed*: D-11 (no in-tree example), D-22 (no front-page
router), partly D-37 (no `examples/`).
*Why high-leverage*: installability. The current README routes a new
user to `config/README.md` (Perlmutter-only) or `uv sync` (which
gives a venv but no input file). A 100-line doc that goes
`module load` → `lxalloc` → `lxpre` → `lxrun gw_jax -i` →
`grep "Σ_X" eqp0.dat` is the smallest gap to "first numbers out".
*Risk*: the quickstart needs a stable example. The
`tests/regression/cohsex_debug/` fixture should be promoted to a
real example with a longer-form description; that takes ~half a
day on top of the doc itself.
*CI/test lock-in*: the existing regression test already exercises
the example; add a docs lint that fails if any code block in
QUICKSTART.md is not also exercised by a test.

### B-4. Add a `tests/integration/test_smoke_ffi.py` and a CI runner
*Touches*: `tests/integration/test_smoke_ffi.py` with three tiny
tests — (a) `import _lorrax_ffi`, (b) one Cholesky via
`common.slate_cholesky_trsm_test -n 64`, (c) one
`jax.distributed.initialize()` + `jax.process_count() == 1`
under a single-process invocation. Then `.github/workflows/ci.yml`
or equivalent that runs `pytest -q tests/` on a Perlmutter or
Cirun-style runner that has the FFI built. Even one weekly
scheduled run is a step up from zero.
*Defects addressed*: D-26, D-27, D-28, D-29, partly D-31.
*Why high-leverage*: maintainability. Catches the entire family of
"vendor bumped, ABI shifted, modulefile drifted" failures *before*
the author hits them in a real run. The c52fbd2 commit ("dispatch
CAL vs NCCL comm at runtime, 0.7+ ABI shift") is exactly the kind
of regression a smoke test prevents.
*Risk*: needs a CI runner with GPU + Shifter access — neither is
free. A scheduled weekly run on Perlmutter via NERSC's
`@nersc/setup-perlmutter`-style runner is the realistic version.
Fallback: a CPU-only Linux runner + a `JAX_PLATFORMS=cpu` smoke
test (skips FFI but catches Python import / parser regressions).
*CI/test lock-in*: the workflow file itself.

### B-5. Wire scikit-build-core as the actual `[build-system]` in pyproject.toml
*Touches*: `pyproject.toml` (~10 LoC adding `[build-system]` with
`scikit-build-core>=0.11.0` and `cmake-args` pointing at
`src/ffi/common/cpp/CMakeLists.txt`). Move `bash build.sh` logic
into a CMake target that scikit-build-core invokes. Add a
`pyproject.toml` plugin spec for `nanobind` if needed.
*Defects addressed*: D-36, D-13/D-14 (only insofar as a real build
backend will surface the JAX version mismatch at install time).
*Why high-leverage*: installability. After this, `pip install -e .`
inside the container builds `liblorrax_ffi.so` — the FFI build is
no longer a separate manual step. Aligns the author with the
scientific-Python community standard (cf. nanobind / scikit-build
docs at https://nanobind.readthedocs.io/en/latest/building.html
and https://scikit-build-core.readthedocs.io/en/latest/getting_started.html
— note both stress that the build backend MUST be declared in
`[build-system]`, not in a dependency-group).
*Risk*: the FFI build needs CMake to find NVHPC / CUDA / SLATE.
The current `build.sh` does this via env vars; scikit-build-core
passes these via `cmake.args` in `pyproject.toml`. Some refactor
of the auto-detect block in `CMakeLists.txt` may be needed.
*CI/test lock-in*: a CI job that runs `pip install -e .` and
imports `_lorrax_ffi`.

### B-6. Consolidate the two memory models (gflat + aot)
*Touches*: per the zeta-rchunk team's converged round-4 plan
(`reports/zeta_rchunk_memory_model_2026-05-13/round4_*`), pick one
package (probably `aot_memory_model/` since it's already
chooser-shaped) and fold `gflat_memory_model.py`'s 4-peak HBM
budget into it as another preset. Drop the cohsex.in switches
`use_aot_chunk_chooser` and `chunk_chooser_mode` once the merged
chooser is the only path.
*Defects addressed*: D-19, D-20.
*Why high-leverage*: maintainability. Two memory models with
overlapping responsibilities is the single biggest [LOC-COST] in
this slice — and the parallel team has done most of the analysis.
This blitz is *act on their report*.
*Risk*: behavioural change in chunk sizing — the merged chooser
must reproduce both old paths' picks on the existing test
fixtures within a small tolerance. The zeta-rchunk team has
this in scope (round4_memory_model_state.md mentions it).
*CI/test lock-in*: extend
`tests/test_aot_memory.py` to assert the chooser picks the same
chunks as the existing `gflat_memory_model.py` on a small fixture
matrix.

### B-7. Promote `lorrax_agent` overlay coordination upstream (or document the gap)
*Touches*: either (a) move `lxstatus` / `lxattach` / `lxreap` into
`config/modulefiles/lorrax/0.1.0.lua` (no Lua change to existing
4 functions, just add 3 more), or (b) add a one-paragraph
"multi-user concurrency" warning to `config/README.md` saying
"upstream module assumes single-user-per-allocation; concurrent
agentic use needs the sandbox `lorrax_agent` overlay".
*Defects addressed*: D-33, D-34 (partial — concurrency aspect),
D-35 (the canonical-source pattern).
*Why high-leverage*: maintainability. As soon as a second user
adopts LORRAX, this becomes a real issue and the sandbox-overlay
solution disappears with the sandbox. Even option (b) — pure
documentation — is enough to alert the porter.
*Risk*: option (a) requires the pool-coordination Python
(`lx_pool.py`) to live somewhere upstream; that module currently
has sandbox-specific assumptions about scratch paths.
*CI/test lock-in*: not strictly needed; a unit test that
`module show lorrax` enumerates the expected shell functions
catches future drift.

### B-8. Add a `MAINTENANCE_TODO.md` at repo root, linked from AGENTS.md
*Touches*: new `MAINTENANCE_TODO.md` at repo root. Move the
*real* maintenance items out of `docs/AGENT_TODO.md` (which
explicitly disowns its own contents) into here. Track:
- Sentinel `_LORRAX_JAX_DISTRIBUTED_DONE` (undocumented invariant
  per `runtime/__init__.py`).
- The deprecated-keys allow-list from B-1.
- D-13/D-14 jax-version-vs-container reconciliation.
- The "lxshell holds no allocation lock" gotcha (per `0.1.0.lua`
  comment block).
- The `bare_coulomb_cutoff` / BGW default mismatch (D-6).
- Anything else accumulated over the next year.
*Defects addressed*: D-25 (the stale "TODO suggestions" doc),
D-15 (informally enforced rules need a real ledger), generally
the maintainability bookkeeping.
*Why high-leverage*: maintainability — closing the loop on the
"author maintaining LORRAX over the next year" prompt. Cheap,
explicit, gets defects out of agents' heads and into a file that
git tracks.
*Risk*: bit-rot on the TODO itself; mitigate by deleting items
when fixed (no "this was a problem in March 2026" archeology).
*CI/test lock-in*: not needed.

### B-9. Add `docs/_index.md` (or rewrite `docs/index.md`) as an explicit router
*Touches*: rewrite `docs/index.md` as a router: "if you're a
new user, read README → QUICKSTART → COHSEX_INPUT → run the
example. If you're porting, read PORTING.md. If you're hacking
on physics, read PHYSICS_COMPREHENSIVE §11 (current) and §1-§2
(theory). If you're tuning memory, read MEMORY_MODEL.md and the
chooser docstring. If you're an AI agent, read AGENTS.md."
Delete the stale code-path references (D-9).
*Defects addressed*: D-9, D-10, D-22, D-37, D-39 (single index
of truth, replacing two indices in README and AGENTS.md).
*Why high-leverage*: installability — for a new user, four
500-line `_COMPREHENSIVE` docs is daunting, and the router cuts
the choice-paralysis time. Maintainability — the current
`index.md` actively misleads.
*Risk*: low. Deleting the stale prose and replacing with a
link map is straightforward.
*CI/test lock-in*: a markdown-link-check CI step (e.g.
`lychee` — https://lychee.cli.rs) that fails the build on a
broken in-repo link. This *also* catches every future doc-vs-code
path-rename drift (D-9, D-10).

### B-10 (lower-priority). Move `mkdocs*` deps to a `docs` dependency group
*Touches*: `pyproject.toml` — move `mkdocs`, `mkdocs-material`,
`mkdocstrings`, `mkdocstrings-python` from `[project.dependencies]`
into a new `[dependency-groups.docs]` block.
*Defects addressed*: D-38 (only the unnecessary-dep half).
*Why high-leverage*: tiny installability win, mostly a hygiene
fix. Reduces the `uv sync` runtime install for non-doc users.
*Risk*: none.
*CI/test lock-in*: not needed.

### Ranking summary

| # | Blitz | Primary axis | Effort | Net leverage |
|---|---|---|---|---|
| B-1 | Cohsex.in schema validation | maintainability | 1 d | very high |
| B-2 | Generate COHSEX_INPUT.md from parser | both | 1 d | very high |
| B-3 | QUICKSTART.md | installability | 0.5 d | high |
| B-4 | Smoke-test CI for FFI / Shifter / dist init | maintainability | 1–2 d | high (needs runner) |
| B-5 | Wire scikit-build-core in `[build-system]` | installability | 1–1.5 d | high |
| B-6 | Consolidate the two memory models | maintainability | 1 d | high (act on parallel team) |
| B-7 | Promote `lorrax_agent` upstream OR document gap | maintainability | 0.5 d | medium |
| B-8 | MAINTENANCE_TODO.md ledger | maintainability | 0.25 d | medium |
| B-9 | docs/_index.md router + link-check CI | both | 0.5 d | medium |
| B-10 | Move mkdocs to docs group | installability | 0.1 d | low (hygiene) |

If the author has one day, do **B-1 + B-2** as a unit (schema +
generated doc). If two days, add **B-3** and a stub **B-8**. If a
week, B-4 unblocks the rest.

---

## 6. Open questions

The honest "I don't knows", in priority order:

- **Q-1** *Is the parser test (B-1's diff/coverage assertion) the
  right level of strictness?* Strict: every cohsex.in key must be
  documented OR explicitly deprecated. Loose: warnings only. The
  strict version will break every existing run dir until `templates/`
  and historical sandbox cohsex.ins are cleaned. I lean strict +
  one-shot mass-rewrite of templates, but the author runs hundreds of
  cohsex.ins across runs/ — rebuilding all of them may not be
  appetising.
- **Q-2** *Does the regression test (`tests/test_gw_jax_regression.py`)
  actually cover the FFI paths?* It uses `ISDF_COHSEX_TEST_PLATFORM=cpu`
  which presumably skips FFI. Without running it I can't tell whether
  the per-platform branch-coverage on `gw_jax.py` paths is 30% or 80%.
  This affects whether B-4's CPU smoke test is a meaningful safety
  net or theatre.
- **Q-3** *Does scikit-build-core (B-5) actually wire cleanly through
  the existing CMake autodetect?* The `CMakeLists.txt` probes
  `$NVHPC_ROOT`, `/lorrax_nvhpc`, etc. — these are env-var-driven,
  which scikit-build-core supports via `cmake.define` in
  `pyproject.toml`. But the staging scripts (which set up
  `/lorrax_nvhpc` as a bind-mount path that doesn't exist outside
  Shifter) live outside the build process. The pip-install path may
  end up needing to know about Shifter, which defeats the point.
  Worth a one-day spike before committing.
- **Q-4** *What is the actual JAX version that runs at production
  time?* The `pyproject.toml` says `jax[cuda13]>=0.9.0`. The
  PORTING.md table says `JAX with jax.ffi: 0.5+, container
  nvcr.io/nvidia/jax:25.04-py3`. The
  `LORRAX_SITE_PACKAGES`-bind-mount comment (ENVIRONMENT
  COMPREHENSIVE.md:86) hints that `h5py / scipy / matplotlib` are
  bind-mounted from the host into the container — but does the
  *host's JAX* override the container's JAX too? If yes, what
  version? If no, what does `jax[cuda13]>=0.9.0` actually constrain?
  A 30-second `python -c "import jax; print(jax.__version__)"`
  inside the container would resolve this; I cannot run it (constraint
  §5 of CONTEXT). This is probably a single-line doc fix, but I
  don't know which line.
- **Q-5** *Does `tests/active/test_reshard_all_to_all.py` get
  collected by default pytest?* `pyproject.toml:43` has
  `testpaths = ["tests"]` and `norecursedirs = ["archive"]`. The
  `active/` directory should be collected (no `_active.py` magic in
  pytest), but the directory name strongly implies "not part of
  default suite". Without running pytest I can't tell. If it is
  collected, D-29 is wrong; if not, why not.
- **Q-6** *Is there a non-NERSC cluster the author has actually
  tested LORRAX on?* The PORTING.md path lists the steps assuming
  none has been tried. If yes (a private workstation? a previous
  group cluster?), the breakage-points are known and one-line
  callouts would convert all of §4.3's [COMPAT] → "documented".
  If no, then PORTING.md is *aspirational*, and a future second
  user is a research project. The doc reads as if it had been
  validated; my read is that it hasn't.
- **Q-7** *Should `lorrax_agent` (sandbox overlay) graduate
  upstream now or wait for a second user?* B-7 option (a) is the
  pre-emptive move; option (b) is the reactive one. The trade-off
  is: graduating now adds maintenance surface for a feature with
  one user; not graduating means the sandbox is the only place
  pool-aware coordination exists, and any production multi-user
  install will collide. I lean option (b) because there's no second
  user yet, but I'm not sure whether the author plans to invite
  one in the next 6 months.
- **Q-8** *Are the "deprecated keys" in §4.1 (D-1, D-5, D-12)
  actually deprecated, or are they intended? `sigma_debug_split_contrib`
  and `write_no_head_vw` may be auto-derived flags that the doc
  documents under the wrong heading. I treated them as drift; the
  author can confirm by reading the doc and the parser side-by-side.
  This affects whether B-2's regen produces the right doc on first
  pass.
- **Q-9** *Is the GitHub-Actions / NERSC-runner CI scenario (B-4)
  realistic?* The infrastructure for "GitHub-hosted runner that
  loads a Perlmutter-style modulefile" is non-trivial. NERSC has a
  `@nersc/setup-perlmutter` action equivalent (per
  https://docs.nersc.gov/services/jupyter/) but I'd need to verify
  it covers Shifter + GPU + multi-rank Slurm. Realistic fallback:
  a self-hosted runner on a workstation with one GPU, running a
  cut-down test. Either way, the cost of B-4 is dominated by the
  runner setup, not the test code.

Agent 4 done — see agent_4.md
