# Onboarding lens — zero to first calculation

**Agent:** onboarding analyst (4-agent LORRAX docs review)
**Repo:** `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D` (branch `agent/dep-home-migration`)
**Scope:** the actual happy path a newcomer follows, the cliffs on it, and the doc fixes / structure that would most reduce onboarding friction. READ + PROPOSE ONLY.

All line citations verified against the live checkout on 2026-06-29.

---

## 1. The real "zero to first calculation" sequence (traced, not assumed)

There are **two distinct happy paths** and the docs blur them into one. Distinguishing them is the single biggest clarity win available.

### Path A — laptop / login-node CPU smoke test (no FFI, no Shifter, no GPU)

This is the only path a brand-new person can complete on a fresh clone with nothing pre-staged. It is **not** described as a path anywhere; you have to reconstruct it.

1. `curl -LsSf https://astral.sh/uv/install.sh | sh` (one-time) — `ENVIRONMENT_COMPREHENSIVE.md:64`
2. `uv sync` — installs the project **editable** (verified: `.venv/.../__editable__.lorrax-0.1.0.pth` adds `src/` to `sys.path`; this is what makes `python -m gw.gw_jax` resolvable). README.md:21.
3. `uv run python -m pytest -q` — README.md:22 claims "~15s". The only collected non-archive test is `tests/test_gw_jax_regression.py`, marked `@pytest.mark.regression` (`tests/test_gw_jax_regression.py:79`), which **shells out** to `python -m gw.gw_jax` on the bundled fixture and is CPU-capable (`_gpu_available()` → falls back; `tests/test_gw_jax_regression.py:21-24, 82-83`).
4. First real calculation: `python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in` (the command `docs/index.md` "Quick runs" actually gives).

**Why Path A works without the FFI cliff:** the fixture input sets `use_ffi_io = false` (`tests/regression/cohsex_debug/cohsex_test.in:18`) and the fixture ships its own wavefunction `WFNsmall.h5`, `centroids_frac_60.txt`, `dipole.h5`, `kin_ion.h5` (verified present in `tests/regression/cohsex_debug/`). FFI imports are **lazy** — every `from ffi...` is inside a function body (verified: `gw_driver_helpers.py:156`, `w_isdf.py:282`, `file_io/wfn_loader.py:249,284,569`), so `import gw.gw_jax` never touches `liblorrax_ffi.so`. A single-rank static-COHSEX run with `use_ffi_io=false` therefore never calls `ffi_loader.get_lib()`.

> This is the **golden onboarding path** and it is buried. Nobody is told "this exact command runs end-to-end on a fresh clone with zero native build." That sentence belongs at the top of the README.

### Path B — Perlmutter production GW (Shifter + 3 native stacks + FFI build)

1. `vi config/perlmutter/site_config.sh` (account, QoS, paths) — `config/README.md:9-11`, `ENVIRONMENT_COMPREHENSIVE.md:143-146`
2. Stage three native trees: `stage_nvhpc.sh`, `stage_cray.sh` (phdf5), `stage_cray.sh` (slate) — `ENVIRONMENT_COMPREHENSIVE.md:253-259`. **One-time, mandatory, undocumented prerequisite for everything below.**
3. Build SLATE host install (`$HOME/software/slate/install`) — *referenced* by `site_config.sh:108` but **no build instructions exist anywhere in the docs.**
4. **Build `liblorrax_ffi.so`:** `src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh` — `ENVIRONMENT_COMPREHENSIVE.md:265-267`. This is gitignored (`.gitignore:71-72`) so it is absent from a fresh clone; the failure is `FileNotFoundError … Build with: bash src/ffi/common/cpp/build.sh` (`ffi_loader.py:89-92`).
5. `bash config/perlmutter/install.sh` → `module load lorrax` (`config/README.md:13-16`)
6. `lxalloc` → **3 preprocessing steps** → `lxrun python3 -u -m gw.gw_jax -i cohsex.in`

**The 3 preprocessing steps are the most under-documented part of the user journey.** `lxpre cohsex.in 640` is shown as one opaque line (`config/README.md:39`, `ENVIRONMENT_COMPREHENSIVE.md:166`). What it actually runs (modulefile body, `config/modulefiles/lorrax/0.1.0.lua:312-337`):
- `[1/3] python3 -m centroid.kmeans_cli <N> --seed 42` → `centroids_frac_<N>.txt`
- `[2/3] python3 -m psp.get_dipole_mtxels -i <in>` → `dipole.h5`
- `[3/3] python3 -m gw.kin_ion_io_chunked -i <in>` → `kin_ion.h5`

None of those three module names, and none of the three output artifacts, appear in any user-facing doc. A newcomer who can't or won't use `lxpre` (e.g. a non-Perlmutter cluster, or debugging) has no documented way to produce centroids/dipoles/kin_ion.

---

## 2. The cliffs, in the order a newcomer hits them

| # | Cliff | Where it bites | Current doc state | Evidence |
|---|---|---|---|---|
| C1 | **`gw_isdf/` path is fiction** — module is `src/gw/`, not `src/gw_isdf/` | First time anyone tries to open the file the README names | README.md:14 says `gw_isdf/gw_jax.py`; actual is `src/gw/gw_jax.py` | `src/gw/gw_jax.py` exists; `src/gw_isdf/` does not (verified `ls src/`) |
| C2 | **README quick-start uses `gw.gw_jax` but text uses `gw_isdf`** — internally contradictory | First command | README.md:23 says `python -m gw.gw_jax` (correct); README.md:14 says `gw_isdf/gw_jax.py` (wrong) | both lines in one file |
| C3 | **`uv sync --no-install-project` breaks `python -m gw.gw_jax`** | Anyone following the ENV doc instead of the README | `ENVIRONMENT_COMPREHENSIVE.md:67` says `uv sync --no-install-project --locked`; README.md:21 says bare `uv sync` | `package = true` + editable `.pth` is what puts `src/` on `sys.path`; `--no-install-project` skips it. Verified `.venv/.../__editable__.lorrax-0.1.0.pth` → `src` |
| C4 | **JAX version contradiction** | First confusing moment debugging GPU | `pyproject.toml:9` pins `jax[cuda13]>=0.9.0`; `ENVIRONMENT_COMPREHENSIVE.md:29` repeats it as authoritative. The production image `nvcr.io/nvidia/jax:25.04-py3` (`site_config.sh:32`) ships JAX ~0.5.3 | a newcomer who `uv sync`s gets 0.9.x locally but the cluster runs 0.5.x — no doc reconciles these |
| C5 | **Stale FFI default paths (`$SCRATCH` vs `$HOME/software`)** | First FFI run after the relocation | `ENVIRONMENT_COMPREHENSIVE.md:238-240` and `config/README.md:109-111` still print `$SCRATCH/lorrax_nvhpc`, `$SCRATCH/lorrax_phdf5_cray/stage`, `$SCRATCH/lorrax_slate_cray/stage` | actual defaults are `$HOME/software/...` (`site_config.sh:102-104`, comment dated 2026-06-24 explains scratch purge). The §7 table at `ENVIRONMENT_COMPREHENSIVE.md:361` also says "under `$SCRATCH`" |
| C6 | **`liblorrax_ffi.so` not in a fresh clone; build is the single biggest from-scratch cliff** | First production (FFI-I/O or distributed-eigh) run | README.md says nothing about it; only `ENVIRONMENT_COMPREHENSIVE.md:263-269` documents the build, and only deep in §5 | `.gitignore:71-72` excludes it; `ffi_loader.py:89-92` is the error |
| C7 | **`run_shifter.sh`/`build.sh` need a GPU alloc + staged NVHPC first** | Trying to build the FFI | `build.sh:6` example shows `lxalloc` first but README/ENV §5.4 omits that the stages (C-cliff above) and an allocation are prerequisites | `build.sh:35-47` hard-fails if `LORRAX_MPI_INCLUDE_DIR`/`LORRAX_MPICH_LIB_DIR` unset — only `run_shifter.sh` sets them |
| C8 | **SLATE host install has no build doc** | Setting up `$HOME/software/slate/install` | referenced at `site_config.sh:108`, `ENVIRONMENT_COMPREHENSIVE.md:362`; no build recipe anywhere | grep for slate build instructions: only `stage_cray.sh` (which stages, not builds) |
| C9 | **3 preprocessing module names undocumented** | First non-`lxpre` preprocessing | `lxpre cohsex.in 640` shown opaquely (`config/README.md:39`) | actual: `centroid.kmeans_cli`, `psp.get_dipole_mtxels`, `gw.kin_ion_io_chunked` (`0.1.0.lua:325-334`) |
| C10 | **`docs/index.md` "Key modules" cites a nonexistent path** | Browsing the landing page | `docs/index.md` lists `src/isdf/common/wfnreader.py` and "See formalism details in formalism.md" and "see examples/" | none exist: no `src/isdf/`, no `docs/formalism.md`, no `examples/` (all verified absent). Actual wfn loader is `src/common/load_wfns.py` (README.md:13) |
| C11 | **Console-command name drift** | Trying the advertised CLIs | README.md:16 advertises `lorrax-gw`, `gw_jax`, `lorrax-centroids`, `lorrax-bse`; these DO install (verified `.venv/bin/`). But README.md:16 also calls `gw_jax` and `lorrax-gw` interchangeably while line 14 still says `gw_isdf`. Minor, but the README contradicts itself within 2 lines | `pyproject.toml:22-26`; `.venv/bin/{gw_jax,lorrax-gw,lorrax-bse,lorrax-centroids}` all present |
| C12 | **"For AI agents" framing** | Immediately — sets the wrong audience | `ENVIRONMENT_COMPREHENSIVE.md:2` "**For AI agents**"; README.md:39 "For AI agents: read AGENTS.md first" | a human newcomer is told the docs aren't for them |

**Severity ranking for onboarding:** C5 + C6 + C3 are the genuine *blockers* (you cannot complete Path B without resolving them, and they are wrong or buried). C1/C2/C10/C11 are *credibility* bugs — a newcomer who opens the README, follows a path that doesn't exist, and finds the file isn't where it's claimed will distrust the rest of the docs. C12 is *framing* — cheap to fix, high signal.

---

## 3. Small, immediately-actionable doc fixes (no restructuring)

Each is (current → problem → fix), with file:line.

1. **README.md:14** — `gw_isdf/gw_jax.py` → the directory is `gw/`. Fix to `gw/gw_jax.py` (and `w_isdf.py` is `gw/w_isdf.py`). *Problem:* points newcomers at a path that does not exist. *Fix:* `s|gw_isdf/|gw/|g` on this line.

2. **README.md:21-24 (Quick start block)** — `uv sync` then `python -m gw.gw_jax -i cohsex.in`. *Problem:* there is no `cohsex.in` in the repo root, so the third line fails for a newcomer; the genuinely-working command is the fixture one. *Fix:* change the run line to the bundled fixture: `uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in` and add a one-line note "runs end-to-end on CPU on a fresh clone — no GPU or native build required."

3. **ENVIRONMENT_COMPREHENSIVE.md:67** — `uv sync --no-install-project --locked`. *Problem:* skips the editable install, so `python -m gw.gw_jax` then raises `ModuleNotFoundError: gw`. *Fix:* drop `--no-install-project` (keep `--locked` if reproducibility is wanted); align with README.md:21.

4. **ENVIRONMENT_COMPREHENSIVE.md:238-240** and **config/README.md:109-111** and **ENVIRONMENT_COMPREHENSIVE.md:361** — FFI default host paths shown as `$SCRATCH/...`. *Problem:* relocated to `$HOME/software/...` on 2026-06-24 (`site_config.sh:99-104`); docs now point at a purged/empty scratch dir. *Fix:* replace all three default paths with `$HOME/software/lorrax_nvhpc`, `$HOME/software/lorrax_phdf5_cray/stage`, `$HOME/software/lorrax_slate_cray/stage`; update §7 line 361 from "under `$SCRATCH`" to "under `$HOME/software`".

5. **ENVIRONMENT_COMPREHENSIVE.md:103** and **config/README.md:65** — `JAX_COMPILATION_CACHE_DIR = $SCRATCH/.jax_cache`. *Problem:* same scratch-purge logic that drove the FFI relocation applies; at minimum flag it as ephemeral. *Fix:* add a one-line admonition that this dir is on purgeable scratch and is safe to lose (it's a cache), so newcomers don't panic when it vanishes.

6. **README.md:14, ENVIRONMENT_COMPREHENSIVE.md:2** — "**For AI agents**" framing. *Problem:* tells a human the doc isn't for them. *Fix:* make the doc human-first; move the "agents read AGENTS.md" pointer to a single line at the bottom.

7. **README.md §Quick start / §4 (ENVIRONMENT_COMPREHENSIVE.md:164-172)** — `lxpre cohsex.in 640` opaque. *Problem:* the three modules and three output files are invisible; non-`lxpre` users are stuck. *Fix:* add a 3-line expansion right under the `lxpre` example: "`lxpre` runs (1) `centroid.kmeans_cli N` → `centroids_frac_N.txt`, (2) `psp.get_dipole_mtxels` → `dipole.h5`, (3) `gw.kin_ion_io_chunked` → `kin_ion.h5`." (Source: `0.1.0.lua:325-334`.)

8. **ENVIRONMENT_COMPREHENSIVE.md:263-269 (§5.4)** — FFI build shown as a single command. *Problem:* omits the two hard prerequisites (an `lxalloc` GPU allocation and the staged NVHPC tree) that `build.sh:6,35-47` require. *Fix:* prepend "Prereqs: (a) `stage_nvhpc.sh` has run; (b) you hold a GPU allocation (`lxalloc`). Then:" before the command.

9. **docs/index.md ("Key modules", "see examples/", "See formalism details in formalism.md")** — *Problem:* `src/isdf/common/wfnreader.py`, `docs/formalism.md`, and `examples/` do not exist. *Fix:* repoint wfn loader to `src/common/load_wfns.py`; remove the `formalism.md` and `examples/` references or replace with `tests/regression/cohsex_debug/` as the worked example.

10. **README.md:22 ("~15s")** — *Problem:* the lone collected test is a `@pytest.mark.regression` end-to-end subprocess run; on CPU it is materially slower than 15s and is not a "unit test." *Fix:* either re-label it "regression smoke test (CPU, ~1–2 min)" or add a genuinely fast `pytest -m "not regression"` lane and document that.

11. **config/README.md:29-33** — tells the user to `export SLURM_JOBID=<jobid>` from a second terminal. *Problem:* a newcomer's first reaction; works but is fragile (stale JOBID, wrong job). *Fix:* note that `lxalloc` already exports `SLURM_JOBID` in the shell it runs in, and the manual export is only for *other* shells / IDEs.

---

## 4. Structural proposals (newcomer-facing, not agent-facing)

### S1. Split the install story into the two real paths, with a support matrix
The docs today present one tangled path that is partly local-uv, partly Shifter, partly bare-venv. Mirror JAX/PySCF: a single **Installation** page with a matrix.

| Path | Runtime | Native FFI? | Hardware | First command |
|---|---|---|---|---|
| **Smoke (Path A)** | `uv venv` | no | any CPU | `uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in` |
| **Perlmutter (Path B)** | Shifter via `module load lorrax` | yes (build `liblorrax_ffi.so`) | A100×4 | `lxpre` → `lxrun … gw.gw_jax` |
| **Generic SLURM** | bare venv / Apptainer | optional | any CUDA | per §7 |

The matrix tells a newcomer *in one glance* which path their machine supports and that Path A needs no native build. This single table dissolves cliffs C3, C4, C6 by making the "no FFI needed here / FFI needed there" boundary explicit.

### S2. A real "Getting Started" page (proposed outline)
```
Getting Started
├─ What you need (Python ≥3.12; for GPU: CUDA 12; for production: Perlmutter + Shifter)
├─ 60-second smoke test  (Path A — copy-paste, runs on a laptop)
│    uv sync ; uv run python -m pytest -q
│    uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
│    → expected output: eqp_test.dat, matches eqp_ref.dat
├─ Understanding the inputs  (WFN.h5, centroids_frac_N.txt, cohsex.in — what each is, where it comes from)
├─ Your first real calculation  (the 3 preprocessing steps → GW), explicitly:
│    1. centroids   (centroid.kmeans_cli N)
│    2. dipoles     (psp.get_dipole_mtxels)
│    3. kin_ion     (gw.kin_ion_io_chunked)
│    4. GW          (gw.gw_jax)
└─ Next: scaling to GPU / multi-node → Environment Setup page
```
Anchor it on the **already-shipped fixture** as the worked example — it is the one thing guaranteed to exist and run.

### S3. An "Environment Setup" page that separates "what the module gives you" from "what you must build once"
Today `ENVIRONMENT_COMPREHENSIVE.md` interleaves env-var reference, FFI internals, multi-host theory, and troubleshooting. Restructure to:
```
Environment Setup
├─ Local (uv)               — Path A, exhaustive, no native deps
├─ Perlmutter (module)      — module load + lxalloc/lxpre/lxrun, one screen
├─ One-time native setup    — the 4 things you build/stage ONCE:
│     stage_nvhpc · stage phdf5 · stage slate · build liblorrax_ffi.so
│     (+ the missing SLATE host-install recipe — currently undocumented, C8)
├─ Porting to another cluster  — §7 as-is, but cross-linked from the matrix
└─ Troubleshooting          — §8 as-is (it is genuinely good)
```
The "build once" box is what newcomers most need and currently must reverse-engineer from `build.sh` comments and `site_config.sh`.

### S4. Make the golden Path-A command the literal first thing in the README
The README's lead is three paragraphs of physics prose (README.md:3-9) before any command. JAX/PySCF lead with "here's how to install and run one thing." Move a 4-line copy-paste smoke test (Path A) to immediately under the title; physics prose moves below it.

### S5. Single source of truth for paths/versions
C1/C4/C5 are all "doc prose drifted from `pyproject.toml` / `site_config.sh`." The FFI bind-mount table and the dep table are duplicated verbatim in `ENVIRONMENT_COMPREHENSIVE.md` *and* `config/README.md`, and both drifted. Propose: the bind-mount/default-path table lives in exactly one place (or is generated from `site_config.sh`), and the JAX version is stated once with an explicit "local-resolve vs container-shipped" note rather than asserted as `>=0.9.0` in two docs that the container contradicts.

---

## 5. Single highest-leverage change

**Add a "60-second smoke test" at the very top of the README that is the bundled fixture command, and label it as the path that needs no GPU and no native build** — i.e. fix README.md:14/21-24 to point at `tests/regression/cohsex_debug/cohsex_test.in` with `use_ffi_io=false`, and say so explicitly.

Rationale: it is the *only* command a fresh-clone newcomer can run to success today, it already exists and passes, and surfacing it (a) gives an immediate win that builds trust, (b) draws the bright line between "Path A needs nothing native" and "Path B needs the FFI build," which is the conceptual confusion underneath cliffs C3, C4, C6, and (c) costs ~5 lines of edits. Everything else in the onboarding story is easier to absorb once a newcomer has seen LORRAX produce a correct `eqp` table on their own machine.
