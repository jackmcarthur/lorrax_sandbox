# LORRAX sandbox — agent orientation

Single orientation point for LORRAX GW-BSE development on TACC Frontera.
Rebuilt and verified 2026-07-31; it replaces all earlier versions of this
file. Read it once, completely.

## Which tree is truth

The LORRAX repository is `/work2/08271/jackmc/frontera/lorrax`, branch
`fix/zq-band-gather-device-invariance` at `ecf461e`, ~150 commits ahead of
origin, NOT pushed. Work lands on that branch the same day it is validated.
Jobs never read the live tree: they read frozen source bundles built by
`config/frontera/build_cpu_runtime_bundle.sh`. Pinned 4x4 physics baselines
live under `/scratch2/08271/jackmc/mos2_4x4_test` (in active use by another
agent; treat read-only). This sandbox holds decks, tools, and records —
never a competing copy of the source.

## Current state (2026-07-31)

- Target: Frontera CPU (CLX, 2x28 cores/node, 192 GB/node). Scaling rule:
  thousands of low-memory processes; no N_mu^2 tile on any single rank.
- Full 4x4 GW pipeline green (jobs 7884609/7884612): eqp parity exact-0,
  sigma_mnk.h5 parity 1.1e-13 eV vs pinned baseline.
- Production transport: jax CPU collectives implementation `mpi`;
  gloo is banned at distributed tiers (`CLAIMS.md` rows 3-4).
- Certified opt-in perf stack: `LORRAX_FFT_FFI`(+`_FUSED`), vendor GEMM
  FFI, `slab_io=auto` parallel-HDF5 writer. Details in `GATES.md`.
- Architecture: three AST-gated levels — L1 physics drivers, L2 numerical
  routines, L3 substrate. Repo `docs/architecture/{layers,services,ffi_layout}.md`.

## Where truth lives

| Question | Source |
|---|---|
| Was X measured or decided? | `CLAIMS.md` (ledger — append, never re-derive) |
| Gate defaults / certified settings | `GATES.md` |
| Cross-cutting preconditions | `INVARIANTS.md` |
| How to launch a job | repo `config/frontera/templates/gw_dev.sbatch` + `config/frontera/mpi_transport_env.sh`; recipe in `skills/execute_workflow/SKILL.md` |
| Machine, container, 9-layer env stack | repo `docs/environment/overview.md`, `.../machines/frontera.md` |
| Portability (MKL vs Cray FFTW, GPU legs) | `docs/PORTABILITY.md` (pointers to repo docs) |
| Verifying sharding / collectives | `docs/HLO_HOWTO.md` + `tools/hlo/` |
| Full 2026 campaign history | `/scratch2/08271/jackmc/lorrax_setup/docs/SPEEDUP_SCORECARD.md` (9400+ lines — grep, never read linearly) |
| BGW / QE input semantics | `docs/docs_bgw/`, `docs/docs_qe/`, `docs/docs_gwjax/` |
| BGW-vs-LORRAX matching conventions | `docs/BGW_LORRAX_MATCHING.md`; output parsers in `PARSE_OUTPUTS.md` |

## Operational rules

1. Every claim carries a jobid and an on-disk artifact path. Job outputs
   are read from disk, never predicted. New verdicts are appended to
   `CLAIMS.md` in the same session that produced them.
2. Verify the instrument before trusting its verdict: broken harnesses
   have historically produced more false results than the code under test.
3. Login nodes: uid-wide RLIMIT_NPROC is 300 — at most ~3 concurrent
   agents, no process storms, `make -j4`. No containers and no `srun` on
   login; `sbatch` works (dev queue: 2 jobs / 40 nodes).
4. Login `python3` is 3.7. The repo AST gate suites run on it via their
   `__main__` runners: `tests/test_layering.py`,
   `tests/test_crossfile_requests.py`, `tests/test_env_registry.py`.
   The venv python works only inside the container.
5. Login HDF5 tools are 1.8 and cannot open 1.14 files; inspect `.h5`
   with a small in-container job.
6. apptainer binds must include `/opt/intel` and must never include `/dev`.
7. Collective/HLO tables are valid cache-cold only
   (`ISDF_JAX_CACHE_DIR=""`); warm caches under-report silently.
8. Git: feature branches in the repo; commit locally with explicit
   pathspecs; never push — the owner pushes.

## Do not trust

- Anything under `_archive/` — historical; wrong wherever it disagrees
  with this file, `CLAIMS.md`, or the repo docs.
- Any Perlmutter remnant (`/pscratch`, `lxrun`, Shifter, GPU memory
  tables, `uv run`): that environment is gone.
- Any claim without a `CLAIMS.md` row and jobid: treat as hypothesis.

## Sandbox map

| Path | Contents |
|---|---|
| `CLAIMS.md`, `GATES.md`, `INVARIANTS.md` | the three ledgers (see table above) |
| `fastloop/` | single-process mini-deck fast loop — scaffold only; `fastloop/PLAN.md` states what exists vs planned |
| `tools/` | working analysis tools (`hlo/analyze_hlo_dump.py`, `compare_bgw_gwjax.py`) |
| `skills/` | task recipes: build_inputs, execute_workflow, compare, checkpoint |
| `templates/`, `assets/` | QE/BGW input templates; pseudopotentials |
| `runs/`, `reports/`, `scripts/` | empty after the 2026-07-31 purge; READMEs state what belongs |
| `docs/` | reference documentation (see table above) |
| `_archive/` | dated historical record, non-normative |

New session read order: this file, then `CLAIMS.md`, `GATES.md`,
`INVARIANTS.md`, then whatever the task needs (~165 lines total).
