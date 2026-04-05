# AGENTS.md

This sandbox allows debugging and testing of all modules of LORRAX (Low-scaling Real-space Real-Axis eXcited state package), primarily via comparisons to BerkeleyGW. Our immediate focus is `gw.gw_jax`, LORRAX's GW driver module. LORRAX and GWJAX will occasionally be used interchangeably like how PW/QE are used to describe `pw.x`, the plane-wave SCF driver in Quantum ESPRESSO. You will use local or cluster GPUs to run Quantum ESPRESSO for DFT reference wavefunctions, followed by BerkeleyGW and/or LORRAX, and various output parsing tools. Your purpose is to execute an autonomous agentic loop to develop the LORRAX source code and execute numerical experiments to compare with BGW. (heavy modifications of LORRAX are welcomed; QE/BGW you may read and perhaps add debug prints under extreme circumstances).

This is a structured environment with skills for executing any of the three codes and scripts for extracting values from output files. There are rules for organizing output runs, logging changes and gradual bug repair status, and git habits for LORRAX. Note the **bolded** rules in this document which must always be followed.

## Read order when starting a session

1. This file (`AGENTS.md`).
2. Skills, in order: `skills/build_inputs/SKILL.md` → `skills/execute_workflow/SKILL.md` → `skills/compare/SKILL.md` → `skills/checkpoint/SKILL.md`.
3. `CHANGELOG.md` and the most recent `reports/*/report.md` for current status.
4. The active `runs/*/manifest.yaml`.
5. If modifying LORRAX source: `sources/lorrax/AGENTS.md` (module map, coding standards).
6. Then start work.

## Other documents

### Skills (tool-call documents for agents)

These are the structured instructions an agent reads before performing a specific task.

| Document | Location | Purpose |
|----------|----------|---------|
| **Build Inputs** | `skills/build_inputs/SKILL.md` | How to construct all QE/BGW/LORRAX input files from a material specification. Read this before creating any run directory. This is not a place to cut corners or link old calculation outputs, and you are required to construct these files and directories in the prescribed, thorough and organized way. |
| **Execute Workflow** | `skills/execute_workflow/SKILL.md` | Exact execution order, Perlmutter and local commands, full paths for all executables, verification checks after each step. Read this before running anything. |
| **Compare** | `skills/compare/SKILL.md` | Output parsing, BGW-vs-LORRAX comparison, plotting. Contains all parsers, the physics of SX-X+CH vs Σ⁺+Σ⁻, and comparison procedures. **Use these parsers for all output extraction.** |
| **Checkpoint** | `skills/checkpoint/SKILL.md` | The checkpoint routine: pytest, git commit, report update, CHANGELOG. **Follow this at every major checkpoint** (see "Checkpoints and reports" below). |

**If you encounter any error when carrying out these steps due to incorrect information in the skills (or anywhere else), you must stop what you're doing and report it in KNOWN_SANDBOX_ERRORS.md before continuing.**

### Reference documentation

| Document | Location | Purpose |
|----------|----------|---------|
| `CHANGELOG.md` | top level | Shared memory across sessions. What's done, what's broken, what to try next. **Read every session; update before ending every session.** |
| `KNOWN_SANDBOX_ERRORS.md` | top level | Log of documentation errors, broken paths, and stale references found by agents. **If you encounter a sandbox infrastructure problem, record it here immediately before continuing your work.** |
| `compare_bgw_gwjax.py` | top level | Multi-k comparison script. Matches BGW sigma_hp.log to LORRAX eqp0.dat via WFN.h5 k-points. Produces table + plot. See `skills/compare/SKILL.md` for usage. |
| `isdf_sos_debug.py` | top level | ISDF sum-over-states debug script. Projects BGW's G-space ε⁻¹ into ISDF basis. Requires `eps0mat.h5`. See `skills/compare/SKILL.md` §3 for when to use. |

### Code and input documentation

| Directory | Contents |
|-----------|----------|
| `docs/docs_gwjax/` | `COHSEX_INPUT.md` — exhaustive reference for flags in `cohsex.in`, output file format specs (every column in .dat, every dataset/shape in .h5). |
| `docs/docs_bgw/` | BerkeleyGW input file specs (`epsilon.inp`, `sigma.inp`, `pw2bgw.inp`), HDF5 file format specs (`wfn.h5.spec`, `epsmat.h5.spec`), and overview documents for each code. |
| `docs/docs_qe/` | Quantum ESPRESSO `INPUT_PW.txt` (full pw.x reference), `INPUT_pw2bgw.txt`, and parallel tuning notes. |
| `templates/` | Base input file templates (`scf.in`, `nscf.in`, `nscfq.in`, `pw2bgw.in`, `pw2bgwq.in`, `epsilon.inp`, `sigma.inp`, `cohsex.in`, `manifest.yaml`). The Build Inputs skill modifies copies of these; do not edit templates directly unless the default physics changes. |
| `assets/pseudopotentials/` | FR-ONCVPSP PBE pseudopotentials from PseudoDojo. `standard/` and `stringent/` sets. See `assets/pseudopotentials/README.md`. |

### Source code

| Symlink | Target | What it is |
|---------|--------|------------|
| `sources/lorrax` | `/home/jackm/projects/lorrax` | LORRAX source. **Read `sources/lorrax/AGENTS.md` for module map, run commands, and coding standards before editing any code.** |
| `sources/BerkeleyGW` | `/home/jackm/SOURCES/BerkeleyGW` | BGW source. Read-only (debug prints ok). Key: `Sigma/mtxel_cor.f90`, `Common/fixwings.f90`. |
| `sources/q-e-qe-7.4` | `/home/jackm/SOURCES/q-e-qe-7.4` | Quantum ESPRESSO. Read-only. |

### Memory budget

| Platform | GPU | VRAM | `memory_per_device_gb` |
|----------|-----|------|------------------------|
| Local WSL | RTX 5070 | 8 GB | 6.0 |
| Perlmutter | A100 | 40 GB | 28 |

Use `XLA_PYTHON_CLIENT_PREALLOCATE=false` on the local GPU if OOM occurs.

## Git discipline

**Before modifying any LORRAX source code, you must create a feature branch.** Naming convention: `agent/<initiative>` (e.g. `agent/head-correction-fix`). Never commit directly to `main`.

**Commit at every major checkpoint** (see below). Each commit should be a self-contained, testable unit. Run `uv run python -m pytest -q` before committing. Do not push to remote without explicit instruction.

This rule connects directly to checkpoints: git commits are the durable record of code changes, and reports are the durable record of what those changes achieved.

## Run directories

All calculations live under `runs/`. Structure:

```text
runs/
  SYSTEM/
    00_{RUN_NAME}/
      manifest.yaml
      qe/
      00_bgw/
      01_bgw_{variant_description}/
      00_lorrax/
      01_lorrax_{variant_description}/
    01_{RUN_NAME}/
```

`qe/` is a static reference shared by BGW and LORRAX variants in the same run. Variants are enumerated: `01_bgw_30Ry_epscutoff`, `01_lorrax_600centroids`, etc. New top-level runs (`01_`, `02_`) are for different k-grids, lattice parameters, or band counts.

## `manifest.yaml`

Every run directory must have one. Use `templates/manifest.yaml` or this template:

```yaml
run_id: mos2_3x3_g0w0_2026-04-02
system: MoS2
pipeline: qe_bgw_lorrax   # qe_only | qe_bgw | lorrax_only | qe_bgw_lorrax
platform: perlmutter       # perlmutter | local_wsl | local_mac

variant_of: null           # parent run_id, or null
reuse_from_parent: []      # list of reused paths, e.g. [qe/WFN.h5]
overrides: {}              # what changed vs parent

# Directory-level status. States: pending | running | complete | failed | reused | skipped
steps:
  qe:          { state: complete }
  00_bgw:      { state: complete }
  01_bgw_30Ry: { state: running, note: "testing lower epsilon cutoff" }
  00_lorrax:   { state: failed, note: "OOM on 8GB GPU" }
```

Omit steps that don't apply. The manifest tracks directory-level status, not individual sub-steps — the directory names and their contents are self-documenting.

## Invalidation

Dependency chain: `SCF → NSCF → pw2bgw → epsilon → sigma`, and `NSCF → lorrax`.

- **SCF change**: invalidate everything. New top-level run.
- **NSCF/k-grid/nbnd change**: new top-level run.
- **epsilon.inp change**: new BGW variant (`01_bgw_...`).
- **sigma.inp change**: new BGW variant.
- **LORRAX input change**: new LORRAX variant (`01_lorrax_...`).

Never mutate a completed run. Make a variant.

## Checkpoints and reports

**A major checkpoint** is triggered by any of:
- Completing a top-to-bottom code feature
- ~5 commits of incremental work
- A novel comparison result (pass/fail, new plot)
- Before ending a session

**At each checkpoint, follow `skills/checkpoint/SKILL.md`.** In brief:
1. Run pytest if LORRAX source was modified.
2. Commit to the feature branch.
3. Update or create `reports/<initiative>/report.md` with tables, plots, status.
4. Update `CHANGELOG.md`.

Reports replace STATUS.md as the human-readable orientation point. They should be legible standalone: formatted tables, embedded plots when available, clear next-steps. See `reports/head_fix_2026-04-04/report.md` for an example.

## Debugging

1. Read `manifest.yaml` for the failing run.
2. Tail the failing step's output (10-30 lines).
3. Record hypothesis, then make a variant to test it.
4. Log concisely: command, tail of output, one sentence.

---

## Non-negotiable rules

The following rules are bolded throughout this document. They are restated here because agents routinely forget them. **All of these must be followed in every session, without exception.**

1. **Use the provided parsers for all output extraction.** Avoid ad-hoc parsing code; if novel but important, add new parsing code to the Skill. See `skills/compare/SKILL.md`.
2. **Follow the checkpoint routine every ~5 commits/after minor milestones.** Pytest, commit, report, CHANGELOG. See `skills/checkpoint/SKILL.md`.
3. **Report sandbox infrastructure errors in `KNOWN_SANDBOX_ERRORS.md` immediately** — before continuing your work.
4. **Read `CHANGELOG.md` at the start of every session. Update it before ending every session.**
5. **Read `sources/lorrax/AGENTS.md` before editing any LORRAX source code.**
6. **Create a feature branch before modifying LORRAX source.** Never commit directly to `main`.
