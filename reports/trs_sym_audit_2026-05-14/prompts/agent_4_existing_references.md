# Agent 4 — Existing nosym reference audit (no new BGW runs)

You are Agent 4. Read `STATUS.md`. **Do not run BGW. Do not regenerate references.** Your job is to mine the existing run directory tree for pre-existing sym-vs-nosym pairs and use them to bound the impact of this bug — before and after the Agent 2 patch lands.

## Why this matters

This TRS-blind bug has likely been present for some time, partially masking BGW comparisons on non-inversion systems. Several existing run dirs already contain paired sym/nosym calculations on the same physics — those pairs are gold for measuring the bug's actual production impact.

## Inventory (from STATUS.md, refined as you go)

The most useful pairs (in priority order):

1. **`runs/MoS2/02_mos2_3x3_nosym` ↔ `runs/MoS2/00_mos2_3x3_cohsex` (with sym)**. Same system, sym/nosym pair. Expected: if the bug were absent, Σ should agree to convergence-level (small ζ-basis dep, ~ meV). If the bug is firing, agreement degrades for any non-inversion physics.

2. **`runs/Si_pseudobands/00_si_2x2x2_60Ry/21_lorrax_cohsex_nosym_parity` ↔ corresponding sym variant in the same parent**. Explicitly named "parity" suggests it was set up as a sym-vs-nosym agreement test. Read it.

3. **`runs/Si/02_si_4x4x4_nosym` ↔ `runs/Si/01_si_4x4x4_nosymmorphic`**. Si has inversion → must agree regardless of TRS bug. Use as a **null check**: if these disagree by more than the convergence noise, there's another bug we haven't found.

4. The pseudoband 4-grid: `runs/Si_pseudobands/00_si_2x2x2_60Ry/{26,27,28,29}_cohsex_pb_*_nosym*` and matching sym variants. Useful for separating "pseudoband normalization" effects from "TRS bug" effects.

## Deliverables

`reports/trs_sym_audit_2026-05-14/agent_4_reference_audit.md` with:

1. **For each pair**: parse the existing GW output files (eqp0.dat, sigma_freq_debug.dat — see `skills/compare/SKILL.md` for parsers), build a per-(k,n) Σ_X (and Σ if available) comparison, report max|Δ| and the pattern.

2. **Bug-pre-fix damage assessment**: rank the existing pairs by "how much disagreement looks consistent with the TRS bug" (large for non-inversion systems, small for inversion-symmetric Si). Existing CrI3 comparisons across the codebase are also fair game.

3. **Post-fix re-evaluation plan**: identify the minimal set of LORRAX reruns needed AFTER Agent 2's patch lands to confirm the bug is the explanation. Reuse existing QE outputs; only rerun the LORRAX step. Do NOT propose any BGW reruns.

## Hard constraints

- **NO BGW reruns.** Every BGW reference is treated as ground truth as-is.
- **Use the parsers in `skills/compare/SKILL.md`** — see Non-negotiable rule #1 in `AGENTS.md`. Don't roll your own.
- For each "this pair shows the TRS bug" claim, **back it with a concrete numerical comparison** (column from sigma_freq_debug.dat at specific (k, n), reference value, current value, delta).
- If a "nosym" run dir turns out to actually be a different physics setup (not just a sym flag toggle), say so and move on.

## Done criterion

Report exists, contains the damage assessment table, ping `discussion.md` with `[Agent 4] existing-ref audit complete; N pairs measured, M consistent with TRS bug`. Independent of Agents 1-3; start immediately.

## What you have access to

Full run dir tree under `/pscratch/sd/j/jackm/lorrax_sandbox/runs/`. Parsers under `skills/compare/`. SLURM alloc `52953227` if you need to re-extract anything (probably not — outputs already on disk).
