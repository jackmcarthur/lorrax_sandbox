# Target harness architecture — capstone summary, 2026-08-05

The intended end-state across sandbox + repo, consolidating every design
in this report directory. Five functions; each piece listed with what it
contains and which document specifies it. Items marked [exists] are live
today; [built] were produced this session; everything else is specified
but not yet implemented.

## Function 1 — Orientation & context (what an agent reads, and when)

| Piece | Contains | Spec |
|---|---|---|
| repo `docs/architecture/DESIGN.md` | Read-first, ~2 pages: three-layer spec (drivers/algorithms/plumbing), microservice rule, O(N³) two-sums identity, design-envelope statement, "traced graph is the bill" JAX discipline, mesh/process/env rules, sharding verification recipe, flag-don't-bend | DESIGN_draft.md [built, needs owner edit] |
| Scoped AGENTS.md / path-scoped rules per package | Local idioms only (FFT rules in src/common, FFI in src/ffi, symmetry incl. the k_irr↔k_full idiom pair), each ≤ a screen, every rule a pointer to an exemplar | RULES_v2 §0 placement map |
| `EXEMPLARS.md` | Operation → blessed implementation index, one line each; the routing fix for the propagation failure | ANTIPATTERN_ENFORCEMENT R2 |
| sandbox `AGENTS.md` | Small stable entry: tree-of-truth, read order, operational rules; current-state section becomes a pointer to STATE.md | report.md §3.1 [exists, to shrink] |
| `STATE.md` | Curated ≤100-line playbook: live rulings, blockers, 5 latest verdicts, in-flight work; regenerated at every checkpoint | report.md §3.1 |

## Function 2 — Rules & enforcement (how conventions bind)

| Piece | Contains | Spec |
|---|---|---|
| ast-grep rule corpus (+ rules_gate ratchet) | Structural antipattern rules, one per incident, message names the blessed replacement; 13 seed rules from the idiom audit; allowlists only shrink | ANTIPATTERN_ENFORCEMENT R4; rules_gate.py [built, 2 rules live] |
| PostToolUse hook dispatcher | Runs fast checkers on each edited file, feeds failures + fix pointers back into the agent loop — detection while intent is still in context | ANTIPATTERN_ENFORCEMENT R3 (highest leverage) |
| import contracts (import-linter or tach) | L1/L2/L3 layering as declarative config; per-module public interfaces block private-helper imports | ANTIPATTERN_ENFORCEMENT R5 |
| Runtime refusals | `refuse(rule_id, got, want, fix, doc)` helper + physics preconditions bound at driver startup: centroid⊇sigma window, dipole.h5 provenance stamp, htransform all-valence, degeneracy-respecting subspaces, TRS scoping | RULES_v2 §F |
| jaxtyping annotations | Shape/axis contracts on public array functions, trace-time-only cost; conventions travel inside copied signatures | ANTIPATTERN_ENFORCEMENT R7 |
| Capability-parity gate | Every L1 stage must register: dotted timing.section, planner peak row, precompile entry, docstring sharding table | ANTIPATTERN_ENFORCEMENT R7 |
| Scaffolds (`new_stage`, `new_ffi_target`) | Stamp the exemplar skeleton so idioms are generated, not remembered | ANTIPATTERN_ENFORCEMENT R1 |
| Extraction worklist | AsyncShardCollector+Sink → common/; `_cached_jit` promotion; parameterized `make_rs_contraction`; adopt `aot_kernel_peak_bytes` | ANTIPATTERN_ENFORCEMENT R6 |
| `TASTE.md` | Judgment-only rules (altitude, register, flag-don't-bend) + adjudicated-objection registry feeding the critic pass; recurrent entries graduate to gates | RULES_v2 §0; report.md §3.5 |

## Function 3 — Verification (the ladder)

| Piece | Contains | Spec |
|---|---|---|
| `tools/gate0.sh` | One login-safe command: AST suites + py_compile + rules_gate + ledger lint (seconds) | report.md §3.3 |
| fastloop | Mini-deck full-chain semantic check, both legs, pinned refs (~3 min) [exists]; extend: per-stage HLO forbid, BSE stage, checkpoint wiring | fastloop/PLAN.md + report.md §3.3 |
| Invariance-gate idiom | "Two paths must agree, from prepared state" — preferred shape for new gates over frozen refs | RULES_seed §8b [exists in repo] |
| Campaign tiers | b300 A/B (~15 min) → b600/P=64 → production; cheapest-sufficient-rung rule written in AGENTS.md | report.md §3.3 |
| `tools/selftest/` | Fixtures + known answers for every sandbox instrument; discharges ASSERTIONS caveats | report.md §3.6 |
| Planted-defect battery | Seeded known defects + fresh agent localization trials; the eval of the whole harness | report.md discussion; ANTIPATTERN_ENFORCEMENT R8 |

## Function 4 — Evidence & memory

| Piece | Contains | Spec |
|---|---|---|
| `runs/records/*.jsonl` | One machine-written record per job, appended by the sbatch template: jobid, src sha (dirty refusal), deck content hash, system:partition + env fingerprint, exit, sacct+VmHWM, stage walls, artifact paths with durability class (retrieve/stash/expendable), declared context breaks | EVIDENCE_DESIGN §3 |
| CLAIMS.md + `claims/NNNN.md` | One-line judgment rows (claim, verdict, jobid) + per-claim detail files embedding retrieve-class excerpts captured at landing; append-only, REFUTED forever | report.md §3.1 + EVIDENCE_DESIGN |
| GATES as reference dict | Metric, baseline, ±tolerance, unit, scope, blessed-by — read by the parity harness; scripted re-pin | EVIDENCE_DESIGN §3 |
| KNOWN_LORRAX_ISSUES | Thin live rows (status + smallest fix + mechanism-class tag + pointer); fix narratives live in claim files | report.md §3.1 + review feedback |
| `PATTERNS.md` | Numbered transferable-mechanism registry (instrument traps, gauge classes, provenance rules), ≤10 lines each, fed by the checkpoint Reflector step | report.md §3.5 |
| `tools/ledger_lint.py` + `tools/trend.py` | Ledger structure enforcement; per-stage wall/memory history over the jsonl | report.md §3.1/§3.7 |

## Function 5 — Execution & job management

| Piece | Contains | Spec |
|---|---|---|
| `harness/` instrument library | Certified-once composable pieces: exec-python launcher, VmHWM sampler, per-case timeouts + per-rank capture, `ab_legs.py` N-leg matrix runner (fresh env per leg), `parity.py` comparator | report.md §3.2 |
| `RUNS_INFLIGHT.md` | Jobid, agent, purpose, scratch dir, predicted outcome, claimed shared resources; append on submit, strike on landing | report.md §3.4 |
| Sentinel fastloop | Scheduled run of the fixed deck, tagged in the records — machine-drift detector | EVIDENCE_DESIGN §3 |
| Skills | build_inputs / execute_workflow / compare / checkpoint [exist]; checkpoint gains Reflector+Curator+lint steps; mechanical steps promoted to refusing CLIs | report.md §3.5 + owner directives |

## The loops that make it compound

1. **Edit loop**: hook catches violation → fix in one turn (R3).
2. **Incident loop**: caught antipattern → ast-grep rule or TASTE entry,
   ledgered → 0-recurrence class (R4 discipline).
3. **Checkpoint loop**: Reflector extracts lessons → PATTERNS/STATE
   regenerated under budget → next session orients O(1).
4. **Campaign loop**: hypothesis + predicted impact → packed matrix job →
   records jsonl + claim verdict → trend/regression detection.
5. **Harness eval loop**: planted defects measure whether 1-4 work.

## Standing constraints (the anti-goals, unchanged)

Plain files + sbatch + exit codes only — no daemons, DBs, frameworks, or
hosted services (environment-disqualified and strategy-disqualified).
Verdicts are never automated. Prose never grows: new conventions land as
rule/template/helper + one pointer line. Every harness investment must
name the measured cost or error class it removes.
