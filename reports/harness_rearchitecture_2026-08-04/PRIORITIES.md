# Priority triage — 2026-08-05

Owner directive: reassess every piece for information density and real
priority. Bar applied: does it remove a MEASURED cost or error class
(jobid/incident on record), or only a plausible one? This file SUPERSEDES
the adoption-order sections in report.md §5, RULES_v2 §J,
EVIDENCE_DESIGN §3 ordering, and ANTIPATTERN_ENFORCEMENT §4 — the fact
that four such lists existed is itself the disease being treated.

Self-criticism first, because the assessment applies to this session's
output: several proposals below fail the measured-cost bar (planted-defect
battery, selftest suite, sentinel cron — plausible, not measured); three
overlapping registries were proposed where one suffices (PATTERNS + TASTE
+ EXEMPLARS); and STATE.md duplicates a section AGENTS.md already has.
Cut accordingly.

## The standing design-process rule (owner directive, permanent)

**Major design changes are designed against the question: "what design
criteria would Jack be most likely to object to?"** — answered in writing
BEFORE implementation, with the anticipated objections either resolved or
surfaced for adjudication. The adjudicated-objection registry (TASTE.md)
is the training set for this question and every adjudication appends to
it. Permanent homes: DESIGN.md §5 (now edited in) and the TASTE.md
preamble. This is the design-time form of the critic loop; it subsumes
several weaker process proposals.

## Tier 0 — precondition (everything below compounds on it)

| Piece | Why it survives the bar |
|---|---|
| Frontera push + repo AGENTS.md truth repair + sandbox pointer reconciliation | Measured: origin at 2026-07-22, entry doc actively wrong, two certified branches unmerged. Not harness work — the precondition for all of it. CONSOLIDATION.md phases 0-2. |

## Tier 1 — first week after the push (each: measured incident, cheap)

| Piece | Evidence | Scope note |
|---|---|---|
| CLAIMS split: one-line rows + claims/NNNN.md with excerpts-at-landing | CLAIMS.md = 68 lines / ~30k tokens; read-truncation already occurred; duplicate row 40 | ledger_lint shrinks to ~20 lines inside gate0 — not its own tool |
| AGENTS.md current-state under a hard line budget, refreshed at checkpoint | AGENTS says 07-31 while CLAIMS is at 08-04 | **STATE.md as a separate file: CUT** — it duplicates an existing AGENTS.md section; fix the section, don't add a file |
| run_record.py in the sbatch template (per-job JSONL) | CLAIMS-38 assembled by hand; sacct 700x trap; hand-logging varies by agent | ~100 lines. trend.py DEFERRED until the jsonl exists and a real question needs it |
| DESIGN.md lands (owner-edited) | The placement anchor every rule cites; encodes owner's three-layer spec | Hard 2-page cap — it is the context-budget flagship |
| gate0.sh (AST suites + rules_gate + ledger check, one command) | Folklore multi-step login ritual today | Wrapper only — no new checks beyond what exists |
| fastloop→checkpoint wiring | Open since 07-31; one docs edit | — |

## Tier 2 — first month, attached to work that touches them anyway

| Piece | Evidence | Scope note |
|---|---|---|
| PostToolUse hook running gate0-fast on edited files | Parity matrix: violations detected post-hoc cost full re-sessions | Ship with the EXISTING checks; grow rules later |
| First ~5 ast-grep rules (time-arithmetic, getattr-default, except-swallow, blocking-read-in-loop, cache-without-precompile) | Each has a named incident/live violation in the audit | One-per-incident thereafter. The 13-rule list is a BACKLOG, not a launch set |
| F0 refuse() helper + F1-F5 runtime refusals | dipole-staleness & band-window classes have shipped wrong-but-plausible results; rank_criterion proves the pattern | — |
| σ^B/v_q_bispinor timing.section fix, THEN the capability-parity gate | KNOWN row: audit had to project a stage it couldn't measure | Fix the 3 files first; the gate prevents recurrence |
| harness/: exec launcher + VmHWM sampler + ab_legs matrix runner | sacct trap (7885315), coordinator leak (7885122), 97-false-reds (7885150) — all instrument incidents | parity.py stays inside fastloop until a second consumer exists |
| fastloop: per-stage HLO forbid + BSE stage | Blockers closed (62ba395); BSE ring class was a late catch | — |
| Scoped AGENTS.md (common/, ffi/) with exemplar pointers | Propagation study: routing failure, zero-caller helpers | **EXEMPLARS.md as separate file: FOLD** into these scoped files — one less root file to know about |
| TASTE.md — the ONE judgment registry (objection-question preamble + adjudicated objections + transferable judgment lessons) | feedback_* pins exist as dangling references; owner runs the objection question ad hoc today | **PATTERNS.md as separate file: FOLD in** — three registries → one |
| RUNS_INFLIGHT.md | mos2_4x4_test contention; cross-leg interference on record | One file, append/strike |

## Tier 3 — opportunistic (do when adjacent work makes them free)

- new_stage scaffold: pays when bispinor phase 2 creates stages; build it
  THEN, from the then-current exemplar.
- Extractions (AsyncShardCollector, _cached_jit, make_rs_contraction,
  aot_kernel_peak_bytes adoption): real, but they are REPO PERF WORK that
  belongs to the σ^B/KNOWN backlog — do them while fixing those rows, and
  the corresponding ast-grep rules turn on afterward.
- jaxtyping: adopt on exemplar signatures as they're touched; the
  new-code rule only after a critical mass exists.
- GATES→machine-read reference dict: GATES.md is 37 lines and fastloop
  pins already carry tolerances — defer until a second parity consumer
  actually reads it.
- Stale-docstring-crossref check: 20 lines, three known hits — fold into
  gate0 when convenient.

## Cut or indefinitely deferred (fails the measured-cost bar today)

| Piece | Why cut |
|---|---|
| Planted-defect battery | Expensive (agent sessions + jobs); measures a risk not yet observed under the new stack. Revisit only if the stack demonstrably leaks |
| Sentinel cron fastloop | Spends a scarce queue slot for drift detection every campaign run already provides implicitly; no machine-drift incident on record |
| tools/selftest suite | Over-built for 2 tools; existing practice (certify instrument at creation with a jobid) covers it. Exception: one fixture for the HLO analyzer's --forbid path, which IS flagged never-run |
| import-linter / tach | The Frontera AST layering gates exist and pass (68/34/9); replacing working bespoke gates with a dependency is negative value. Revisit only for the public-interface feature if private-helper imports become an observed class |
| Semgrep, asv step detection, metrics dashboards | No engine second to ast-grep; no statistics before a history exists; no dashboard before a reader |
| STATE.md, PATTERNS.md, EXEMPLARS.md as separate files | Folded (above): registry and file count are themselves context costs |

## Net architecture after triage

Tier 0-2 leaves: the push; a 4-file sandbox record layer (thin CLAIMS +
claim files, KNOWN, GATES, TASTE) + jsonl records; DESIGN.md + scoped
AGENTS.md in the repo; gate0 + hook + ~7 gates/rules; F-refusals; 3
harness instruments; fastloop completed. Roughly HALF the pieces the
TARGET_ARCHITECTURE capstone lists — the other half is now explicitly
tier-3/cut with its reinstatement condition stated. The five compounding
loops survive intact; they never depended on the cut pieces.
