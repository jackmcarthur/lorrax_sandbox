# Harness re-architecture proposal — 2026-08-04

Grounded in Lilian Weng, "Harness Engineering for Self-Improvement"
(lilianweng.github.io/posts/2026-07-04-harness/, read in full) against the
current state of this sandbox (AGENTS.md, the three ledgers,
KNOWN_LORRAX_ISSUES, fastloop/PLAN.md, ASSERTIONS.md as of commit
739e4960). Written off-cluster; nothing here carries a jobid, so per rule 1
every item below is a PROPOSAL, not a claim. Adopt piecewise; each item
states its evidence and its expected effect on dev-cycle speed, dev-cycle
quality, or code performance.

## 1. The post, compressed to what is actionable here

Weng's frame: a harness is "the system surrounding a base model that
orchestrates execution and decides how the model thinks and plans, calls
tools and acts, perceives and manages context, stores artifacts, and
evaluates results" — and it moves capability as much as the model does.
Three foundational design patterns: (1) **workflow automation** — a
goal-oriented plan/execute/test/improve loop through an agent runtime;
(2) **file system as persistent memory** — durable state in files, never
in transient context, because long-horizon artifacts outgrow any context
window; (3) **sub-agents and backend jobs** — parallel hypothesis testing
with a small process manager, where the key design choice is that
parallelism is "explicit and inspectable ... stored as files, logs, and
status records" so the system can recover after interruption.

The optimization frontier she surveys, in ascending order of what is being
optimized (prompts → structured context → workflow → harness code →
optimizer code):

- **ACE** (Agentic Context Engineering): context as an evolving playbook
  of itemized bullets, maintained by three roles — Generator (produces
  trajectories), Reflector (distills insights from success/failure),
  Curator (merges itemized entries with deterministic logic, periodic
  dedup/refine). The key anti-pattern it names: append-only context piles
  and full-blob rewrites (which cause "context collapse and brevity bias").
- **Self-Harness**: a propose-evaluate-accept loop — *weakness mining*
  (cluster failures into verifier-grounded patterns; a rich failure record
  is required because two runs can share a surface error with different
  causal mechanisms), *bounded harness proposal* (editable surfaces made
  explicit; records of passing behavior that must be preserved; summaries
  of previously attempted edits), *proposal validation* (held-in tests for
  "is the weakness fixed", held-out tests for "did we break anything";
  accept only with no regression on both).
- **AHE** (Agentic Harness Engineering): the bottleneck of harness
  evolution is **observability**, with three pillars — *component*
  observability (every editable component has a file-system
  representation), *experience* observability (raw trajectories summarized
  into a layered hierarchy: per-task root-cause reports → aggregated
  overview, raw traces still accessible — "layered access is more token
  efficient"), *decision* observability (every edit is an evidence-driven,
  falsifiable claim carrying: the failure evidence, inferred root cause,
  targeted fix, and a **predicted impact** — expected fixes AND at-risk
  regressions — validated next round; the verifier and runs directory are
  read-only to the loop, so gains stay attributable).
- **ADAS / AFlow / AlphaEvolve / DGM**: workflow and harness design as
  evolutionary/search problems. Her own caveat: this family "works well
  when candidate solutions are automatically evaluable ... it struggles
  with domains where evaluation is slow, ambiguous, or mostly
  heuristic-based."
- **STOP**'s cautionary result: recursive improvement helped with a strong
  base model and *degraded* with weaker ones — "recursive structure alone
  is not enough."

Her future-challenges list reads like this project's founding documents:
weak evaluators are the bottleneck; "long-horizon projects lose critical
details unless logs are written as persistent artifacts"; models
exhibit over-optimism ("numerical duct tape and declare victory while
signals are still noise" — Bubeck's "p-hacking and eureka-ing"); a
research harness "should make failed attempts easy to preserve"; the
evaluator and permission control must sit outside the loop being evolved;
humans "move up the stack, not out of the loop."

## 2. Scorecard: what this sandbox already is

Read against that taxonomy, this sandbox is already an unusually complete
harness — several of the post's prescriptions exist here in mature form,
and that is worth stating before changing anything:

| Post concept | Sandbox implementation | Status |
|---|---|---|
| File system as persistent memory | CLAIMS/GATES/INVARIANTS/KNOWN ledgers; scorecard; run trees + artifact paths | STRONG — but the memory is outgrowing its read path (§3.1) |
| Workflow automation loop | fastloop (~3 min, both legs, pinned refs, exit-code contract); AST gates; A/B campaigns | STRONG at the ends; the ladder between rungs is implicit (§3.3) |
| Evaluator outside the loop | pinned baselines re-pinned only from certified runs with jobid; BGW reference read-only; "a fastloop failure is a record, a pass is just a gate" | STRONG — this is AHE's read-only-verifier rule, independently derived |
| Against over-optimism / "declare victory on noise" | rule 1 (outputs read from disk, never predicted), rule 2 (verify the instrument), significance bounds (1e-3 eV) vs gauge classes | STRONG |
| Negative results preserved | REFUTED verdicts kept forever (CLAIMS 4, 7, 21-premise, BC-reading, docstring refutations) | STRONG — the post calls this out as rare and load-bearing |
| Decision observability | CLAIMS rows ARE evidence-driven manifesto entries (evidence, root cause, fix, measurement) — but the *predicted impact before the run* is only sometimes present | PARTIAL (§3.5) |
| Experience observability | raw logs on scratch + RESULTS.md per campaign; but no layered per-stage aggregate — the 9400-line scorecard is the "overview" | PARTIAL (§3.7) |
| Sub-agents / backend jobs, explicit + inspectable | ~3 concurrent agents; hand-authored sbatch harnesses; intent is invisible across agents | AD HOC (§3.2, §3.4) |
| Curator role / context playbook | the 2026-07-31 rebuild was a one-shot Curator pass; nothing keeps it curated as claims accrete | EPISODIC (§3.1, §3.5) |
| Weakness mining | KNOWN_LORRAX_ISSUES is a verifier-grounded failure-pattern register per driver | STRONG in content, drifting in form (§3.1) |

The overall shape is right. The proposals below are about scaling laws:
every mechanism that works at 20 claims and 3 agents has a term that blows
up at 100 claims and 5 agents, and several are already visibly binding.

## 3. Proposals, in priority order

### 3.1 P1 — Split evidence from memory: the ledgers are becoming context bombs

**Evidence.** CLAIMS.md is 68 lines but ~30k tokens — a single row (48) is
~700 words; reading the file now exceeds a 25k-token tool read in one
pass, and the "new session read order ... ~225 lines total" promise in
AGENTS.md is already false in token terms. KNOWN_LORRAX_ISSUES rows carry
entire fix narratives inline (the FIXED rows are the longest things in the
file — the opposite of what a live defect register needs). Two CLAIMS rows
are both numbered 40: hand-maintained structure is drifting exactly as
volume grows. This is the failure mode ACE is built against: context kept
as an ever-lengthening blob instead of itemized entries merged by
deterministic logic — and the post's warning that long-horizon work
degrades "unless logs are written as persistent artifacts" has a dual:
persistent artifacts degrade the *read* path unless curated.

**Proposal.** Keep the append-never-delete evidence guarantee; move the
bulk out of the hot read path (ACE's itemized-playbook structure):

- CLAIMS.md rows become **one line each**: date, ≤25-word claim, verdict,
  jobid, pointer to `claims/NNNN.md` holding the full narrative (current
  long rows migrate verbatim — nothing is lost, it just stops being
  mandatory reading). Same for KNOWN rows: status + smallest-fix +
  pointer; a FIXED row shrinks to one line pointing at its claim file.
- Add a curated `STATE.md` with a **hard budget (~100 lines)**: the
  playbook view — current architecture rulings, live blockers, the 5 most
  recent verdicts, what is in flight in which worktree. Regenerated at
  every checkpoint (Curator pass, §3.5); AGENTS.md's "Current state"
  section becomes a pointer at it, ending the drift where AGENTS.md says
  2026-07-31 while CLAIMS is at 2026-08-04.
- Add `tools/ledger_lint.py` (login-python-3.7 clean, like the AST
  gates): refuses duplicate row numbers, rows over a length cap, claims
  without a jobid, KNOWN rows whose status contradicts their pointer
  target. ACE's point that the Curator merges "with deterministic logic"
  is exactly this: the harness already enforces code structure
  mechanically (AST gates); it should enforce its own memory structure the
  same way.

**Effect.** Restores O(1) session orientation as the ledger grows O(n);
speeds every future session's spin-up and keeps "was X measured?"
answerable by grep against one-line rows.

### 3.2 P2 — A certified instrument library: stop re-deriving harnesses per campaign

**Evidence.** Rule 2 exists because "broken harnesses have historically
produced more false results than the code under test" — and the ledger
keeps proving it: the sacct-vs-VmHWM ~700x undersample (CLAIMS 40) came
from a campaign harness that didn't `exec` python; job 7885150's 97 false
reds came from LD_LIBRARY_PATH dropping inside the container; job
7885122's ladder was invalidated by a leaked JAX_COORDINATOR_ADDRESS; the
phdf5 hang class (CLAIMS 48) was only catchable because that harness
happened to carry per-rank stderr files and external timeouts. Every
campaign currently re-authors `inner_common.sh`-style plumbing from
scratch, and every re-authoring re-rolls these dice. In the post's terms:
the runtime layer of this harness is being regenerated per rollout instead
of being a fixed, verified component.

**Proposal.** Promote the patterns that recur in the evidence columns into
a `harness/` toolkit in this sandbox — small, boring, certified once:

- `lorrax_exec.sh` — the exec-python launch (the one reason historical
  sacct numbers are real), transport env sourcing, container binds.
- `vmhwm_sampler.sh` — /proc VmHWM sampling for any harness that cannot
  exec (mechanizes the CLAIMS-40 RULE).
- `with_timeout_per_case.sh` + per-rank stdout/stderr capture — the
  certified phdf5_padrank method (CLAIMS 48 names it as the instrument,
  "because the failure under test is a hang").
- `ab_legs.py` — the A/B/N-leg matrix runner: takes N (env, deck-key)
  variants, runs them in one allocation with a fresh env per leg, emits
  the parity table and a RESULTS.md skeleton. This is the single
  most-repeated structure in the ledger (legs appear in CLAIMS 23, 24,
  27, 29, 34, 40, 44, 45, 47, 49...) and it is rewritten every time.
- `parity.py` — the compare_valsmoke-lineage comparator fastloop already
  embeds, importable instead of copy-adapted.

Certify the toolkit once with its own tiny job (fixtures + known-answer
checks), record the jobid, and make "campaign harnesses compose certified
pieces" a rule. ASSERTIONS.md's standing caveat ("some are syntax-checked,
never run") then applies to a shrinking set.

**Effect.** Dev-cycle quality: the instrument-error class — historically
the largest false-result source — gets amortized to zero marginal cost.
Speed: campaign setup drops from authoring to composition.

### 3.3 P3 — Name the verification ladder, and finish fastloop's own TODO list

**Evidence.** The post's Self-Harness validation stage is a two-split
regression discipline: held-in (does the change do what it claims) and
held-out (did it break anything else), accept only when both are clean.
This sandbox has the pieces — AST gates (seconds, login) → fastloop
(~3 min, 1 node, the held-out split for driver changes) → b300 A/B
(~15 min) → b600/P=64 → bispinor production — but the ladder is implicit
folklore, and fastloop/PLAN.md's honest list has three items open since
2026-07-31: wiring into skills/checkpoint as the mandatory pre-commit gate
(item 3, a docs change); per-stage HLO forbid gates (item 2 — the blocking
open gathers in htransform closed with 62ba395, so the per-stage scoping
it describes is now feasible); no BSE stage (item 4 — the BSE
ring-transport red class in KNOWN would have been a commit-time catch
instead of a pytest-census discovery).

**Proposal.**

- Write the ladder down in AGENTS.md as the harness's verification
  hierarchy, with cost and what each rung can and cannot falsify
  ("cheapest sufficient rung before escalating" as an operational rule).
- Close fastloop item 3 (checkpoint wiring — one docs edit).
- Implement item 2 now that it is unblocked: `--hlo-forbid` per stage,
  hard exit, with the documented 2-all-to-all reshard exemption.
- Extend the chain with a minimal BSE/absorption stage (item 4) — the
  open decision is just which outputs to pin; propose: mini-deck
  absorption eps2 at 2 pinned frequencies + exciton ground-state energy,
  same tolerance regime as the existing pins.
- Add a **tier-0 that runs where the agent lives**: one login-safe
  command (`tools/gate0.sh`) bundling the AST suites + py_compile +
  ledger lint (§3.1), so the seconds-scale rung is one invocation instead
  of a remembered list.

**Effect.** Speed: fewer full-job round trips for defect classes a lower
rung catches (the two 2026-07-31 fastloop catches were exactly this).
Quality: sharding regressions become commit-time failures, not campaign
archaeology.

### 3.4 P4 — Backend-job management: make parallelism explicit and inspectable

**Evidence.** The post's Pattern 3 asks for a "small process manager" and
for parallel work to be "stored as files, logs, and status records" so it
survives interruption and stays visible. Here, the binding resource is not
node-hours but **queue slots** (dev queue: 2 jobs; RLIMIT_NPROC 300; ~3
concurrent agents) and wall-clock latency per round trip. The ledger shows
hypothesis-packing already being improvised — job 7885987 packed five GW
legs into one allocation; 7888568 packed a 3-leg gate + reader matrix —
while cross-agent coordination is informal prose ("in active use by
another agent; treat read-only") and one cross-leg interference class
(the 7885122 coordinator-address leak) is already on record.

**Proposal.**

- Make hypothesis-packing the default, not the improvisation: the
  `ab_legs.py` runner (§3.2) plus a note in skills/execute_workflow —
  "one allocation, N legs, one RESULTS.md" — so each queue slot retires
  several hypotheses. Bake the env-hygiene lesson (fresh env per leg, no
  leaked coordinator/cache variables) into the runner itself.
- A tiny in-flight board, `RUNS_INFLIGHT.md`: jobid, agent, purpose,
  scratch dir, expected artifacts, claimed shared resources (e.g.
  mos2_4x4_test). Append on submit, strike on ledger landing. Three
  agents polling squeue independently can see each other's *jobs* but not
  each other's *intent*, and intent is what prevents read-only-tree
  collisions and duplicate work.

**Effect.** Throughput per queue slot goes up severalfold; the
cross-agent interference class gets a mechanical guard.

### 3.5 P5 — Institutionalize the Reflector, and adopt AHE's predicted-impact discipline

**Evidence.** The ledger's most valuable long-term content is not the
numbers but the *transferable mechanism lessons*: sacct lies unless you
exec; a positive diagonal ridge on an indefinite matrix is not a
regularizer; warm HLO tables under-report; content-hash beats count in
provenance (row 41 cites "pattern #10" as if a numbered pattern registry
existed — it doesn't, yet). Today these live as RULE verdicts scattered
through CLAIMS rows; a new session finds them only by grepping the right
word. Separately, AHE's decision-observability pillar — every edit
carries a **prediction** (expected fixes AND at-risk regressions) that
the next round validates — is only sometimes practiced here: INVARIANTS
row 9 requires stating scaling over the design envelope, and the best
campaigns state expectations (CLAIMS 38's "vs mu^3 = 4.13x" columns), but
it is a habit, not a contract.

**Proposal.**

- `PATTERNS.md` — the numbered registry row 41 already believes exists:
  one entry per transferable mechanism (instrument traps, numerical-gauge
  classes, provenance rules, sharding idioms), each ≤10 lines with
  pointers to the claims that earned it. This is ACE's playbook of
  itemized bullets, applied to judgment rather than facts.
- Extend `skills/checkpoint/SKILL.md` with an explicit end-of-session
  Reflector step: new RULE-class findings go to PATTERNS.md or
  INVARIANTS.md; STATE.md is regenerated within its budget; ledger lint
  runs. Checkpoint is already the natural Curator hook; this makes the
  roles explicit instead of relying on each agent's taste.
- Adopt the predicted-impact convention for perf/architecture changes:
  the claim file (§3.1) for a change opens with its prediction —
  expected win, expected scaling, at-risk regressions — written BEFORE
  the job runs, with the measurement appended after. CLAIMS 38 shows the
  payoff: expectation columns are what let it rank the three
  worse-than-expected terms instantly. Costs one paragraph; makes every
  run a falsifiable experiment rather than a fishing trip.

**Effect.** Quality compounding: the harness accumulates *judgment* in a
place cheap enough to always read, instead of re-purchasing it with jobs
(the ridge-is-inert result took an 11-minute 32-node leg; its lesson
should never need re-deriving).

### 3.6 P6 — Mechanize rule 2: a self-test for the harness itself

**Evidence.** ASSERTIONS.md exists precisely because sandbox tools have
shipped syntax-checked-but-never-run (the HLO analyzer's forbid gate is
flagged "first user should treat its output with rule-2 skepticism").
The post's Self-Harness/AHE loops all assume the verifier itself is
trustworthy and *outside* the evolving surface — which requires the
verifier to have its own tests.

**Proposal.** `tools/selftest/` — checked-in fixtures (a captured HLO
dump with known collectives; a pair of .h5/.dat files with a known
injected delta; a fake sacct/VmHWM transcript) + one small in-container
job that runs every sandbox tool against them and diffs known answers.
Run it when a tool changes and at every re-pin. ASSERTIONS.md's per-tool
caveats collapse to one line: "selftest green at jobid X".

**Effect.** The false-verdict class from unverified instruments becomes a
gated impossibility rather than a standing warning.

### 3.7 P7 — Layered experience observability: machine-readable run metrics

**Evidence.** AHE's experience-observability pillar prescribes exactly
the structure this sandbox lacks in one dimension: raw trajectories →
per-task analysis → aggregated overview, each layer smaller than the
last, raw layer still reachable. For *failures* that structure exists
(logs → RESULTS.md → CLAIMS row). For *performance* it does not:
timing tables live in prose inside job logs and a 9400-line grep-only
scorecard, so questions the campaign asks constantly — "what did
sigma.exec cost at each certified size", "is the planner still 7x over
at low P", "which stage regressed between bundles" — each require
archaeology. CLAIMS 38 (the b300→b600 scaling map) had to be assembled by
hand and is already the most-cited row in KNOWN.

**Proposal.**

- Every certified run appends one JSON line (jobid, src hash, deck, P,
  per-stage walls, per-rank VmHWM, planner estimate, parity verdict) to
  `runs/metrics.jsonl` — emitted by the §3.2 toolkit so it costs nothing
  per campaign. The timing tables already exist in every log; this is a
  parser plus a contract, not new instrumentation.
- `tools/trend.py`: given a stage name, print its wall/memory across
  certified runs. The planner-calibration ratio (est/measured VmHWM —
  7.3x over at b300/P=16 vs 1.06x at b600/P=64, KNOWN gw row) becomes a
  standing column, so the open "re-fit the binder" item gets calibration
  data for free from every future run.

**Effect.** Code performance directly: regressions surface at the next
run instead of the next audit, and every perf claim's "vs what baseline"
question gets a one-command answer.

## 4. Anti-goals — where the post itself says stop

- **No evolutionary/meta-search over this harness.** Weng: the
  ADAS/AlphaEvolve/DGM family "struggles with domains where evaluation is
  slow, ambiguous, or mostly heuristic-based" — here one evaluation is a
  32-node job behind a 2-slot queue. Harness edits should stay
  Self-Harness-shaped instead: mined from actual failures, bounded,
  validated against fastloop + a production A/B (held-in/held-out), one
  at a time.
- **No framework, no database.** The sandbox's strength is plain files +
  sbatch + exit codes, readable by login python 3.7 and a human under
  deadline. The post's OS analogy cuts this way: encapsulate complexity,
  keep the interface simple; and "smarter models keep harnesses simple" —
  every mechanism above is a text file plus at most one small script.
- **Never automate the verdict.** The evaluator must sit outside the
  loop being improved (the post's reward-hacking section; AHE makes the
  verifier read-only). Here that means: automate *collection* (metrics,
  lint, selftest, matrix runs) but a claim's verdict is always written by
  an agent/human against on-disk artifacts, and pins are only re-pinned
  from certified runs — exactly as rule 1 and the fastloop re-pin rule
  already state. Nothing proposed above touches this.
- **STOP's caveat applies to ambition level.** Recursive structure alone
  is not enough; harness investment pays off only where it removes a
  measured cost or error class. Every proposal above cites its ledger
  evidence; anything that can't should stay out of the harness.

## 5. Suggested order of adoption

1. §3.1 ledger split + lint and §3.3's checkpoint wiring — docs-and-
   scripts only, no jobs, immediate context relief.
2. §3.2 toolkit + §3.6 selftest — one small certification job together.
3. §3.3 fastloop items (HLO forbid, BSE stage) — the next time a driver
   change needs a fastloop run anyway.
4. §3.7 metrics — start emitting from the next campaign; backfill later
   from existing logs only if ever needed.
5. §3.4 in-flight board and §3.5 PATTERNS.md + predicted-impact
   convention — adopt at the next checkpoint; they cost one file each.
