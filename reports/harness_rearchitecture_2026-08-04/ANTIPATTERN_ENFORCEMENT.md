# Antipattern enforcement & idiom propagation — study synthesis, 2026-08-05

Owner question: how do we enforce agents not falling into common
antipatterns — infrastructure going far beyond grep gates — given that the
lessons in gw/psp keep failing to propagate to other areas?

Two studies feed this: (A) an idiom audit of the codebase — 18 named
exemplar idioms cataloged from gw/psp/common with file:line, checked for
presence across 9 lagging modules (full catalog + parity matrix in the
study output; highlights inline below); (B) a research sweep of
enforcement mechanisms (ast-grep/Semgrep, import-linter/tach, jaxtyping,
scaffolding/paved-road practice, Claude Code hooks, and the 2025-26
agent-adherence literature). Origin-tree caveat applies as usual.

## 1. The diagnosis: three propagation failures, not one

The parity matrix is stark — of 18 exemplar idioms, the median lagging
module has ~2. But WHY they're missing splits into three mechanically
distinct classes, and each needs different infrastructure:

**Class 1 — importable helpers that simply aren't imported.** The idiom
is a zero-domain-coupling shared helper; absence is literally a missing
import. Evidence of how bad this is:
- `runtime/aot_memory.py::aot_kernel_peak_bytes` has **zero callers**
  repo-wide, despite a docstring naming four intended call sites —
  and `bse/exciton_bands.py:753` hand-rolls the exact inferior version
  it exists to correct.
- `common/progress.py::LoopProgress` has 2 callers; `jax_profile` 8 files
  of ~300; meanwhile `exciton_bands` and `htransform` each REBUILT a
  worse inline timer (`t0 = time.time()` arithmetic) instead of
  importing `timing.section`.
This class includes: timing/profile/progress, `ensure_jax_compile_cache`,
`runtime.padding` helpers, `plan_gflat_chunks`, canonical `*_SPEC`
sharding constants, and the negative idioms (no `getattr(cfg,k,default)`,
no `except Exception`+fallback).

**Class 2 — inline patterns propagated by manual copy.** The idiom is a
SHAPE, not a symbol: the psum_scatter fused contraction is a closure
factory hard-bound to one einsum; the `precompile_*` bodies must
duplicate their caller's exact arg construction (that's the point). These
propagate only by copying — and the fingerprints of copy-decay are
already in the tree: `bse/bse_simple.py:27` and
`wavefunction_bundle.py:380` both cite the sharded project_ri variant at
a module where it does NOT live (it moved to ppm_tau_kernel), while
`bse_ring_comm.py:828` has the correct path and a faithful copy. A
docstring-cross-reference-resolution check would catch all of these.

**Class 3 — idioms needing extraction before they CAN propagate.** The
async-D2H accumulator (`ppm_accumulators._TauAccumulator`) is ~90 lines
of fully domain-free transport machinery welded to ~30 lines of
Σ-specific projection. Nobody has ever copied it — `copy_to_host_async`
appears in exactly 2 files — and its absence is the direct cause of the
serialized `block_until_ready()` inside σ^B's 9-tile loop. Same story
for the sink protocol and (partially) the signature-keyed kernel cache,
whose generic form `wfn_transforms._cached_jit` exists but was never
promoted to common/.

One encouraging data point: where a laggard DID internalize an idiom, it
wrote down why — `bse_ring_comm` documents both its faithful
psum_scatter copy (naming the exemplar) and its deliberate donation
removal (with the reason). Understanding travels fine when the exemplar
is actually seen. The problem is routing, not ability — consistent with
the external evidence (Anthropic: real example beats description; RACG
literature: convention violations are mostly retrieval failures;
instruction-following degrades uniformly past ~150-200 rules, so MORE
PROSE IS NOT AN OPTION).

## 2. The enforcement stack for LORRAX

Ordered by rung (generation-time → hard gate). Each rung names its
day-one content derived from the studies.

**R1 — Scaffolds (kills blank-page re-derivation).** A `new_stage`
skill/script stamping the exemplar skeleton: module-teleology docstring
header, `ensure_jax_compile_cache()` at factory top, signature-keyed
cache dict + sibling `precompile_*` stub, `timing.section` +
`jax_profile` paired nesting, `LoopProgress` in the loop stub, docstring
sharding-table template, refusal-at-entry block. Same for `new_ffi_target`
(the TEMPLATE.md 7-step checklist becomes a script). Platform-engineering
evidence: golden paths work because the right thing is generated, not
remembered.

**R2 — Routed context (kills discovery failure).** 
- `EXEMPLARS.md`: operation → blessed implementation, one line each
  ("async host accumulation → ppm_accumulators._TauAccumulator; fused
  reduce-scatter contraction → ppm_tau_kernel:37; host-cache streaming →
  psi_G_store"). Loaded via path-scoped rules (`.claude/rules` with
  paths: frontmatter) so it costs context only when relevant.
- Every gate/hook error message points at the exemplar (the teaching-
  refusal idiom, now systematized).
- Fix the 3 stale cross-references found (mechanical, day one).

**R3 — Edit-time hooks (the highest-leverage single change).** A
PostToolUse hook on Edit|Write runs the fast checkers (ast-grep rules,
rules_gate, import contracts for the touched package) on the edited file
only, feeding failures back into the agent loop with fix pointers while
intent is still in context. This converts every lower rung from
post-hoc gate (fix = a second session) to in-loop correction (fix = one
turn). Practitioner consensus: feedback-with-next-step beats blocking;
keep it fast; directive message text ("fix autonomously").

**R4 — ast-grep rule corpus (replaces and transcends grep bans).**
Structural YAML rules, one per observed incident, each with `message`
naming the blessed replacement and `fix:` where mechanical. Seed rules,
straight from the audit's detectors:
1. `time.time()`/`perf_counter` arithmetic outside timing/progress/bench
   → use `timing.section` (autofixable).
2. `print` with an i/n counter inside a loop → `LoopProgress`.
3. `getattr(<cfg-like>, '<key>', <default>)` → hard attribute read
   (the w_isdf "opt-out-by-omission" lesson).
4. `except Exception` followed by fallback assignment, not re-raise →
   refuse loudly (σ^B's nan-swallow is the live example).
5. jit dispatch + blocking host read (`block_until_ready`/`device_get`/
   `np.asarray`) in the same loop body → AsyncShardCollector (post-R6
   extraction).
6. kernel factory returning a jit with no module-level cache-dict lookup
   → `_cached_jit`.
7. module-level `_*_cache` holding a jit with no sibling `precompile_*`
   → the pairing rule (catches cohsex + htransform today).
8. inline `(n+d-1)//d`, `math.lcm` mesh alignment, local `_pad_*`
   helpers → `runtime.padding`.
9. ≥3 repeated inline `NamedSharding(mesh, P(...))` literals → import
   the canonical `*_SPEC`.
10. `lax.psum` whose result is resharded on the same axis → the fused
    psum_scatter template.
11. `.addressable_shards[0]`/`addressable_data(0)` without an adjacent
    replication justification (Bug-C class).
12. `main()`/factory dispatching jits without `ensure_jax_compile_cache`.
13. Existing: raw `jnp.fft`, `device_put` (rules_gate migrates in).
Deployed-loop evidence: one-rule-per-incident discipline measured 0%
recurrence of ruled-against classes (arXiv 2607.13091).

**R5 — Architectural contracts.** import-linter (or tach) encoding
L1/L2/L3 as a declarative contract; tach's per-module public-interface
declaration additionally blocks importing a good module's private
internals instead of its blessed entry point. Plus the docstring
cross-reference resolution check (every `module.symbol` cited in a
docstring must resolve) — cheap, and it catches copy-decay.

**R6 — Extraction worklist (turns Class 2/3 into Class 1).** Priority
order, each converts a template-only idiom into an import-checkable one:
1. `AsyncShardCollector(lag=2)` + `Sink` protocol → common/ (the
   `_MemoryTileSink` moves verbatim; its docstring carries the
   make_array_from_single_device_arrays correctness argument that a
   re-implementer would lose).
2. `_cached_jit` (wfn_transforms:84) → common/, becoming THE kernel
   cache (rule 6's autofix target).
3. `make_rs_contraction(mesh, einsums, scatter_dims, specs)` —
   parameterize the psum_scatter factory (3 existing call sites already
   share the structure).
4. Adopt `aot_kernel_peak_bytes` at its four named intended call sites.

**R7 — Shape contracts + capability parity.**
- jaxtyping annotations (`Float[Array, "nk nb nmu"]`) on exemplar public
  functions first — trace-time-only cost under jit, and the convention
  then travels inside every signature an agent copies. New L1/L2 code
  requires them (R4 rule).
- A capability-parity gate for L1 stages (the AHE component-observability
  idea made concrete by the matrix): every registered stage must have
  timing.section (dotted name), a planner peak row, a precompile entry,
  and a docstring sharding table. This is the structural checklist that
  would have flagged σ^B's 0-timing-sections before a 32-node audit had
  to discover it.

**R8 — Behavioral gates (exist; extend).** fastloop, invariance gates
("two paths must agree"), per-stage HLO forbid. The planted-defect
battery (report.md) doubles as the EVAL for this whole stack: seed a
known antipattern, check which rung catches it and how fast.

## 3. What NOT to do

- No second rule engine (Semgrep only if a taint/dataflow rule is ever
  actually needed; one corpus beats two half-corpora).
- No LLM style-judge as primary enforcement (over-flagging erodes trust;
  deterministic rules first, critic pass for judgment-only rules).
- No prose growth: every new convention lands as a rule/template/helper +
  ≤1 pointer line, never a paragraph. The instruction budget is real and
  the sandbox is at it.

## 4. Adoption order

1. R2 stale-ref fixes + EXEMPLARS.md; R4 rules 1-4, 8, 12-13 (pure
   ast-grep, no code changes) with allowlists regenerated post-
   consolidation.
2. R3 hook dispatcher wiring R4+rules_gate (hours; multiplies the rest).
3. R6 extractions 1-2, then enable rules 5-6.
4. R1 new_stage scaffold + R7 capability-parity gate.
5. R5 import contracts; jaxtyping on exemplars, then the new-code rule.
6. R8 planted-defect eval to measure the stack.
