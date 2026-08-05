# RULES_v2 — curated design-rule set with owner verdicts, 2026-08-05

Second pass over RULES_seed.md incorporating the owner's rule-by-rule
review. RULES_seed.md stays as the raw mined evidence (file:line pointers
live there); this file is the proposed FINAL structure: categories anchored
by one overarching principle each, subsidiary rules assessed against it,
and an explicit PLACEMENT for every rule — because a rule agents don't
encounter at the right moment is prose, not a rule.

Dropped by owner verdict: the `np.concatenate` ban (subsumed by the memory
principle), "no new API layers" (superseded by the microservice rule, C2),
the ISDF-rank-excuse rule, "verify the instrument" (sandbox ops epistemics,
not a code rule), "frozen source bundles" (Frontera ops mechanics — lives
in the sandbox launch skill, not the repo rule set).

## 0. Placement map — where a rule lives determines whether it works

| Placement | What goes there | Agent encounter |
|---|---|---|
| `docs/architecture/DESIGN.md` (new, ~2 pages, the read-first doc) | The three-layer split, the scaling doctrine, the O(N³) identity — the WHY that makes every other rule predictable | read once at orientation; linked from AGENTS.md line 1 |
| Scoped `AGENTS.md` per package (`src/common/`, `src/file_io/`, `src/ffi/`, `src/gw/`, `src/bse/`) | Local idiom rules for that layer | auto-loaded when editing there |
| `RULES.md` registry + mechanical gates | Token-checkable bans and structure rules, each row: rule → gate → allowlist | enforced; the gate error message teaches the rule |
| Runtime refusals at driver startup | Run-coupled physics preconditions (category F) | enforced at gwjax/bse runtime; refusal names rule + fix + doc |
| `TASTE.md` | Judgment rules needing review (register, altitude, "flag when a convention fights the task") | critic-pass checklist |

## A. Architecture (→ DESIGN.md, top of file)

**A0 — the three-layer spec (owner's formulation).** Every top-level
driver reads as a description of physics-justified operations. Three
parts: (1) physics-oriented DRIVERS carrying top-level calls; (2)
ALGORITHMS — frequency integration, self-consistent iteration, eigensolver
strategies; (3) PLUMBING — all basic, reusable jax / FFI / communication /
IO machinery. Code moves DOWN this stack as it generalizes, never up.
(The old "main() is a physics outline" rule is the layer-1 corollary; the
Frontera tree's L1/L2/L3 AST gates are the eventual enforcement.)

**A1 — the microservice rule (replaces "no new API layers").** A kernel
routine used in 2+ places with a clear physics or jax-plumbing raison
d'être becomes a microservice: one module, documented API (shapes, units,
shardings, backend behavior), single implementation. Conversely: bespoke
classes as bags of variables are avoided — a class earns existence by
owning an invariant or a resource, not by grouping arguments. The ~30
"single source of truth" docstrings inventoried in RULES_seed §8e are the
existing microservice roster; new candidates join it explicitly.

**A2 — duplicate logic is a defect (generalized).** When a change would
create the 2nd copy of any logic, invoke A1 instead; when you FIND a 2nd
copy, collapsing it is in scope for the current task. (Owner: "good rule,
maybe too specific" — this is the general form.)

**A3 — the O(N³) identity / two-sums rule.** No quantity is ever
assembled by summing over valence-conduction PAIRS; every object is built
from band sums over a single occupancy class combined at an O(N⁰) set of
shared time/frequency nodes. This is the scaling feature of the whole
code; a vc-pair sum anywhere silently reintroduces O(N⁴). Placement:
DESIGN.md headline + TASTE.md checklist item (it is a review-detectable
shape, and a grep for paired vc loop idioms is a plausible soft gate).

**A4 — design-envelope statement (owner: keep, place here).** A perf or
architecture candidate states its expected scaling over the design
envelope — natoms to hundreds, N_mu to tens of thousands, P to thousands,
both backends — BEFORE implementation. Deck-tuned wins invert at scale.

**A5 — flag-don't-bend (reworded from "when a convention forces a bigger
change than the task, flag it").** If honoring a rule in this file would
grow the current task's diff substantially, STOP and surface the conflict
rather than (a) silently violating the rule or (b) silently ballooning
the task. The conflict itself is signal: either the rule needs an owner
exception or the task was mis-scoped. Placement: TASTE.md preamble.

## B. JAX execution discipline ("jaxthonic") (→ DESIGN.md §2 + gates)

**Overarching: the traced graph is the bill.** Everything you write inside
jit is either compiled (pay once per SHAPE) or materialized (pay per
DEVICE per SLOT). All subsidiary rules are instances of minimizing those
two bills; when a new situation isn't covered, reason from the bill.

**B1 — compile economy.** jit the OUTERMOST loop; never jit inside a
Python loop (each iteration with a new shape recompiles). Repetition over
elements is `lax.scan`/jax flow control, not Python iteration. Repeated
operations on slightly-different shapes are PADDED to a common shape when
the op is not flop-dominated — a recompile costs more than the padded
flops. Companion measurement rule: timing/collective tables are taken
cache-cold, but production paths are DESIGNED warm — the number of
distinct compiled signatures per driver is a budget, and the startup
report should print it. (Owner addition 2026-08-05.)

**B2 — flow control and memory interact (moved out of FFT per owner).**
`scan(unroll=1)` inside `shard_map(check_rep=False)` is the pattern for
sharded carries: Python-unrolled loops inside jit pile up N× unsharded
slots; `fori_loop` SPMD-replicates a sharded carry; `scan(unroll>1)`
preallocates N× temporaries. Aliasing of scan-body transients to a single
slot across iterations is load-bearing (the 88 GB solve_zeta OOM).

**B3 — replication threshold made explicit (owner edit).** No intermediate
AS LARGE AS OR LARGER THAN N_mu² or N_b² may materialize replicated (or on
any single rank). Below that threshold, replication is a judgment call,
not a violation. Enforcement: per-stage HLO forbid gate for gather-class
collectives on operands above threshold; `memory_analysis()` assertions in
tests (the `test_bse_stack_matvec` idiom).

**B4 — host arrays and jit boundaries (rewritten for clarity per owner).**
A large host-resident array passed as a jit ARGUMENT is transferred to
every device that runs the computation — P copies of something that only
needed to stream. Large read-only data (ψ(G) etc.) therefore stays on
host in a store and is pulled per-slice INSIDE the jit via `io_callback`
(the PsiGStore pattern). Rule of thumb: if it doesn't fit per-device
comfortably, it enters jit through a callback or a sharded load, never
through the argument list.

**B5 — sharding is declared, then verified.** Functions state shardings
at their boundaries (`in_shardings`/`out_shardings`/
`with_sharding_constraint` at entry and exit — the canonical specs live
with the bundle, not per-consumer). Verification is tracing-based, not
hope-based: AOT-lower and inspect (`jit(f).lower(...).compile()` +
`memory_analysis()`, HLO text scan for gather/transpose classes) whenever
a change touches layouts. Owner directive: build a first-principles
sharding/memory hints page in DESIGN.md §2 — the current scattered hints
under-serve this; include the rematerialization-detection recipe.

**B6 — the 5-second question.** Any operation measured >~5 s gets asked
once: does sharding everything (operands, intermediates, the loop axis)
make it faster? Record the answer where the number is recorded. This is
the standing heuristic form of the campaign's repeated discovery that
unscaled sections dominate walls at scale.

## C. Meshes, processes, environment (→ DESIGN.md §3 + runtime refusals)

**C1 — one 2-D mesh, axes named ('x','y'), one constructor.** No 1-D
band mesh, no hard-coded shapes, no per-module mesh building. (Square-only
ruling on the Frontera tree extends this; carry it into the consolidated
repo.)

**C2 — one JAX process per GPU, always (owner: high priority).** This is
the process model for safe execution across machines; geometries built on
single-process-multi-GPU were deleted once already. Enforcement: runtime
refusal at startup when local device count ≠ 1 in multi-process mode.

**C3 — environment + mesh startup is ONE sequence, not folklore (owner:
warm_mesh_cliques must not stand alone).** Distributed init, device/mesh
resolution, collective warm-up (`warm_mesh_cliques` x/y/world), backend
selection, and the startup report are a single canonical entry point
(`initialize_communicator_stack` on the Frontera tree) that every driver
calls at module top. Drivers never hand-roll any step of it. The rule for
new code: if it touches process/mesh/env state, it belongs inside the
stack, not in a driver.

**C4 — no single-device fallback under multi-host** (the other ranks
deadlock in collectives) — refusal, not warning.

## D. FFT (consolidated per owner; → src/common/AGENTS.md + gates)

**Overarching: one FFT path, shaped for NUFFT.**

- D1: all G↔r transforms through the `common/fft_helpers.py` factories; a
  raw `jnp.fft.*` in stage code is a bug. GATE: banned-token + allowlist.
- D2: k/q are FLAT leading axes, never folded into the FFT grid; the 3-D
  k-shape exists only inside fft_helpers.
- D3: norm='ortho' for physics transforms; 'forward' reserved for the
  CCT/ZCT convolution identity.
- D4: sparse-G ↔ FFT-box moves are precomputed-index GATHERS
  (`jnp.take`), never `.at[].set()` scatters on GPU; the sentinel scheme
  requires the guaranteed zero slot. (Moved here from I/O per owner.)
- D5: chunk FFT batches only over local batch axes, never spatial axes;
  chunk counts are static with a hard ==1 fast path.

## E. Linalg, FFI, backends (→ src/ffi/AGENTS.md + contract tests)

**Overarching: one dispatch seam, safe on both backends, safe at any
rank count.**

**E1 — unified linalg wrappers (owner: high priority).** Main gw/bse code
NEVER calls a backend library directly; it calls the linalg dispatch layer
which selects CPU/GPU/distributed implementations safely and REFUSES
loudly where a backend lacks the capability. New linalg needs = extend the
dispatcher, both backends or an explicit loud gap.

**E2 — divisibility safety is a contract, not a hope (owner: extremely
important).** Every FFI I/O and linalg path must be correct for matrix
ranks NOT divisible by the process count, via carefully-implemented
padding (logical-extent solves, exactly-zero pad rows, bounds tested on
the logical slab). Standing test policy: examples and test geometries are
CONSTRUCTED with non-divisible ranks so the padded path is always the
exercised path. (The Frontera SlabIO audit — one-refusing-rank collective
hang, silent-zero reads — is the incident record behind this rule.)

**E3 — batched/distributed eigh retention.** The distributed eigh path
may be SLOWER than independent per-process eighs at small size; it is
kept and maintained regardless when requested, because it is the only
path that scales to large systems. Do not delete or de-prioritize it on
the strength of small-deck timings (record: the documented-backwards
tier framing incident).

**E4 — FFI subpackage literacy + no reinvention.** `src/ffi/common/`
primitives (loader, broadcast, descriptors, helpers) are never
reimplemented; every agent touching FFI reads the subpackage index first
(place a 20-line inventory at the top of src/ffi/AGENTS.md). Related
mechanical rules: caller-varying values are runtime Args never
compile-time Attrs; layout preconditions validated Python-side BEFORE the
FFI call (native exceptions kill all ranks); collective HDF5 paths
byte-identical across ranks (broadcast from rank 0).

## F. Run-coupled physics preconditions (→ RUNTIME REFUSALS, per owner)

Owner directive: these must be read AT GWJAX RUNTIME, not in docs. Each
becomes a startup/stage-entry check with the standard refusal idiom (F0).

**F0 — the standard refusal (owner: "a standard way will make it work
more often").** One helper — `refuse(rule_id, got, want, fix, doc)` — so
every refusal names the rule, the observed value, the required condition,
the concrete fix, and the doc anchor, uniformly. Unsupported paths always
refuse loudly through it; a parsed-but-ignored config key is a defect
class with its own gate (unknown keys raise, the typo'd-name-must-raise
principle generalized from the PPM path).

- F1: centroid-selection band window spans the sigma band window (exists:
  rank_criterion on the Frontera tree — port pattern to all below).
- F2: `dipole.h5` provenance-stamped with its band window; consumers
  refuse on mismatch (replaces the "regenerate on change" convention).
- F3: htransform requires ALL valence bands, never a subset (owner
  addition).
- F4: band subspaces must not break degeneracies — existing code does
  this; NEW code gets the check as a shared helper + refusal (owner
  addition).
- F5: TRS scoping (owner's formulation): TRS augmentation is legitimate
  inside irreducible-wedge unfolding routines; it is NEVER applied inside
  GW convolution steps — anything entering a k-grid convolution must
  already be on the unfolded k set. Enforcement: the wedge helpers are
  the only importers of TRS machinery (AST-checkable) + the existing
  loud-raise on unmapped k.

## G. Symmetry & BZ (→ src/common/AGENTS.md + DESIGN.md pointer)

- G1: one IBZ table (`SymMaps`) + one sym-action helper; no per-object
  rotate-at-q variants. **Owner addition: document the canonical idiom
  pair — how k_irr quantities are extracted from k_full, and how k_full
  quantities are reconstructed from k_irr — as named functions with a
  worked example, in the same place the rule is stated.** (The rule
  without the how-to is why 6 parallel helpers accumulated.)
- G2: BGW conventions are the composition target: r-action form, inverse
  permutation direction (never argsort-of-forward), q-wrap, G=0 in slot
  0, zeroed pad slots, translations stored in raw 2π·τ form with
  consumers dividing.
- G3: centroid sets are orbit-closed; recovered symmetry groups may only
  ever be ENLARGED relative to the stored group, never downgraded.

## H. Comparisons & reporting (→ skills/compare + TASTE.md)

Kept intact from the sweep (owner raised no objections): explicit
`bare_coulomb_cutoff` in every BGW comparison; compared runs share
windows/cutoffs/budget AND the physical centroid file; the six BSE-vs-BGW
conventions; total Σ_c only, never branch-by-branch; gauge-invariant
scalars only for eigenvector comparisons; gate metrics must respond to a
perturbation of the thing they claim to measure.

## I. Evidence linking — the jobid question (owner asked for a view)

Current practice: claims and docs cite raw jobids + /scratch paths.
Assessment: right EVIDENCE key, wrong REFERENCE unit for agents — a jobid
is opaque off-cluster, dereferenceable only via a 9,400-line grep, and its
artifacts sit on purgeable scratch. Proposal:

1. The unit of citation becomes the claim file (`claims/NNNN.md`), which
   CONTAINS the jobid(s), artifact paths, and — critically — an EXCERPT of
   the load-bearing numbers (the timing rows, the parity deltas), captured
   at landing time while the artifacts still exist. Docs and code comments
   then cite `CLAIMS-48`, never a bare jobid: one hop, readable anywhere,
   purge-proof.
2. `runs/metrics.jsonl` keyed by jobid becomes the machine-readable
   dereference for numbers (report.md §3.7), so trend queries never touch
   scratch either.
3. Bare jobids remain in exactly one place: inside claim files, as the
   ground-truth pointer for audits.

This keeps rule 1's discipline (every claim → job → disk) while making the
chain survivable and one-hop for the agents who actually follow it.

## J. Adoption order

1. DESIGN.md (A0–A5, B overarching + B1–B6, C1–C4) — one doc, the top of
   the read order; this is where the owner's three-layer spec and the
   first-principles sharding/memory page live.
2. F0 refusal helper + F1–F5 runtime checks (port the rank_criterion
   pattern) — these bind at runtime regardless of what anyone reads.
3. Gates: D1 (jnp.fft ban), device_put ban, E-layer direct-backend-call
   ban, unknown-config-key raise. Each with allowlist ratchet.
4. Scoped AGENTS.md drops (D→common, E→ffi, G→common) + the G1 idiom
   documentation.
5. TASTE.md seeding (A3, A5, H) and the claim-file citation switch (I).
