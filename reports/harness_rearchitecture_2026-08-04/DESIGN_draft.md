# LORRAX design spec (DRAFT — proposed docs/architecture/DESIGN.md)

Drafted 2026-08-05 from RULES_v2 categories A-C. This is the top of the
read order: the WHY that makes the enforced rules predictable. Target
length after owner edit: ~2 pages. Everything mechanical referenced here
is (or will be) enforced by a gate or a runtime refusal; this doc exists
so those refusals make sense the first time you hit one.

## 1. The three layers

Every piece of LORRAX belongs to exactly one layer:

1. **Drivers** (gw_jax, bse_jax, kmeans_cli, htransform, ...): read as a
   description of physics-justified operations. A driver's main() is a
   sequence of named stage calls (ζ-fit → V_q → χ₀/W → Σ → eqp) — no
   inlined machinery, no environment plumbing (that comes from the
   runtime stack, §4).
2. **Algorithms**: frequency integration, self-consistent iteration,
   eigensolver strategies, screening pipelines. Physics-aware, but
   reusable across drivers.
3. **Plumbing**: reusable jax / FFI / communication / IO machinery with
   no physics content — FFT factories, loaders, SlabIO, collectives,
   padding, dispatch.

Code moves DOWN this stack as it generalizes, never up. Imports point
down only (a plumbing module never imports an algorithm or driver).

**The microservice rule.** A kernel routine used in 2+ places with a
clear physics or jax-plumbing raison d'être becomes a microservice: one
module, one implementation, documented API (shapes, units, shardings,
backend behavior). Conversely, bespoke classes as bags of variables are
avoided — a class earns existence by owning an invariant or a resource,
not by grouping arguments. The current roster is the ~30 "single source
of truth" modules (fft_helpers, WfnLoader, SlabIO, SymMaps, units,
kq_mapping, coulomb_sphere, runtime.padding, ...); new candidates join
it explicitly, and the second copy of any logic is the signal to do so.

## 2. The scaling identity

**The two-sums rule: no quantity is ever assembled by summing over
valence-conduction PAIRS.** Every object is built from band sums over a
single occupancy class, combined at an O(N⁰) set of shared time or
frequency nodes. This is what makes LORRAX an O(N³) code; one vc-pair
loop anywhere silently reintroduces O(N⁴). BSE matvecs and bispinor
channels inherit the rule unchanged.

**The design envelope.** natoms to hundreds, N_mu to tens of thousands,
P to thousands, CPU and GPU. Any performance or architecture candidate
states its expected scaling over this envelope BEFORE implementation —
deck-tuned wins invert at scale, and the ledger holds several examples.

## 3. JAX discipline: the traced graph is the bill

Everything inside jit is either compiled (paid once per SHAPE) or
materialized (paid per DEVICE per SLOT). Every rule below is an instance
of minimizing one of those two bills; for situations not covered, reason
from the bill.

**Compile economy.**
- jit the outermost loop. Never jit inside a Python loop; repetition
  over elements is lax.scan / jax flow control.
- Repeated operations on slightly different shapes are PADDED to a
  common shape when the op is not flop-dominated: a recompile costs more
  than the padded flops.
- The number of distinct compiled signatures per driver is a budget;
  the startup report prints it. Measure timings cache-cold, design
  production paths warm.

**Flow control × memory.**
- scan(unroll=1) inside shard_map(check_rep=False) is the pattern for
  sharded carries. Python-unrolled loops inside jit pile up N× unsharded
  slots; fori_loop SPMD-replicates a sharded carry; scan(unroll>1)
  preallocates N× temporaries. Scan-body transients aliasing to a single
  slot across iterations is load-bearing (the 88 GB solve_zeta OOM).

**Memory.**
- No intermediate ≥ N_mu² or N_b² materializes replicated or on any
  single rank. Below that threshold, replication is a judgment call.
- Large read-only host data (ψ(G), ...) never enters jit through the
  argument list — a jit argument is transferred to every device. It
  stays in a host store and streams per-slice via io_callback
  (the PsiGStore pattern).
- Device data ingress goes through the FFI I/O layer / sharded loaders;
  jax.device_put is banned (gate B4).

**Sharding is declared, then verified.**
- Functions state shardings at their boundaries: in_shardings /
  out_shardings / with_sharding_constraint at entry and exit. Canonical
  specs live with the bundle (wavefunction_bundle), not per-consumer, so
  a mismatch is caught at import time.
- Verification recipe (run it whenever a change touches layouts):
  1. lowered = jax.jit(f, in_shardings=..., out_shardings=...).lower(*abstract_args)
  2. compiled = lowered.compile(); compiled.memory_analysis() — check
     temp_size against the slot budget (the test_bse_stack_matvec idiom:
     assert peak temp flat in the batch axis).
  3. Scan compiled.as_text() for gather-class ops on ≥N_mu²-class
     operands (all-gather, all-reduce on full tiles) and for
     transpose/copy layout churn — the head_wing_schur idiom, and
     tools/hlo/analyze_hlo_dump.py --forbid for whole-driver dumps
     (cache-cold only: warm caches skip modules and under-report).
- **The 5-second question**: any operation measured >~5 s gets asked
  once — does sharding everything (operands, intermediates, the loop
  axis) beat it? Record the answer next to the number.

## 4. Meshes, processes, environment

- ONE 2-D mesh with axes named ('x','y'), built by one constructor.
  No 1-D band mesh, no hard-coded shapes, no per-module mesh building.
  Meshes are square-only; non-square device counts refuse, naming the
  count to request.
- ONE JAX process per GPU, always. Geometries built on
  single-process-multi-GPU were deleted once already; refusal at startup.
- Environment + mesh startup is ONE sequence — distributed init, mesh
  resolution, collective warm-up, backend gates, startup report — via
  the runtime initialization stack, called once at driver module top.
  New code touching process/mesh/env state belongs inside the stack,
  never in a driver.
- No single-device fallback under multi-host (the other ranks deadlock
  in collectives): refusal, not warning.
- Main gw/bse code calls the unified linalg dispatch layer, never a
  backend library directly; unsupported combinations refuse loudly
  through the standard refusal helper. Every FFI I/O and linalg path is
  correct for ranks NOT divisible by the process count, and test
  geometries are constructed non-divisible so the padded path is always
  the exercised path.

## 5. When a rule fights the task

If honoring a rule in this document would substantially grow the current
task's diff: STOP and surface the conflict. Do not silently violate the
rule; do not silently balloon the task. The conflict is signal — either
the rule needs an owner exception (record it), or the task was
mis-scoped.
